# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Gradients of whole queries on DataFusion (ddx v2).

Write a model's forward pass and loss as SQL, and take its gradient in SQL::

    from ddxdb import ad

    ad.sql(ctx, '''
        WITH loss AS (SELECT SUM(power(x.v * w.val - y.v, 2)) AS l
                      FROM x JOIN w ON x.i = w.i JOIN y ON x.s = y.s)
        SELECT w.i, w.val - 0.1 * g.val AS val
        FROM w JOIN grad(loss, w.val) g ON w.i = g.i
    ''')

``grad(loss, table.column)`` in a ``FROM`` clause is the gradient of the loss
the CTE ``loss`` computes: a relation shaped like the table, its dims and the
named columns' gradients. An SGD step is then a join, ``params - lr *
grad(loss)(params)``. :class:`ddxdb.Context` understands the same syntax in its
``.sql()``.

Underneath, :func:`grad` turns a loss query into a :class:`BackwardProgram` of
Substrait plans, and :func:`run` runs them on the context in order, registering
each result as a table. :func:`vjp` pulls back a cotangent of any query's
output. Nothing in the query is labelled for any of this; the one function ddx
gives a meaning to is ``ddx_stop_gradient(x)``, JAX's ``lax.stop_gradient``
(:func:`register_stop_gradient`).

Every table a program writes is named with a prefix unique to that program,
``__ddx_{id}_``, so two programs on one context never read each other's tables,
and a user's table is never replaced unless its name starts with ``__ddx_``,
which is reserved. After :func:`run`, only the value and the gradients remain on
the context; :func:`release` drops those too.

One limit Rust's ``ddx_datafusion::ad`` does not share: datafusion-python's
Substrait consumer names a computed column by its whole expression, so a deep
chain of row-wise maps that each read their input twice (twenty
``sin(v) + 0.1 * v`` layers, say) makes a very large plan here. The Rust
adapter consumes steps with short names.

Importing this module requires DataFusion.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator, Sequence

try:
    import pyarrow as pa
    from datafusion import SessionContext, udf
    from datafusion.substrait import Consumer, Serde
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError("ddxdb.ad needs DataFusion: pip install 'ddxdb[datafusion]'") from e

from ._ddxdb import (
    InvalidColumn,
    _bind_reads,
    _grad,
    _Statements,
    _unbound_reads,
    _vjp,
)

__all__ = [
    "STOP_GRADIENT",
    "BackwardProgram",
    "Check",
    "Gradient",
    "Step",
    "grad",
    "register_stop_gradient",
    "release",
    "run",
    "run_checks",
    "run_step",
    "sql",
    "sql_all",
    "vjp",
]

#: The SQL name of the stop-gradient function.
STOP_GRADIENT = "ddx_stop_gradient"


@dataclasses.dataclass(frozen=True)
class Step:
    """A plan to run, and the table name to register its result as."""

    name: str
    plan: bytes  # a serialized Substrait plan


@dataclasses.dataclass(frozen=True)
class Check:
    """A plan that must return no rows, and what a row means."""

    plan: bytes  # a serialized Substrait plan
    message: str


@dataclasses.dataclass(frozen=True)
class Gradient:
    """Where a ``wrt`` table's gradient lands.

    ``columns`` are the table's dims, then its ``wrt`` values, named as in the
    table; each value column holds the gradient: ``0`` where none reached, and
    ``NULL`` where the value itself is ``NULL``.
    """

    table: str
    step: str
    columns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class BackwardProgram:
    """The steps that compute a query's value and its gradient."""

    forward_steps: tuple[Step, ...]
    value: str  # the step holding the query's own result
    cotangent_table: str  # vjp: the table the caller registers the cotangent as
    cotangent: tuple[str, ...]  # vjp: that table's columns
    checks: tuple[Check, ...]  # run before the steps; each must return no rows
    backward_steps: tuple[Step, ...]
    gradients: tuple[Gradient, ...]
    # The same program on the Rust side, which runs the run protocol
    # (ddx_ad::Runner) and plans sql_all's statements.
    _handle: object = dataclasses.field(default=None, repr=False, compare=False)

    def steps(self) -> Iterator[Step]:
        """Every step, in the order they must run."""
        yield from self.forward_steps
        yield from self.backward_steps

    def intermediate_steps(self) -> Iterator[Step]:
        """The steps only other steps read: the saved aggregates and the
        cotangents. :func:`run` drops them once the gradients are written."""
        keep = {self.value, *(g.step for g in self.gradients)}
        return (s for s in self.steps() if s.name not in keep)


def register_stop_gradient(ctx: SessionContext) -> None:
    """Register ``ddx_stop_gradient`` on ``ctx`` as the identity on DOUBLE.

    DataFusion keeps one UDF per name, and a Python UDF takes exact types, so
    this one takes DOUBLE: an argument of another numeric type is cast to
    DOUBLE, and so is the result. The gradient is unaffected, since ddx treats
    the argument as a constant whatever it contains, but the type is not:
    ``val - ddx_stop_gradient(val)`` on a ``REAL`` column is ``DOUBLE`` here,
    where the Rust UDF (``ddx_datafusion::stop_gradient_udf``), which accepts
    any type, keeps it ``REAL``. Cast back if the type matters.
    """
    ctx.register_udf(udf(lambda x: x, [pa.float64()], pa.float64(), "immutable", name=STOP_GRADIENT))


def _program(raw_and_handle) -> BackwardProgram:
    raw, handle = raw_and_handle
    forward, value, cotangent_table, cotangent, checks, backward, gradients = raw
    return BackwardProgram(
        forward_steps=tuple(Step(n, p) for n, p in forward),
        value=value,
        cotangent_table=cotangent_table,
        cotangent=tuple(cotangent),
        checks=tuple(Check(p, m) for p, m in checks),
        backward_steps=tuple(Step(n, p) for n, p in backward),
        gradients=tuple(Gradient(t, s, tuple(c)) for t, s, c in gradients),
        _handle=handle,
    )


def grad(ctx: SessionContext, sql: str, wrt: Sequence[tuple[str, str]]) -> BackwardProgram:
    """The gradient of the loss ``sql`` computes, with respect to the
    ``(table, column)`` pairs in ``wrt``. The query must return one row and
    one column.

    Raises a :class:`ddxdb.DdxError` subclass when ddx cannot differentiate
    the query: :class:`ddxdb.NotScalar` for a query that is not a loss,
    :class:`ddxdb.UnknownColumn` for a ``wrt`` it does not read, and so on.
    """
    return _program(_grad(Serde.serialize_bytes(sql, ctx), [tuple(w) for w in wrt]))


def vjp(ctx: SessionContext, sql: str, wrt: Sequence[tuple[str, str]]) -> BackwardProgram:
    """The vector-Jacobian product of the query ``sql``. Before running the
    backward steps, register the cotangent as the table
    ``program.cotangent_table`` names, with the columns ``program.cotangent``
    lists."""
    return _program(_vjp(Serde.serialize_bytes(sql, ctx), [tuple(w) for w in wrt]))


def run(ctx: SessionContext, program: BackwardProgram) -> None:
    """Run ``program``: its checks, then every step, registering each result on
    ``ctx``. Once the gradients are written, the intermediate tables are
    dropped; the value and the gradients stay until :func:`release` or the next
    run replaces them.

    Raises :class:`ddxdb.InvalidColumn` when a check fails: a ``wrt`` table's
    rows are not what the program assumed, for instance two share their dims.

    A program depends on the tables' names and schemas, not their values, so
    build it once and run it on every training step. One program's runs must
    not overlap, since they write the same tables.
    """
    # The order of a run, and what it drops when, are ddx_ad::Runner's: this
    # loop only does what it is asked.
    runner = program._handle.runner()
    steps = list(program.steps())
    while (action := runner.next()) is not None:
        kind, arg = action
        try:
            if kind == "check":
                runner.report(_returns_rows(ctx, program.checks[arg].plan))
            elif kind == "step":
                run_step(ctx, steps[arg])
                runner.report(False)
            else:
                ctx.deregister_table(arg)
                runner.report(False)
        except Exception as e:  # handed to the runner, which raises it in finish()
            runner.fail(e)
    runner.finish()


def run_checks(ctx: SessionContext, program: BackwardProgram) -> None:
    """Run ``program``'s checks, raising :class:`ddxdb.InvalidColumn` on the
    first that returns a row."""
    for check in program.checks:
        if _returns_rows(ctx, check.plan):
            raise InvalidColumn(f"invalid wrt column: {check.message}")


def _returns_rows(ctx: SessionContext, plan: bytes) -> bool:
    return ctx.create_dataframe_from_logical_plan(_consume(ctx, plan)).limit(1).count() > 0


def release(ctx: SessionContext, program: BackwardProgram) -> None:
    """Drop every table ``program`` registered on ``ctx``, the value and the
    gradients included."""
    for step in program.steps():
        ctx.deregister_table(step.name)


def run_step(ctx: SessionContext, step: Step) -> None:
    """Run one step and register its result as a table, replacing any table of
    that name. Every step it reads must already be registered. Unlike
    :func:`run`, this neither runs the checks nor drops anything."""
    table = ctx.create_dataframe_from_logical_plan(_consume(ctx, step.plan)).to_arrow_table()
    _register(ctx, step.name, table)


def _consume(ctx: SessionContext, plan: bytes):
    """``plan`` (a step or a check) as a DataFusion logical plan. Its reads of
    earlier steps, or of a vjp's cotangent, name their columns but not their
    types; the types come from the tables themselves, as the engine states
    them."""
    schemas = {
        name: Serde.serialize_bytes(f'SELECT * FROM "{name}"', ctx)
        for name in _unbound_reads(plan)
    }
    return Consumer.from_substrait_plan(ctx, Serde.deserialize_bytes(_bind_reads(plan, schemas)))


def _register(ctx: SessionContext, name: str, table: pa.Table) -> None:
    # Registered as record batches (an in-memory table), not a view:
    # DataFusion's Substrait consumer can only pick columns out of a table scan.
    batches = table.to_batches() or [pa.RecordBatch.from_pylist([], schema=table.schema)]
    ctx.deregister_table(name)
    ctx.register_record_batches(name, [batches])


def sql(ctx: SessionContext, statement: str):
    """Run ``statement``, in which ``grad(loss, table.column, …)`` in a ``FROM``
    clause is the gradient of the loss the CTE ``loss`` computes, as a relation
    shaped like ``table``. Returns a DataFrame.

    Each loss's program runs first, once, however many calls use it. A
    statement with no such call is planned as it is. The programs' tables are
    dropped once the statement is planned; the DataFrame keeps what it reads.
    A loss defined in a ``WITH RECURSIVE`` clause is refused.
    """
    return sql_all(ctx, [statement])[0]


def sql_all(ctx: SessionContext, statements: Sequence[str]) -> list:
    """:func:`sql` for several statements. Statements that take ``grad`` of the
    same loss query share one run of its program, so a training step can update
    each parameter table with its own statement and pay for the backward pass
    once."""
    # Which programs to run and how each statement reads their gradients are
    # ddx_ad::sql::Statements'; running them and planning the result are
    # DataFusion's.
    planned = _Statements(list(statements))
    ran: list[BackwardProgram] = []
    try:
        for query, wrt in planned.jobs():
            program = grad(ctx, query, wrt)
            ran.append(program)
            run(ctx, program)
        return [ctx.sql(s) for s in planned.rewrite([p._handle for p in ran])]
    finally:
        # A planned DataFrame holds the tables it reads, so the programs'
        # tables can leave the catalog now.
        for program in ran:
            release(ctx, program)
