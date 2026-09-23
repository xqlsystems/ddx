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

Importing this module requires DataFusion.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Iterator, Sequence

try:
    import pyarrow as pa
    from datafusion import SessionContext, udf
    from datafusion.substrait import Consumer, Serde
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError("ddxdb.ad needs DataFusion: pip install 'ddxdb[datafusion]'") from e

from ._ddxdb import (
    _bind_reads,
    _find_grad_calls,
    _grad,
    _rewrite_grad_calls,
    _unbound_reads,
    _vjp,
)

__all__ = [
    "COTANGENT",
    "STOP_GRADIENT",
    "VALUE",
    "BackwardProgram",
    "Gradient",
    "Step",
    "grad",
    "register_stop_gradient",
    "run",
    "run_step",
    "sql",
    "sql_all",
    "vjp",
]

#: The SQL name of the stop-gradient function.
STOP_GRADIENT = "ddx_stop_gradient"
#: The step holding a program's value: the query's own result.
VALUE = "__ddx_value"
#: The table a vjp program reads the output's cotangent from.
COTANGENT = "__ddx_cotangent"


@dataclasses.dataclass(frozen=True)
class Step:
    """A plan to run, and the table name to register its result as."""

    name: str
    plan: bytes  # a serialized Substrait plan


@dataclasses.dataclass(frozen=True)
class Gradient:
    """Where a ``wrt`` table's gradient lands.

    ``columns`` are the table's dims, then its ``wrt`` values, named as in the
    table; each value column holds the gradient, and ``0`` where none reached.
    """

    table: str
    step: str
    columns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class BackwardProgram:
    """The steps that compute a query's value and its gradient."""

    forward_steps: tuple[Step, ...]
    value: str  # the step holding the query's own result
    cotangent: tuple[str, ...]  # vjp: the columns of the COTANGENT table it reads
    backward_steps: tuple[Step, ...]
    gradients: tuple[Gradient, ...]

    def steps(self) -> Iterator[Step]:
        """Every step, in the order they must run."""
        yield from self.forward_steps
        yield from self.backward_steps


def register_stop_gradient(ctx: SessionContext, types: Iterable[pa.DataType] = (pa.float64(),)) -> None:
    """Register ``ddx_stop_gradient`` on ``ctx`` as the identity, for ``types``."""
    for t in types:
        ctx.register_udf(udf(lambda x: x, [t], t, "immutable", name=STOP_GRADIENT))


def _program(raw) -> BackwardProgram:
    forward, value, cotangent, backward, gradients = raw
    return BackwardProgram(
        forward_steps=tuple(Step(n, p) for n, p in forward),
        value=value,
        cotangent=tuple(cotangent),
        backward_steps=tuple(Step(n, p) for n, p in backward),
        gradients=tuple(Gradient(t, s, tuple(c)) for t, s, c in gradients),
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
    backward steps, register the cotangent as the table :data:`COTANGENT`, with
    the columns ``program.cotangent`` lists."""
    return _program(_vjp(Serde.serialize_bytes(sql, ctx), [tuple(w) for w in wrt]))


def run(ctx: SessionContext, program: BackwardProgram) -> None:
    """Run every step of ``program``, registering each result on ``ctx``.

    A program depends on the tables' names and schemas, not their values, so
    build it once and run it on every training step.
    """
    for step in program.steps():
        run_step(ctx, step)


def run_step(ctx: SessionContext, step: Step) -> None:
    """Run one step and register its result as a table, replacing any table of
    that name. Every step it reads must already be registered."""
    # A step's reads of earlier steps name their columns but not their types;
    # the types come from the tables themselves, as the engine states them.
    schemas = {
        name: Serde.serialize_bytes(f'SELECT * FROM "{name}"', ctx)
        for name in _unbound_reads(step.plan)
    }
    bound = _bind_reads(step.plan, schemas)
    logical = Consumer.from_substrait_plan(ctx, Serde.deserialize_bytes(bound))
    table = ctx.create_dataframe_from_logical_plan(logical).to_arrow_table()
    _register(ctx, step.name, table)


def _register(ctx: SessionContext, name: str, table: pa.Table) -> None:
    # Registered as record batches (an in-memory table), not a view:
    # DataFusion's Substrait consumer can only pick columns out of a table scan.
    batches = table.to_batches() or [pa.RecordBatch.from_pylist([], schema=table.schema)]
    ctx.deregister_table(name)
    ctx.register_record_batches(name, [batches])


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql(ctx: SessionContext, statement: str):
    """Run ``statement``, in which ``grad(loss, table.column, …)`` in a ``FROM``
    clause is the gradient of the loss the CTE ``loss`` computes, as a relation
    shaped like ``table``. Returns a DataFrame.

    Each loss's program runs first, once, however many calls use it. A
    statement with no such call is planned as it is.
    """
    return sql_all(ctx, [statement])[0]


def sql_all(ctx: SessionContext, statements: Sequence[str]) -> list:
    """:func:`sql` for several statements. Statements that take ``grad`` of the
    same loss query share one run of its program, so a training step can update
    each parameter table with its own statement and pay for the backward pass
    once."""
    found = [_find_grad_calls(s) for s in statements]

    # One program per distinct loss query, with respect to every column any
    # statement asks about.
    queries: list[str] = []
    wrts: list[list[tuple[str, str]]] = []
    program_of: dict[tuple[int, int], int] = {}
    for s, calls in enumerate(found):
        if calls is None:
            continue
        losses, _ = calls
        for l, (_, query, wrt) in enumerate(losses):
            if query not in queries:
                queries.append(query)
                wrts.append([])
            p = queries.index(query)
            for w in wrt:
                if tuple(w) not in wrts[p]:
                    wrts[p].append(tuple(w))
            program_of[(s, l)] = p

    # Run each, keeping its gradients under names of their own: the next
    # program reuses the step names.
    kept: dict[tuple[int, str], tuple[str, tuple[str, ...]]] = {}
    for p, (query, wrt) in enumerate(zip(queries, wrts)):
        program = grad(ctx, query, wrt)
        run(ctx, program)
        for g in program.gradients:
            name = f"__ddx_grad_{p}_{g.table.replace('.', '_')}"
            _register(ctx, name, ctx.table(g.step).to_arrow_table())
            kept[(p, g.table)] = (name, g.columns)

    frames = []
    for s, statement in enumerate(statements):
        if found[s] is None:
            frames.append(ctx.sql(statement))
            continue
        _, calls = found[s]
        relations = []
        for loss, table, columns in calls:
            p = program_of[(s, loss)]
            match = next(
                (v for (q, t), v in kept.items()
                 if q == p and (t == table or t.split(".")[-1].lower() == table.lower())),
                None,
            )
            if match is None:
                raise RuntimeError(f"no gradient was computed for {table!r}")
            name, all_columns = match
            values = sum(1 for t, c in wrts[p] if t.lower() == table.lower())
            dims = list(all_columns[: len(all_columns) - values])
            asked = [v for c in columns for v in all_columns[len(dims):] if v.lower() == c.lower()]
            picked = ", ".join(_quote(c) for c in dims + asked)
            relations.append(f"(SELECT {picked} FROM {_quote(name)})")
        frames.append(ctx.sql(_rewrite_grad_calls(statement, relations)))
    return frames
