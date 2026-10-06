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
grad(loss)(params)``. ``jvp(f, table.column, tangent)`` is the CTE ``f``'s
output with, beside each column, its tangent along ``tangent`` (a CTE or a
table shaped like ``table``), as ``{column}_tangent``::

    ad.sql(ctx, '''
        WITH loss AS (SELECT SUM(val * val) AS l FROM w),
             v AS (SELECT i, 1.0 AS val FROM w)
        SELECT l, l_tangent FROM jvp(loss, w.val, v)
    ''')

Several tables' tangents go in one call, each after its columns:
``jvp(f, w.val, dw, b.val, db)``. :class:`ddxdb.Context` understands the
same syntax in its ``.sql()``.

Underneath, :func:`grad` turns a loss query into a :class:`BackwardProgram` of
Substrait plans, and :func:`run` runs them on the context in order, registering
each result as a table. :func:`vjp` pulls back a cotangent of any query's
output, and :func:`jvp` pushes a tangent forward (a :class:`ForwardProgram`).
Each takes SQL or a program, so they compose: ``jvp(ctx, grad(ctx, loss,
wrt), wrt)`` gives Hessian-vector products. Both kinds of program share one
vocabulary: the tables the caller registers (:class:`InputTable`) and the
tables a program leaves (:class:`OutputTable`). Nothing in the query is
labelled for any of this; the one function ddx gives a meaning to is
``ddx_stop_gradient(x)``, JAX's ``lax.stop_gradient``
(:func:`register_stop_gradient`).

Every table a program writes is named with a prefix unique to that program,
``__ddx_{id}_``, so two programs on one context never read each other's tables,
and a user's table is never replaced unless its name starts with ``__ddx_``,
which is reserved. After :func:`run`, only the value and the gradients remain on
the context; :func:`release` drops those too.

Two limits Rust's ``ddx_datafusion::ad`` does not share. datafusion-python's
Substrait consumer names a computed column by its whole expression, so a deep
chain of row-wise maps that each read their input twice (twenty
``sin(v) + 0.1 * v`` layers, say) makes a very large plan here; the Rust
adapter consumes steps with short names. And datafusion-python decodes a plan
with protobuf's default limit of 100 nested messages, two per relation, so a
step more than about 48 relations deep is refused
(:class:`ddxdb.UnsupportedExpression`): ``jvp`` of a three-layer MLP's loss
fits, nn.py's does not yet.

Importing this module requires DataFusion.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

try:
    import pyarrow as pa
    from datafusion import SessionContext, udf
    from datafusion.substrait import Consumer, Serde
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError("ddxdb.ad needs DataFusion: pip install 'ddxdb[datafusion]'") from e

from ._ddxdb import (
    InvalidColumn,
    UnsupportedExpression,
    _bind_reads,
    _not_in_subquery,
    _Statements,
    _unbound_reads,
)
from . import Backend
from ._ddxdb import (
    BackwardProgram,
    Check,
    ForwardProgram,
    InputTable,
    OutputTable,
    Step,
    Tangent,
    grad_plan,
    jvp_plan,
    vjp_plan,
)
from ._ddxdb import run as _run

#: A program of either kind.
Program = Union[BackwardProgram, ForwardProgram]

__all__ = [
    "STOP_GRADIENT",
    "Backend",
    "BackwardProgram",
    "DataFusionBackend",
    "Check",
    "ForwardProgram",
    "InputTable",
    "OutputTable",
    "Program",
    "Step",
    "Tangent",
    "grad",
    "jvp",
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


def grad(
    ctx: SessionContext,
    of: Union[str, Program],
    wrt: Sequence[tuple[str, str]],
    *,
    namespace: Optional[str] = None,
) -> BackwardProgram:
    """The gradient of ``of`` with respect to the ``(table, column)`` pairs in
    ``wrt``: of SQL, the loss the query computes, which must be one row and
    one column; of a program, its one output table, which must be too (its
    gradient if it has one, otherwise its value).

    Raises a :class:`ddxdb.DdxError` subclass when ddx cannot differentiate
    the query: :class:`ddxdb.NotScalar` for a query that is not a loss,
    :class:`ddxdb.UnknownColumn` for a ``wrt`` it does not read, and so on.
    ``namespace`` fixes the prefix of the tables the program writes (see
    :func:`ddxdb.grad_plan`). The same plan then gives the same
    program; the same query need not, since DataFusion's producer can number
    its functions differently from one call to the next.
    """
    return grad_plan(_plan_of(ctx, of), wrt, namespace=namespace)


def vjp(
    ctx: SessionContext,
    of: Union[str, Program],
    wrt: Sequence[tuple[str, str]],
    *,
    namespace: Optional[str] = None,
) -> BackwardProgram:
    """The vector-Jacobian product of ``of``, SQL or a program's one output
    table. Before it runs, register the cotangent as the input table
    ``program.inputs`` gives for the output (``of`` is ``None``), with the
    columns it lists; of a program, that program's own inputs come first."""
    return vjp_plan(_plan_of(ctx, of), wrt, namespace=namespace)


def jvp(
    ctx: SessionContext,
    of: Union[str, Program],
    wrt: Sequence[tuple[str, str]],
    *,
    namespace: Optional[str] = None,
) -> ForwardProgram:
    """The Jacobian-vector product of ``of`` with respect to the ``(table,
    column)`` pairs in ``wrt``: of SQL, a program computing the query's output
    and, beside it, its tangent (``program.value.tangents`` says which column
    holds which); of a program, its steps rewritten, so that of a
    :func:`grad` program each gradient's tangent is the Hessian-vector product
    ``H·v`` (``program.gradients``). There is no ``hvp``: it is ``jvp(ctx,
    grad(ctx, loss, wrt), wrt)``, and one ``grad`` program serves every
    direction.

    Before it runs, register each ``wrt`` table's tangent as the input table
    ``program.inputs`` gives for it (``of`` is the table's name parts): its
    dims, then a tangent under each ``wrt`` column's name. A row it lacks has
    tangent 0. Of a program, that program's own inputs come first.
    """
    return jvp_plan(_plan_of(ctx, of), wrt, namespace=namespace)


def _plan_of(ctx: SessionContext, of: Union[str, Program]):
    """``of`` as ``grad_plan`` and the rest take it: SQL serialized as the
    context plans it, or a program as it is."""
    if isinstance(of, str):
        _refuse_not_in(of)
        return Serde.serialize_bytes(of, ctx)
    return of


def _refuse_not_in(sql: str) -> None:
    """Refuse ``x NOT IN (subquery)``: DataFusion's Substrait producer drops
    its NULL semantics (ddx issue #104), so ddx would differentiate a query
    that keeps rows a NULL in the subquery excludes."""
    if _not_in_subquery(sql):
        raise UnsupportedExpression(
            "not supported by ddx-ad yet: `NOT IN` over a subquery: DataFusion's Substrait "
            "producer writes it as a plain anti-join, which keeps the rows a NULL in the "
            "subquery should exclude (ddx issue #104). Write `NOT EXISTS`, or filter the NULLs "
            "out of the subquery"
        )


class DataFusionBackend:
    """A DataFusion ``SessionContext`` as a :class:`ddxdb.Backend`:
    each step's result is registered as an in-memory table."""

    def __init__(self, ctx: SessionContext):
        self.ctx = ctx

    def select_all(self, name: str) -> bytes:
        return Serde.serialize_bytes(f'SELECT * FROM "{name}"', self.ctx)

    def returns_rows(self, plan: bytes) -> bool:
        lp = Consumer.from_substrait_plan(self.ctx, _deserialize(plan))
        return self.ctx.create_dataframe_from_logical_plan(lp).limit(1).count() > 0

    def materialize(self, name: str, plan: bytes) -> None:
        lp = Consumer.from_substrait_plan(self.ctx, _deserialize(plan))
        _register(self.ctx, name, self.ctx.create_dataframe_from_logical_plan(lp).to_arrow_table())

    def drop_table(self, name: str) -> None:
        self.ctx.deregister_table(name)


def run(target: Union[SessionContext, Backend], program: Program) -> None:
    """Run ``program``, of either kind, on a DataFusion context, or on any
    :class:`ddxdb.Backend`: its checks, then every step, registering
    each result. Its input tables must be registered first. Once the outputs
    are written, the intermediate tables are dropped; the value and the
    gradients stay until :func:`release` or the next run replaces them. A run
    that fails leaves none of the program's tables.

    Raises :class:`ddxdb.InvalidColumn` when a check fails: a ``wrt`` table's
    rows are not what the program assumed, for instance two share their dims.

    A program depends on the tables' names and schemas, not their values, so
    build it once and run it on every training step. One program's runs must
    not overlap, since they write the same tables.
    """
    _run(DataFusionBackend(target) if isinstance(target, SessionContext) else target, program)


def run_checks(ctx: SessionContext, program: Program) -> None:
    """Run ``program``'s checks, raising :class:`ddxdb.InvalidColumn` on the
    first that returns a row."""
    for check in program.checks:
        if _returns_rows(ctx, check.plan):
            raise InvalidColumn(f"invalid wrt column: {check.message}")


def _returns_rows(ctx: SessionContext, plan: bytes) -> bool:
    return ctx.create_dataframe_from_logical_plan(_consume(ctx, plan)).limit(1).count() > 0


def release(ctx: SessionContext, program: Program) -> None:
    """Drop every table ``program`` registered on ``ctx``, the value and the
    gradients included. Its input tables are the caller's, and stay."""
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
    earlier steps, or of an input table, name their columns but not their
    types; the types come from the tables themselves, as the engine states
    them."""
    schemas = {
        name: Serde.serialize_bytes(f'SELECT * FROM "{name}"', ctx)
        for name in _unbound_reads(plan)
    }
    return Consumer.from_substrait_plan(ctx, _deserialize(_bind_reads(plan, schemas)))


def _deserialize(plan: bytes):
    """``plan`` decoded by datafusion-python, which refuses one nested more
    than protobuf's default 100 messages deep (two per relation): a step of a
    deep enough query. Said plainly, and typed, rather than as a decode
    error."""
    try:
        return Serde.deserialize_bytes(plan)
    except Exception as e:
        if "recursion limit" not in str(e):
            raise
        raise UnsupportedExpression(
            "not supported by ddxdb yet: a program step nests more than about 48 relations "
            "deep, past the protobuf limit datafusion-python decodes Substrait with. Run it "
            "from Rust (ddx-datafusion's ad module), or split the query"
        ) from e


def _register(ctx: SessionContext, name: str, table: pa.Table) -> None:
    # Registered as record batches (an in-memory table), not a view:
    # DataFusion's Substrait consumer can only pick columns out of a table scan.
    batches = table.to_batches() or [pa.RecordBatch.from_pylist([], schema=table.schema)]
    ctx.deregister_table(name)
    ctx.register_record_batches(name, [batches])


def sql(ctx: SessionContext, statement: str, *, dialect: str = "datafusion"):
    """Run ``statement``, in which, in a ``FROM`` clause, ``grad(loss,
    table.column, …)`` is the gradient of the loss the CTE ``loss`` computes,
    as a relation shaped like ``table``, and ``jvp(f, table.column, …,
    tangent, …)`` is the CTE ``f``'s output with each column's tangent beside
    it, as ``{column}_tangent`` (see the module docs). Returns a DataFrame.

    Each loss's program runs first, once, however many calls use it, and each
    distinct ``jvp`` call's. A statement with no such call is planned as it
    is. The programs' tables are
    dropped once the statement is planned; the DataFrame keeps what it reads.
    A loss defined in a ``WITH RECURSIVE`` clause is refused. ``dialect``
    is how the statement is parsed to find the calls, one of the names
    :func:`ddxdb.rewrite_sql` accepts.
    """
    return sql_all(ctx, [statement], dialect=dialect)[0]


def sql_all(ctx: SessionContext, statements: Sequence[str], *, dialect: str = "datafusion") -> list:
    """:func:`sql` for several statements. Statements that take ``grad`` of the
    same loss query share one run of its program, so a training step can update
    each parameter table with its own statement and pay for the backward pass
    once; identical ``jvp`` calls share one run too."""
    # Which programs to run and how each statement reads their gradients are
    # ddx_ad::sql::Statements'; running them and planning the result are
    # DataFusion's.
    planned = _Statements(list(statements), dialect)
    ran: list[BackwardProgram] = []
    ran_jvps: list[ForwardProgram] = []
    try:
        for query, wrt, restrict in planned.jobs():
            # Only the gradient rows the statements read, where they say.
            selects = []
            for table, predicate in restrict:
                try:
                    selects.append((table, Serde.serialize_bytes(f"SELECT * FROM {table} WHERE {predicate}", ctx)))
                except Exception:  # every row is computed, which is always right
                    pass
            _refuse_not_in(query)
            program = grad_plan(Serde.serialize_bytes(query, ctx), wrt, restrict=selects)
            ran.append(program)
            run(ctx, program)
        for j, (query, wrt) in enumerate(planned.jvp_jobs()):
            program = jvp(ctx, query, wrt)
            ran_jvps.append(program)
            # Each tangent, as the columns the program asks for, by name.
            for name, columns, tangent in planned.jvp_inputs(j, program):
                picked = ", ".join('"' + c.replace('"', '""') + '"' for c in columns)
                table = ctx.sql(f"SELECT {picked} FROM ({tangent}) AS __ddx_tangent").to_arrow_table()
                _register(ctx, name, table)
            run(ctx, program)
        return [ctx.sql(s) for s in planned.rewrite(ran, ran_jvps)]
    finally:
        # A planned DataFrame holds the tables it reads, so the programs'
        # tables can leave the catalog now, and the tangents registered here.
        for program in ran:
            release(ctx, program)
        for program in ran_jvps:
            release(ctx, program)
            for table in program.inputs:
                ctx.deregister_table(table.name)
