# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Gradients of whole queries on DataFusion (ddx v2).

Write a model's forward pass and loss as SQL, and take its gradient in SQL::

    from ddxdb import ad

    ad.sql(ctx, '''
        WITH loss AS (SELECT SUM(power(x.v * w.val - y.v, 2)) AS l
                      FROM x JOIN w ON x.i = w.i JOIN y ON x.s = y.s)
        SELECT i, val - 0.1 * grad(loss, val) AS val FROM w
    ''')

In a ``SELECT`` over a table, ``grad(loss, column)`` is the gradient at each
row of the loss the CTE ``loss`` computes (``table.column`` when the
``SELECT`` reads more than one table), so an SGD step, ``params - lr *
grad(loss)(params)``, is one expression. ``vjp(f, column, cotangent)`` is the
same for any CTE ``f``, pulled back along ``cotangent`` (a CTE or a table
shaped like ``f``'s output). ``jvp(e, table.column, tangent)``, a column of the
statement's ``SELECT`` or a CTE's, is the tangent at each row of its
expression ``e`` along ``tangent`` (a CTE or a table shaped like ``table``,
or a number: ``jvp(sin(x), t.x, 1)`` is ``cos(x)``, row by row)::

    ad.sql(ctx, '''
        WITH v AS (SELECT i, 1.0 AS val FROM w)
        SELECT SUM(val * val) AS l, jvp(SUM(val * val), w.val, v) AS dl FROM w
    ''')

One ``SELECT``'s ``jvp`` columns share a tangent; tangents along two tables
are two ``SELECT`` statements, and add. :class:`ddxdb.Context` understands the
same syntax in its ``.sql()``.

Underneath, :func:`grad` turns a loss query into a :class:`BackwardProgram` of
Substrait plans, and :func:`run` runs them on the context and returns what
they computed as DataFrames (:class:`Outputs`). :func:`vjp` pulls back a
cotangent of any query's output, and :func:`jvp` pushes a tangent forward (a
:class:`ForwardProgram`); :func:`run` takes the cotangent or tangents.
Each takes SQL or a program, so they compose: ``jvp(ctx, grad(ctx, loss,
wrt), wrt)`` gives Hessian-vector products. Both kinds of program share one
vocabulary: the tables the caller registers (:class:`InputTable`) and the
tables a program leaves (:class:`OutputTable`). Nothing in the query is
labelled for any of this; the one function ddx gives a meaning to is
``ddx_stop_gradient(x)``, JAX's ``lax.stop_gradient``
(:func:`register_stop_gradient`).

While a program runs, every table it writes is named with a prefix unique to
that program, ``__ddx_{id}_``, so two programs on one context never read each
other's tables, and a user's table is never replaced unless its name starts
with ``__ddx_``, which is reserved. None of them stays once :func:`run`
returns.

One limit Rust's ``ddx_datafusion::ad`` does not share: datafusion-python's
Substrait consumer names a computed column by its whole expression, so a deep
chain of row-wise maps that each read their input twice (twenty
``sin(v) + 0.1 * v`` layers, say) makes a very large plan here. The Rust
adapter consumes steps with short names. (datafusion-python also decodes a
plan with protobuf's default nesting limit; ddx splits every step deeper
than it into steps, so none reaches it.)

Importing this module requires DataFusion.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Union

try:
    import pyarrow as pa
    from datafusion import SessionContext, udf
    from datafusion.substrait import Consumer, Serde
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError("ddxdb.ad needs DataFusion: pip install 'ddxdb[datafusion]'") from e

from ._ddxdb import (
    UnsupportedExpression,
    _not_in_subquery,
)
from . import Backend, rewrite
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
    "Outputs",
    "run",
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
    wrt: Union[str, Sequence[str]],
    *,
    namespace: Optional[str] = None,
) -> BackwardProgram:
    """The gradient of ``of`` with respect to ``wrt``, a column as
    ``"table.column"`` or a list of them (one program, one backward pass, a
    gradient for each table): of SQL, the loss the query computes, which must be one row and
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
    wrt: Union[str, Sequence[str]],
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
    wrt: Union[str, Sequence[str]],
    *,
    namespace: Optional[str] = None,
) -> ForwardProgram:
    """The Jacobian-vector product of ``of`` with respect to ``wrt``, a column
    as ``"table.column"`` or a list of them: of SQL, a program computing the query's output
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

    def plan_sql(self, sql: str) -> bytes:
        _refuse_not_in(sql)
        return Serde.serialize_bytes(sql, self.ctx)

    def returns_rows(self, plan: bytes) -> bool:
        lp = Consumer.from_substrait_plan(self.ctx, _deserialize(plan))
        return self.ctx.create_dataframe_from_logical_plan(lp).limit(1).count() > 0

    def materialize(self, name: str, plan: bytes) -> None:
        lp = Consumer.from_substrait_plan(self.ctx, _deserialize(plan))
        _register(self.ctx, name, self.ctx.create_dataframe_from_logical_plan(lp).to_arrow_table())

    def drop_table(self, name: str) -> None:
        self.ctx.deregister_table(name)


class Outputs(NamedTuple):
    """What a program computed, as DataFrames that hold their rows."""

    #: The query's output: its columns, then for a ``jvp`` program each
    #: column's tangent as ``{column}_tangent``, as ``jvp(…)`` in SQL returns
    #: them.
    value: object
    #: One per ``wrt`` table, in the program's order: the table's dims, then
    #: each value's gradient under the value's name (and, for ``jvp`` of a
    #: program with gradients, each one's tangent, named as in ``value``).
    gradients: list


def run(ctx: SessionContext, program: Program, inputs: Sequence = ()) -> Outputs:
    """Run ``program``, of either kind, on ``inputs``, one per
    ``program.inputs`` in order (a ``vjp``'s cotangent, a ``jvp``'s tangents;
    none for ``grad``), each any object that exports an Arrow stream (a
    DataFrame, a ``pyarrow.Table``, a Polars or DuckDB result: the Arrow
    PyCapsule interface) with at least the columns its table names, and
    return what it computed.

    First its checks, then every step. Nothing it writes stays on ``ctx``,
    whether it succeeds or fails. Raises :class:`ddxdb.InvalidColumn` when a
    check fails: a ``wrt`` table's or an input's rows are not what the
    program assumed, for instance two share their dims.

    A program depends on the tables' names and schemas, not their values, so
    build it once and run it on every training step. One program's runs must
    not overlap, since they write the same tables while they run. To run a
    program on another engine, see :func:`ddxdb.run`.
    """
    if not isinstance(ctx, SessionContext):
        raise TypeError(
            "ad.run runs a program on a DataFusion SessionContext; ddxdb.run(backend, program) "
            "runs one on any ddxdb.Backend"
        )
    tables = program.inputs
    if len(inputs) != len(tables):
        raise ValueError(
            f"the program takes {len(tables)} input tables "
            f"({', '.join(t.name for t in tables)}), and was given {len(inputs)}"
        )
    try:
        for table, given in zip(tables, inputs):
            # Anything that exports an Arrow stream (DataFusion, Polars, DuckDB,
            # pandas, pyarrow), read without copying where it can be.
            arrow = given if isinstance(given, pa.Table) else pa.table(given)
            _register(ctx, table.name, arrow.select(list(table.columns)))
        _run(DataFusionBackend(ctx), program)
        return Outputs(ctx.sql(program.value.query()), [ctx.sql(g.query()) for g in program.gradients])
    finally:
        # A planned DataFrame holds the tables it reads, so they can leave
        # the catalog now.
        _release(ctx, program)
        for table in tables:
            ctx.deregister_table(table.name)


def _release(ctx: SessionContext, program: Program) -> None:
    """Drop every table ``program`` registered on ``ctx``."""
    for step in program.steps():
        ctx.deregister_table(step.name)


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
    """Run ``statement``, in which, as columns of a ``SELECT``, ``grad(loss,
    column)`` is the gradient at each row of the loss the CTE ``loss``
    computes; ``vjp(f, column, cotangent)`` the same for any CTE ``f`` along
    a cotangent; and ``jvp(e, table.column, …, tangent)`` the tangent at each
    row of the expression ``e`` (see the module docs). Calls nest: ``f``,
    ``e``'s ``SELECT``, or the cotangent or tangent may read another call,
    and is differentiated through it. Returns a DataFrame.

    Each program runs first, once, however many calls use it. A statement
    with no such call is planned as it is. The programs' tables are
    dropped once the statement is planned; the DataFrame keeps what it reads.
    A loss defined in a ``WITH RECURSIVE`` clause is refused. ``dialect``
    is how the statement is parsed to find the calls: a dialect name
    ``sqlparser`` knows, or ``"datafusion"``.
    """
    return sql_all(ctx, [statement], dialect=dialect)[0]


def sql_all(ctx: SessionContext, statements: Sequence[str], *, dialect: str = "datafusion") -> list:
    """:func:`sql` for several statements. Statements that take ``grad`` of the
    same loss query share one run of its program, so a training step can update
    each parameter table with its own statement and pay for the backward pass
    once; ``vjp`` calls along one cotangent share one too, and identical
    ``jvp`` calls."""
    # Which programs to run, how each is built and run, and how each
    # statement reads them are ddx-ad's (ddx_ad::rewrite); DataFusion plans and
    # runs.
    # A planned DataFrame holds the tables it reads, so they can leave the
    # catalog once every statement is planned.
    with rewrite(DataFusionBackend(ctx), statements, dialect) as rewritten:
        # SessionContext's own sql, not a ddxdb.Context's, which would look for
        # calls again.
        return [SessionContext.sql(ctx, s) for s in rewritten.statements]
