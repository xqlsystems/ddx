# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Gradients of whole queries on DataFusion.

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

A training loop runs the same statements on every batch: :func:`prepare`
prepares them once, and each later call runs the same plans again, without
planning, keying or reading views.

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

import threading
import warnings
import weakref
from typing import Callable, NamedTuple, Optional, Sequence, Union

try:
    import pyarrow as pa
    from datafusion import SessionContext, udf
    from datafusion.plan import ExecutionPlan
    from datafusion.substrait import Consumer, Serde
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError("ddxdb.ad needs DataFusion: pip install 'ddxdb[datafusion]'") from e

from ._ddxdb import (
    UnsupportedExpression,
    _not_in_subquery,
    _view_probe,
)
from . import Backend, Prepared, Programs, _name_parts, rewrite
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
    "prepare",
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


#: The settings under which DataFusion pushes into a plan's scans a filter
#: its first run finds. ``ExecutionPlan.reset_state`` keeps that filter, so
#: a plan run again would leave out rows that this run's tables hold.
_DYNAMIC_FILTERS = tuple(
    f"datafusion.optimizer.enable{kind}_dynamic_filter_pushdown" for kind in ("", "_join", "_topk", "_aggregate")
)


class DataFusionBackend:
    """A DataFusion ``SessionContext`` as a :class:`ddxdb.Backend`:
    each step's result is registered as an in-memory table.

    With ``in_place``, the caller promises that between runs of a
    :class:`ddxdb.Prepared`, the tables its statements read change only by
    writes into the memory they hold, as ``xarray_sql``'s prepared queries
    write them; never by being registered again. Then each plan's physical
    plan is compiled once and run again (:meth:`compile`), rather than
    planned on every run. See :func:`prepare`."""

    def __init__(self, ctx: SessionContext, *, in_place: bool = False):
        self.ctx = ctx
        self.in_place = in_place
        # Each view's plan and table's schema as session_key last saw them,
        # and how many times they have changed.
        self._catalog: list = []
        self._changes = 0
        # How many times a compiled step's rows replaced the ones it had
        # stored rather than being written there: a plan compiled before
        # then may read the old ones.
        self._stored = 0
        self._warned = False

    def select_all(self, name: str) -> bytes:
        return Serde.serialize_bytes(f'SELECT * FROM "{name}"', self.ctx)

    def plan_sql(self, sql: str) -> bytes:
        _refuse_not_in(sql)
        return Serde.serialize_bytes(sql, self.ctx)

    def plan_key(self, sql: str) -> str:
        # DataFusion expands each view, and each view it reads, as it turns
        # SQL into a plan, so the plan before optimization holds every
        # view's definition, and its schemas every table's columns, types
        # and nullability: all an optimized plan's meaning depends on.
        return self.ctx.sql(sql).logical_plan().display_indent_schema()

    def view_sql(self, name: str) -> Optional[str]:
        """The SQL of the view ``name``, read from DataFusion's information
        schema. The information schema is on in a bare ``SessionContext()``;
        a context made from its own ``SessionConfig`` needs
        ``.with_information_schema(True)``, or every view is left to the
        engine, with a warning. A view registered from a DataFrame has no
        SQL, and is left to the engine too.

        A DataFusion view keeps the tables it was made over, while ddxdb
        reads tables by name. After a table a view reads is registered again
        under its name, the view still reads the old one, and ddxdb would
        differentiate the new one. DataFusion's Python API doesn't say which
        table a view reads, so this can't be refused here as it is from
        Rust: create a view again after registering what it reads."""
        try:
            # All of them: how the plans compare depends on the optimizer's.
            settings = dict(
                (r["name"], r["value"])
                for r in self.ctx.sql("SELECT name, value FROM information_schema.df_settings")
                .to_arrow_table()
                .to_pylist()
            )
        except ValueError:
            # DataFusion raises ValueError for a table it can't find: here,
            # the information schema is off.
            parts = _name_parts(name)
            if parts is not None and 1 <= len(parts) <= 3:
                _warn_if_view(self.ctx, parts)
            return None
        # A name is resolved as DataFusion resolves it: an unquoted part in
        # lower case only if the session normalizes them.
        normalize = settings["datafusion.sql_parser.enable_ident_normalization"] == "true"
        parts = _name_parts(name, normalize)
        if parts is None or not 1 <= len(parts) <= 3:
            return None
        defaults = [
            settings["datafusion.catalog.default_catalog"],
            settings["datafusion.catalog.default_schema"],
        ]
        catalog, schema, table = _qualified(parts, defaults)
        rows = (
            self.ctx.sql(
                "SELECT definition FROM information_schema.views "
                "WHERE table_catalog = $1 AND table_schema = $2 AND table_name = $3",
                param_values={"1": catalog, "2": schema, "3": table},
            )
            .to_arrow_table()
            .column(0)
            .to_pylist()
        )
        if not rows or rows[0] is None:
            return None
        # DataFusion keeps the plan a view was made with, so once a view it
        # reads is redefined, its SQL means something else than what runs.
        # The SQL is given only if its query, read in the view's place,
        # plans as reading the view does.
        probe = _view_probe(rows[0])
        if probe is None:
            return None
        try:
            same = _same_plans(self.ctx, name, probe, settings)
        except ValueError:
            # The view's tables, or its SQL's, no longer plan.
            return None
        return rows[0] if same else None

    def session_key(self) -> Optional[str]:
        """The session's settings, which hold where names are searched and
        how they are read, and how many times this backend has seen its
        context's views or tables' schemas change. Each view's plan is the
        plan DataFusion runs for it, the one it kept when it was made, which
        reads a view redefined since as it was then. Plans compare in Rust,
        and a plan unchanged shares its kept part, so the comparison is
        quick.

        ``None`` without the information schema, which holds the settings:
        each call then prepares the statements again. Unlike Rust's, this
        can't tell a view that reads a table registered again since (see
        :meth:`view_sql`)."""
        try:
            settings = (
                self.ctx.sql("SELECT name, value FROM information_schema.df_settings ORDER BY name")
                .to_arrow_table()
                .to_pylist()
            )
        except ValueError:
            # DataFusion raises ValueError for a table it can't find: here,
            # the information schema is off.
            return None
        # Each view's plan, and every other table's schema, which the plans
        # were bound to; ddx's own tables come and go with each run.
        catalog = []
        for c in sorted(self.ctx.catalog_names()):
            cat = self.ctx.catalog(c)
            for s in sorted(cat.schema_names()):
                if s == "information_schema":
                    continue
                schema = cat.schema(s)
                for t in sorted(schema.table_names()):
                    table = schema.table(t)
                    path = ".".join('"' + p.replace('"', '""') + '"' for p in (c, s, t))
                    if table.kind == "view":
                        catalog.append((path, self.ctx.table(path).logical_plan()))
                    elif not t.startswith("__ddx_"):
                        catalog.append((path, table.schema))
        if catalog != self._catalog:
            self._catalog, self._changes = catalog, self._changes + 1
        return f"{self._changes} {settings!r}"

    def returns_rows(self, plan: bytes) -> bool:
        lp = Consumer.from_substrait_plan(self.ctx, _deserialize(plan))
        return self.ctx.create_dataframe_from_logical_plan(lp).limit(1).count() > 0

    def materialize(self, name: str, plan: bytes) -> None:
        lp = Consumer.from_substrait_plan(self.ctx, _deserialize(plan))
        _register(self.ctx, name, self.ctx.create_dataframe_from_logical_plan(lp).to_arrow_table())

    def drop_table(self, name: str) -> None:
        self.ctx.deregister_table(name)

    def compile(self, plan: bytes) -> Optional["_Compiled"]:
        """``plan``'s physical plan, compiled to run again, with ``in_place``;
        otherwise ``None``. Also ``None``, with a warning, if this
        datafusion has no ``ExecutionPlan.reset_state`` (the xqlsystems
        fork's, branch ``reset-plan-state-55.0``), or a setting in
        :data:`_DYNAMIC_FILTERS` is on, or the information schema that
        holds the settings is off.

        A compiled step's rows are stored in memory of its own, and each later
        run writes its rows there, so the compiled plans that read the step
        read them. When a run's rows can't be written there (another number
        of rows, another schema, a NULL, or a column that isn't an integer
        or a float and changed), they are stored anew, and every plan
        compiled before then is planned again."""
        if not self.in_place or not self._can_compile():
            return None
        return _Compiled(*self._physical(plan), self._stored)

    def materialize_compiled(self, name: str, plan: bytes, compiled: "_Compiled") -> None:
        if compiled.stored != self._stored:
            compiled.plan, compiled.schema = self._physical(plan)
        rows = _rows(_execute(self.ctx, compiled.plan), compiled.schema)
        if not compiled.write(rows):
            # Rows stored for the first time make no compiled plan stale: a
            # plan that reads this step's table is compiled after the step
            # first runs.
            replaced = compiled.batch is not None
            compiled.store(rows)
            self._stored += replaced
        compiled.stored = self._stored
        self.ctx.deregister_table(name)
        self.ctx.register_record_batches(name, [[compiled.batch]])

    def returns_rows_compiled(self, plan: bytes, compiled: "_Compiled") -> bool:
        # A check may read a seed (a tangent, say), whose rows may have been
        # stored anew since.
        if compiled.stored != self._stored:
            compiled.plan, compiled.schema = self._physical(plan)
            compiled.stored = self._stored
        return any(b.num_rows for b in _execute(self.ctx, compiled.plan))

    def _physical(self, plan: bytes):
        """``plan``'s physical plan, and the schema of its rows."""
        lp = Consumer.from_substrait_plan(self.ctx, _deserialize(plan))
        df = self.ctx.create_dataframe_from_logical_plan(lp)
        return df.execution_plan(), df.schema()

    def _can_compile(self) -> bool:
        """Whether a plan can be compiled: see :meth:`compile`."""
        if not hasattr(ExecutionPlan, "reset_state"):
            why = (
                "this datafusion has no ExecutionPlan.reset_state; install the xqlsystems fork "
                "(branch reset-plan-state-55.0)"
            )
        else:
            try:
                rows = (
                    self.ctx.sql(
                        "SELECT name, value FROM information_schema.df_settings WHERE name IN "
                        f"({', '.join(repr(n) for n in _DYNAMIC_FILTERS)})"
                    )
                    .to_arrow_table()
                    .to_pylist()
                )
            except ValueError:
                # DataFusion raises ValueError for a table it can't find:
                # here, the information schema is off.
                rows = None
            if rows is None:
                why = "this SessionContext has no information_schema, which holds its settings"
            else:
                on = sorted(r["name"] for r in rows if r["value"] != "false")
                if not on:
                    return True
                why = f"{', '.join(on)} {'is' if len(on) == 1 else 'are'} on; set them to false"
        if not self._warned:
            self._warned = True
            warnings.warn(f"in_place: ddxdb plans every step on every run, since {why}", stacklevel=2)
        return False


def _rows(batches: list, schema: pa.Schema) -> pa.Table:
    """``batches`` as a table, as ``DataFrame.to_arrow_table`` makes one:
    of the batches' own schema, which may differ from the plan's (in which
    columns are nullable, say), and of ``schema`` only if there are none."""
    return pa.Table.from_batches(batches) if batches else schema.empty_table()


def _execute(ctx: SessionContext, plan) -> list:
    """The rows of a compiled physical plan, run again: every node's state
    reset, so a hash join reads its build side again."""
    plan = plan.reset_state()
    return [b.to_pyarrow() for p in range(plan.partition_count) for b in ctx.execute(plan, p)]


class _Compiled:
    """A plan :meth:`DataFusionBackend.compile` compiled: its physical plan,
    made when the backend had replaced stored rows ``stored`` times, and a
    step's rows, in memory each run writes into."""

    def __init__(self, plan, schema: pa.Schema, stored: int):
        self.plan, self.schema, self.stored = plan, schema, stored
        self.batch: Optional[pa.RecordBatch] = None
        # Per column, the memory a run writes its values into; None for a
        # column whose values must stay the same.
        self._buffers: list = []

    def store(self, rows: pa.Table) -> None:
        """Store ``rows``, each integer or float column without a NULL in
        memory of its own."""
        columns, self._buffers = [], []
        for column in rows.columns:
            column = column.combine_chunks()
            buffer = None
            if column.null_count == 0 and (pa.types.is_integer(column.type) or pa.types.is_floating(column.type)):
                buffer = pa.allocate_buffer(len(column) * column.type.byte_width)
                memoryview(buffer)[:] = _values(column)
                column = pa.Array.from_buffers(column.type, len(column), [None, buffer])
            columns.append(column)
            self._buffers.append(buffer)
        self.batch = pa.RecordBatch.from_arrays(columns, schema=rows.schema)

    def write(self, rows: pa.Table) -> bool:
        """Write ``rows`` into the stored ones, if they fit: as many, of the
        same schema, no NULL where a value is written, and the rest as they
        were."""
        if self.batch is None or rows.num_rows != self.batch.num_rows or rows.schema != self.batch.schema:
            return False
        columns = [c.combine_chunks() for c in rows.columns]
        for buffer, new, old in zip(self._buffers, columns, self.batch.columns):
            if new.null_count if buffer is not None else not new.equals(old):
                return False
        for buffer, new in zip(self._buffers, columns):
            if buffer is not None:
                memoryview(buffer)[:] = _values(new)
        return True


def _values(column: pa.Array) -> memoryview:
    """The bytes of ``column``'s values: an integer or float column's."""
    width = column.type.byte_width
    start = column.offset * width
    return memoryview(column.buffers()[1])[start : start + len(column) * width]


#: Per context, each view's last optimized comparison: the two unoptimized
#: plans and the settings it was made for, and whether the optimized plans
#: matched.
_compared: "weakref.WeakKeyDictionary[SessionContext, dict]" = weakref.WeakKeyDictionary()


def _same_plans(ctx: SessionContext, name: str, probe: str, settings: dict) -> bool:
    """Whether reading the view ``name`` plans as ``probe``, its SQL read in
    its place, does.

    DataFusion coerces the plan it keeps for a view (``val * 2`` over a
    DOUBLE becomes ``val * CAST(2 AS DOUBLE)``), and Python can't coerce the
    probe's plan alone, so a view that needs coercion plans unlike its SQL
    until both are optimized, which coerces both alike. Optimizing a model's
    view takes long, and a training loop asks on every step, so the plans
    are compared unoptimized first, and an optimized comparison is kept for
    the unoptimized plans and the session ``settings`` it was made under:
    the optimizer gives the same plans for the same ones."""
    view = _plan_under_alias(ctx, f"SELECT * FROM {name}", optimized=False)
    query = _plan_under_alias(ctx, probe, optimized=False)
    if view == query:
        return True
    compared = _compared.setdefault(ctx, {})
    made = (view, query, settings)
    kept = compared.get(name)
    if kept is not None and kept[0] == made:
        return kept[1]
    same = _plan_under_alias(ctx, f"SELECT * FROM {name}", optimized=True) == _plan_under_alias(
        ctx, probe, optimized=True
    )
    compared[name] = (made, same)
    return same


def _plan_under_alias(ctx: SessionContext, sql: str, *, optimized: bool) -> str:
    """The plan of ``sql``, a ``SELECT *`` from one aliased relation, under
    the alias: what that relation reads. Optimizing may remove the ``SELECT
    *`` projection above the alias, so the plan is read from below the
    first alias."""
    df = ctx.sql(sql)
    plan = df.optimized_logical_plan() if optimized else df.logical_plan()
    while plan.to_variant().__class__.__name__ != "SubqueryAlias":
        inputs = plan.inputs()
        if len(inputs) != 1:
            raise ValueError("not a SELECT * from one aliased relation")
        plan = inputs[0]
    return plan.inputs()[0].display_indent_schema()


def _qualified(parts: list, defaults: list) -> list:
    """A name of one to three parts as catalog, schema and table. A name
    leaves out leading parts, which are taken from the front of
    ``defaults``, the default catalog and schema: ``t`` is in both,
    ``s.t`` in the default catalog."""
    missing = 3 - len(parts)
    return defaults[:missing] + parts


_warned: "weakref.WeakSet[SessionContext]" = weakref.WeakSet()


def _warn_if_view(ctx: SessionContext, parts: list) -> None:
    """Warns, once per context, that a view is left to the engine because
    the information schema that holds its SQL is off. The name is resolved
    against DataFusion's default catalog and schema, which can't be read
    without it; a view it misses only goes unwarned."""
    if ctx in _warned:
        return
    catalog, schema, table = _qualified(parts, ["datafusion", "public"])
    try:
        tables = ctx.catalog(catalog).schema(schema)
    except KeyError:
        return
    if tables.table_exist(table) and tables.table(table).kind == "view":
        _warned.add(ctx)
        warnings.warn(
            f"ddxdb can't read the SQL of view {'.'.join(parts)!r}, because this "
            "SessionContext has no information_schema, so the view is planned as "
            "the engine expands it: a view that reads its layers more than once "
            "is copied that many times. Create the context with "
            "SessionConfig().with_information_schema(True).",
            stacklevel=2,
        )


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


def _duckdb(con) -> bool:
    """Whether ``con`` is a DuckDB connection, without importing DuckDB."""
    return type(con).__name__ == "DuckDBPyConnection"


def sql(ctx: SessionContext, statement: str, *, dialect: Optional[str] = None):
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
    ``sqlparser`` knows, or ``"datafusion"``, the default.

    ``ctx`` may be a DuckDB connection instead, which :func:`ddxdb.duckdb.sql`
    runs on. Its statements are DuckDB's, so ``dialect`` is ``"duckdb"``
    or left out.
    """
    return sql_all(ctx, [statement], dialect=dialect)[0]


#: The programs :func:`sql_all` keeps for each context, gone with it.
_programs: "weakref.WeakKeyDictionary[SessionContext, Programs]" = weakref.WeakKeyDictionary()


def sql_all(ctx: SessionContext, statements: Sequence[str], *, dialect: Optional[str] = None) -> list:
    """:func:`sql` for several statements. Statements that take ``grad`` of the
    same loss query share one run of its program, so a training step can update
    each parameter table with its own statement and pay for the backward pass
    once; ``vjp`` calls along one cotangent share one too, and identical
    ``jvp`` calls.

    Each program is built once per context and run again by later calls, as
    a training loop makes them (:class:`ddxdb.Programs`): it depends on the
    schemas of the tables its query reads and on the views' definitions, not
    on the tables' values, and is built again when one of those changes. One
    caveat: DataFusion's optimizer folds ``now()`` into a literal, so a
    program whose query reads it keeps the time of the call that built it."""
    if _duckdb(ctx):
        from . import duckdb as on_duckdb

        if dialect not in (None, "duckdb"):
            raise ValueError(f"a DuckDB connection's statements are in DuckDB's dialect, not {dialect!r}")
        return on_duckdb.sql_all(ctx, statements)
    # Which programs to run, how each is built and run, and how each
    # statement reads them are ddx-ad's (ddx_ad::rewrite); DataFusion plans and
    # runs.
    # A planned DataFrame holds the tables it reads, so they can leave the
    # catalog once every statement is planned.
    programs = _programs.setdefault(ctx, Programs())
    with rewrite(DataFusionBackend(ctx), statements, dialect or "datafusion", programs) as rewritten:
        # SessionContext's own sql, not a ddxdb.Context's, which would look for
        # calls again.
        return [SessionContext.sql(ctx, s) for s in rewritten.statements]


def prepare(
    ctx: SessionContext,
    statements: Sequence[str],
    *,
    dialect: Optional[str] = None,
    in_place: bool = False,
) -> Callable[[], list]:
    """:func:`sql_all`'s ``statements``, prepared to run again on ``ctx``, as
    ``jax.jit`` prepares a function. Returns a function of no arguments that
    runs them and returns what :func:`sql_all` would::

        step = ad.prepare(ctx, statements)
        for batch in batches:
            ctx.deregister_table("batch")
            ctx.register_record_batches("batch", [batch])
            new_weights = step()

    Its first call computes the calls as :func:`sql_all` does. Every later
    one runs the same plans again (:class:`ddxdb.Prepared`), so it skips
    what a call of :func:`sql_all` repeats: planning the objective, keying
    each program, and checking that each view's SQL is still what
    DataFusion runs for it.

    Between calls, a table may change its rows, or be registered again under
    its name. Each call compares every table's schema, every view's plan
    and the session's settings with those it was prepared with, and
    prepares the statements again once one has changed. As with
    :func:`sql_all`'s kept programs, a ``now()`` in a loss keeps the time
    of the call that prepared it: DataFusion folds it into a constant. Calls
    wait for each other, since each writes the same tables.

    Experimental: ``in_place=True`` makes later calls faster still, by running each
    step's physical plan again rather than planning it (see
    :class:`DataFusionBackend`). A physical plan holds the tables it read
    when it was made, so it is for a caller that changes its tables only by
    writing into the memory they hold, as ``xarray_sql``'s prepared queries
    do. A table registered again under its name would go unread: the plans
    would read the one it replaced, and give wrong results without an
    error. A call also writes into the memory of the tables the last call's
    DataFrames read, so collect those first. It needs the xqlsystems fork
    of datafusion (``ExecutionPlan.reset_state``), the information schema,
    and the four ``enable*_dynamic_filter_pushdown`` settings off; without
    them, it warns and plans each step as before.

    ``ctx`` may be a DuckDB connection instead, which
    :func:`ddxdb.duckdb.prepare` prepares on, without ``in_place``."""
    if _duckdb(ctx):
        from . import duckdb as on_duckdb

        if dialect not in (None, "duckdb"):
            raise ValueError(f"a DuckDB connection's statements are in DuckDB's dialect, not {dialect!r}")
        if in_place:
            raise ValueError("in_place is for a DataFusion SessionContext; DuckDB plans each step on every call")
        return on_duckdb.prepare(ctx, statements)
    prepared = Prepared(DataFusionBackend(ctx, in_place=in_place), statements, dialect or "datafusion")
    lock = threading.Lock()

    def run() -> list:
        with lock, prepared.run() as rewritten:
            # As in sql_all: SessionContext's own sql, and the tables leave
            # the catalog once every statement is planned.
            return [SessionContext.sql(ctx, s) for s in rewritten.statements]

    return run
