# ddxdb

Calculus in SQL: the derivative of a query with respect to a table's columns,
as JAX takes the derivative of a function, computed by the engine.

```python
from ddxdb import Context

ctx = Context()
ctx.sql("CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)")
ctx.sql("""
WITH f AS (SELECT SUM(val * val) AS l FROM w)
SELECT i, val, grad(f, val) AS df_dval FROM w     -- 2·val, row by row
""")
```

In a `SELECT` over a table, `grad(f, val)` is, at each row, the derivative of
the number the CTE `f` computes with respect to that row's `val` (`w.val` when
the `SELECT` reads several tables). The query is ordinary SQL; nothing in it
is labelled for ddx, and the one function ddx gives a meaning to is
`ddx_stop_gradient(x)`, JAX's `lax.stop_gradient`.

A gradient is what an optimizer steps along, so a query scoring anything (a
model's loss, a curve's misfit, an energy) can be optimized in SQL: a step,
`params - lr * grad(f)(params)`, is one expression.

## `jvp` and `vjp` in SQL

Forward mode is `jvp(e, table.column, tangent)`, a column of the `SELECT` that
computes `e`: its tangent at each row along `tangent`, a CTE or table shaped
like `table` (its dims, then the column's tangent), or a number, the same at
every row, as `jax.jvp` returns it beside the value. Along 1, it is the
derivative of an expression row by row: `SELECT x, jvp(sin(x), t.x, 1) AS dx
FROM t` is `cos(x)`. Along a table:

```python
ctx.sql("""
WITH h AS (
  SELECT x.sample, w.out, tanh(SUM(x.val * w.val)) AS val
  FROM x JOIN w ON x.inp = w.inp GROUP BY x.sample, w.out),
dw AS (SELECT inp, out, 0.01 AS val FROM w)
SELECT sample, out, val, jvp(val, w.val, dw) AS val_tangent FROM h
""")
```

One `SELECT`'s `jvp` columns share a tangent; tangents along two tables are
two `SELECT`s, and add. `vjp(f, column, cotangent)` is the reverse: in a
`SELECT` over a table, the pull-back to each row of a cotangent shaped like
the CTE `f`'s output. Calls nest: `f`, `e`'s `SELECT`, or a tangent can read
another call, so a `jvp` of a CTE of gradients is a Hessian-vector product, in
SQL.

This is the Python distribution of [`ddx`](https://github.com/xqlsystems/ddx),
a thin wrapper over the `ddx-ad` engine.

## Install

```bash
pip install ddxdb                  # the programs, for any engine
pip install "ddxdb[datafusion]"    # + Context and ddxdb.ad, on DataFusion
```

## `Context`, for DataFusion

A real `SessionContext` subclass whose `.sql()` runs `grad` first: every
inherited method, property and constructor argument works unchanged. It lives
in `ddxdb.datafusion` (a subclass needs its base class at import time) and is
re-exported as `ddxdb.Context`, imported on first use, so `import ddxdb` needs
no engine.

On a plain `SessionContext`, `ddxdb.ad.sql(ctx, statement)` does the same, and
`ddxdb.ad.sql_all` runs several statements that take `grad` of one loss for
the price of one backward pass.

## Programs, and how they compose

`ddxdb.ad.grad`, `vjp` and `jvp` give the programs underneath, and each takes
SQL or a program, and the columns to differentiate with respect to (`wrt`): a
column as `"table.column"`, or a list of them, which one backward pass
differentiates together. They compose: `ad.jvp(ctx, ad.grad(ctx, loss, wrt),
wrt)` gives Hessian-vector products, checked against `jax.jvp(jax.grad(f))`. Both
kinds of program (`BackwardProgram`, `ForwardProgram`) list the tables you
register before they run (`inputs`, as `InputTable`s) and the tables they
leave (`value` and `gradients`, as `OutputTable`s). `ad.run(ctx, program,
inputs)` runs one on its inputs (a cotangent or tangents, as DataFrames or
Arrow tables, in the order of `program.inputs`) and returns what it computed,
`value` and `gradients`, as DataFrames, tangents named `{column}_tangent`;
nothing stays on the context:

```python
hvp = ad.jvp(ctx, ad.grad(ctx, "SELECT SUM(val * val * val) AS l FROM w", "w.val"), "w.val")
out = ad.run(ctx, hvp, [ctx.sql("SELECT i, 1.0 AS val FROM w")])
out.gradients[0]   # i, val (the gradient), val_tangent (H·v)
```

## Other engines

`ddxdb.grad_plan(plan_bytes, wrt)` differentiates the serialized Substrait plan
an engine's producer writes, and `ddxdb.run(backend, program)` runs the result
on any object with four methods, `select_all`, `returns_rows`, `materialize`
and `drop_table` (the `ddxdb.Backend` protocol); `vjp_plan` and `jvp_plan`
are the same for the other two. They are ddx-ad's own Rust, the same code the
Rust adapter runs. Pass `namespace="__ddx_mine_"` to get the
same program from the same plan every time.

The SQL calls run on another engine too: `ddxdb.rewrite(backend, statements)`
runs the programs the statements' `grad`, `vjp` and `jvp` calls need on a
backend that also has `plan_sql` (SQL to a serialized Substrait plan), and
returns the statements rewritten to read what they computed. Run them on the
engine inside the `with` block; the tables they read are dropped when it ends:

```python
with ddxdb.rewrite(backend, [step_w, step_b]) as rewritten:
    for sql in rewritten.statements:
        engine.sql(sql)
```

In a training loop, pass the same `ddxdb.Programs()` each step
(`ddxdb.rewrite(backend, statements, programs=programs)`), and each program
is built once and run again on later steps. A program is kept under the
backend's `plan_key(sql)`, a text that changes whenever the engine's plan of
the query could (its tables' schemas, its views' definitions), so a backend
without that method gets nothing kept. `ddxdb.ad.sql` and `Context.sql` keep
programs for each context this way on their own.

A loss may read a view whose layers read each other more than once, as a
model defined with `CREATE VIEW` does. A backend with a `view_sql(name)` method
returns the view's SQL. ddxdb then writes the view into the loss as CTEs, so
each layer is planned once rather than copied at every read.
`DataFusionBackend` reads the SQL from DataFusion's information schema. The
information schema is on in a bare `SessionContext()`. A context made from its
own `SessionConfig` needs `.with_information_schema(True)`; without it, ddxdb
warns and leaves the view to the engine.

## What you can write

Inside the loss: `+ - * /`; the chain rule for the trig / inverse-trig / exp /
log / hyperbolic set plus `abs`; `power` with a constant base or exponent;
joins, `GROUP BY`, `SUM`, `AVG`, `MAX`, `MIN`, `CASE`, `greatest` and `least`.

```python
ddxdb.supported_functions()               # ['abs', 'acos', 'asin', ...]
```

The unary functions ddx has a rule for, read from the engine rather than
restated.

## Errors are typed

An unsupported construct is always an error, never a silently wrong number —
this is a numerical-correctness library, and a plausible-looking wrong
derivative is the worst thing it could produce. The kind of failure is a class,
so you can catch the one you can act on. All of them derive from
`ddxdb.DdxError`:

- `UnsupportedExpression`: no rule for something in the query.
- `NotScalar`: `grad` of a query that is not a loss (one row, one column).
- `UnknownColumn`: a column the loss does not read.
- `InvalidColumn`: a column that cannot be differentiated (not a float, or in
  a table whose rows do not have unique dims).

## Development

```bash
pip install maturin pytest
maturin develop --uv
python -m pytest tests/
```

Building compiles `protoc` from source for the `substrait` crate (it needs
`cmake` and a C++ compiler), which takes a few minutes the first time.

## License

Licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), the
same as the rest of [ddx](https://github.com/xqlsystems/ddx).
