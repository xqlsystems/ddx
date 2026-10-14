# ddxdb

Calculus in SQL: the derivative of a query with respect to a table's columns,
as JAX takes the derivative of a function, computed by the engine.

```python
from ddxdb import Context

ctx = Context()
ctx.sql("CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)")
ctx.sql("""
WITH f AS (SELECT SUM(val * val) AS l FROM w)
SELECT w.i, g.val AS df_dval
FROM w JOIN grad(f, w.val) g ON w.i = g.i     -- 2·val, row by row
""")
```

`grad(f, table.column)` in a `FROM` clause is the gradient of the number the
CTE `f` computes, as a relation shaped like the table. The query is ordinary
SQL; nothing in it is labelled for ddx, and the one function ddx gives a
meaning to is `ddx_stop_gradient(x)`, JAX's `lax.stop_gradient`.

A gradient is what an optimizer steps along, so a query scoring anything (a
model's loss, a curve's misfit, an energy) can be optimized in SQL: a step,
`params - lr * grad(f)(params)`, is a join of the table with its gradient.

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
the price of one backward pass. `ddxdb.ad.grad` and `ddxdb.ad.vjp` give the
underlying program.

## Other engines

`ddxdb.grad_plan(plan_bytes, wrt)` differentiates the serialized Substrait plan
an engine's producer writes, and `ddxdb.run(backend, program)` runs the result
on any object with four methods, `select_all`, `returns_rows`, `materialize`
and `drop_table` (the `ddxdb.Backend` protocol). Both are ddx-ad's own Rust,
the same code the Rust adapter runs. Pass `namespace="__ddx_mine_"` to get the
same program from the same plan every time.

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
