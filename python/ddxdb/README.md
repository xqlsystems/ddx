# ddxdb

Autograd for SQL: write a model's forward pass and loss as a query, and take
its gradient in SQL, as JAX takes the gradient of a function.

```python
from ddxdb import Context

ctx = Context()
# ... register x(sample, inp, val), w(inp, out, val), y(sample, out, val) ...

ctx.sql("""
WITH h AS (
  SELECT x.sample, w.out, tanh(SUM(x.val * w.val)) AS val
  FROM x JOIN w ON x.inp = w.inp GROUP BY x.sample, w.out),
loss AS (
  SELECT SUM(power(h.val - y.val, 2)) AS l
  FROM h JOIN y ON h.sample = y.sample AND h.out = y.out)
SELECT w.inp, w.out, w.val - 0.1 * g.val AS val
FROM w JOIN grad(loss, w.val) g ON w.inp = g.inp AND w.out = g.out
""")
```

`grad(loss, table.column)` in a `FROM` clause is the gradient of the loss the
CTE computes, as a relation shaped like the table. That is one SGD step:
`params - lr * grad(loss)(params)`, written as a join. Nothing in the loss is
labelled for ddx; the one function it gives a meaning to is
`ddx_stop_gradient(x)`, JAX's `lax.stop_gradient`.

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
