# ddx-datafusion

The [DataFusion](https://datafusion.apache.org/) adapter for
[`ddx`](https://github.com/xqlsystems/ddx), "autograd for composable databases."
Differentiate a whole query, as JAX differentiates a function. DataFusion plans
the query and writes it as Substrait; [`ddx-ad`](https://crates.io/crates/ddx-ad)
differentiates the plan, and DataFusion runs the plans it returns.

## Gradients of whole queries: `ad`

In SQL, `grad(loss, val)` in a `SELECT` over a table is, at each row, the
gradient of the number the CTE `loss` computes; `vjp(f, val, cotangent)` the
same for any CTE `f` along a cotangent; and `jvp(e, w.val, tangent)`, a column
of the `SELECT` computing `e`, its tangent at each row, along a tangent table
or a number (`jvp(sin(x), t.x, 1)` is `cos(x)`):

```rust
use ddx_datafusion::ad;

let step = ad::sql(&ctx, "
    WITH loss AS (SELECT SUM(val * val) AS l FROM w)
    SELECT i, val - 0.1 * grad(loss, val) AS val FROM w").await?;
```

Nothing in the loss is labelled; the one function ddx claims is
`ddx_stop_gradient` (`register_stop_gradient`). `ad::sql_all` runs several
statements that take `grad` of one loss for one backward pass, and
`ad::prepare` prepares them to run on every batch of a training loop,
planning nothing after the first. Calls nest,
so `jvp` of a CTE over `grad(loss, w.val)` is a Hessian-vector product.
`ad::grad`, `ad::vjp` and `ad::jvp` give the program underneath, and
`ad::run(ctx, &program, &inputs)` returns what it computed as DataFrames.
While it runs, a program's tables are named under a prefix of its own
(`__ddx_{id}_`, a reserved prefix), so programs never touch each other's
tables or yours, and neither `ad::sql` nor `ad::run` leaves any behind.
[`examples/nn`](examples/nn) trains a small MLP written entirely in SQL
(adapted from [a neural network written entirely in SQL](https://github.com/xqlsystems/xarray-sql/pull/196) in xarray-sql) with one SQL statement per
parameter table: `cargo run -p ddx-datafusion --example nn`.

## Version compatibility

`ddx-ad` reads and writes the plans `datafusion-substrait` produces and
consumes, so the two must resolve the same `substrait`, and
`tests/substrait_pin.rs` asserts it.

| `ddx-datafusion` | `datafusion` | `substrait` |
|---|---|---|
| 0.1, 0.2 | 54 | 0.63 |
| 0.3 | 55 | 0.63 |

## Status

Alpha, and versioned independently of the other crates. Anything ddx cannot
differentiate is a typed error rather than a silently wrong number.

Licensed under Apache-2.0.
