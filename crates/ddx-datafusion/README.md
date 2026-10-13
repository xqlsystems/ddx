# ddx-datafusion

The [DataFusion](https://datafusion.apache.org/) adapter for
[`ddx`](https://github.com/xqlsystems/ddx), "autograd for composable databases."
Differentiate a whole query, as JAX differentiates a function. DataFusion plans
the query and writes it as Substrait; [`ddx-ad`](https://crates.io/crates/ddx-ad)
differentiates the plan, and DataFusion runs the plans it returns.

## Gradients of whole queries: `ad`

In SQL,
`grad(loss, table.column)` in a `FROM` clause is the gradient of the loss a CTE
computes, as a relation shaped like the table:

```rust
use ddx_datafusion::ad;

let step = ad::sql(&ctx, "
    WITH loss AS (SELECT SUM(val * val) AS l FROM w)
    SELECT w.i, w.val - 0.1 * g.val AS val
    FROM w JOIN grad(loss, w.val) g ON w.i = g.i").await?;
```

Nothing in the loss is labelled; the one function ddx claims is
`ddx_stop_gradient` (`register_stop_gradient`). `ad::sql_all` runs several
statements that take `grad` of one loss for one backward pass. Calls nest,
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
| 0.1 | 54 | 0.63 |

## Status

Alpha, and versioned independently of the other crates. Anything ddx cannot
differentiate is a typed error rather than a silently wrong number.

Licensed under Apache-2.0.
