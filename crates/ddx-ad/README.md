# ddx-ad

Query-level reverse-mode automatic differentiation for SQL: the v2 engine of
[ddx](https://github.com/xqlsystems/ddx).

`ddx-core` differentiates one scalar expression. `ddx-ad` differentiates a whole
query, the way `jax.grad` differentiates a function. Write a model's forward pass
and loss as one SQL query, and `grad` returns the queries that compute the
loss's gradient with respect to table columns:

```sql
WITH h AS (
  SELECT x.sample, w.out, SUM(x.val * w.val) AS z
  FROM x JOIN w ON x.inp = w.inp
  GROUP BY x.sample, w.out)
SELECT SUM(power(tanh(h.z) - y.val, 2)) AS loss
FROM h JOIN y ON h.sample = y.sample AND h.out = y.out
```

No part of the query is labelled for this. A query is a composition of
relational primitives (map, select, join, grouped aggregate), and each has a
transpose rule, as each JAX primitive does: a `SUM`'s transpose broadcasts, a
join's transpose sums back, and a projected expression's local derivatives come
from `ddx-core`. `SUM`, `AVG`, `MAX`, `MIN` and `COUNT` have rules. The one
function ddx claims is `ddx_stop_gradient(x)`, JAX's `lax.stop_gradient`.

As in JAX, `grad` needs a loss (one row, one column) and `vjp` pulls back a
cotangent of any output. A gradient comes back shaped like its table: the
table's dims (the columns not differentiated) and the gradient of each `wrt`
column under the column's own name.

`ddx-ad` reads and writes [Substrait](https://substrait.io/) plans, so it
depends on no engine. An engine adapter produces the forward query's plan and
runs the plans that come back, in order, materializing each under its name.
For DataFusion that adapter is
[`ddx-datafusion`](https://docs.rs/ddx-datafusion)'s `ad` module.

An unsupported construct on the path from a `wrt` column to the loss is a typed
error, never a wrong or silently zero gradient. See
[`docs/design.md`](https://github.com/xqlsystems/ddx/blob/main/docs/design.md)
§4 for the design.

`substrait` is pinned exactly and re-exported as `ddx_ad::substrait`: an adapter
must pass the same `substrait::proto::Plan` type its engine's producer uses.

Building needs `protoc`, the Protocol Buffers compiler.

## License

Apache-2.0.
