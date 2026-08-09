# ddx-datafusion

The [DataFusion](https://datafusion.apache.org/) adapter for
[`ddx`](https://github.com/xqlsystems/ddx), "autograd for composable databases."
Write calculus directly in SQL and let DataFusion evaluate the derivative per row
— the relational equivalent of `jax.vmap(jax.grad(f))`:

```sql
SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g
```

`grad`/`jvp` are **markers**, not row functions. They carry a differentiation
request through planning and are always rewritten away *before* execution, so
what DataFusion runs is an ordinary expression it already knows how to evaluate.

All the calculus lives in [`ddx-core`](https://crates.io/crates/ddx-core); this
crate only connects it to an engine.

## Two routes to the same rewrite

```rust,ignore
use datafusion::prelude::SessionContext;

let ctx = SessionContext::new();
ddx_datafusion::install(&ctx);          // registers the markers + the rule

// ...then use the context normally; `grad` needs no special call site.
let df = ctx.sql("SELECT grad(sin(x), x) AS d FROM t").await?;
```

`install` registers two things and needs both: UDFs so the marker calls *parse
and plan*, and the analyzer rule that actually differentiates. The UDFs alone
would let a marker reach execution, where it deliberately errors rather than
computing something.

| | `install` (in-engine) | `ddx_sql` (text rewrite) |
|---|---|---|
| How | an `AnalyzerRule` on the bound plan | rewrites the SQL string first |
| Call style | bare `grad()`, anywhere | `ddx_sql(&ctx, sql)` |
| Works with | SQL **and** the DataFrame API | SQL strings |
| Column identity | resolved by the planner | syntactic |
| Correlated subqueries | ✗ (loud error) | ✓ |
| Errors surface at | `collect()` | the `ddx_sql` call |

**Prefer `install`.** Running after binding means columns arrive already
resolved, so the qualification-ambiguity errors a pre-binding text rewrite has to
raise simply cannot occur.

**Reach for `ddx_sql` when a marker sits inside a correlated subquery** — the one
query shape the in-engine path cannot carry, because re-planning the derivative
against the subquery's own inputs loses the outer reference. It detects this and
says so rather than guessing.

Recursive CTEs are *not* such a shape: `install` carries a marker in a recursive
term perfectly well, which is what makes a training loop expressible as one
query.

## What the two paths disagree about

They drive the same engine, so they agree on the calculus. They can disagree
about **what may be differentiated with respect to**, because they disagree about
what a "column" is:

```sql
SELECT grad(sum(x) * sum(x), sum(x)) AS d FROM t
```

`ddx_sql` refuses it — syntactically `sum(x)` is a function call, and the `wrt`
must be a bare column. `install` answers `2·sum(x)`, because by the time it sees
the plan the planner has lowered the aggregate to a bound column. That is
deliberate: differentiating with respect to a computed alias is already correct
and supported (`grad(s*s, s)` is `2s`), and an aggregate output is the same shape
one level down.

## Version compatibility

This adapter bridges two crates that must agree on a third: it unparses a bound
DataFusion `Expr` into a `sqlparser::ast::Expr` for `ddx-core`. If the two
resolve *different* `sqlparser` versions they are unrelated Rust types and the
bridge does not compile, so the dependency is pinned exactly and a test asserts
the resolved tree still contains exactly one `sqlparser` — a future bump fails at
the pin, with an explanation, rather than confusingly at the bridge.

| `ddx-datafusion` | `datafusion` | `sqlparser` |
|---|---|---|
| 0.1 | 54 | 0.62 |

## Status

Alpha, and versioned independently of `ddx-core` — the two publish on their own
cadence. The differentiation surface is `ddx-core`'s: `+ - * /`, the unary chain
rule for the trig / inverse-trig / exp / log / hyperbolic set plus `abs`, `power`
with a constant base or exponent, higher-order by nesting, and through-aggregate
by linearity. Anything else is a typed error rather than a silently wrong number.

Licensed under Apache-2.0.
