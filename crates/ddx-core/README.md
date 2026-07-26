# ddx-core

Engine-neutral symbolic differentiation of SQL scalar expressions — the v1
core of [`ddx`](https://github.com/xqlsystems/ddx), "autograd for composable
databases." Write calculus directly in SQL and let the engine evaluate the
derivative per row (the relational equivalent of `jax.vmap(jax.grad(f))`):

```sql
SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g
```

`grad`/`jvp` are **markers**, not row functions: they carry a differentiation
request through parsing and are always rewritten away *before* execution.

```rust
use ddx_core::Ddx;
use ddx_core::sqlparser::dialect::GenericDialect;

let ddx = Ddx::new();
let out = ddx.rewrite_sql("SELECT grad(sin(x), x) AS d FROM t", &GenericDialect {})?;
assert_eq!(out, "SELECT (cos(x)) AS d FROM t");
```

The engine differentiates [`sqlparser::ast::Expr`](https://docs.rs/sqlparser)
directly — the AST *is* the IR, there is no bespoke representation. The primary 
dependency is `sqlparser`, **re-exported** as `ddx_core::sqlparser`
so downstream adapters cannot link a mismatched version.

## What it supports

`+ - * /`; the unary chain rule for the trig / inverse-trig / exp / log /
hyperbolic set plus `abs`; `power` with a constant base or exponent;
higher-order via nesting; through-aggregate via linearity
(`AVG(grad(loss, theta))`). Custom unary rules are registrable
(`ddx.register("myfn", rule)` — the rule supplies `f'(u)`, the engine applies
the chain rule). Anything else is a typed `DiffError`, never a silently-wrong
number.

Scalar `vjp` is deliberately **not** here: the name is reserved for the
query-level reverse-mode operation in `ddx-ad` (design.md §3.6, §4).

## Correctness properties worth knowing

- **Identifier folding is per-dialect** (`Ddx::for_datafusion()` vs
  `Ddx::for_duckdb()`): unquoted identifiers always fold case; DuckDB folds
  quoted ones too. `grad(Temp*Temp, temp)` matches.
- **Ambiguity is a hard error, not a guess**: a `wrt` that can't be pinned
  syntactically (`grad(a.x*b.x, x)`) errors rather than differentiating the
  wrong column.
- **`div` forces floating-point division** (`CAST(<numerator> AS DOUBLE)`), so
  integer columns don't silently truncate the derivative.
- **0/1-folding follows JAX's `Zero`-tangent convention** and differs from
  unfolded SQL only on NULL-bearing rows (documented, tested).

See [`../../docs/design.md`](../../docs/design.md) §3 for the full rationale and
the decision log (`F#`/`G#`) behind each of these.

## Status

M0 (the scalar core) — implemented. The per-engine adapters (`ddx-datafusion`,
`ddx-duckdb`), the Python wheel (`ddxdb`), and the query-level v2 engine
(`ddx-ad`) are later milestones; see the design doc's §8.

Licensed under Apache-2.0.
