# ddx
_[JAX](https://docs.jax.dev/en/latest/)-style [automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html) in SQL_

[![crates.io](https://img.shields.io/crates/v/ddx-core.svg?label=ddx-core)](https://crates.io/crates/ddx-core)
[![crates.io](https://img.shields.io/crates/v/ddx-datafusion.svg?label=ddx-datafusion)](https://crates.io/crates/ddx-datafusion)
[![PyPI](https://img.shields.io/pypi/v/ddxdb.svg?label=ddxdb)](https://pypi.org/project/ddxdb/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Write calculus directly in SQL and get derivatives back as ordinary columns,
evaluated row by row by the engine alongside everything else:

```sql
SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g
```

`grad`/`jvp` are compile-time **markers**: they carry a differentiation request
through parsing and are rewritten away — to plain derivative SQL — before the
query ever runs. Differentiating a column expression and letting the engine
evaluate it per row is the relational equivalent of `jax.vmap(jax.grad(f))`, with
the rows as the batch dimension.

One engine-neutral Rust core, thin per-engine adapters.

## Try it

```sh
pip install ddxdb
```

```python
import ddxdb

ddxdb.rewrite_sql("SELECT grad(sin(x), x) AS d FROM t")
# 'SELECT (cos(x)) AS d FROM t'
```

`rewrite_sql` is text in, text out, so it works with **any** engine that accepts
SQL — pass the result wherever you would have passed the original:

```python
con.sql(ddxdb.rewrite_sql(q, "duckdb"))        # DuckDB
session.sql(ddxdb.rewrite_sql(q, "spark"))     # Spark
ctx.sql(ddxdb.rewrite_sql(q))                  # DataFusion
```

It has no required runtime dependencies — the engines are optional extras — so
depending on it costs nothing until you use it.

From Rust:

```rust
use ddx_core::Ddx;
use ddx_core::sqlparser::dialect::GenericDialect;

let out = Ddx::new()
    .rewrite_sql("SELECT grad(sin(x), x) AS d FROM t", &GenericDialect {})
    .unwrap();
assert_eq!(out, "SELECT (cos(x)) AS d FROM t");
```

On DataFusion, `ddx-datafusion` installs an `AnalyzerRule` so bare `grad()` works
in ordinary SQL *and* through the DataFrame API, with columns resolved by the
planner rather than syntactically.

## Status

**M2 landed and released.** The scalar engine, the DataFusion adapter and the
Python wheel are all published.

**DataFusion is the engine with native support**: `ddx-datafusion` installs an
`AnalyzerRule`, so bare `grad()` works in ordinary SQL and through the DataFrame
API. Every other engine — DuckDB included — goes through `rewrite_sql` today: you
rewrite the text and hand the result to your own connection. A native DuckDB
extension, with `grad()` understood in-database, comes eventually (M5).

| | | |
|---|---|---|
| [`ddx-core`](crates/ddx-core) | the v1 engine | [crates.io](https://crates.io/crates/ddx-core) |
| [`ddxdb`](python/ddxdb) | Python wheel — `rewrite_sql` + a DataFusion `Context` | [PyPI](https://pypi.org/project/ddxdb/) |
| [`ddx-datafusion`](crates/ddx-datafusion) | DataFusion adapter: `AnalyzerRule` + `ddx_sql` | [crates.io](https://crates.io/crates/ddx-datafusion) |
| [`ddx-ad`](crates/ddx-ad) | v2 — query-level reverse-mode AD over Substrait | M3/M4 |
| `ddx-duckdb` | DuckDB community extension | M5 |

Next is **M3/M4** — reverse-mode AD over whole *queries*, where a gradient step
becomes a query rather than a column. See [docs/design.md](docs/design.md) §8.

## Correctness

The governing constraint is **fail loud, never silently wrong**: an expression
ddx cannot differentiate is a typed error naming what it could not handle, never
a plausible number. What backs that up:

- **A JAX oracle.** [`tests/`](tests) generates a function, traces it to a jaxpr,
  and hands that *one* object to both sides — rendered as SQL for ddx,
  differentiated by JAX for the oracle — then compares the columns DataFusion and
  DuckDB actually produce. Central differences cross-check independently, since
  JAX and ddx share a structure and a common misconception would be invisible
  between them.
- **A property suite** over randomly generated expressions, with conditioning
  gates so float noise is not mistaken for a bug, plus a soak that runs nightly.
- **Pinned conventions** where ddx and JAX differ on purpose rather than one
  being wrong — `abs` at its kink, missing values, domain edges — asserted from
  both sides so a change to either is visible.

## Layout

```
crates/
  ddx-core/         # v1 engine — differentiate sqlparser::ast::Expr + rewrite_sql
  ddx-ad/           # v2 engine — query-level reverse-mode AD over Substrait
  ddx-datafusion/   # DataFusion adapter: AnalyzerRule (bare grad) + ddx_sql
python/ddxdb/       # PyO3/maturin wheel: rewrite_sql + a DataFusion Context
tests/              # cross-engine numeric-agreement suites (vs JAX)
docs/spikes/        # runnable evidence for every design claim
docs/design.md      # the design
```

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Copyright 2026 Alexander Merose

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
