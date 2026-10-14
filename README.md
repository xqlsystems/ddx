# ddx
_[JAX](https://docs.jax.dev/en/latest/)-style [automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html) in SQL_

[![crates.io](https://img.shields.io/crates/v/ddx-core.svg?label=ddx-core)](https://crates.io/crates/ddx-core)
[![crates.io](https://img.shields.io/crates/v/ddx-datafusion.svg?label=ddx-datafusion)](https://crates.io/crates/ddx-datafusion)
[![PyPI](https://img.shields.io/pypi/v/ddxdb.svg?label=ddxdb)](https://pypi.org/project/ddxdb/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Write calculus in SQL. ddx takes the derivative of a query with respect to a
table's columns, the way JAX takes the derivative of a function, and the
engine computes it alongside everything else:

```sql
WITH f AS (SELECT SUM(val * val) AS l FROM w)
SELECT w.i, g.val AS df_dval
FROM w JOIN grad(f, w.val) g ON w.i = g.i     -- 2·val, row by row
```

The query being differentiated is ordinary SQL: joins, sums, averages, max and
min, window rankings. The engine plans it and writes it as Substrait; ddx
differentiates the plan and hands back Substrait plans for the engine to run.

## Try it

```sh
pip install "ddxdb[datafusion]"
```

```python
from ddxdb import Context

ctx = Context()
ctx.sql("CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)")
ctx.sql("""
    WITH loss AS (SELECT SUM(val * val) AS l FROM w)
    SELECT i, val FROM grad(loss, w.val) ORDER BY i
""").collect()   # 2.0, 4.0
```

From Rust, `ddx_datafusion::ad::sql(&ctx, statement)` does the same.

## Optimization: training, fitting, equilibrium

A gradient is what an optimizer steps along, so anything a query can score, a
query can optimize: a model's loss, a curve's misfit, a physical system's
energy. A step, `params - lr * grad(f)(params)`, is a join of the parameter
table with its gradient. Training a neural network is one such case:

```sql
WITH h AS (
  SELECT x.sample, w.out, tanh(SUM(x.val * w.val)) AS val
  FROM x JOIN w ON x.inp = w.inp GROUP BY x.sample, w.out),
loss AS (
  SELECT SUM(power(h.val - y.val, 2)) AS l
  FROM h JOIN y ON h.sample = y.sample AND h.out = y.out)
SELECT w.inp, w.out, w.val - 0.1 * g.val AS val
FROM w JOIN grad(loss, w.val) g ON w.inp = g.inp AND w.out = g.out
```

That runs as written through `ddxdb.Context` in Python and
`ddx_datafusion::ad::sql` in Rust. ddx differentiates the loss the way
`jax.grad` differentiates a function: reverse mode, one transpose rule per
relational operator, and nothing in the query is labelled. The backward pass is a sequence of
plain Substrait plans the engine runs.
[`examples/nn`](crates/ddx-datafusion/examples/nn) trains a small classifier
this way: a two-layer MLP over 6×6 images, adapted from [a neural network written entirely in SQL](https://github.com/xqlsystems/xarray-sql/pull/196)
in xarray-sql, which computed its gradients with hand-written backward queries.
Here its loss query is used unchanged, and ddx's gradients equal those
hand-written ones to 1e-12. The spikes' MLP, attention and max-pool gradients
equal `jax.grad` to 1e-12.

## Status

Query-level AD (`ddx-ad`, with `ddx_datafusion::ad` and `ddxdb.ad` on top) is
built: it matches `jax.grad` on every fixture, trains an MLP on DataFusion, and
ships with the next release. ddx's first design, which rewrote `grad(x * y, x)` in a
select list to derivative SQL one row at a time, has been removed; its
derivative rules live on as `ddx-core`, which `ddx-ad` uses for each
expression.

**DataFusion is the engine with native support.** Any other engine whose
Substrait producer and consumer ddx-ad understands can run the programs
through `ddxdb.run` and the `ddxdb.Backend` protocol.

| | | |
|---|---|---|
| [`ddx-ad`](crates/ddx-ad) | query-level AD over Substrait | next release |
| [`ddx-core`](crates/ddx-core) | the scalar derivative rules `ddx-ad` uses | [crates.io](https://crates.io/crates/ddx-core) |
| [`ddx-datafusion`](crates/ddx-datafusion) | DataFusion adapter: `ad` | [crates.io](https://crates.io/crates/ddx-datafusion) |
| [`ddxdb`](python/ddxdb) | Python wheel: a DataFusion `Context`, `ad`, and the programs for any engine | [PyPI](https://pypi.org/project/ddxdb/) |
| `ddx-duckdb` | DuckDB community extension | planned |

Next is DuckDB: a community extension that runs ddx's programs through
DuckDB's own Substrait consumer. The milestone plan is in
[docs/design.md](docs/design.md).

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
  Query-level AD has its own: generated loss queries whose gradients must match
  a finite difference of the query, and a set of identities that must hold
  between ddx's own programs.
- **Pinned conventions** where ddx and JAX differ on purpose rather than one
  being wrong — `abs` at its kink, missing values, domain edges — asserted from
  both sides so a change to either is visible.

## Layout

```
crates/
  ddx-ad/           # query-level AD over Substrait
  ddx-core/         # the scalar derivative rules ddx-ad uses
  ddx-datafusion/   # DataFusion adapter: ad
python/ddxdb/       # PyO3/maturin wheel: a DataFusion Context + ad
tests/              # numeric-agreement suites (vs JAX)
docs/spikes/        # runnable evidence for every design claim
docs/design.md      # the design
docs/fast-linalg-notes.md  # notes seeding a future fast linear-algebra project
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
