<!--
SPDX-FileCopyrightText: 2026 OpenAI
SPDX-License-Identifier: Apache-2.0
-->

# Exact GELU and reusable primal outputs for SQL training

These changes let [xarray-sql](https://github.com/xqlsystems/xarray-sql),
a SQL interface to labeled arrays, train a
[nanoGPT](https://github.com/karpathy/nanoGPT) character model through
ddx-generated gradients. ddx remains the automatic differentiation engine:
the downstream GPU executor evaluates its SQL programs and does not replace
them with a framework backward pass.

## Error-function differentiation

The scalar registry now supports `erf`, the error function used in exact
GELU: `GELU(x) = x * (1 + erf(x / sqrt(2))) / 2`. Its derivative is
`2 / sqrt(pi) * exp(-x*x)`, composed with the existing chain rule.
The SQL plan reader/writer preserves `erf` in generated programs, so a
derivative containing the original primal function can be serialized and
executed. No tanh approximation is substituted for exact GELU.

The engine executing the SQL must register an `erf` implementation. This
change adds differentiation and round-tripping, not an engine-specific
SQL function. xarray-sql registers the DataFusion function and an
equivalent trusted array primitive; replacing that function invalidates
array lowering permission there.

The Rust scalar-rule test checks chain-rule composition and the analytic
second derivative, rather than only matching one first derivative. Existing
unsupported-expression and derivative semantics remain in effect. Exact
GELU support is a feature/correctness change; no isolated speedup or memory
reduction was measured for the rule itself.

## Read the primal already computed by a prepared derivative

`ddx_ad::Prepared::values()` and Python `ddxdb.Prepared.values` expose
`OutputTable` metadata for the primal of each derivative job, in job order.
The table rows can be read **during a successful run, before release**:

```python
prepared = ddxdb.prepare(backend, gradient_statements)
with prepared.run() as rewritten:
    outputs = prepared.values
    loss_sql = outputs[0].query()
    # Execute loss_sql on backend while this run's tables exist.
```

Before the first run, Python raises `RuntimeError`. Metadata is not a
materialized snapshot or permission to retain temporary tables after
release. Statements without derivative jobs produce no primal entries.
All existing release/error cleanup and prepared-session invalidation rules
continue to apply. Callers must release one run before beginning another.

Previously, downstream training ran a separate forward query solely to
report loss. The prepared derivative already computes that primal. Exposing
its output descriptor permits reuse without reading private program state,
changing differentiation, retaining tables indefinitely, or rerunning loss.
The binding returns metadata only; it does not add GPU dependencies to ddx.

The Python test `python/ddxdb/tests/test_prepared_values.py` checks the
public contract: access before running fails; the primal and gradient
reflect changed input rows on a second run; the temporary primal table is
unavailable after release. Downstream tests check changing compiled batch
and learning-rate values, loss snapshots and failed-step atomicity.

## Measured downstream effect and attribution

November 1, 2026: Apple M5 Ultra, 30 CPU cores, 64 GPU cores, 256 GiB,
macOS 27.0.1. xarray-sql's real Shakespeare recipe has 10,745,088 parameters,
batch 12, context 256, 6 layers/heads, width 384, exact GELU, dropout 0.2,
global clipping and AdamW. These are training updates from scratch, not
inference or a 123M-parameter model.

| Downstream SQL implementation | First step | Warm step |
|---|---:|---:|
| Compact GPU with Arrow state outputs and separate loss | 2.523 s | 2.024 s, updates 2–5 |
| Owned resident state and shared primal, uncompiled | 1.840 s | 1.356 s, updates 2–5 |
| Complete compiled GPU step with GPU dropout | 1.468 s | 101.3 ms, updates 2–100 |

Primal reuse contributes to the resident implementation's **1.49×**
five-step warm improvement, together with state ownership and removal of
Arrow state export/registration. It was not measured in isolation.
Whole-step compilation, compact index operations and GPU dropout bring
the combined downstream improvement to **19.98×** over the old 2.024 s
GPU path. These gains must not be attributed to this small ddx API alone.

All 100 timed compiled steps exported zero arrays and read back zero array
bytes, with one validation flag and one loss scalar per update. A separate
five-update run checked every loss, raw gradient, weight and AdamW moment
against original nanoGPT/PyTorch (float32 tolerance 3e-5). The longer run
measured performance without per-step oracle exports. Startup and shader
cache effects were not isolated; no ddx-specific RSS reduction is claimed.

See xarray-sql's `NANOGPT_PERFORMANCE.md` and `benchmarks/apple/resident-summary.json`
for commands, timing scope, memory counters, native PyTorch comparison and
raw evidence. The downstream non-integration suite passed 714 tests.

## Reproduce and review

The tested base is ddx release commit
`af71683f6678bb093e600f948c212490a4d8b74b` (Python 0.4.6), rather than an
unreleased main snapshot. The local downstream build was labeled
`0.4.6+xqlmath.3`; upstream package versions are not changed by this work.

The changes are split into error-function differentiation and the prepared
primal-output API. Neither requires an MLX dependency or relaxed derivative
checks in ddx. Run the Rust workspace checks per `CONTRIBUTING.md`, then
build the Python extension and run its tests, including the new lifetime
and current-input test.

The persistent checkout passed the Rust workspace suite (535 passed,
10 ignored), workspace and Python-extension Clippy with warnings denied,
the repository-pinned nightly formatter and warning-free Rust docs. Its
rebuilt wheel passed the combined ddx/EinFold Python binding suite
(126 passed, 6 optional skips). Rebuilt persistent dependency wheels also
passed xarray-sql's full 714-test non-integration suite.

After integrating EinFold's newer upstream runtime changes, a final wheel
rebuild/recheck passed 128 combined binding tests (6 optional skips) and
again all 714 downstream tests. The two additional binding tests came
from upstream EinFold; ddx numerical code was unchanged.
