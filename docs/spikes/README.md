# Spikes — the evidence behind the design

Every load-bearing claim in [`../docs/design.md`](../docs/design.md) that could be
checked with a small program was checked with one. These are those programs. Each is
self-contained and prints a pass/fail; they double as regression fixtures for the
crates once they exist. `design.md`'s Decision Log (`F#`/`G#`/`R#`/`S#` tags) is the
audit trail for why each of these was run; the "Design ref" column below points at
the main narrative section each spike backs.

| Spike | Verifies | Design ref |
| --- | --- | --- |
| `relational_ad_spike.py` | An MLP's whole backward pass = mechanical application of **five transpose rules** over relational primitives; all 6 param grads match `jax.grad` to ~1e-18. Reproduces xarray-sql#196's hand-written `delta*`/`g*` queries exactly. | §4.1, §4.3 |
| `attention_ad_spike.py` | Same rules cover **scaled dot-product attention** (Q/K/V projections, QKᵀ, softmax over the key axis, A@V); grads w.r.t. Wq/Wk/Wv/X match `jax.grad` to ~1e-16. Generality beyond the MLP, and the fan-in accumulation case (`X` feeds `Wq`/`Wk`/`Wv`). | §4.1, §4.4 |
| `attention_causal_mask_check.py` | The transformer **causal mask** is just elementwise — masked attention grads still match `jax.grad` to ~1e-16, no new rule. | §4.1 |
| `sqlparser-spike/` (Rust) | `sqlparser`'s `Display` drops precedence parens on *constructed* trees (`(a+b)*c` → `a + b * c`), and `Nested`-wrapping fixes it. Spans are 1-based *characters*, not byte offsets. Decision log: `G1`, `G3`. | §3.2, §5, M0 |
| `substrait_ad_marker_spike.py` | Adopting Substrait + custom extension-function markers (not a bespoke Rust IR) for v2: a `ddx_contract_mark(...)` marker wrapped around an aggregate's operand survives DuckDB's own `get_substrait`→`from_substrait` round-trip AND a genuine cross-engine hop (DataFusion produces the marker-tagged plan, DuckDB consumes and executes it) — numerically exact both ways. DuckDB→DataFusion deserializes cleanly (execution not yet exercised). DuckDB's `substrait` extension is community-maintained, not core, as of 1.5.4 (`INSTALL substrait` 404s; `INSTALL substrait FROM community` works). Decision log: `S1`, `S2`. | §4.2 |
| `route_ad_spike.py` | The Route (argmax/max-pool) transpose rule vs. `jax.grad`: machine-exact (0.00e+00) away from ties. **At an exact tie**, our SQL-idiom's deterministic first-index tiebreak diverges from `jax.grad(jnp.max)`'s tie-splitting convention — both defensible, must be pinned explicitly (same treatment as the `abs`-at-0 kink), not assumed to agree with JAX. Decision log: `S3`. | §4.3, §5 |
| `duckdb_substrait_window_bug.py` | Route's forward SQL (`ROW_NUMBER()` top-1-per-group) through Substrait: a plain window column round-trips fine through DuckDB, but the **full top-1-per-group idiom silently returns the wrong (unfiltered) rows** — no exception — because DuckDB's own optimizer rewrites it into an `arg_max` join before Substrait export, and that rewritten form doesn't survive the round-trip. Reproduces with no ddx marker involved. DataFusion round-trips the identical idiom correctly, isolating this as a DuckDB-specific bug, not a general Substrait-window gap. A two-step workaround (Substrait-round-trip the window column, then filter with plain engine-native SQL) is verified to produce the correct result — Route does not need to wait on an upstream fix to ship. Decision log: `S4`, `S5`. | §4.3, §4.6 |
| `duckdb_reentrancy_r1b.py` | A query on a 2nd connection to the same DuckDB DB, run during an outer query, is safe (reads, DML, no deadlock) but runs in its own transaction (can't see uncommitted state). Decision log: `R1b`. | §3.4 |
| `datafusion_python_analyzer_rule_r2.py` | **M1.** `datafusion-python` (54.0.0) cannot inject an `AnalyzerRule`, checked three ways: no `add_analyzer_rule` on the public `SessionContext`, none on the `_internal` PyO3 class (which exposes no `SessionState` handle either), and no `__datafusion_analyzer_rule__` in the compiled **FFI capsule vocabulary** — so a compiled Rust extension can't either. The logical optimizer is closed the same way (`remove_optimizer_rule`, no `add`); the one rule capsule that exists is *physical*, i.e. post-planning. Also pins §3.1 live: a `grad` *scalar* UDF receives `[1.0, 4.0, 9.0]` for `grad(x*x, x)` — evaluated values, never the expression. Three seams enumerated: plan serialization and `add_physical_optimizer_rule` (both rejected), plus **table functions**, which *do* receive an unevaluated `RawExpr` — the DataFusion analogue of DuckDB's `ddx('<sql>')`, worth adopting in M2 (→ #73), but not a route to bare `grad()`: composite args arrive constant-folded, and args resolve against an *empty schema*. That second limit is narrower than "columns are rejected" — a bare `x` passes through as an unresolved `Expr(x)`; what dies is anything needing type resolution, so `ddx(x+1)` and `ddx(grad(x*x,x))` both fail with "No field named x". Only opaque leaves survive, so the seam carries a SQL *string*. Exits non-zero and withholds its verdict if any check fails, so a future upstream change can't be read as confirmation. Decision log: `R2`. | §3.3, §3.4, §8 M1 |
| `substrait_limitation_repro.py` | `datafusion-substrait`'s producer rejects recursive CTEs and DML (`Unsupported plan type: RecursiveQuery` / `DmlStatement`) — why Substrait isn't v1's transport, and why v1 and v2 need different mechanisms. | §1.1, §4.2, ddx#1 |

## Running them

Python spikes (a venv with the deps):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy jax duckdb datafusion pyarrow  # jax for the AD spikes; duckdb/datafusion/pyarrow for the engine ones
python docs/spikes/relational_ad_spike.py             # → W2..b0 max|rule - jax.grad| ~1e-18  OK
python docs/spikes/attention_ad_spike.py              # → Wq/Wk/Wv/X ~1e-16  OK
python docs/spikes/attention_causal_mask_check.py     # → causal Wq/Wk/Wv ~1e-16  OK
python docs/spikes/duckdb_reentrancy_r1b.py
python docs/spikes/datafusion_python_analyzer_rule_r2.py  # → 19/19 checks OK (R2 confirmed; exits 1 if not)
python docs/spikes/substrait_limitation_repro.py
python docs/spikes/substrait_ad_marker_spike.py       # → 4/4 checks OK (DuckDB round-trip + cross-engine)
python docs/spikes/route_ad_spike.py                  # → 0.00e+00 vs jax.grad; ties diverge (pin explicitly)
python docs/spikes/duckdb_substrait_window_bug.py     # → A/C/D OK, B silently wrong (DuckDB bug, workaround verified)
```

Rust spike (`sqlparser` 0.62):

```bash
cd docs/spikes/sqlparser-spike && cargo run
# G1 constructed (a+b)*c   Display => a + b * c      (WRONG — reparses as a+(b*c))
# G1 fixed  Nested(a+b)*c  Display => (a + b) * c    (correct)
# G3 'grad' byte offset=17, char offset=16           (spans are characters)
```

## Note on the AD spikes

`relational_ad_spike.py` and `attention_ad_spike.py` implement **only** the transpose
(VJP) rules for the relational primitives in `design.md` §4.3 — contraction
(`JOIN`+`GROUP BY SUM`), elementwise map (whose local derivative is `ddx-core`'s
scalar `grad`), per-group `SUM`, and broadcast/bias — and compose them in reverse.
Nothing else is hand-written; the softmax/softmax-cross-entropy deltas *fall out* of
the primitives. That they match `jax.grad` to machine precision is the concrete
evidence that query-level reverse-mode AD (`design.md` §4) is an engineering
project, not research. The published precedent is Tang et al., *Auto-Differentiation
of Relational Computations …*, ICML 2023.
