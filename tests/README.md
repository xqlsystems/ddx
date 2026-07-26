# Cross-engine tests (vs JAX)

This directory holds the **cross-engine numeric-agreement suites** (design.md
§5, §6): the same `grad`/`jvp` expression, rewritten per-dialect by `ddx-core`,
must evaluate to numerically equal columns in DuckDB and DataFusion — checked
against **JAX** (`jax.grad`) as the oracle, since the whole design mirrors JAX's
forward/reverse structure and the same seed/cotangent semantics.

**Status: scaffold.** The minimal JAX-oracle harness is pulled forward into M2
(its exit gate — "xarray-sql green on `ddx-core` vs. JAX, no regressions" —
depends on it); the broader suites land in M6 (design.md §8). Planned coverage:

- **Numeric agreement vs JAX** for every rule, with finite-difference as a
  cheap independent cross-check where a JAX equivalent is awkward.
- **Cross-engine equivalence**: DuckDB vs DataFusion on the identical
  per-dialect rewrite.
- **Convention-pinning tests** (not blind oracle comparison) where a convention
  genuinely differs rather than one side being wrong (design.md §5):
  - *Kinks* — `abs` at 0 gives `0` from the `signum` rule; pin the value.
  - *Domain-widening* — a derivative can fail where the primal doesn't
    (`sqrt(x)` fine at 0, `1/(2*sqrt(x))` divides by zero); sample away from
    edges or pin per-engine behavior.
  - *NULL/folding* — folded and unfolded derivatives agree everywhere except
    the documented NULL-row cases (the JAX-`Zero`-tangent convention, F11).

The Rust unit and integration tests for the engine itself live with the crate,
in [`../crates/ddx-core/tests`](../crates/ddx-core/tests) (the ported rule
tests, span splicing, the guards, identifier folding, and the semantic
round-trip property test). The runnable spikes that back each design claim are
in [`../spikes`](../spikes).
