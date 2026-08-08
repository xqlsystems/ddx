# Changelog

All notable changes to `ddx-core` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries below the first release are maintained automatically by
[release-plz](https://release-plz.dev/).

## [Unreleased]

## [0.1.2](https://github.com/xqlsystems/ddx/compare/ddx-core-v0.1.1...ddx-core-v0.1.2) - 2026-08-08

### Fixed

- *(ddx-core)* parenthesize a right operand that binds as tightly as its parent ([#53](https://github.com/xqlsystems/ddx/pull/53))

## [0.1.1](https://github.com/xqlsystems/ddx/compare/ddx-core-v0.1.0...ddx-core-v0.1.1) - 2026-08-08

### Added

- `power` with a constant base or exponent now accepts one written as a *cast*
  literal — `power(x, CAST(3 AS DOUBLE))` differentiates where it previously
  returned `NotImplemented`. A cast to a numeric type is recognised as the
  constant it wraps, which matters because query engines inject these: type
  coercion rewrites `power(x, 3)` over a `DOUBLE` column into
  `power(CAST(x AS DOUBLE), CAST(3 AS DOUBLE))` before ddx ever sees it. Casts
  to non-numeric types are still not constants (`CAST(1 AS VARCHAR)` is the
  string `'1'`).
- `ddx_core::test_utils`, behind the off-by-default `test-utils` feature: the
  expression generator, reference interpreter, numeric conditioning gates and
  failure reporter that `ddx-core`'s own property suite runs on. Exposed so that
  crates building on `ddx-core` can fuzz against the *same* generator rather
  than inventing their own. Test support, not API — **semver-exempt**, and
  compiled only when you turn the feature on, so a default build of `ddx-core`
  is byte-for-byte unaffected by it.

### Notes

- The engine itself is unchanged: no differentiation rule was added, removed or
  altered, and every derivative `ddx-core` emitted at 0.1.0 it still emits.
- This release accompanies the first real `ddx-datafusion` adapter
  ([#49](https://github.com/xqlsystems/ddx/pull/49)), which is not yet published.

## [0.1.0] - 2026-07-26

### Added

- Initial release: the v1 scalar differentiation engine (design.md Milestone 0).
- `Ddx` — the engine object: `rewrite_sql` (the whole `grad`/`jvp` marker path,
  byte-identical outside the marker), `explain` (preview a rewrite without
  running it), `differentiate` / `jvp` / `differentiate_sql` (the lower-level
  "calculus compiler" surface), and `register` for user-defined rules.
- Per-dialect identifier folding (`Ddx::for_datafusion()` / `for_duckdb()`) and
  an extensible, name-keyed rule registry.
- Differentiation surface: `+ - * /`; the unary chain rule for the trig /
  inverse-trig / exp / log / hyperbolic set plus `abs`; `power` with a constant
  base or exponent; higher-order via nesting; through-aggregate via linearity.
  Unsupported constructs are typed errors with actionable guidance — never a
  silently-wrong number.
- Depends on `sqlparser` only (pinned `=0.62.0`, re-exported as
  `ddx_core::sqlparser`).

[Unreleased]: https://github.com/xqlsystems/ddx/compare/ddx-core-v0.1.0...HEAD
[0.1.0]: https://github.com/xqlsystems/ddx/releases/tag/ddx-core-v0.1.0
