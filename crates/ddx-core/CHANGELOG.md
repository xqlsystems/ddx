# Changelog

All notable changes to `ddx-core` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries below the first release are maintained automatically by
[release-plz](https://release-plz.dev/).

## [Unreleased]

## [0.2.0](https://github.com/xqlsystems/ddx/compare/ddx-core-v0.1.3...ddx-core-v0.2.0) - 2026-08-09

### Added

- *(ddxdb)* the Python wheel — rewrite_sql, a Context shim, a DuckDB path ([#58](https://github.com/xqlsystems/ddx/pull/58))

## [0.1.3](https://github.com/xqlsystems/ddx/compare/ddx-core-v0.1.2...ddx-core-v0.1.3) - 2026-08-08

### Other

- *(ddx-core)* skip points where a divisor has cancelled to rounding noise ([#56](https://github.com/xqlsystems/ddx/pull/56))

## [0.1.2](https://github.com/xqlsystems/ddx/compare/ddx-core-v0.1.1...ddx-core-v0.1.2) - 2026-08-08

### Fixed

- **Rendering a derivative could re-associate it, producing a wrong number in
  valid SQL.** A derivative whose right operand bound as tightly as its parent
  lost its parentheses: `a * (b / c)` was written as `a * b / c`, which reads
  back as `(a * b) / c`. The two agree in exact arithmetic and diverge without
  bound as `c` approaches zero, so a result could be off by any amount — the
  continuous fuzz caught one wrong by twenty-four orders of magnitude
  ([#53](https://github.com/xqlsystems/ddx/pull/53)).

  **Who is affected.** Anyone who takes the *text* of a derivative and reparses
  it — `differentiate_sql`, or `rewrite_sql` output fed to an engine, which is
  the normal path. Derivatives consumed as an `Expr` in memory were never
  affected, because nothing reparsed them. The trigger needs a quotient inside a
  product or a nested quotient, which the quotient and chain rules build
  routinely, so upgrading is worthwhile even if you have not seen a bad value:
  the error is silent and only large where a denominator is near zero.

  Emitted SQL now carries a few more parentheses on same-precedence chains.
  Nothing else about the derivatives changed.

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
