# Changelog

All notable changes to `ddx-core` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries below the first release are maintained automatically by
[release-plz](https://release-plz.dev/).

## [Unreleased]

## [0.1.1](https://github.com/xqlsystems/ddx/compare/ddx-core-v0.1.0...ddx-core-v0.1.1) - 2026-08-08

### Added

- *(ddx-datafusion)* bare grad() via an AnalyzerRule, plus the ddx_sql helper ([#49](https://github.com/xqlsystems/ddx/pull/49))

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
