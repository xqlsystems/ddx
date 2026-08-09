# Changelog

All notable changes to `ddx-datafusion` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries below the first release are maintained automatically by
[release-plz](https://release-plz.dev/).

## [Unreleased]

## [0.1.0] - 2026-08-09

First release. The DataFusion adapter for `ddx-core`: `grad`/`jvp` markers in
SQL, rewritten to derivative expressions before execution.

### Added

- **`install(&ctx)`** — the in-engine path. Registers the marker UDFs so
  `grad(...)` parses and plans, plus an `AnalyzerRule` that differentiates on the
  *bound* plan. Because it runs after binding, columns arrive already resolved by
  the planner, so the qualification ambiguities a pre-binding text rewrite has to
  refuse cannot arise. Works with the DataFrame API as well as SQL, and carries a
  marker inside a recursive CTE — the shape a training loop needs.
- **`ddx_sql(&ctx, sql)`** — the text-rewrite path, for the one query shape the
  in-engine path cannot carry: a marker inside a *correlated subquery*, where
  re-planning the derivative against the subquery's own inputs would lose the
  outer reference. The in-engine path detects that case and errors rather than
  guessing.
- **`install_with` / `ddx_sql_with`** — the same two routes driven by a
  caller-configured engine, for picking up custom differentiation rules.
  User-defined DataFusion UDFs need no registration with ddx: a function called
  inside a marker is read off the bound expression when the derivative is
  re-planned.

### Notes

- **The two paths agree on the calculus and can disagree on what may be the
  `wrt`.** `grad(sum(x) * sum(x), sum(x))` is refused by `ddx_sql` — syntactically
  `sum(x)` is a function call, and the `wrt` must be a bare column — and answered
  by `install` as `2·sum(x)`, because the planner has already lowered the
  aggregate to a bound column. Differentiating with respect to a computed alias
  is an operation ddx already supports, so refusing it one level down would be
  the inconsistency.
- **`datafusion` and `ddx-core` must resolve the same `sqlparser`.** The bridge
  unparses a bound DataFusion `Expr` into a `sqlparser::ast::Expr`; two different
  `sqlparser` versions are two unrelated Rust types and it stops compiling. The
  pin is exact and `tests/sqlparser_pin.rs` asserts the resolved tree still holds
  exactly one, so a future bump fails at the pin with an explanation rather than
  confusingly at the bridge. This release: `datafusion` 54, `sqlparser` 0.62.
