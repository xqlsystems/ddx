# Changelog

All notable changes to `ddx-datafusion` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries below the first release are maintained automatically by
[release-plz](https://release-plz.dev/).

## [Unreleased]

## [0.2.3](https://github.com/xqlsystems/ddx/compare/ddx-datafusion-v0.2.2...ddx-datafusion-v0.2.3) - 2026-10-20

### Fixed

- *(ddx-ad)* count a near-tie at MAX or MIN in jvp as grad does ([#180](https://github.com/xqlsystems/ddx/pull/180))
- *(ddx-datafusion)* refuse a view over a table registered again since the view was made ([#175](https://github.com/xqlsystems/ddx/pull/175))
- *(ddx-ad)* give a join with no condition the condition true ([#174](https://github.com/xqlsystems/ddx/pull/174))

### Other

- *(ddx-ad)* walk plans with optics, each walk said once ([#177](https://github.com/xqlsystems/ddx/pull/177))
- *(ddx-datafusion)* read a constant loss's missing tangent as 0 in the jvp soak ([#178](https://github.com/xqlsystems/ddx/pull/178))
- *(ddx-datafusion)* fuzz the SQL surface's fast paths ([#173](https://github.com/xqlsystems/ddx/pull/173))
- *(ddx-ad)* write the views an objective reads as its CTEs ([#169](https://github.com/xqlsystems/ddx/pull/169))
- *(ddx-ad)* plan each CTE an objective reads more than once on its own ([#168](https://github.com/xqlsystems/ddx/pull/168))
- *(ddx-ad)* forward mode and composition over shared relations ([#167](https://github.com/xqlsystems/ddx/pull/167))
- *(ddx-ad)* lower a relation shared through ReferenceRel once ([#166](https://github.com/xqlsystems/ddx/pull/166))
- *(ddx-ad)* keep the SQL surface's programs between calls ([#164](https://github.com/xqlsystems/ddx/pull/164))
- *(ddx-ad)* contract through masks and sides of several inputs ([#162](https://github.com/xqlsystems/ddx/pull/162))

## [0.2.2](https://github.com/xqlsystems/ddx/compare/ddx-datafusion-v0.2.1...ddx-datafusion-v0.2.2) - 2026-10-18

### Changed

- Faster gradients of products. A `SUM` of a product of two joined inputs is now transposed as a contraction, as `jax.grad` transposes a matrix product, instead of rebuilding the forward join in every gradient step to test whether the product was NULL. `matmul`'s gradient at n = 50,000 takes 0.83 s instead of 1.79 s ([#126](https://github.com/xqlsystems/ddx/pull/126))
- A statement rewritten for its `grad` or `vjp` calls drops the CTEs it no longer reads, which an engine would otherwise plan again for each statement: in a GPT training step of one statement per parameter table, 0.44 s of a 2.4 s step ([#160](https://github.com/xqlsystems/ddx/pull/160))

## [0.2.1](https://github.com/xqlsystems/ddx/compare/ddx-datafusion-v0.2.0...ddx-datafusion-v0.2.1) - 2026-10-18

### Added

- grad and vjp of window functions, from ddx-ad 0.2.1: `SUM` and `AVG` over any frame, and `MAX` and `MIN` over each whole partition ([#145](https://github.com/xqlsystems/ddx/pull/145))
- jvp of an ordered aggregate, from ddx-ad 0.2.1 ([#139](https://github.com/xqlsystems/ddx/pull/139))

## [0.2.0](https://github.com/xqlsystems/ddx/compare/ddx-datafusion-v0.1.1...ddx-datafusion-v0.2.0) - 2026-10-17

### Added

- [**breaking**] grad, vjp and jvp in SQL are columns of a SELECT, and nest ([#132](https://github.com/xqlsystems/ddx/pull/132))
- [**breaking**] engines implement an async Backend; ddx-ad owns the only run loop ([#136](https://github.com/xqlsystems/ddx/pull/136))
- [**breaking**] every run checks its tables ([#135](https://github.com/xqlsystems/ddx/pull/135))
- [**breaking**] run returns values, as a JAX function does ([#129](https://github.com/xqlsystems/ddx/pull/129))
- [**breaking**] remove v1's SQL surface; ddx-core is the scalar rules ddx-ad uses ([#133](https://github.com/xqlsystems/ddx/pull/133))
- *(ddx-ad)* [**breaking**] grad, vjp and jvp compose with each other ([#121](https://github.com/xqlsystems/ddx/pull/121))
- *(ddx-ad)* jvp of a query and of a program; Hessian-vector products ([#113](https://github.com/xqlsystems/ddx/pull/113))

### Fixed

- *(ddx-ad)* an ungrouped aggregate keeps its row; jvp of a constant is 0 ([#138](https://github.com/xqlsystems/ddx/pull/138))
- *(ddx-ad)* a MAX of a grouped MAX is exact at a near tie ([#116](https://github.com/xqlsystems/ddx/pull/116))
- *(ddx-ad)* write an emit only on a projection ([#117](https://github.com/xqlsystems/ddx/pull/117))
- *(ddx-ad)* an ordering by an expression breaks no ties; two smaller findings ([#114](https://github.com/xqlsystems/ddx/pull/114))

### Other

- fuzz grad, vjp and jvp as SQL columns ([#137](https://github.com/xqlsystems/ddx/pull/137))
- every doc and docstring stands on its own ([#134](https://github.com/xqlsystems/ddx/pull/134))
- a soak oracle for jvp, against reverse mode ([#130](https://github.com/xqlsystems/ddx/pull/130))
- *(ddx-ad)* an opt module that fuses projections and splits deep steps ([#123](https://github.com/xqlsystems/ddx/pull/123))
- tangents by forward mode over each expression; the plan walked once; checks at once ([#119](https://github.com/xqlsystems/ddx/pull/119))
- *(ddx-ad)* [**breaking**] one vocabulary of input and output tables for every program ([#118](https://github.com/xqlsystems/ddx/pull/118))

## [0.1.1](https://github.com/xqlsystems/ddx/compare/ddx-datafusion-v0.1.0...ddx-datafusion-v0.1.1) - 2026-10-04

### Added

- *(ddx-datafusion)* nn.py trained with grad in SQL (M4 3) ([#80](https://github.com/xqlsystems/ddx/pull/80))
- grad(loss, table.column) in SQL (M4 2) ([#83](https://github.com/xqlsystems/ddx/pull/83))
- *(ddx-datafusion)* grad, vjp and run, the v2 API on DataFusion (M4 1) ([#79](https://github.com/xqlsystems/ddx/pull/79))
- *(ddx-ad)* save each distinct aggregate once; the attention fixture (M3 9) ([#77](https://github.com/xqlsystems/ddx/pull/77))
- *(ddx-ad)* AVG, MAX and MIN rules; rank-select and stop-gradient (M3 8) ([#76](https://github.com/xqlsystems/ddx/pull/76))
- *(ddx-ad)* one transpose per relational primitive; grad and vjp (M3 6) ([#74](https://github.com/xqlsystems/ddx/pull/74))
- *(ddx-ad)* relations as dims and values; the forward pass read (M3 5) ([#73](https://github.com/xqlsystems/ddx/pull/73))
- *(ddx-ad)* emit the plans of a backward program (M3 4) ([#72](https://github.com/xqlsystems/ddx/pull/72))
- *(ddx-ad)* a plan's function table; ddx_stop_gradient on DataFusion (M3 2, re-land of #70) ([#105](https://github.com/xqlsystems/ddx/pull/105))

### Fixed

- a NULL loss sends no gradient; two smaller review findings ([#109](https://github.com/xqlsystems/ddx/pull/109))

### Other

- describe the v2 API in ddx-datafusion's metadata; package all three crates in the dry-run ([#110](https://github.com/xqlsystems/ddx/pull/110))
- faster backward plans for deep chains and CTE reuse; column and row pushdown ([#108](https://github.com/xqlsystems/ddx/pull/108))
- aim the soak at today's changes; a cost benchmark; three findings (M4 11) ([#102](https://github.com/xqlsystems/ddx/pull/102))
- a forward-mode oracle, a cost bound and a SQL-text fuzz; what they found (M4 10) ([#98](https://github.com/xqlsystems/ddx/pull/98))
- mutation testing for the v2 soak, and the blind spots it found (M4 9) ([#93](https://github.com/xqlsystems/ddx/pull/93))
- grad in SQL under awkward names, and SGD loops in SQL (M4 8) ([#92](https://github.com/xqlsystems/ddx/pull/92))
- ties, NULLs, extreme values and big tables in the v2 soak (M4 7) ([#91](https://github.com/xqlsystems/ddx/pull/91))
- vary the plan ddx reads — optimizer rules and Substrait rewrites (M4 6) ([#90](https://github.com/xqlsystems/ddx/pull/90))
- a simulation soak for query-level AD, and three bugs it found (M4 5) ([#89](https://github.com/xqlsystems/ddx/pull/89))
- M4 in the READMEs and the design (M4 4) ([#82](https://github.com/xqlsystems/ddx/pull/82))
- *(ddx-ad)* contraction fixtures, written as plain SQL (M3 7) ([#75](https://github.com/xqlsystems/ddx/pull/75))

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
