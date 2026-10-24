# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.4](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.2.3...ddx-ad-v0.2.4) - 2026-10-24

### Added

- *(ddxdb)* grad, vjp and jvp in SQL on DuckDB ([#192](https://github.com/xqlsystems/ddx/pull/192))
- *(ddx-ad)* write a program's steps as SQL (to_sql) ([#189](https://github.com/xqlsystems/ddx/pull/189))

### Other

- *(deps)* upgrade DataFusion to 55 ([#185](https://github.com/xqlsystems/ddx/pull/185))

## [0.2.3](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.2.2...ddx-ad-v0.2.3) - 2026-10-20

### Changed

- Faster gradients of causal attention. The contraction rule now covers a join filtered by its grouping keys (a causal mask) and a product of several joined inputs, as `jax.grad` transposes attention. At 256 tokens the gradient's steps take 283 ms instead of 1,119 ms ([#162](https://github.com/xqlsystems/ddx/pull/162))
- `rewrite` keeps the programs it builds between calls, in a `Programs` the caller holds, so a training loop builds its gradient program once instead of on every step. A program is found again by its query and `wrt`, and by the new optional `Backend::plan_key`, which changes whenever the engine's plan could ([#164](https://github.com/xqlsystems/ddx/pull/164))
- A CTE read more than once is differentiated once. Repeated aggregates are found by structural id instead of by encoding their subtrees (76 ms instead of 394 ms for 2,048 copies) ([#165](https://github.com/xqlsystems/ddx/pull/165)); relations shared through `ReferenceRel` are lowered once in reverse mode ([#166](https://github.com/xqlsystems/ddx/pull/166)) and in forward mode and composition ([#167](https://github.com/xqlsystems/ddx/pull/167)); `rewrite` plans each CTE an objective reads more than once on its own ([#168](https://github.com/xqlsystems/ddx/pull/168)), including the CTEs of a view it reads, through the new optional `Backend::view_sql` ([#169](https://github.com/xqlsystems/ddx/pull/169))
- Every walk over a plan is now written once, as optics, so the walks agree on what a relation's inputs and expressions are; building a program is 2–15% faster ([#177](https://github.com/xqlsystems/ddx/pull/177))

### Fixed

- `jvp` treats values within a few ulps of a `MAX` or `MIN` as tied where `grad` does (an argument computed by a sum or a mean), so `jvp` along `t` equals ⟨∇L, t⟩ at a near-tie too ([#180](https://github.com/xqlsystems/ddx/pull/180))
- A semi or anti join with no condition, which DataFusion plans for an uncorrelated `EXISTS` or `IN`, is written with the condition `true`, which DataFusion's Substrait consumer accepts ([#174](https://github.com/xqlsystems/ddx/pull/174))

## [0.2.2](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.2.1...ddx-ad-v0.2.2) - 2026-10-18

### Changed

- Faster gradients of products. A `SUM` of a product of two joined inputs is now transposed as a contraction, as `jax.grad` transposes a matrix product, instead of rebuilding the forward join in every gradient step to test whether the product was NULL. `matmul`'s gradient at n = 50,000 takes 0.83 s instead of 1.79 s ([#126](https://github.com/xqlsystems/ddx/pull/126))
- A statement rewritten for its `grad` or `vjp` calls drops the CTEs it no longer reads, which an engine would otherwise plan again for each statement: in a GPT training step of one statement per parameter table, 0.44 s of a 2.4 s step ([#160](https://github.com/xqlsystems/ddx/pull/160))

## [0.2.1](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.2.0...ddx-ad-v0.2.1) - 2026-10-18

### Added

- grad and vjp of window functions: `SUM` and `AVG` over any frame, and `MAX` and `MIN` over each whole partition, so softmax written with `MAX(x) OVER` and `SUM(exp(x)) OVER` has a gradient ([#145](https://github.com/xqlsystems/ddx/pull/145))
- jvp of an ordered aggregate (`MAX(x ORDER BY y)`) ([#139](https://github.com/xqlsystems/ddx/pull/139))

### Fixed

- A window's gradient no longer depends on what happens to its rows above it: a filter, semi-join, duplicating join or group with no gradient above a window cannot change its counts, ties or frames ([#145](https://github.com/xqlsystems/ddx/pull/145))
- `ROW_NUMBER` and `FIRST_VALUE` with no `ORDER BY` are refused unless their ties are broken, since their values depend on the engine's row order ([#145](https://github.com/xqlsystems/ddx/pull/145))

### Changed

- One linear form per reduction, read by forward and reverse mode, so both modes have the same aggregate and window rules ([#139](https://github.com/xqlsystems/ddx/pull/139), [#145](https://github.com/xqlsystems/ddx/pull/145))

## [0.2.0](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.1.0...ddx-ad-v0.2.0) - 2026-10-17

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

- every doc and docstring stands on its own ([#134](https://github.com/xqlsystems/ddx/pull/134))
- *(ddx-ad)* an opt module that fuses projections and splits deep steps ([#123](https://github.com/xqlsystems/ddx/pull/123))
- tangents by forward mode over each expression; the plan walked once; checks at once ([#119](https://github.com/xqlsystems/ddx/pull/119))
- *(ddx-ad)* [**breaking**] one vocabulary of input and output tables for every program ([#118](https://github.com/xqlsystems/ddx/pull/118))

## [0.1.0](https://github.com/xqlsystems/ddx/releases/tag/ddx-ad-v0.1.0) - 2026-10-04

### Added

- grad(loss, table.column) in SQL (M4 2) ([#83](https://github.com/xqlsystems/ddx/pull/83))
- *(ddx-datafusion)* grad, vjp and run, the v2 API on DataFusion (M4 1) ([#79](https://github.com/xqlsystems/ddx/pull/79))
- *(ddx-ad)* save each distinct aggregate once; the attention fixture (M3 9) ([#77](https://github.com/xqlsystems/ddx/pull/77))
- *(ddx-ad)* AVG, MAX and MIN rules; rank-select and stop-gradient (M3 8) ([#76](https://github.com/xqlsystems/ddx/pull/76))
- *(ddx-ad)* one transpose per relational primitive; grad and vjp (M3 6) ([#74](https://github.com/xqlsystems/ddx/pull/74))
- *(ddx-ad)* relations as dims and values; the forward pass read (M3 5) ([#73](https://github.com/xqlsystems/ddx/pull/73))
- *(ddx-ad)* emit the plans of a backward program (M3 4) ([#72](https://github.com/xqlsystems/ddx/pull/72))
- *(ddx-ad)* the map primitive's local derivatives, via ddx-core (M3 3) ([#71](https://github.com/xqlsystems/ddx/pull/71))
- *(ddx-ad)* a plan's function table; ddx_stop_gradient on DataFusion (M3 2, re-land of #70) ([#105](https://github.com/xqlsystems/ddx/pull/105))

### Fixed

- a NULL loss sends no gradient; two smaller review findings ([#109](https://github.com/xqlsystems/ddx/pull/109))

### Other

- faster backward plans for deep chains and CTE reuse; column and row pushdown ([#108](https://github.com/xqlsystems/ddx/pull/108))
- *(ddx-ad)* add the substrait dependency and protoc (M3 1) ([#69](https://github.com/xqlsystems/ddx/pull/69))
- Created automatic release process. ([#46](https://github.com/xqlsystems/ddx/pull/46))
- Milestone 0 - `ddx-core` + scaffolding ([#3](https://github.com/xqlsystems/ddx/pull/3))
