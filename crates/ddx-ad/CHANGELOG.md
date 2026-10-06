# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.1.0...ddx-ad-v0.2.0) - 2026-10-06

### Added

- skip a check whose fact a caller already holds (Verified) ([#120](https://github.com/xqlsystems/ddx/pull/120))
- *(ddx-ad)* [**breaking**] grad, vjp and jvp compose with each other ([#121](https://github.com/xqlsystems/ddx/pull/121))
- *(ddx-ad)* jvp of a query and of a program; Hessian-vector products (M4.5 1) ([#113](https://github.com/xqlsystems/ddx/pull/113))

### Fixed

- *(ddx-ad)* a MAX of a grouped MAX is exact at a near tie ([#116](https://github.com/xqlsystems/ddx/pull/116))
- *(ddx-ad)* write an emit only on a projection ([#117](https://github.com/xqlsystems/ddx/pull/117))
- *(ddx-ad)* an ordering by an expression breaks no ties; two smaller findings ([#114](https://github.com/xqlsystems/ddx/pull/114))

### Other

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
