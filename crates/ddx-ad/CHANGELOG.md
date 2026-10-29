# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.3.1...ddx-ad-v0.3.2) - 2026-10-29

### Added

- `grad` and `vjp` differentiate through `UNION ALL`, as `jvp` already did. Each input's rows carry its own columns, and a union column's cotangent goes to the input that gave the row, as `jnp.concatenate`'s transpose is a slice. `UNION` (distinct) is still refused ([#229](https://github.com/xqlsystems/ddx/pull/229))

### Fixed

- A loss that joins one-row aggregates is taken as a scalar loss when some of them read only constant data. DataFusion writes a `CROSS JOIN` as a join on true, and only a cross join was read as one row; inner, left, right, semi- and anti-joins of one-row inputs now are too ([#230](https://github.com/xqlsystems/ddx/pull/230))

### Changed

- A step that exactly one later step reads, once, is computed inside that step rather than written as a table: xarray-sql's GPT training step has 64 steps rather than 77, about 1.5% faster on DuckDB ([#231](https://github.com/xqlsystems/ddx/pull/231))
- The program cache treats a lock that a panic poisoned as empty, rather than reading what the panic left. The code otherwise follows the Google Rust style guide, with no change in behaviour or speed ([#234](https://github.com/xqlsystems/ddx/pull/234), [#239](https://github.com/xqlsystems/ddx/pull/239), [#240](https://github.com/xqlsystems/ddx/pull/240))

## [0.3.1](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.3.0...ddx-ad-v0.3.1) - 2026-10-28

### Added

- `inlined_size(plan)`: the size, in expression nodes, of a plan's largest column written out down to the tables it reads, or `None` for a plan holding a relation it cannot size. A Substrait consumer that names each computed column after its expression, as datafusion-python's does, gives names about this long, and they grow exponentially in a chain of columns that each read the one before twice, as stacked softmaxes do. An adapter can run such a step another way, as `to_sql`'s SQL ([#221](https://github.com/xqlsystems/ddx/pull/221))

### Changed

- Batched, multi-head attention's backward no longer pairs every two sequences and heads. The transpose of a contraction joins the cotangent on the keys its join holds equal to a grouping key, transitively: attention's output joins `v` to the weights on the sequence, and groups by the weights' sequence. At xarray-sql's GPT size, the contraction of `v` with the output's cotangent has 196,608 rows rather than 9.4 million ([#220](https://github.com/xqlsystems/ddx/pull/220))
- A contraction's side is rebuilt with the equalities its join implies between that side's own columns. A plan may join the softmax's normalizer to the weights only through `v`, as DuckDB's does; the side `weights ⋈ norm` then joined each score to every sequence's normalizer. nanoGPT's CPU training step on DuckDB takes about 27 s rather than 34–48 s, and peaks at 2.3 GB rather than 5.8 GB ([#225](https://github.com/xqlsystems/ddx/pull/225))
- A run drops each intermediate table once the last step that reads it has run, rather than all of them at the end, so it holds only the tables later steps still read: a GPT training step on DuckDB holds 49,089 step rows at its peak rather than 70,595. A prepared program works out when to drop each table once. A failed drop now fails the run, which drops every table left, and a program two of whose steps write one table is refused before it runs ([#223](https://github.com/xqlsystems/ddx/pull/223))

## [0.3.0](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.2.6...ddx-ad-v0.3.0) - 2026-10-28

### Added

- `prepare` and `Prepared`: statements whose `grad`, `vjp` and `jvp` calls are computed once, to compute again, as `jax.jit` traces a function once. A later `Prepared::run` runs each program's plans as they were bound, asking the engine only for its new optional `Backend::session_key`; once a table's schema, a view or a setting changes, it runs nothing, for the statements to be prepared again ([#203](https://github.com/xqlsystems/ddx/pull/203))
- An engine may compile each plan `prepare` runs, with the new optional `Backend::compile`, and `Prepared::run` runs what it compiled (`Backend::materialize_compiled`, `Backend::returns_rows_compiled`) rather than handing it the plan to plan again. The default compiles nothing ([#214](https://github.com/xqlsystems/ddx/pull/214))

### Changed

- **Breaking:** `Backend` has an associated type `Compiled`, what `compile` returns: `type Compiled = ();` for an engine that compiles nothing. `Backend::returns_rows` takes `&[&Bound]` ([#214](https://github.com/xqlsystems/ddx/pull/214))

## [0.2.6](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.2.5...ddx-ad-v0.2.6) - 2026-10-26

### Fixed

- A softmax over a window in a view, its weights read by the output and by their sum, is differentiated on DuckDB. Each CTE such a view reads more than once is planned on its own, and the pieces' window functions are now declared as a projection's are; before, the view reached DuckDB's Substrait producer whole and failed with `Not implemented Error: CTE` (#204, [#207](https://github.com/xqlsystems/ddx/pull/207))
- The contraction rule groups by each join key once. A plan that repeats a key column, as DuckDB's plan of attention does, made a step whose root named more columns than its relation had ([#207](https://github.com/xqlsystems/ddx/pull/207))

## [0.2.5](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.2.4...ddx-ad-v0.2.5) - 2026-10-25

### Fixed

- Windows in DuckDB's plans are differentiated. DuckDB writes a window as a `ConsistentPartitionWindowRel`, which `grad`, `vjp`, `jvp` and a query over another call now read as the projection of its window functions, as DataFusion writes one. A `RANGE` frame with no `ORDER BY` that ends at the current row, DuckDB's `OVER ()`, is the whole partition. Before, only a window DuckDB happened to rewrite as a join to an aggregate was differentiated ([#200](https://github.com/xqlsystems/ddx/pull/200))
- `rewrite` finds the calls in a statement whose CTE has a materialization hint (`AS MATERIALIZED`, `AS NOT MATERIALIZED`) in any dialect, and keeps the hint in the statement it returns. sqlparser parses the hint only in its Postgres dialect, so such a statement was passed over with its calls unrewritten ([#200](https://github.com/xqlsystems/ddx/pull/200))
- A view's SQL with a materialization hint is read too, so its CTEs read more than once are planned once each ([#200](https://github.com/xqlsystems/ddx/pull/200))

## [0.2.4](https://github.com/xqlsystems/ddx/compare/ddx-ad-v0.2.3...ddx-ad-v0.2.4) - 2026-10-24

### Added

- `to_sql` writes a step's Substrait plan as SQL, in DataFusion's or DuckDB's dialect (`Dialect`), so an engine can run ddx's programs without a Substrait consumer. A plan it can't write exactly is refused ([#189](https://github.com/xqlsystems/ddx/pull/189))
- `tables_read` and `rename_tables`: the tables a plan reads, and a way to rename them, for an engine whose producer writes a table without its schema ([#192](https://github.com/xqlsystems/ddx/pull/192))

### Changed

- The minimum supported Rust version is 1.94, DataFusion 55's ([#185](https://github.com/xqlsystems/ddx/pull/185))
- The `jvp` rewrite writes an expression repeated in its `SELECT` once, since DuckDB's Substrait producer drops a repeated output column ([#192](https://github.com/xqlsystems/ddx/pull/192))

### Fixed

- A plan whose root names don't match its output columns is refused with the reason, instead of panicking ([#192](https://github.com/xqlsystems/ddx/pull/192))

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
