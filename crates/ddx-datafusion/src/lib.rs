// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-datafusion` — the DataFusion adapter for [ddx](https://github.com/xqlsystems/ddx).
//!
//! Write calculus directly in SQL and let DataFusion evaluate the derivative
//! per row — the relational equivalent of `jax.vmap(jax.grad(f))`:
//!
//! ```sql
//! SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g
//! ```
//!
//! All the calculus lives in [`ddx_core`]; this crate only connects it to an
//! engine. It offers the same rewrite by two routes (design.md §3.3):
//!
//! | | [`install`] (Path B) | [`ddx_sql`] (Path A) |
//! |---|---|---|
//! | How | in-engine `AnalyzerRule` on the bound plan | rewrite the SQL text first |
//! | Call style | bare `grad()`, anywhere | `ddx_sql(&ctx, sql)` |
//! | Works with | SQL **and** the DataFrame API | SQL strings |
//! | Column identity | resolved by the planner | syntactic (guards may fire) |
//! | Correlated subqueries | ✗ (loud error) | ✓ |
//! | Errors surface at | `collect()` | the `ddx_sql` call |
//!
//! **Prefer [`install`].** Because it runs after binding, columns arrive
//! already resolved, so the qualification-ambiguity errors a pre-binding text
//! rewrite must raise (design.md §3.5) simply cannot occur.
//!
//! **Reach for [`ddx_sql`] when a marker sits inside a correlated subquery.**
//! That is the one query shape Path B genuinely cannot carry: the bridge
//! re-plans the derivative against the subquery's own inputs, and an outer
//! reference does not survive that. Path B detects it and says so.
//!
//! Recursive CTEs are *not* such a shape — Path B carries a marker in a
//! recursive term perfectly well (`LogicalPlan::RecursiveQuery` is an ordinary
//! node with ordinary inputs). This doc previously claimed otherwise, and it was
//! never true.
//!
//! # Where the two paths genuinely differ
//!
//! They drive the same [`ddx_core`] engine, so they agree on the calculus. They
//! do not always agree on **what may be the `wrt`**, because they disagree about
//! what a "column" is:
//!
//! ```sql
//! SELECT grad(sum(x) * sum(x), sum(x)) AS d FROM t
//! ```
//!
//! Path A refuses this — syntactically `sum(x)` is a function call, not a bare
//! column, and design.md §3.6 requires a bare column. Path B answers `2·sum(x)`,
//! because by the time it sees the plan the planner has already lowered the
//! aggregate to the bound column `sum(t.x)`, and differentiating with respect to
//! a column is exactly what it does.
//!
//! **Path B's `wrt` is any column of the node's input schema, including
//! planner-derived ones** — aggregate outputs, window outputs, computed aliases.
//! That is deliberate rather than accidental: design.md §3.5's carve-out (`G4`)
//! already endorses differentiating with respect to a *computed alias*
//! (`grad(s*s, s)` is `2s`, and is right), and an aggregate output is the same
//! shape one level down. Rejecting it would contradict that carve-out.
//!
//! # Path B
//!
//! ```
//! # use datafusion::prelude::SessionContext;
//! # #[tokio::main]
//! # async fn main() -> datafusion::error::Result<()> {
//! let ctx = SessionContext::new();
//! ddx_datafusion::install(&ctx);
//!
//! ctx.sql("CREATE TABLE t AS VALUES (1.0), (2.0), (3.0)").await?.collect().await?;
//!
//! // bare grad() — no wrapper
//! let df = ctx.sql("SELECT grad(column1 * column1, column1) AS d FROM t").await?;
//! # let _ = df.collect().await?;
//! # Ok(())
//! # }
//! ```
//!
//! # What it supports
//!
//! Whatever [`ddx_core`] supports: `+ - * /`; the unary chain rule for the trig
//! / inverse-trig / exp / log / hyperbolic set plus `abs`; `power` with a
//! constant base or exponent; higher-order via nesting; through-aggregate via
//! linearity (`AVG(grad(loss, theta))`). Anything else is a typed error, never
//! a silently-wrong number (design principle 5). Errors from the engine arrive
//! as [`DataFusionError::External`] boxing a [`ddx_core::DiffError`], so you can
//! downcast and match on the variant.
//!
//! [`DataFusionError::External`]: datafusion::error::DataFusionError::External

#![forbid(unsafe_code)]

mod analyzer;
mod error;
mod markers;
mod replan;
mod sql;

use std::sync::Arc;

use datafusion::prelude::SessionContext;

pub use analyzer::DdxAnalyzer;
pub use markers::{grad_udf, jvp_udf, GRAD, JVP};
pub use sql::{ddx_sql, ddx_sql_with, rewrite_sql, rewrite_sql_with};

/// The engine this adapter drives, re-exported so downstream code links the
/// same version — and, through it, the same `sqlparser` (design.md §6, G2).
pub use ddx_core;

/// Install ddx on `ctx`: register the `grad`/`jvp` marker UDFs and the analyzer
/// rule that rewrites them away (Path B).
///
/// Both halves are required and neither is useful alone. The UDFs exist only so
/// the marker calls *parse and plan*; the analyzer rule is what actually
/// differentiates. Registering the UDFs without the rule would let a marker
/// reach execution, where it deliberately errors (design.md §3.1).
///
/// ```
/// # use datafusion::prelude::SessionContext;
/// let ctx = SessionContext::new();
/// ddx_datafusion::install(&ctx);
/// ```
pub fn install(ctx: &SessionContext) {
    install_with(ctx, DdxAnalyzer::new());
}

/// [`install`] with a caller-configured analyzer — use this to pick up custom
/// differentiation rules (see [`DdxAnalyzer::with_engine`]).
///
/// Your own UDFs need no registration with ddx: a function called inside a
/// marker is read off the bound expression when the derivative is re-planned,
/// so `grad(my_udf(y) * x, x)` works whenever `my_udf` was registered, before or
/// after this call.
pub fn install_with(ctx: &SessionContext, analyzer: DdxAnalyzer) {
    ctx.register_udf(grad_udf());
    ctx.register_udf(jvp_udf());
    ctx.add_analyzer_rule(Arc::new(analyzer));
}
