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
//! | Query shapes | whatever a `LogicalPlan` carries | anything that parses |
//!
//! **Prefer [`install`].** Because it runs after binding, columns arrive
//! already resolved, so the qualification-ambiguity errors a pre-binding text
//! rewrite must raise (design.md §3.5) simply cannot occur.
//!
//! **Reach for [`ddx_sql`] when the marker sits somewhere a bound plan can't
//! carry it** — most importantly inside a recursive CTE, which is exactly where
//! a whole training loop lives.
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
pub fn install_with(ctx: &SessionContext, analyzer: DdxAnalyzer) {
    ctx.register_udf(grad_udf());
    ctx.register_udf(jvp_udf());
    ctx.add_analyzer_rule(Arc::new(analyzer));
}
