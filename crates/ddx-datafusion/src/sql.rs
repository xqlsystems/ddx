// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The SQL source-to-source rewrite.
//!
//! Rewrite every `grad`/`jvp` marker in the SQL *text* before it reaches the
//! engine, then hand plain SQL to a stock [`SessionContext`]. This is the
//! universal path: it runs before planning, so it works for every query shape
//! the parser accepts — recursive CTEs, DML, subqueries — which is what lets a
//! whole training loop live in one query. Path B (in-engine, [`crate::analyzer`])
//! is the more ergonomic one but is bounded by what a `LogicalPlan` can carry.

use datafusion::error::Result;
use datafusion::prelude::{DataFrame, SessionContext};
use ddx_core::sqlparser::dialect::GenericDialect;
use ddx_core::Ddx;

use crate::error::to_df_err;

/// Rewrite `grad`/`jvp` markers in `sql` and run the result on `ctx` — the
/// one-liner form of the text rewrite.
///
/// The context needs no ddx setup at all: no marker UDFs, no analyzer rule. By
/// the time the engine sees the statement the markers are gone, replaced by
/// ordinary derivative SQL.
///
/// A statement containing no marker is passed through byte-identical and is
/// never even parsed by ddx, so wrapping every query in
/// `ddx_sql` costs essentially nothing.
///
/// ```
/// # use datafusion::prelude::SessionContext;
/// # use ddx_datafusion::ddx_sql;
/// # #[tokio::main]
/// # async fn main() -> datafusion::error::Result<()> {
/// let ctx = SessionContext::new();
/// ctx.sql("CREATE TABLE t AS VALUES (1.0), (2.0), (3.0)").await?.collect().await?;
///
/// // d(x*x)/dx = 2x, computed by the engine as an ordinary column.
/// let df = ddx_sql(&ctx, "SELECT grad(column1 * column1, column1) AS d FROM t").await?;
/// let batches = df.collect().await?;
/// assert_eq!(batches[0].num_rows(), 3);
/// # Ok(())
/// # }
/// ```
pub async fn ddx_sql(ctx: &SessionContext, sql: &str) -> Result<DataFrame> {
    ddx_sql_with(ctx, sql, &Ddx::for_datafusion()).await
}

/// [`ddx_sql`] driven by a caller-supplied engine — use this when you have
/// registered custom differentiation rules via [`Ddx::register`].
pub async fn ddx_sql_with(ctx: &SessionContext, sql: &str, ddx: &Ddx) -> Result<DataFrame> {
    ctx.sql(&rewrite_sql_with(sql, ddx)?).await
}

/// Rewrite the markers in `sql` and return the derivative SQL as text, without
/// running it.
///
/// Useful for logging what will execute, for feeding another tool, or for the
/// `ddxdb` Python shim, which does exactly this and then calls a stock
/// `Context.sql()`.
pub fn rewrite_sql(sql: &str) -> Result<String> {
    rewrite_sql_with(sql, &Ddx::for_datafusion())
}

/// [`rewrite_sql`] driven by a caller-supplied engine.
///
/// `GenericDialect` is the parser DataFusion itself uses for SQL, and
/// [`Ddx::for_datafusion`] supplies the matching identifier-folding policy
/// (unquoted folds, quoted keeps case) — the two must agree or column matching
/// silently diverges from the engine's own.
pub fn rewrite_sql_with(sql: &str, ddx: &Ddx) -> Result<String> {
    ddx.rewrite_sql(sql, &GenericDialect {}).map_err(to_df_err)
}
