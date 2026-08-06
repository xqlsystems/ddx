// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Path A end-to-end: the SQL source-to-source rewrite on a stock engine
//! (design.md §3.3).
//!
//! The interesting cases are the ones Path B *cannot* reach, which is the whole
//! reason this path exists alongside it.

use datafusion::arrow::array::Float64Array;
use datafusion::error::Result;
use datafusion::prelude::SessionContext;
use ddx_datafusion::{ddx_sql, rewrite_sql};

/// A **stock** context — deliberately no `install()`. Path A needs no engine
/// setup at all: the markers are gone before the engine sees the statement.
async fn ctx() -> Result<SessionContext> {
    let ctx = SessionContext::new();
    ctx.sql(
        "CREATE TABLE t AS
         SELECT * FROM (VALUES (1.0, 4.0), (2.0, 5.0), (3.0, 6.0)) AS v(x, y)",
    )
    .await?
    .collect()
    .await?;
    Ok(ctx)
}

async fn col(ctx: &SessionContext, sql: &str) -> Result<Vec<f64>> {
    let batches = ddx_sql(ctx, sql).await?.collect().await?;
    let mut out = Vec::new();
    for b in &batches {
        let a = b
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("derivatives are always Float64");
        out.extend((0..a.len()).map(|i| a.value(i)));
    }
    Ok(out)
}

#[tokio::test]
async fn grad_runs_on_a_context_with_no_ddx_setup() -> Result<()> {
    let ctx = ctx().await?;
    assert_eq!(
        col(&ctx, "SELECT grad(x * x, x) AS d FROM t ORDER BY x").await?,
        vec![2.0, 4.0, 6.0],
    );
    Ok(())
}

#[tokio::test]
async fn agrees_with_path_b_on_the_same_query() -> Result<()> {
    // Both paths drive the same ddx-core engine, so they must agree
    // numerically. This is the cheap in-repo version of the cross-engine
    // equivalence discipline of design.md §5.
    let sql = "SELECT grad(sin(x * y), y) AS d FROM t ORDER BY x";

    let a_ctx = ctx().await?;
    let via_a = col(&a_ctx, sql).await?;

    let b_ctx = ctx().await?;
    ddx_datafusion::install(&b_ctx);
    let batches = b_ctx.sql(sql).await?.collect().await?;
    let arr = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let via_b: Vec<f64> = (0..arr.len()).map(|i| arr.value(i)).collect();

    assert_eq!(via_a.len(), via_b.len());
    for (a, b) in via_a.iter().zip(&via_b) {
        assert!((a - b).abs() < 1e-12, "Path A gave {a}, Path B gave {b}");
    }
    Ok(())
}

#[tokio::test]
async fn newton_iteration_in_a_recursive_cte() -> Result<()> {
    // THE case Path B cannot serve: a marker inside a recursive CTE — a whole
    // iterative solve as one query (design.md §3.6). Newton's method for
    // sqrt(2): x <- x - (x²-2)/grad(x²-2, x), which rewrites to /(x + x).
    let ctx = ctx().await?;
    // The anchor's columns are aliased explicitly rather than via the
    // `newton(i, x)` column list — DataFusion does not apply that list here,
    // and the anchor's fields would otherwise be named `Int64(0)`/`Float64(1)`.
    let sql = "
        WITH RECURSIVE newton AS (
            SELECT 0 AS i, 1.0 AS x
            UNION ALL
            SELECT i + 1 AS i, x - (x * x - 2.0) / grad(x * x - 2.0, x) AS x
            FROM newton WHERE i < 6
        )
        SELECT x FROM newton WHERE i = 6";

    let got = col(&ctx, sql).await?;
    assert_eq!(got.len(), 1);
    assert!(
        (got[0] - std::f64::consts::SQRT_2).abs() < 1e-12,
        "Newton did not converge to sqrt(2): got {}",
        got[0]
    );
    Ok(())
}

#[tokio::test]
async fn a_marker_free_statement_is_returned_byte_identical() -> Result<()> {
    // design.md §3.2 (F5): the parse-free pre-gate means a statement with no
    // marker is never parsed, so ddx can neither fail it nor reformat it.
    let sql = "SELECT   *   FROM t  /* odd  spacing preserved */";
    assert_eq!(rewrite_sql(sql)?, sql);
    Ok(())
}

#[tokio::test]
async fn rewrite_is_inspectable_without_running_it() -> Result<()> {
    let out = rewrite_sql("SELECT grad(sin(x), x) AS d FROM t")?;
    assert_eq!(out, "SELECT (cos(x)) AS d FROM t");
    Ok(())
}

#[tokio::test]
async fn path_a_fails_at_the_call_not_at_collect() -> Result<()> {
    // The ergonomic difference from Path B worth pinning: Path A rewrites
    // before the engine is involved, so an unsupported construct is reported
    // immediately rather than during execution.
    let ctx = ctx().await?;
    let err = ddx_sql(&ctx, "SELECT grad(atan2(x, y), x) FROM t")
        .await
        .expect_err("atan2 has no rule yet");
    let msg = err.to_string();
    assert!(msg.contains("not implemented"), "unexpected: {msg}");
    assert!(msg.contains("atan2"), "must name the culprit: {msg}");
    Ok(())
}
