// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Path B end-to-end: bare `grad()` through the `AnalyzerRule` on a live
//! DataFusion engine.
//!
//! These assert on *executed numbers*, not on rewritten SQL text. A rewrite
//! that looks right but plans to the wrong expression is exactly the failure
//! mode this milestone exists to rule out.

use datafusion::arrow::array::Float64Array;
use datafusion::error::Result;
use datafusion::prelude::SessionContext;

/// A context with ddx installed and a table `t(x, y)` of three rows.
async fn ctx() -> Result<SessionContext> {
    let ctx = SessionContext::new();
    ddx_datafusion::install(&ctx);
    ctx.sql(
        "CREATE TABLE t AS
         SELECT * FROM (VALUES (1.0, 4.0), (2.0, 5.0), (3.0, 6.0)) AS v(x, y)",
    )
    .await?
    .collect()
    .await?;
    Ok(ctx)
}

/// Run `sql` and return the first column as `f64`s.
async fn col(ctx: &SessionContext, sql: &str) -> Result<Vec<f64>> {
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut out = Vec::new();
    for b in &batches {
        let a = b
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("every derivative is emitted DOUBLE-typed");
        out.extend((0..a.len()).map(|i| a.value(i)));
    }
    Ok(out)
}

#[tokio::test]
async fn bare_grad_runs_end_to_end() -> Result<()> {
    // THE M2 EXIT CRITERION: no wrapper, no rewritten string — just grad() in
    // ordinary SQL against a stock engine with the analyzer installed.
    let ctx = ctx().await?;
    assert_eq!(
        col(&ctx, "SELECT grad(x * x, x) AS d FROM t ORDER BY x").await?,
        vec![2.0, 4.0, 6.0], // d(x²)/dx = 2x
    );
    Ok(())
}

#[tokio::test]
async fn product_rule_picks_the_right_variable() -> Result<()> {
    let ctx = ctx().await?;
    // d(xy)/dx = y and d(xy)/dy = x — the "full gradient as tidy columns" case.
    assert_eq!(
        col(&ctx, "SELECT grad(x * y, x) AS d FROM t ORDER BY x").await?,
        vec![4.0, 5.0, 6.0],
    );
    assert_eq!(
        col(&ctx, "SELECT grad(x * y, y) AS d FROM t ORDER BY x").await?,
        vec![1.0, 2.0, 3.0],
    );
    Ok(())
}

#[tokio::test]
async fn chain_rule_through_a_function() -> Result<()> {
    let ctx = ctx().await?;
    let got = col(&ctx, "SELECT grad(sin(x * y), x) AS d FROM t ORDER BY x").await?;
    // d/dx sin(xy) = y·cos(xy)
    let want: Vec<f64> = [(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]
        .iter()
        .map(|(x, y): &(f64, f64)| y * (x * y).cos())
        .collect();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-12, "got {g}, want {w}");
    }
    Ok(())
}

#[tokio::test]
async fn higher_order_falls_out_of_nesting() -> Result<()> {
    // Bottom-up rewriting gives higher-order differentiation for free.
    let ctx = ctx().await?;
    assert_eq!(
        col(
            &ctx,
            "SELECT grad(grad(x * x * x, x), x) AS d FROM t ORDER BY x"
        )
        .await?,
        vec![6.0, 12.0, 18.0], // d²(x³)/dx² = 6x
    );
    Ok(())
}

#[tokio::test]
async fn jvp_is_the_directional_derivative() -> Result<()> {
    let ctx = ctx().await?;
    // jvp(x*x, x, y) = 2x·y
    assert_eq!(
        col(&ctx, "SELECT jvp(x * x, x, y) AS d FROM t ORDER BY x").await?,
        vec![8.0, 20.0, 36.0],
    );
    Ok(())
}

#[tokio::test]
async fn grad_inside_an_aggregate_is_one_descent_step() -> Result<()> {
    // Differentiating through an aggregate is linearity, so the marker goes
    // INSIDE it — which is what makes a gradient step expressible in SQL.
    let ctx = ctx().await?;
    let got = col(&ctx, "SELECT AVG(grad(x * x, x)) AS g FROM t").await?;
    assert_eq!(got, vec![4.0]); // mean of [2,4,6]
    Ok(())
}

#[tokio::test]
async fn works_through_the_dataframe_api_too() -> Result<()> {
    // The reason Path B exists rather than only Path A: no SQL string involved.
    use datafusion::logical_expr::col as c;

    let ctx = ctx().await?;
    let grad = ddx_datafusion::grad_udf();
    let df = ctx
        .table("t")
        .await?
        .select(vec![grad.call(vec![c("x") * c("x"), c("x")]).alias("d")])?
        .sort(vec![c("d").sort(true, false)])?;

    let batches = df.collect().await?;
    let a = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(
        (0..a.len()).map(|i| a.value(i)).collect::<Vec<_>>(),
        vec![2.0, 4.0, 6.0]
    );
    Ok(())
}

#[tokio::test]
async fn a_query_without_markers_is_untouched() -> Result<()> {
    let ctx = ctx().await?;
    assert_eq!(
        col(&ctx, "SELECT x * 2 AS d FROM t ORDER BY x").await?,
        vec![2.0, 4.0, 6.0],
    );
    Ok(())
}

#[tokio::test]
async fn unsupported_construct_is_a_loud_error() -> Result<()> {
    // Never a silently-wrong number: an unsupported construct must error.
    //
    // Note where the error surfaces: `sql()` only builds the logical plan, and
    // analyzer rules run during optimization, so a ddx failure appears at
    // `collect()` (or `create_physical_plan`), NOT at `sql()`. Path A differs —
    // there the rewrite happens before the engine is involved at all, so the
    // same mistake fails at the `ddx_sql` call.
    let ctx = ctx().await?;
    let err = ctx
        .sql("SELECT grad(atan2(x, y), x) FROM t")
        .await?
        .collect()
        .await
        .expect_err("atan2 has no rule yet — this must fail, not guess");

    // ddx-core's own diagnosis has to reach the user, not just the rule name.
    // DataFusion wraps a rule failure as `Context(rule_name, err)`, which
    // Displays as "ddx_markers\ncaused by\n<the real message>".
    let msg = err.to_string();
    assert!(msg.contains("not implemented"), "unexpected error: {msg}");
    assert!(
        msg.contains("atan2"),
        "the message must name the culprit: {msg}"
    );

    // ...and the typed error must survive for programmatic matching.
    let mut source = std::error::Error::source(&err);
    let mut found_typed = false;
    while let Some(s) = source {
        if let Some(d) = s.downcast_ref::<ddx_datafusion::ddx_core::DiffError>() {
            assert!(matches!(
                d,
                ddx_datafusion::ddx_core::DiffError::NotImplemented(_)
            ));
            found_typed = true;
        }
        source = s.source();
    }
    assert!(
        found_typed,
        "the DiffError must be downcastable from the error chain"
    );
    Ok(())
}

#[tokio::test]
async fn a_marker_that_reaches_execution_errors() -> Result<()> {
    // Without install(), the UDF is unknown; register it WITHOUT the analyzer
    // rule and the marker survives planning — which must fail loudly at
    // execution rather than return a number.
    let ctx = SessionContext::new();
    ctx.register_udf(ddx_datafusion::grad_udf());
    ctx.sql("CREATE TABLE t AS SELECT * FROM (VALUES (1.0)) AS v(x)")
        .await?
        .collect()
        .await?;

    let err = ctx
        .sql("SELECT grad(x * x, x) FROM t")
        .await?
        .collect()
        .await
        .expect_err("an unrewritten marker must never produce a value");
    let msg = err.to_string();
    assert!(msg.contains("reached execution"), "unexpected: {msg}");
    Ok(())
}
