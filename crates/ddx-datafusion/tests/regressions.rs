// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for Path B correctness bugs.
//!
//! Every one of these was invisible to reading the code and only showed up by
//! executing a query against a live engine. Two of them were **silently wrong**
//! rather than loud — the class ddx must never produce — so they get permanent
//! coverage here rather than a fix and a shrug.
//!
//! The common thread: an `AnalyzerRule` runs *after* the planner has already
//! bound, coerced, and cached schemas, so a rewrite has to put back an
//! expression that is consistent with all three.

use datafusion::arrow::array::Float64Array;
use datafusion::arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::prelude::SessionContext;

mod common;

async fn ctx() -> Result<SessionContext> {
    let ctx = SessionContext::new();
    ddx_datafusion::install(&ctx);
    for ddl in [
        "CREATE TABLE t AS SELECT * FROM (VALUES (1.0,4.0),(2.0,5.0),(3.0,6.0)) AS v(x,y)",
        // A capitalized, quoted column — what any Parquet/CSV file with
        // capitalized headers produces.
        "CREATE TABLE tu AS SELECT column1 AS \"X\" FROM (VALUES (1.0),(2.0),(3.0))",
        // An INTEGER column: derivatives are emitted DOUBLE, so this is where
        // mixed-type arithmetic shows up.
        "CREATE TABLE ti AS SELECT * FROM (VALUES (1),(2),(3)) AS v(x)",
    ] {
        ctx.sql(ddl).await?.collect().await?;
    }
    Ok(ctx)
}

/// Run `sql`, returning the first column as `f64` plus its Arrow type.
async fn col(ctx: &SessionContext, sql: &str) -> Result<(Vec<f64>, DataType)> {
    let batches = ctx.sql(sql).await?.collect().await?;
    Ok((common::f64_column(&batches), common::column_type(&batches)))
}

#[tokio::test]
async fn uppercase_column_is_not_silently_zero() -> Result<()> {
    // WAS SILENTLY WRONG: returned 0.0 for every row.
    //
    // The body is unparsed by `Unparser`, whose DefaultDialect quotes any
    // identifier with an uppercase letter; the wrt was built unquoted straight
    // off the bound Column. Under FoldUnquoted the quoted occurrence keeps its
    // case and the unquoted wrt folds to lowercase, so they never matched and
    // every occurrence classified as "not the wrt" — derivative 0, no error.
    let ctx = ctx().await?;
    let (vals, ty) = col(&ctx, "SELECT grad(\"X\" * \"X\", \"X\") AS d FROM tu").await?;
    assert_eq!(
        vals,
        vec![2.0, 4.0, 6.0],
        "uppercase column silently zeroed"
    );
    assert_eq!(ty, DataType::Float64);

    // Path A is the oracle: it must agree.
    assert_eq!(
        ddx_datafusion::rewrite_sql("SELECT grad(\"X\" * \"X\", \"X\") AS d FROM tu")?,
        "SELECT (\"X\" + \"X\") AS d FROM tu"
    );
    Ok(())
}

#[tokio::test]
async fn derivatives_over_integer_columns_execute() -> Result<()> {
    // WAS A LOUD FAILURE, but only at execution: the re-planned expression
    // never went through TypeCoercion (this rule runs after it), so ddx-core's
    // DOUBLE literals met an Int64 column and Arrow rejected the op.
    let ctx = ctx().await?;

    // Was: Arrow "Invalid arithmetic operation: Float64 / Int64".
    let (vals, ty) = col(&ctx, "SELECT grad(x / 2, x) AS d FROM ti").await?;
    assert_eq!(vals, vec![0.5, 0.5, 0.5]);
    assert_eq!(ty, DataType::Float64);

    // Was: Arrow "Invalid comparison operation: Int64 > Float64" — abs emits a
    // portable CASE-based sign, which compares against literals.
    let (vals, _) = col(&ctx, "SELECT grad(abs(x), x) AS d FROM ti").await?;
    assert_eq!(vals, vec![1.0, 1.0, 1.0]);
    Ok(())
}

#[tokio::test]
async fn derivative_is_always_double_and_ancestors_agree() -> Result<()> {
    // WAS TWO BUGS AT ONCE.
    //
    // The marker UDF declares Float64, and that type is already cached in every
    // ancestor's schema. `map_children` preserves a parent's cached schema while
    // swapping its input, so only the rewritten node was recomputed: when the
    // derivative planned to Int64 (`x + x` over an integer column), ancestors
    // kept a stale Float64 and the optimizer's invariant check blew up with an
    // internal error. The single-level form failed more quietly — it just
    // returned Int64, violating the "derivatives are always DOUBLE" policy
    // that the marker's own return_type asserts.
    let ctx = ctx().await?;

    let (vals, ty) = col(&ctx, "SELECT grad(x * x, x) AS d FROM ti").await?;
    assert_eq!(vals, vec![2.0, 4.0, 6.0]);
    assert_eq!(ty, DataType::Float64, "derivatives must always be DOUBLE");

    // Was: "Internal error: Assertion failed: compatible: ... original schema:
    // d: Float64, new schema: d: Int64".
    let (vals, ty) = col(&ctx, "SELECT d FROM (SELECT grad(x * x, x) AS d FROM ti)").await?;
    assert_eq!(vals, vec![2.0, 4.0, 6.0]);
    assert_eq!(ty, DataType::Float64);
    Ok(())
}

#[tokio::test]
async fn power_with_a_constant_exponent_survives_type_coercion() -> Result<()> {
    // WAS A WRONG DIAGNOSIS on a supported, documented case.
    //
    // TypeCoercion runs before this rule, so `power(x, 3)` arrived as
    // `power(CAST(x AS DOUBLE), CAST(3 AS DOUBLE))`. ddx-core's `as_const` saw
    // through `Nested` and unary minus but not `Cast`, so the exponent read as
    // variable and the call was rejected with "both the base and the exponent
    // depend on the differentiation variable" — which is simply untrue of `3`.
    //
    // Note this bit BOTH integer and float tables: the literal `3` is Int64, so
    // coercion inserts a cast even when the column is already DOUBLE.
    let ctx = ctx().await?;
    for table in ["ti", "t"] {
        let (vals, ty) = col(
            &ctx,
            &format!("SELECT grad(power(x, 3), x) AS d FROM {table}"),
        )
        .await?;
        assert_eq!(vals, vec![3.0, 12.0, 27.0], "power(x,3) failed on {table}");
        assert_eq!(ty, DataType::Float64);
    }
    Ok(())
}

#[tokio::test]
async fn markers_inside_subqueries_are_rewritten() -> Result<()> {
    // WAS A LOUD FAILURE: `LogicalPlan::map_children` visits only direct
    // relational inputs, never a plan embedded in an expression, so a marker in
    // a scalar subquery was skipped by both the pre-gate and the walk and
    // survived to execution.
    let ctx = ctx().await?;

    // AVG(grad(y*y, y)) over y = [4,5,6] is AVG([8,10,12]) = 10; /10 = 1.0.
    let (vals, _) = col(
        &ctx,
        "SELECT x FROM t WHERE x > (SELECT AVG(grad(y * y, y)) FROM t) / 10.0 ORDER BY x",
    )
    .await?;
    assert_eq!(vals, vec![2.0, 3.0]);
    Ok(())
}

#[tokio::test]
async fn path_a_and_path_b_agree_on_every_regression_case() -> Result<()> {
    // The cheapest guard against the two paths drifting: they drive the same
    // ddx-core, so wherever both can run a query they must produce equal
    // numbers. Each of these is a case that once differed.
    let b_ctx = ctx().await?;
    let a_ctx = ctx().await?;

    for sql in [
        "SELECT grad(\"X\" * \"X\", \"X\") AS d FROM tu",
        "SELECT grad(x / 2, x) AS d FROM ti",
        "SELECT grad(abs(x), x) AS d FROM ti",
        "SELECT grad(power(x, 3), x) AS d FROM ti",
        "SELECT grad(power(x, 3), x) AS d FROM t",
    ] {
        let (via_b, _) = col(&b_ctx, sql).await?;
        let batches = ddx_datafusion::ddx_sql(&a_ctx, sql)
            .await?
            .collect()
            .await?;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap_or_else(|| panic!("Path A returned a non-Float64 column for: {sql}"));
        let via_a: Vec<f64> = (0..arr.len()).map(|i| arr.value(i)).collect();
        assert_eq!(via_a, via_b, "paths disagree on: {sql}");
    }
    Ok(())
}
