// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Bugs `ad_simulation.rs` found, each reduced to the smallest query that
//! shows it and pinned against a gradient worked by hand.
//!
//! Each is `#[ignore]`d as a known bug until its fix lands (CONTRIBUTING.md,
//! "a failing test first"); `cargo test --test ad_findings -- --ignored` runs
//! them, and every one fails on this branch. Its fix removes the `ignore`.
//!
//! - **NULL rows leak gradient.** SUM, AVG, MAX and MIN skip a row whose
//!   argument is NULL, so nothing in that row can move the loss. ddx still
//!   broadcasts the group's cotangent to the row and pushes it through the
//!   row's other inputs, where a partial that does not read the NULL (the 1 of
//!   `p + q`) is not NULL. The gradient is silently wrong at every parameter
//!   joined to a NULL, in a parameter table or in constant data.
//! - **Two rank filters in one region collide.** Rebuilding the region keeps
//!   both window columns, and the optimizer has given them the same name, so
//!   DataFusion's consumer refuses the step. The program was accepted.
//! - **An unoptimized plan's join condition and a CASE.** `grad_plan` accepts
//!   any `LogicalPlan`, a DataFrame's included; this unoptimized one is
//!   accepted and its backward step then fails DataFusion's schema check.

use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, ColumnRef};

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// `(key, gradient)` rows of `table`'s gradient of `loss`.
async fn grad(ctx: &SessionContext, loss: &str, table: &str) -> Vec<(i64, Option<f64>)> {
    let program = ad::grad(ctx, loss, &[ColumnRef::new(table, "val")])
        .await
        .unwrap();
    ad::run(ctx, &program).await.unwrap();
    let batches = ctx
        .sql(&format!(
            "SELECT * FROM {} ORDER BY 1",
            program.gradients[0].step
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    use datafusion::arrow::array::{Array, AsArray};
    use datafusion::arrow::datatypes::{Float64Type, Int64Type};
    let mut out = Vec::new();
    for b in &batches {
        let k = b.column(0).as_primitive::<Int64Type>();
        let v = b.column(1).as_primitive::<Float64Type>();
        for r in 0..b.num_rows() {
            out.push((k.value(r), (!v.is_null(r)).then(|| v.value(r))));
        }
    }
    out
}

#[tokio::test]
#[ignore = "known bug: a row an aggregate skips as NULL still sends its cotangent to its other inputs"]
async fn a_row_an_aggregate_skips_as_null_sends_no_gradient() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE q (i BIGINT, val DOUBLE) AS VALUES (0, 5.0), (1, NULL)",
    )
    .await;
    // Row 1's term is p + NULL = NULL, which SUM skips: the loss is p(0) + 5,
    // so ∂loss/∂p is 1 at row 0 and 0 at row 1.
    let got = grad(
        &ctx,
        "SELECT SUM(p.val + q.val) AS loss FROM p JOIN q ON p.i = q.i",
        "p",
    )
    .await;
    assert_eq!(got, vec![(0, Some(1.0)), (1, Some(0.0))]);
}

#[tokio::test]
#[ignore = "known bug: two rankings in one recomputed region share a column name"]
async fn a_rank_filter_over_a_rank_filter_runs() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 3.0), (2, 2.0)",
    )
    .await;
    // Top two by val, then the top one of those: row 1, whose val is summed.
    let got = grad(
        &ctx,
        "WITH a AS (SELECT i, v FROM (SELECT i, val AS v, ROW_NUMBER() OVER (ORDER BY val DESC, i) AS rk FROM p) WHERE rk <= 2), \
              b AS (SELECT i, v FROM (SELECT i, v, ROW_NUMBER() OVER (ORDER BY v DESC, i) AS rk FROM a) WHERE rk = 1) \
         SELECT SUM(v) AS loss FROM b",
        "p",
    )
    .await;
    assert_eq!(got, vec![(0, Some(0.0)), (1, Some(1.0)), (2, Some(0.0))]);
}

#[tokio::test]
#[ignore = "known bug: a row an aggregate skips as NULL still sends its cotangent to its other inputs"]
async fn a_null_in_constant_data_sends_no_gradient_through_its_row() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE d (i BIGINT, val DOUBLE) AS VALUES (0, 3.0), (1, NULL)",
    )
    .await;
    // Row 1 is d - p = NULL and skipped: the loss is 3 - p(0), and
    // ∂loss/∂p(1) is 0.
    let got = grad(
        &ctx,
        "SELECT SUM(d.val - p.val) AS loss FROM p JOIN d ON p.i = d.i",
        "p",
    )
    .await;
    assert_eq!(got, vec![(0, Some(-1.0)), (1, Some(0.0))]);
}

#[tokio::test]
#[ignore = "known bug: the backward step does not match its own schema"]
async fn an_unoptimized_join_with_a_constant_condition_under_a_case_runs() {
    // grad_plan takes any LogicalPlan, a DataFrame's included, and this one is
    // accepted, then its backward step does not match its own schema.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE b (j BIGINT, val DOUBLE) AS VALUES (0, 0.5), (1, -0.25)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE m (i BIGINT, j BIGINT, val DOUBLE) AS VALUES (0, 0, 0.6), (1, 0, -0.5), (0, 1, 0.9)",
    )
    .await;
    let sql = "WITH r0 AS (SELECT i, j, CAST(val * 10 AS BIGINT) AS v FROM m), \
                    r2 AS (SELECT a.i, a.j, CASE WHEN b.val > 0 THEN a.v ELSE 0.5 * a.v END AS v \
                           FROM r0 a JOIN b ON a.j = b.j AND a.j <> 1) \
               SELECT SUM(v) AS loss FROM r2";
    let plan = ctx.sql(sql).await.unwrap().into_unoptimized_plan();
    let program = ad::grad_plan(&ctx, &plan, &[ColumnRef::new("b", "val")]).unwrap();
    ad::run(&ctx, &program).await.unwrap();
}
