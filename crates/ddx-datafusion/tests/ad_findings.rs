// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Bugs `ad_simulation.rs` found, each reduced to the smallest query that
//! shows it and pinned against a gradient worked by hand.
//!
//! Each was `#[ignore]`d as a known bug until its fix landed (CONTRIBUTING.md,
//! "a failing test first"). Every ddx bug here is now fixed, and runs as an
//! ordinary test; each entry below names the fix. The upstream DataFusion
//! bugs at the end stay ignored until an upgrade fixes them.
//!
//! - **NULL rows leak gradient.** SUM, AVG, MAX and MIN skip a row whose
//!   argument is NULL, so nothing in that row can move the loss. ddx still
//!   broadcasts the group's cotangent to the row and pushes it through the
//!   row's other inputs, where a partial that does not read the NULL (the 1 of
//!   `p + q`) is not NULL. The gradient is silently wrong at every parameter
//!   joined to a NULL, in a parameter table or in constant data. *Fixed in
//!   #74 and #76:* each reduce rule's seed is NULL where its argument is.
//! - **Two rank filters in one region collide.** Rebuilding the region keeps
//!   both window columns, and the optimizer has given them the same name, so
//!   DataFusion's consumer refuses the step. The program was accepted.
//!   *Fixed in #73:* a window column is renamed in place once computed.
//! - **A CASE over integer data, in an unoptimized plan.** `grad_plan`
//!   accepts any `LogicalPlan`, a DataFrame's included; a CASE choosing
//!   between integer columns on a varied condition is accepted, and its
//!   backward step then holds values of a type its schema does not declare.
//!   *Fixed in #79:* a step's table takes the schema of the plan that ran.

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
async fn an_unoptimized_case_over_integer_data_runs() {
    // grad_plan takes any LogicalPlan, a DataFrame's included. In this
    // unoptimized one a CASE picks between integer data on a condition that
    // depends on b.val: it has no derivative with respect to b (the branches
    // are constant), so the gradient is 0. ddx accepts it, then its backward
    // step fails "Mismatch between schema and batches": a column's declared
    // type is not the type of the values it holds.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE b (val DOUBLE, i BIGINT) AS VALUES (0.5, 0), (-0.25, 1)",
    )
    .await;
    exec(&ctx, "CREATE TABLE m (val DOUBLE) AS VALUES (0.6), (-0.5)").await;
    let sql = "SELECT SUM(CASE WHEN b.val > 0 THEN a.v ELSE 0.5 * a.v END) AS loss \
               FROM (SELECT CAST(val * 10 AS BIGINT) AS v FROM m) a CROSS JOIN b";
    let plan = ctx.sql(sql).await.unwrap().into_unoptimized_plan();
    let program = ad::grad_plan(&ctx, &plan, &[ColumnRef::new("b", "val")]).unwrap();
    ad::run(&ctx, &program).await.unwrap();
}

// ---------------------------------------------------------------------------
// Upstream: DataFusion bugs the soak reached, pinned with no ddx involved so an
// upgrade shows at once whether they are fixed. Each is reached only with an
// optimizer rule set DataFusion does not ship by default.

/// A context whose logical optimizer runs only `rules`.
fn ctx_with_only(rules: &[&str]) -> SessionContext {
    use datafusion::execution::SessionStateBuilder;
    use datafusion::optimizer::Optimizer;
    let rules = Optimizer::new()
        .rules
        .into_iter()
        .filter(|r| rules.contains(&r.name()))
        .collect();
    SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_optimizer_rules(rules)
            .build(),
    )
}

#[tokio::test]
#[ignore = "upstream DataFusion 54: a sort beneath a limit is dropped under a join"]
async fn upstream_a_limit_keeps_its_sort_under_a_join() {
    // Without push_down_limit to fuse the limit into the sort, the physical
    // plan loses the sort once the projection above it drops the sort key,
    // and the limit keeps the first rows in table order. ddx's backward steps
    // recompute such a region under a join, so on a context configured like
    // this its gradient lands on the wrong rows.
    let ctx = ctx_with_only(&[]);
    exec(
        &ctx,
        "CREATE TABLE u (i BIGINT, val DOUBLE) AS VALUES (1, -0.34), (2, -0.57), (0, 0.71)",
    )
    .await;
    exec(&ctx, "CREATE TABLE one (c DOUBLE) AS VALUES (1.0)").await;
    let batches = ctx
        .sql("SELECT t.i FROM (SELECT i FROM u ORDER BY val DESC LIMIT 1) t CROSS JOIN one")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    use datafusion::arrow::array::AsArray;
    use datafusion::arrow::datatypes::Int64Type;
    let i = batches[0].column(0).as_primitive::<Int64Type>().value(0);
    assert_eq!(i, 0, "the row with the largest val is i = 0");
}
