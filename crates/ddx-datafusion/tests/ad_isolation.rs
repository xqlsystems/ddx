// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Adversarial review (🤖😈): `ad::run` materializes every step into the
//! caller's `SessionContext` under fixed, global names (`__ddx_value`,
//! `__ddx_saved_{n}`, `__ddx_cotangent_{n}`, `__ddx_grad_{table}`), replacing
//! whatever is there and leaving it all behind. The first two tests failed
//! before each program got its own name prefix.

use datafusion::arrow::array::{AsArray, RecordBatch};
use datafusion::arrow::datatypes::Float64Type;
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, ColumnRef};

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

async fn f64s(ctx: &SessionContext, sql: &str) -> Vec<f64> {
    let batches: Vec<RecordBatch> = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Float64Type>().values().to_vec())
        .collect()
}

#[tokio::test]
async fn run_does_not_replace_a_users_table() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE __ddx_value (note VARCHAR) AS VALUES ('mine')",
    )
    .await;
    let program = ad::grad(
        &ctx,
        "SELECT SUM(val * val) AS l FROM w",
        &[ColumnRef::new("w", "val")],
    )
    .await
    .unwrap();
    // Refusing to run is acceptable; silently dropping the user's table is not.
    if ad::run(&ctx, &program).await.is_err() {
        return;
    }
    let still_mine = ctx
        .sql("SELECT note FROM __ddx_value")
        .await
        .map(|_| true)
        .unwrap_or(false);
    assert!(
        still_mine,
        "ad::run deregistered the user's table `__ddx_value`"
    );
}

#[tokio::test]
async fn interleaved_programs_do_not_read_each_others_tape() {
    // Two programs on one context (two models, two losses, or two requests to
    // a server sharing a SessionContext) both write `__ddx_saved_0`. If B runs
    // between A's forward and backward steps, A's MAX rule compares w's rows
    // with v's saved maximum, no row matches, and A's gradient is silently
    // all zeros. With tokio::spawn on a shared context this happens on its
    // own; it can also fail with "No table named '__ddx_saved_0'".
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 5.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE v (i BIGINT, val DOUBLE) AS VALUES (0, 2.0), (1, 3.0)",
    )
    .await;
    let a = ad::grad(
        &ctx,
        "SELECT MAX(val) AS l FROM w",
        &[ColumnRef::new("w", "val")],
    )
    .await
    .unwrap();
    let b = ad::grad(
        &ctx,
        "SELECT MAX(val) AS l FROM v",
        &[ColumnRef::new("v", "val")],
    )
    .await
    .unwrap();
    for s in &a.forward_steps {
        ad::run_step(&ctx, s).await.unwrap();
    }
    ad::run(&ctx, &b).await.unwrap();
    for s in &a.backward_steps {
        ad::run_step(&ctx, s).await.unwrap();
    }
    let got = f64s(
        &ctx,
        &format!("SELECT val FROM {} ORDER BY i", a.gradients[0].step),
    )
    .await;
    assert_eq!(got, vec![0.0, 1.0], "d/dw MAX(w.val) is one-hot on the max");
}

#[tokio::test]
async fn run_leaves_the_value_and_gradients_and_release_drops_them() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 5.0)",
    )
    .await;
    let program = ad::grad(
        &ctx,
        "SELECT MAX(val) * SUM(val) AS l FROM w",
        &[ColumnRef::new("w", "val")],
    )
    .await
    .unwrap();
    assert!(program.intermediate_steps().next().is_some());
    ad::run(&ctx, &program).await.unwrap();
    for step in program.intermediate_steps() {
        assert!(
            !ctx.table_exist(step.name.as_str()).unwrap(),
            "{}",
            step.name
        );
    }
    assert!(ctx.table_exist(program.value.as_str()).unwrap());
    assert_eq!(
        f64s(
            &ctx,
            &format!("SELECT val FROM {} ORDER BY i", program.gradients[0].step)
        )
        .await,
        vec![5.0, 11.0]
    );
    // Running again works from a context with no intermediates left.
    ad::run(&ctx, &program).await.unwrap();
    ad::release(&ctx, &program).unwrap();
    for step in program.steps() {
        assert!(
            !ctx.table_exist(step.name.as_str()).unwrap(),
            "{}",
            step.name
        );
    }
}

#[tokio::test]
async fn an_unoptimized_case_over_integer_data_runs() {
    // From the v2 soak (#89, #90). grad_plan takes any LogicalPlan, a
    // DataFrame's included. In this unanalyzed one a CASE picks between a
    // BIGINT and a DOUBLE branch on a condition that depends on b.val, so
    // its type is settled only by type coercion; the step's declared
    // schema disagreed with the batches it produced.
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
    // The branches are constant, so the gradient is 0 at every row.
    let got = f64s(
        &ctx,
        &format!("SELECT val FROM {} ORDER BY i", program.gradients[0].step),
    )
    .await;
    assert_eq!(got, vec![0.0, 0.0]);
}

#[tokio::test]
async fn a_deep_chain_of_maps_plans_to_a_small_plan() {
    // From the v2 soak (#98): each layer `sin(v) + 0.1 * v` reads v twice,
    // and DataFusion's Substrait consumer names a computed column by its
    // expression, so every layer doubled the names: 60 MB of plan at 14
    // layers, and 20 exhausted 13 GB. ad::run consumes a step with short
    // names instead.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 0.3), (1, 0.5)",
    )
    .await;
    let depth = 20;
    let mut ctes = vec!["c0 AS (SELECT i, val AS v FROM p)".to_string()];
    for k in 1..depth {
        ctes.push(format!(
            "c{k} AS (SELECT i, sin(v) + 0.1 * v AS v FROM c{})",
            k - 1
        ));
    }
    let loss = format!(
        "WITH {} SELECT SUM(v) AS l FROM c{}",
        ctes.join(", "),
        depth - 1
    );
    let program = ad::grad(&ctx, &loss, &[ColumnRef::new("p", "val")])
        .await
        .unwrap();
    let mut bytes = 0;
    for step in program.steps() {
        let lp = ad::logical_plan(&ctx, &step.plan).await.unwrap();
        bytes += lp.display_indent().to_string().len();
        ad::run_step(&ctx, step).await.unwrap();
    }
    assert!(
        bytes < 2_000_000,
        "{bytes} bytes of plan for {depth} layers"
    );
    // And the gradient is the chain rule's: Π (cos(v_k) + 0.1).
    let got = f64s(
        &ctx,
        &format!("SELECT val FROM {} ORDER BY i", program.gradients[0].step),
    )
    .await;
    for (g, v0) in got.iter().zip([0.3f64, 0.5]) {
        let (mut v, mut d) = (v0, 1.0);
        for _ in 1..depth {
            d *= v.cos() + 0.1;
            v = v.sin() + 0.1 * v;
        }
        assert!((g - d).abs() < 1e-12, "{g} vs {d}");
    }
}

#[tokio::test]
async fn vjp_refuses_a_cotangent_whose_keys_repeat_through_ad_run() {
    // The check reads the cotangent table, whose types ddx does not know;
    // ad::run binds a check's reads as it binds a step's.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    let program = ad::vjp(
        &ctx,
        "SELECT i, val * val AS s FROM p",
        &[ColumnRef::new("p", "val")],
    )
    .await
    .unwrap();
    exec(
        &ctx,
        &format!(
            "CREATE TABLE \"{}\" (i BIGINT, s DOUBLE) AS VALUES (0, 1.0), (0, 1.0), (1, 1.0)",
            program.cotangent_table
        ),
    )
    .await;
    let err = ad::run(&ctx, &program).await.unwrap_err();
    assert!(err.to_string().contains("share their keys"), "{err}");
}
