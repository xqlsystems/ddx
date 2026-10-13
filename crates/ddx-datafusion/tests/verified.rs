// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `run_verified`: a program's checks skipped where a [`Verified`] already
//! holds what they prove, and run again once a table is forgotten.

mod common;

use common::ad::rows;
use common::substrait_of;
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, ColumnRef, Verified};

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

async fn ctx_with_w() -> SessionContext {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0), (2, 3.0)",
    )
    .await;
    ctx
}

fn wrt() -> [ColumnRef; 1] {
    [ColumnRef::new("w", "val")]
}

/// w with a second row at i = 0: its dims no longer identify its rows.
async fn repeat_a_dim(ctx: &SessionContext) {
    exec(
        ctx,
        "CREATE OR REPLACE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (0, 5.0), (1, 2.0), (2, 3.0)",
    )
    .await;
}

#[tokio::test]
async fn a_run_records_what_its_checks_proved() {
    let ctx = ctx_with_w().await;
    let plan = substrait_of(&ctx, "SELECT SUM(val * val) AS l FROM w", true).await;
    let program = ddx_ad::grad(&plan, &wrt()).unwrap();
    let mut verified = Verified::new();
    assert!(!verified.knows(&program.checks[0]));
    ad::run_verified(&ctx, &program, &[], &mut verified)
        .await
        .unwrap();
    assert!(verified.knows(&program.checks[0]));
    assert_eq!(program.checks[0].table, vec!["w"]);
    assert_eq!(program.checks[0].keys, vec!["i"]);
}

#[tokio::test]
async fn a_known_fact_skips_its_check_and_forgetting_the_table_brings_it_back() {
    let ctx = ctx_with_w().await;
    let plan = substrait_of(&ctx, "SELECT SUM(val * val) AS l FROM w", true).await;
    let program = ddx_ad::grad(&plan, &wrt()).unwrap();
    let mut verified = Verified::new();
    ad::run_verified(&ctx, &program, &[], &mut verified)
        .await
        .unwrap();

    // The caller's promise broken: w's dims repeat, and the fact is kept.
    // The check is skipped, so the run goes ahead; that is the contract.
    repeat_a_dim(&ctx).await;
    ad::run_verified(&ctx, &program, &[], &mut verified)
        .await
        .expect("a known fact skips its check");

    // Forgotten, w is checked again, and refused.
    verified.forget("w");
    let e = ad::run_verified(&ctx, &program, &[], &mut verified)
        .await
        .expect_err("w's dims repeat");
    assert!(e.to_string().contains("share their dims"), "{e}");
    assert!(
        !verified.knows(&program.checks[0]),
        "a failed run records nothing"
    );
    // run, which keeps no facts, refuses it too.
    assert!(ad::run(&ctx, &program, &[]).await.is_err());
}

#[tokio::test]
async fn a_failed_run_records_nothing() {
    let ctx = ctx_with_w().await;
    repeat_a_dim(&ctx).await;
    let plan = substrait_of(&ctx, "SELECT SUM(val * val) AS l FROM w", true).await;
    let program = ddx_ad::grad(&plan, &wrt()).unwrap();
    let mut verified = Verified::new();
    assert!(ad::run_verified(&ctx, &program, &[], &mut verified)
        .await
        .is_err());
    assert!(!verified.knows(&program.checks[0]));
}

#[tokio::test]
async fn conjugate_gradient_checks_the_parameters_once_and_each_direction() {
    // An H·v program run once per direction, as conjugate gradient runs it:
    // w is checked on the first run only; each new direction, an input, is
    // checked every time, and one whose keys repeat is refused.
    let ctx = ctx_with_w().await;
    let loss = "SELECT SUM(val * val * val) AS l FROM w";
    let grad = ad::grad(&ctx, loss, &wrt()).await.unwrap();
    let hvp = ad::jvp(&ctx, &grad, &wrt()).await.unwrap();
    let w_dims = hvp
        .checks
        .iter()
        .find(|c| c.table == vec!["w"])
        .expect("w's dims check")
        .clone();
    let mut verified = Verified::new();
    for k in 1..=3 {
        let v = ctx
            .sql(&format!("SELECT i, CAST({k} AS DOUBLE) AS val FROM w"))
            .await
            .unwrap();
        let out = ad::run_verified(&ctx, &hvp, &[v], &mut verified)
            .await
            .unwrap();
        assert!(verified.knows(&w_dims));
        // H = diag(6 w), so H·v = 6 w v.
        ctx.register_table("hv", out.gradients[0].clone().into_view())
            .unwrap();
        let got = rows(&ctx, "SELECT i, val_tangent FROM hv ORDER BY i").await;
        ctx.deregister_table("hv").unwrap();
        for (row, w) in got.iter().zip([1.0, 2.0, 3.0]) {
            assert_eq!(row[1], 6.0 * w * k as f64);
        }
    }
    // Keys that repeat, in a direction given after its keys were checked.
    let v = ctx
        .sql("SELECT * FROM (VALUES (CAST(0 AS BIGINT), 1.0), (CAST(0 AS BIGINT), 1.0)) AS t(i, val)")
        .await
        .unwrap();
    let e = ad::run_verified(&ctx, &hvp, &[v], &mut verified)
        .await
        .expect_err("the direction's keys repeat");
    assert!(e.to_string().contains("share their dims"), "{e}");
}

#[tokio::test]
async fn a_run_leaves_nothing_on_the_context_and_takes_its_inputs_by_position() {
    let ctx = ctx_with_w().await;
    let grad = ad::grad(&ctx, "SELECT SUM(val * val * val) AS l FROM w", &wrt())
        .await
        .unwrap();
    let hvp = ad::jvp(&ctx, &grad, &wrt()).await.unwrap();
    let tables = |ctx: &SessionContext| -> Vec<String> {
        let mut t = ctx
            .catalog("datafusion")
            .unwrap()
            .schema("public")
            .unwrap()
            .table_names();
        t.sort();
        t
    };
    let before = tables(&ctx);

    // Too few inputs: refused, naming the one it takes.
    let e = ad::run(&ctx, &hvp, &[]).await.unwrap_err();
    assert!(e.to_string().contains(&hvp.inputs[0].name), "{e}");
    // An input without a column its table needs: refused, naming it.
    let v = ctx.sql("SELECT i, 1.0 AS dval FROM w").await.unwrap();
    let e = ad::run(&ctx, &hvp, &[v]).await.unwrap_err();
    assert!(e.to_string().contains("val"), "{e}");
    assert_eq!(tables(&ctx), before, "a failed run leaves nothing");

    // Extra columns are ignored; the outputs hold their rows after the
    // program's tables are gone.
    let v = ctx
        .sql("SELECT 'x' AS note, i, 1.0 AS val FROM w")
        .await
        .unwrap();
    let out = ad::run(&ctx, &hvp, &[v]).await.unwrap();
    assert_eq!(tables(&ctx), before, "a run leaves nothing");
    let value = out.value.collect().await.unwrap();
    assert_eq!(value[0].schema().field(0).name(), "l");
    let names: Vec<String> = out.gradients[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, vec!["i", "val", "val_tangent"]);
    let rows = out.gradients[0].clone().collect().await.unwrap();
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
}

#[tokio::test]
async fn forget_matches_a_table_as_a_column_ref_names_it() {
    let ctx = ctx_with_w().await;
    let plan = substrait_of(&ctx, "SELECT SUM(val * val) AS l FROM w", true).await;
    let program = ddx_ad::grad(&plan, &wrt()).unwrap();
    let mut verified = Verified::new();
    ad::run_verified(&ctx, &program, &[], &mut verified)
        .await
        .unwrap();
    verified.forget("x");
    assert!(verified.knows(&program.checks[0]), "another table");
    verified.forget("W");
    assert!(!verified.knows(&program.checks[0]), "case-insensitive");
}
