// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ad::run`: a program's inputs in, its value and gradients out, nothing
//! left on the context. Before any step, every run checks that each `wrt`
//! table's dims and each input's keys identify their rows, since rows that
//! shared them would get their gradients summed.

mod common;

use common::ad::rows;
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, ColumnRef};

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

#[tokio::test]
async fn every_run_checks_the_table_and_its_input() {
    // An H·v program run once per direction, as conjugate gradient runs it.
    let ctx = ctx_with_w().await;
    let loss = "SELECT SUM(val * val * val) AS l FROM w";
    let grad = ad::grad(&ctx, loss, &wrt()).await.unwrap();
    let hvp = ad::jvp(&ctx, &grad, &wrt()).await.unwrap();
    for k in 1..=3 {
        let v = ctx
            .sql(&format!("SELECT i, CAST({k} AS DOUBLE) AS val FROM w"))
            .await
            .unwrap();
        let out = ad::run(&ctx, &hvp, &[v]).await.unwrap();
        // H = diag(6 w), so H·v = 6 w v.
        ctx.register_table("hv", out.gradients[0].clone().into_view())
            .unwrap();
        let got = rows(&ctx, "SELECT i, val_tangent FROM hv ORDER BY i").await;
        ctx.deregister_table("hv").unwrap();
        for (row, w) in got.iter().zip([1.0, 2.0, 3.0]) {
            assert_eq!(row[1], 6.0 * w * k as f64);
        }
    }
    // A direction whose keys repeat is refused.
    let v = ctx
        .sql("SELECT * FROM (VALUES (CAST(0 AS BIGINT), 1.0), (CAST(0 AS BIGINT), 1.0)) AS t(i, val)")
        .await
        .unwrap();
    let e = ad::run(&ctx, &hvp, &[v])
        .await
        .expect_err("the direction's keys repeat");
    assert!(e.to_string().contains("share their dims"), "{e}");
    // So is a run after w's dims stop identifying its rows, though earlier
    // runs passed.
    exec(
        &ctx,
        "CREATE OR REPLACE TABLE w (i BIGINT, val DOUBLE) AS \
         VALUES (0, 1.0), (0, 5.0), (1, 2.0), (2, 3.0)",
    )
    .await;
    let v = ctx.sql("SELECT i, 1.0 AS val FROM w").await.unwrap();
    let e = ad::run(&ctx, &hvp, &[v])
        .await
        .expect_err("w's dims repeat");
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
