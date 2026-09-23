// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx_stop_gradient` on a real engine: it is the identity, it keeps its
//! argument's type, and it reaches the Substrait plan where `ddx-ad` finds it.

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::SessionContext;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;
use ddx_ad::{Functions, STOP_GRADIENT};

async fn ctx() -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    for sql in [
        "CREATE TABLE a (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
        "CREATE TABLE h (g BIGINT, x REAL) AS VALUES (0, 1.5), (1, -1.0)",
    ] {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }
    ctx
}

async fn run(ctx: &SessionContext, sql: &str) -> String {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

#[tokio::test]
async fn it_is_the_identity() {
    let ctx = ctx().await;
    let with = "SELECT i, exp(val - ddx_stop_gradient(val)) AS v FROM a ORDER BY i";
    let without = "SELECT i, exp(val - (val)) AS v FROM a ORDER BY i";
    assert_eq!(run(&ctx, with).await, run(&ctx, without).await);
}

#[tokio::test]
async fn it_keeps_its_argument_type() {
    // A REAL column stays REAL: no cast is planned around or inside it.
    let ctx = ctx().await;
    let df = ctx
        .sql("SELECT ddx_stop_gradient(x) AS m FROM h")
        .await
        .unwrap();
    let plan = df.logical_plan().display_indent().to_string();
    assert!(!plan.contains("CAST"), "{plan}");
    assert_eq!(
        df.schema().field(0).data_type(),
        &datafusion::arrow::datatypes::DataType::Float32
    );
}

#[tokio::test]
async fn it_survives_into_the_substrait_plan() {
    let ctx = ctx().await;
    let sql = "SELECT SUM(exp(val - ddx_stop_gradient(val))) AS s FROM a";
    let df = ctx.sql(sql).await.unwrap();
    for plan in [df.logical_plan().clone(), df.into_optimized_plan().unwrap()] {
        let substrait = to_substrait_plan(&plan, &ctx.state()).unwrap();
        let functions = Functions::from_plan(&substrait).unwrap();
        let found = functions
            .declarations()
            .iter()
            .any(|d| ddx_ad::normalize(&d.name) == STOP_GRADIENT);
        assert!(found, "no {STOP_GRADIENT} declared");
    }
}
