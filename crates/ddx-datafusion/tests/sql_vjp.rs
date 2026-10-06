// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `vjp(f, table.column, …, cotangent)` in SQL, end to end on DataFusion: a
//! gradient shaped like the table, pulled back from any `f` along a
//! cotangent shaped like `f`'s output, checked against the closed form,
//! against `grad(…)` (a vjp seeded with 1) and against `ad::vjp`'s program.

use datafusion::arrow::array::AsArray;
use datafusion::arrow::datatypes::{Float64Type, Int64Type};
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, ColumnRef};

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// `(i, val)` rows of `sql`, through `ad::sql`, by `i`.
async fn pairs(ctx: &SessionContext, sql: &str) -> Vec<(i64, f64)> {
    let batches = ad::sql(ctx, sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .collect()
        .await
        .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let i = b.column(0).as_primitive::<Int64Type>();
        let v = b.column(1).as_primitive::<Float64Type>();
        out.extend((0..b.num_rows()).map(|r| (i.value(r), v.value(r))));
    }
    out.sort_by_key(|p| p.0);
    out
}

async fn weights() -> SessionContext {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 0.5), (1, -1.0), (2, 2.0)",
    )
    .await;
    ctx
}

#[tokio::test]
async fn a_relation_pulls_its_cotangent_back_to_a_gradient_shaped_like_the_table() {
    // s = val², so the pull-back of c is 2 val c.
    let ctx = weights().await;
    let got = pairs(
        &ctx,
        "WITH f AS (SELECT i, val * val AS s FROM w), \
              c AS (SELECT i, CAST(i + 1 AS DOUBLE) AS s FROM w) \
         SELECT i, val FROM vjp(f, w.val, c)",
    )
    .await;
    assert_eq!(got, vec![(0, 1.0), (1, -4.0), (2, 12.0)]);
}

#[tokio::test]
async fn vjp_of_a_loss_along_one_is_grad() {
    let ctx = weights().await;
    let with = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), one AS (SELECT 1.0 AS l)";
    let vjp = pairs(&ctx, &format!("{with} SELECT * FROM vjp(loss, w.val, one)")).await;
    let grad = pairs(&ctx, &format!("{with} SELECT * FROM grad(loss, w.val)")).await;
    assert_eq!(vjp, grad);
    assert_eq!(vjp, vec![(0, 0.75), (1, 3.0), (2, 12.0)]);
}

#[tokio::test]
async fn a_vjp_s_where_on_its_dims_restricts_its_rows_and_agrees_with_the_program() {
    let ctx = weights().await;
    exec(
        &ctx,
        "CREATE TABLE c (i BIGINT, s DOUBLE) AS VALUES (0, 1.0), (1, 10.0), (2, 100.0)",
    )
    .await;
    let got = pairs(
        &ctx,
        "WITH f AS (SELECT i, val * val AS s FROM w) \
         SELECT g.i, g.val FROM vjp(f, w.val, c) g WHERE g.i > 0",
    )
    .await;
    // The oracle: ad::vjp's program, its cotangent registered by hand.
    let program = ad::vjp(
        &ctx,
        "SELECT i, val * val AS s FROM w",
        &[ColumnRef::new("w", "val")],
    )
    .await
    .unwrap();
    exec(
        &ctx,
        &format!("CREATE TABLE {} AS SELECT * FROM c", program.inputs[0].name),
    )
    .await;
    ad::run(&ctx, &program).await.unwrap();
    let want = pairs(
        &ctx,
        &format!(
            "SELECT i, val FROM {} WHERE i > 0",
            program.gradients[0].step
        ),
    )
    .await;
    assert_eq!(got, want);
    assert_eq!(got, vec![(1, -20.0), (2, 400.0)]);
}

#[tokio::test]
async fn vjps_along_one_cotangent_share_a_backward_pass_across_tables() {
    let ctx = weights().await;
    exec(
        &ctx,
        "CREATE TABLE b (i BIGINT, val DOUBLE) AS VALUES (0, 3.0), (1, 4.0), (2, 5.0)",
    )
    .await;
    // f keeps both tables' dims, as a vjp's output must.
    let with = "WITH f AS (SELECT w.i, b.i AS j, w.val * b.val AS s FROM w JOIN b ON w.i = b.i), \
                c AS (SELECT i, i AS j, 1.0 AS s FROM w)";
    let a = format!("{with} SELECT * FROM vjp(f, w.val, c)");
    let b = format!("{with} SELECT * FROM vjp(f, b.val, c)");
    let planned = ddx_ad::Statements::plan(
        &[a.as_str(), b.as_str()],
        &datafusion::sql::sqlparser::dialect::GenericDialect {},
    )
    .unwrap();
    assert_eq!(planned.jobs().len(), 1);
    let frames = ad::sql_all(&ctx, &[&a, &b]).await.unwrap();
    let mut out = Vec::new();
    for f in frames {
        let batches = f.collect().await.unwrap();
        out.push(
            batches[0]
                .column(1)
                .as_primitive::<Float64Type>()
                .values()
                .to_vec(),
        );
    }
    // d(w·b)/dw = b, d(w·b)/db = w.
    let mut w = out[0].clone();
    w.sort_by(f64::total_cmp);
    assert_eq!(w, vec![3.0, 4.0, 5.0]);
}
