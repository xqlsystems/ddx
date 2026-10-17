// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `vjp(f, column, cotangent)` in SQL, end to end on DataFusion: a column of
//! a SELECT over the column's table, pulled back from any `f` along a
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
async fn a_column_pulls_its_cotangent_back_to_its_table_s_rows() {
    // s = val², so the pull-back of c is 2 val c.
    let ctx = weights().await;
    let got = pairs(
        &ctx,
        "WITH f AS (SELECT i, val * val AS s FROM w), \
              c AS (SELECT i, CAST(i + 1 AS DOUBLE) AS s FROM w) \
         SELECT i, vjp(f, val, c) FROM w",
    )
    .await;
    assert_eq!(got, vec![(0, 1.0), (1, -4.0), (2, 12.0)]);
}

#[tokio::test]
async fn vjp_of_a_loss_along_one_is_grad() {
    let ctx = weights().await;
    let with = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), one AS (SELECT 1.0 AS l)";
    let vjp = pairs(
        &ctx,
        &format!("{with} SELECT i, vjp(loss, val, one) FROM w"),
    )
    .await;
    let grad = pairs(&ctx, &format!("{with} SELECT i, grad(loss, val) FROM w")).await;
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
         SELECT i, vjp(f, val, c) FROM w WHERE i > 0",
    )
    .await;
    // The oracle: ad::vjp's program, run on the cotangent.
    let program = ad::vjp(
        &ctx,
        "SELECT i, val * val AS s FROM w",
        &[ColumnRef::new("w", "val")],
    )
    .await
    .unwrap();
    let c = ctx.sql("SELECT * FROM c").await.unwrap();
    let out = ad::run(&ctx, &program, &[c]).await.unwrap();
    ctx.register_table("oracle", out.gradients[0].clone().into_view())
        .unwrap();
    let want = pairs(&ctx, "SELECT i, val FROM oracle WHERE i > 0").await;
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
    let a = format!("{with} SELECT i, vjp(f, val, c) FROM w");
    let b = format!("{with} SELECT i, vjp(f, val, c) FROM b");
    // That they are one program is ddx-ad's unit test; here, that
    // the one program gives each statement its table's gradient.
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
    let mut b = out[1].clone();
    b.sort_by(f64::total_cmp);
    assert_eq!(b, vec![-1.0, 0.5, 2.0]);
}
