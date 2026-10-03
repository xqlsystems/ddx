// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad(loss, table.column)` in SQL, through `ddx_datafusion::ad::sql`.

mod common;

use common::ad::Table;
use datafusion::arrow::array::{AsArray, RecordBatch};
use datafusion::arrow::datatypes::{Float64Type, Int64Type};
use datafusion::prelude::SessionContext;
use ddx_ad::AdError;
use ddx_datafusion::ad;

fn w() -> Table {
    Table {
        name: "w",
        columns: vec![("i", "BIGINT"), ("val", "DOUBLE")],
        rows: vec![vec![0.0, 1.0], vec![1.0, -2.0], vec![2.0, 0.5]],
    }
}

fn b() -> Table {
    Table {
        name: "b",
        columns: vec![("i", "BIGINT"), ("val", "DOUBLE")],
        rows: vec![vec![0.0, 0.25], vec![1.0, 3.0], vec![2.0, -1.0]],
    }
}

async fn ctx() -> SessionContext {
    let ctx = SessionContext::new();
    w().create(&ctx).await;
    b().create(&ctx).await;
    ctx
}

/// `(i, value)` pairs from the first two columns of `sql`'s result.
async fn pairs(ctx: &SessionContext, sql: &str) -> Vec<(i64, f64)> {
    let batches: Vec<RecordBatch> = ad::sql(ctx, sql).await.unwrap().collect().await.unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let i = b.column(0).as_primitive::<Int64Type>();
        let v = b.column(1).as_primitive::<Float64Type>();
        out.extend((0..b.num_rows()).map(|r| (i.value(r), v.value(r))));
    }
    out.sort_by_key(|p| p.0);
    out
}

#[tokio::test]
async fn grad_is_a_relation_shaped_like_its_table() {
    let ctx = ctx().await;
    let got = pairs(
        &ctx,
        "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w) \
         SELECT * FROM grad(loss, w.val)",
    )
    .await;
    assert_eq!(got, vec![(0, 3.0), (1, 12.0), (2, 0.75)]);
}

#[tokio::test]
async fn an_sgd_step_is_a_join() {
    // params - lr * grad(loss)(params)
    let ctx = ctx().await;
    let got = pairs(
        &ctx,
        "WITH loss AS (SELECT SUM(val * val) AS l FROM w) \
         SELECT w.i, w.val - 0.25 * g.val AS val \
         FROM w JOIN grad(loss, w.val) g ON w.i = g.i",
    )
    .await;
    // val - 0.25 · 2·val = val / 2
    assert_eq!(got, vec![(0, 0.5), (1, -1.0), (2, 0.25)]);
}

#[tokio::test]
async fn one_loss_two_tables() {
    // loss = Σ_i w_i · b_i: ∂/∂w = b and ∂/∂b = w, from one program.
    let ctx = ctx().await;
    let sql = "WITH loss AS (SELECT SUM(w.val * b.val) AS l FROM w JOIN b ON w.i = b.i) \
               SELECT gw.i, gw.val + 10.0 * gb.val AS v \
               FROM grad(loss, w.val) gw JOIN grad(loss, b.val) gb ON gw.i = gb.i";
    let got = pairs(&ctx, sql).await;
    let want: Vec<(i64, f64)> = w()
        .rows
        .iter()
        .zip(&b().rows)
        .map(|(w, b)| (w[0] as i64, b[1] + 10.0 * w[1]))
        .collect();
    assert_eq!(got, want);
}

#[tokio::test]
async fn the_loss_can_use_other_ctes() {
    let ctx = ctx().await;
    let got = pairs(
        &ctx,
        "WITH sq AS (SELECT i, val * val AS s FROM w), \
              loss AS (SELECT SUM(s) AS l FROM sq) \
         SELECT i, val FROM grad(loss, w.val)",
    )
    .await;
    assert_eq!(got, vec![(0, 2.0), (1, -4.0), (2, 1.0)]);
}

#[tokio::test]
async fn a_statement_without_grad_runs_as_it_is() {
    let ctx = ctx().await;
    let got = pairs(&ctx, "SELECT i, val FROM w").await;
    assert_eq!(got, vec![(0, 1.0), (1, -2.0), (2, 0.5)]);
}

#[tokio::test]
async fn grad_of_something_that_is_not_a_loss_is_refused() {
    let ctx = ctx().await;
    let err = ad::sql(
        &ctx,
        "WITH loss AS (SELECT i, val * val AS l FROM w) SELECT * FROM grad(loss, w.val)",
    )
    .await
    .unwrap_err();
    let datafusion::error::DataFusionError::External(boxed) = err else {
        panic!("expected External, got {err}")
    };
    assert!(matches!(
        boxed.downcast_ref::<AdError>(),
        Some(AdError::NotScalar(_))
    ));
}

#[tokio::test]
async fn statements_sharing_a_loss_share_its_program() {
    // One training step, one statement per parameter table.
    let ctx = ctx().await;
    let loss = "WITH loss AS (SELECT SUM(w.val * w.val * b.val) AS l FROM w JOIN b ON w.i = b.i) ";
    let update_w = format!(
        "{loss} SELECT w.i, w.val - 0.5 * g.val AS val FROM w JOIN grad(loss, w.val) g ON w.i = g.i"
    );
    let update_b = format!(
        "{loss} SELECT b.i, b.val - 0.5 * g.val AS val FROM b JOIN grad(loss, b.val) g ON b.i = g.i"
    );
    let frames = ad::sql_all(&ctx, &[&update_w, &update_b]).await.unwrap();
    // Both gradients came from one program: the frames read tables with one
    // program's prefix, and the catalog keeps none of them.
    let prefix = |plan: String| -> Vec<String> {
        plan.split("__ddx_")
            .skip(1)
            .filter_map(|rest| rest.split_once('_').map(|(id, _)| id.to_string()))
            .collect()
    };
    let mut ids: Vec<String> = frames
        .iter()
        .flat_map(|f| prefix(f.logical_plan().display_indent().to_string()))
        .collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 1, "{ids:?}");
    let left: Vec<String> = ctx
        .catalog("datafusion")
        .unwrap()
        .schema("public")
        .unwrap()
        .table_names()
        .into_iter()
        .filter(|n| n.starts_with("__ddx_"))
        .collect();
    assert!(left.is_empty(), "{left:?}");

    let mut got = Vec::new();
    for df in frames {
        let batches = df.collect().await.unwrap();
        let mut rows = Vec::new();
        for b in &batches {
            let i = b.column(0).as_primitive::<Int64Type>();
            let v = b.column(1).as_primitive::<Float64Type>();
            rows.extend((0..b.num_rows()).map(|r| (i.value(r), v.value(r))));
        }
        rows.sort_by_key(|p| p.0);
        got.push(rows);
    }
    for (k, (w, b)) in w().rows.iter().zip(&b().rows).enumerate() {
        // ∂/∂w = 2·w·b, ∂/∂b = w²
        assert_eq!(got[0][k].1, w[1] - 0.5 * 2.0 * w[1] * b[1]);
        assert_eq!(got[1][k].1, b[1] - 0.5 * w[1] * w[1]);
    }
}

#[tokio::test]
async fn case_variants_of_a_column_keep_the_dims() {
    // w.val and W.VAL are one wrt column; the gradient keeps its dim `i`.
    let ctx = ctx().await;
    let got = pairs(
        &ctx,
        "WITH loss AS (SELECT SUM(val * val) AS l FROM w) \
         SELECT a.i, a.val + b.val AS v \
         FROM grad(loss, w.val) a JOIN grad(loss, W.VAL) b ON a.i = b.i",
    )
    .await;
    assert_eq!(got, vec![(0, 4.0), (1, -8.0), (2, 2.0)]);
}

#[tokio::test]
async fn tables_whose_names_join_alike_keep_their_own_gradients() {
    // Adversarial review (🤖😈): `sql_all` kept each gradient as
    // `__ddx_grad_{p}_{parts joined by _}`, so `a_b.c` and `a.b_c` share one
    // name and both calls read the second table's gradient: 20, 20 instead
    // of 2, 20, with no error.
    let ctx = SessionContext::new();
    for sql in [
        "CREATE SCHEMA a_b",
        "CREATE SCHEMA a",
        "CREATE TABLE a_b.c (i BIGINT, val DOUBLE) AS VALUES (0, 1.0)",
        "CREATE TABLE a.b_c (i BIGINT, val DOUBLE) AS VALUES (0, 10.0)",
    ] {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }
    let got = pairs(
        &ctx,
        "WITH loss AS (SELECT SUM(p.val * p.val) + SUM(q.val * q.val) AS l \
                       FROM a_b.c p CROSS JOIN a.b_c q) \
         SELECT g1.i, g1.val * 1000.0 + g2.val AS v \
         FROM grad(loss, a_b.c.val) g1 CROSS JOIN grad(loss, a.b_c.val) g2",
    )
    .await;
    // d/dp = 2p = 2 and d/dq = 2q = 20, packed as 2 * 1000 + 20.
    assert_eq!(got, vec![(0, 2020.0)]);
}

#[tokio::test]
async fn grad_in_sql_of_a_table_with_capitals() {
    // From the v2 soak (#92): the gradient step was registered under a name
    // DataFusion lowercased, then read back quoted, case and all.
    let ctx = SessionContext::new();
    ctx.sql("CREATE TABLE \"W\" (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let got = pairs(
        &ctx,
        "WITH loss AS (SELECT SUM(val * val) AS l FROM \"W\") \
         SELECT i, val FROM grad(loss, \"W\".val) ORDER BY i",
    )
    .await;
    assert_eq!(got, vec![(0, 2.0), (1, 4.0)]);
}

#[tokio::test]
async fn grad_in_sql_reads_comments_as_comments() {
    // From the v2 soak (#98): the call's end was found by counting
    // parentheses in the text, so one inside a comment cut the call short
    // or never closed it, and a comment between `grad` and `(` hid the call.
    let ctx = SessionContext::new();
    for sql in [
        "CREATE TABLE cp (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
        "CREATE TABLE cq (i BIGINT, val DOUBLE) AS VALUES (0, 3.0), (1, 4.0)",
    ] {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }
    for call in [
        "grad(loss /* ) */, cp.val)",
        "grad(loss, cp.val /* ( */)",
        "grad(loss, -- )\n cp.val)",
        "grad /* c */ (loss, cp.val)",
        "GRAD\n(loss, cp.val)",
    ] {
        let got = pairs(
            &ctx,
            &format!(
                "WITH loss AS (SELECT SUM(val * val) AS l FROM cp) \
                 SELECT i, val FROM {call} ORDER BY i"
            ),
        )
        .await;
        assert_eq!(got, vec![(0, 2.0), (1, 4.0)], "{call:?}");
    }
    // Two calls, with a comment that opened a parenthesis inside the first:
    // the first call's span ran into the second's, and rewriting panicked.
    let got = pairs(
        &ctx,
        "WITH loss AS (SELECT SUM(cp.val * cq.val) AS l FROM cp JOIN cq ON cp.i = cq.i) \
         SELECT a.i, a.val + b.val AS v \
         FROM grad(loss, cp.val /* ( */) a JOIN grad(loss, cq.val) b ON a.i = b.i /* ) */ \
         ORDER BY a.i",
    )
    .await;
    assert_eq!(got, vec![(0, 4.0), (1, 6.0)]);
}

#[tokio::test]
async fn not_in_over_a_nullable_subquery_is_refused_not_differentiated_wrongly() {
    // From the adversarial tester's review of #79: DataFusion's Substrait
    // producer writes `NOT IN`'s null-aware anti-join as a plain one (#104),
    // so ddx would differentiate a query that keeps rows a NULL excludes,
    // and say nothing. The adapter sees the null-aware join and refuses.
    use datafusion::prelude::SessionContext;
    use ddx_datafusion::ad::{self, AdError, ColumnRef};
    let ctx = SessionContext::new();
    for sql in [
        "CREATE TABLE ny (s BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0), (2, 4.0)",
        "CREATE TABLE nx (s BIGINT, val DOUBLE) AS VALUES (0, 0.5), (NULL, 0.9), (1, -0.2)",
        "CREATE TABLE nn (s BIGINT NOT NULL, val DOUBLE) AS VALUES (0, 0.5), (1, -0.2)",
        "CREATE TABLE ky (s BIGINT NOT NULL, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0), (2, 4.0)",
    ] {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }
    let loss =
        "SELECT SUM(val * val) AS l FROM ny WHERE s NOT IN (SELECT s FROM nx WHERE val > 0.05)";
    let refused = |e: datafusion::error::DataFusionError| match e {
        datafusion::error::DataFusionError::External(b) => {
            matches!(b.downcast_ref::<AdError>(), Some(AdError::NotImplemented(m)) if m.contains("NOT IN"))
        }
        _ => false,
    };
    let wrt = [ColumnRef::new("ny", "val")];
    let err = ad::grad(&ctx, loss, &wrt).await.unwrap_err();
    assert!(refused(err));
    // Unoptimized, NOT IN is still a subquery expression: refused too.
    let lp = ctx.sql(loss).await.unwrap().into_unoptimized_plan();
    assert!(refused(ad::grad_plan(&ctx, &lp, &wrt).unwrap_err()));
    // In SQL, the same.
    let stmt = format!("WITH loss AS ({loss}) SELECT * FROM grad(loss, ny.val)");
    assert!(refused(ad::sql(&ctx, &stmt).await.unwrap_err()));
    // NOT EXISTS is a plain anti-join, and the round trip keeps it.
    let exists = "SELECT SUM(val * val) AS l FROM ny WHERE NOT EXISTS \
                  (SELECT 1 FROM nx WHERE nx.s = ny.s AND nx.val > 0.05)";
    ad::grad(&ctx, exists, &wrt).await.unwrap();
    // Neither side can be NULL: no null-aware join, nothing lost.
    let keyed =
        "SELECT SUM(val * val) AS l FROM ky WHERE s NOT IN (SELECT s FROM nn WHERE val > 0.05)";
    ad::grad(&ctx, keyed, &[ColumnRef::new("ky", "val")])
        .await
        .unwrap();
}
