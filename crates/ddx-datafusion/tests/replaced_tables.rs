// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A DataFusion view keeps the tables it was made over: after one is
//! registered again under its name, the view still reads the old one, while
//! ddx reads tables by name. ddx refuses such a query rather than
//! differentiate a loss the engine doesn't compute (ddx issue #172).

use datafusion::arrow::array::{AsArray, RecordBatch};
use datafusion::arrow::datatypes::{Float64Type, Int64Type};
use datafusion::error::DataFusionError;
use datafusion::prelude::SessionContext;
use ddx_ad::ColumnRef;
use ddx_datafusion::ad;

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// `w`, `x`, and a view `v` of their product, then `x` registered again.
async fn replaced() -> SessionContext {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE x (i BIGINT, val DOUBLE) AS VALUES (0, 10.0), (1, 20.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE VIEW v AS SELECT w.i, w.val * x.val AS p FROM w JOIN x ON w.i = x.i",
    )
    .await;
    ctx
}

async fn replace_x(ctx: &SessionContext) {
    ctx.deregister_table("x").unwrap();
    exec(
        ctx,
        "CREATE TABLE x (i BIGINT, val DOUBLE) AS VALUES (0, 100.0), (1, 200.0)",
    )
    .await;
}

fn is_refusal(e: &DataFusionError) -> bool {
    // ddx's refusal, however DataFusion wraps it.
    let m = e.to_string();
    m.contains("registered again") && m.contains("Create the view again")
}

async fn pairs(ctx: &SessionContext, sql: &str) -> Result<Vec<(i64, f64)>, DataFusionError> {
    let batches: Vec<RecordBatch> = ad::sql(ctx, sql).await?.collect().await?;
    let mut out = Vec::new();
    for b in &batches {
        let i = b.column(0).as_primitive::<Int64Type>();
        let v = b.column(1).as_primitive::<Float64Type>();
        out.extend((0..b.num_rows()).map(|r| (i.value(r), v.value(r))));
    }
    out.sort_by_key(|p| p.0);
    Ok(out)
}

const LOSS: &str = "WITH loss AS (SELECT SUM(p) AS l FROM v) SELECT i, grad(loss, val) FROM w";

#[tokio::test]
async fn a_view_over_a_table_registered_again_is_refused() {
    let ctx = replaced().await;
    // Before: the gradient of SUM(w.val * x.val) is x.val.
    assert_eq!(pairs(&ctx, LOSS).await.unwrap(), vec![(0, 10.0), (1, 20.0)]);
    replace_x(&ctx).await;
    // The engine still reads the old x through v.
    let wrt = [ColumnRef::new("w", "val")];
    let err = ad::grad(&ctx, "SELECT SUM(p) AS l FROM v", &wrt)
        .await
        .unwrap_err();
    assert!(is_refusal(&err), "{err}");
    // In SQL too, though its program was kept from the call before.
    let err = pairs(&ctx, LOSS).await.unwrap_err();
    assert!(is_refusal(&err), "{err}");
}

#[tokio::test]
async fn a_model_view_over_a_table_registered_again_is_refused() {
    // A view whose CTE is read twice is written into the objective from its
    // SQL, which reads the new x: it is left to the engine, and refused.
    let ctx = replaced().await;
    exec(
        &ctx,
        "CREATE VIEW sq AS WITH a AS (SELECT i, p FROM v) \
         SELECT a.i, a.p * b.p AS p FROM a JOIN a b ON a.i = b.i",
    )
    .await;
    let loss = "WITH loss AS (SELECT SUM(p) AS l FROM sq) SELECT i, grad(loss, val) FROM w";
    // d/dw of SUM((w x)^2) is 2 w x^2.
    assert_eq!(
        pairs(&ctx, loss).await.unwrap(),
        vec![(0, 200.0), (1, 1600.0)]
    );
    replace_x(&ctx).await;
    let err = pairs(&ctx, loss).await.unwrap_err();
    assert!(is_refusal(&err), "{err}");
    // Nor is either view's SQL given to be written out, whatever else
    // refuses the objective first.
    use ddx_ad::Backend;
    let mut backend = ddx_datafusion::engine::DataFusion(ctx.clone());
    assert_eq!(backend.view_sql("v").await.unwrap(), None);
    assert_eq!(backend.view_sql("sq").await.unwrap(), None);
}

#[tokio::test]
async fn a_table_registered_again_and_read_by_name_is_not_refused() {
    let ctx = replaced().await;
    replace_x(&ctx).await;
    let direct = "WITH loss AS (SELECT SUM(w.val * x.val) AS l FROM w JOIN x ON w.i = x.i) \
                  SELECT i, grad(loss, val) FROM w";
    assert_eq!(
        pairs(&ctx, direct).await.unwrap(),
        vec![(0, 100.0), (1, 200.0)]
    );
    // The view made again reads the new x.
    exec(
        &ctx,
        "CREATE OR REPLACE VIEW v AS SELECT w.i, w.val * x.val AS p FROM w JOIN x ON w.i = x.i",
    )
    .await;
    assert_eq!(
        pairs(&ctx, LOSS).await.unwrap(),
        vec![(0, 100.0), (1, 200.0)]
    );
}

/// A recursive CTE reads its own rows through a work table named after
/// it, which no catalog holds: that is not a table registered again.
#[tokio::test]
async fn a_recursive_cte_is_not_taken_for_a_table_registered_again() {
    let ctx = replaced().await;
    let r = "WITH RECURSIVE r AS (SELECT CAST(1 AS BIGINT) AS n UNION ALL \
             SELECT n + 1 FROM r WHERE n < 3)";
    exec(&ctx, &format!("CREATE VIEW rv AS {r} SELECT n FROM r")).await;
    use ddx_ad::Backend;
    let mut backend = ddx_datafusion::engine::DataFusion(ctx.clone());
    assert!(backend.view_sql("rv").await.unwrap().is_some());
    // DataFusion's Substrait producer refuses a recursive query; that, not
    // this check, is what refuses its gradient.
    let wrt = [ColumnRef::new("w", "val")];
    let loss = format!("{r} SELECT SUM(w.val * r.n) AS l FROM w JOIN r ON w.i + 1 = r.n");
    let err = ad::grad(&ctx, &loss, &wrt).await.unwrap_err();
    assert!(!is_refusal(&err), "{err}");
}
