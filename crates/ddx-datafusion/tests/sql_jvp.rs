// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `jvp(f, table.column, …, tangent, …)` in SQL, end to end on DataFusion:
//! `f`'s output with each column's tangent beside it, checked against the
//! closed form and against `ad::jvp`'s program, which `tests/jvp.rs` checks
//! against finite differences.

use std::collections::BTreeMap;

use datafusion::arrow::array::{Array, AsArray, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Float64Type, Int64Type};
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, ColumnRef};

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// Each row of `batches` as numbers, by column name.
fn rows(batches: &[RecordBatch]) -> Vec<BTreeMap<String, f64>> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let mut row = BTreeMap::new();
            for (c, field) in b.schema().fields().iter().enumerate() {
                let a = b.column(c);
                let v = match a.data_type() {
                    DataType::Float64 => {
                        let a = a.as_primitive::<Float64Type>();
                        if a.is_null(r) {
                            f64::NAN
                        } else {
                            a.value(r)
                        }
                    }
                    DataType::Int64 => a.as_primitive::<Int64Type>().value(r) as f64,
                    other => panic!("column {} of type {other}", field.name()),
                };
                row.insert(field.name().clone(), v);
            }
            out.push(row);
        }
    }
    out
}

async fn sql_rows(ctx: &SessionContext, sql: &str) -> Vec<BTreeMap<String, f64>> {
    let df = ad::sql(ctx, sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    rows(&df.collect().await.unwrap())
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * b.abs().max(1.0)
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
async fn the_tangent_of_a_loss_is_its_directional_derivative() {
    // d/dt Σ (val + t·v)³ = Σ 3 val² v.
    let ctx = weights().await;
    let got = sql_rows(
        &ctx,
        "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), \
              v AS (SELECT i, CAST(i + 1 AS DOUBLE) AS val FROM w) \
         SELECT l, l_tangent FROM jvp(loss, w.val, v)",
    )
    .await;
    assert_eq!(got.len(), 1);
    let want = 3.0 * (0.25 * 1.0 + 1.0 * 2.0 + 4.0 * 3.0);
    assert!(close(got[0]["l"], 7.125), "{got:?}");
    assert!(close(got[0]["l_tangent"], want), "{got:?}");
}

#[tokio::test]
async fn a_relation_s_rows_each_get_their_tangent_through_two_tables() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE x (n BIGINT, i BIGINT, v DOUBLE) AS VALUES \
         (0, 0, 1.5), (0, 1, -0.5), (1, 0, 0.25), (1, 1, 1.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT, o BIGINT, val DOUBLE) AS VALUES \
         (0, 0, 0.3), (0, 1, -0.2), (1, 0, 0.7), (1, 1, 0.1)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE b (o BIGINT, val DOUBLE) AS VALUES (0, 0.25), (1, -0.6)",
    )
    .await;
    let h = "SELECT x.n, w.o, tanh(SUM(x.v * w.val) + MAX(b.val)) AS y \
             FROM x JOIN w ON x.i = w.i JOIN b ON w.o = b.o GROUP BY x.n, w.o";
    let got = sql_rows(
        &ctx,
        &format!(
            "WITH h AS ({h}), \
                  dw AS (SELECT i, o, 0.1 * (i + 2 * o + 1) AS val FROM w), \
                  db AS (SELECT o, 1.0 - o AS val FROM b) \
             SELECT n, o, y, y_tangent FROM jvp(h, w.val, dw, b.val, db) ORDER BY n, o"
        ),
    )
    .await;

    // The oracle: the same jvp as a program, with the same tangents.
    let wrt = [ColumnRef::new("w", "val"), ColumnRef::new("b", "val")];
    let program = ad::jvp(&ctx, h, &wrt).await.unwrap();
    for input in &program.inputs {
        let sql = match input.of.table().unwrap().last().unwrap().as_str() {
            "w" => "SELECT i, o, 0.1 * (i + 2 * o + 1) AS val FROM w",
            _ => "SELECT o, 1.0 - o AS val FROM b",
        };
        exec(&ctx, &format!("CREATE TABLE {} AS {sql}", input.name)).await;
    }
    ad::run(&ctx, &program).await.unwrap();
    let tangent = &program.value.tangents[0];
    assert_eq!(tangent.column, "y");
    let want = rows(
        &ctx.sql(&format!(
            "SELECT n, o, y, {} AS t FROM {} ORDER BY n, o",
            tangent.tangent, program.value.step
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap(),
    );
    assert_eq!(got.len(), 4);
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert_eq!((g["n"], g["o"]), (w["n"], w["o"]));
        assert!(close(g["y"], w["y"]), "{g:?} vs {w:?}");
        assert!(close(g["y_tangent"], w["t"]), "{g:?} vs {w:?}");
        assert!(g["y_tangent"] != 0.0);
    }
}

#[tokio::test]
async fn a_tangent_from_a_table_and_a_missing_row_s_tangent_is_zero() {
    let ctx = weights().await;
    // No row for i = 2: its tangent is 0.
    exec(
        &ctx,
        "CREATE TABLE v (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 1.0)",
    )
    .await;
    let got = sql_rows(
        &ctx,
        "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w) \
         SELECT l_tangent FROM jvp(loss, w.val, v)",
    )
    .await;
    assert!(close(got[0]["l_tangent"], 3.0 * (0.25 + 1.0)), "{got:?}");
}

#[tokio::test]
async fn a_column_already_named_like_a_tangent_keeps_its_name() {
    let ctx = weights().await;
    let got = sql_rows(
        &ctx,
        "WITH f AS (SELECT SUM(val) AS l, SUM(val * val) AS l_tangent FROM w), \
              v AS (SELECT i, 1.0 AS val FROM w) \
         SELECT * FROM jvp(f, w.val, v)",
    )
    .await;
    let row = &got[0];
    assert_eq!(
        row.keys().cloned().collect::<Vec<_>>(),
        vec!["l", "l_tangent", "l_tangent_2", "l_tangent_tangent"]
    );
    assert!(close(row["l_tangent"], 0.25 + 1.0 + 4.0));
    assert!(close(row["l_tangent_2"], 3.0), "{row:?}");
    assert!(close(row["l_tangent_tangent"], 2.0 * 1.5), "{row:?}");
}

#[tokio::test]
async fn grad_and_jvp_in_one_statement_and_nothing_left_behind() {
    let ctx = weights().await;
    let got = sql_rows(
        &ctx,
        "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), \
              v AS (SELECT i, 1.0 AS val FROM w) \
         SELECT g.i, g.val AS g, j.l_tangent AS d \
         FROM grad(loss, w.val) g CROSS JOIN jvp(loss, w.val, v) j ORDER BY g.i",
    )
    .await;
    // ∇l = 3 val²; ∇l·1 = Σ 3 val².
    let d = 3.0 * (0.25 + 1.0 + 4.0);
    for (row, val) in got.iter().zip([0.5, -1.0, 2.0]) {
        assert!(close(row["g"], 3.0 * val * val), "{row:?}");
        assert!(close(row["d"], d), "{row:?}");
    }
    let left: Vec<String> = ctx
        .catalog("datafusion")
        .unwrap()
        .schema("public")
        .unwrap()
        .table_names()
        .into_iter()
        .filter(|t| t.starts_with("__ddx_"))
        .collect();
    assert!(left.is_empty(), "left on the context: {left:?}");
}

#[tokio::test]
async fn a_tangent_missing_a_column_is_an_error_that_names_it() {
    let ctx = weights().await;
    let e = ad::sql(
        &ctx,
        "WITH loss AS (SELECT SUM(val * val) AS l FROM w), \
              v AS (SELECT i, 1.0 AS dval FROM w) \
         SELECT * FROM jvp(loss, w.val, v)",
    )
    .await
    .unwrap_err();
    assert!(e.to_string().contains("val"), "{e}");
}

#[tokio::test]
async fn statements_planned_together_share_one_jvp() {
    let ctx = weights().await;
    let with = "WITH loss AS (SELECT SUM(val * val) AS l FROM w), \
                v AS (SELECT i, 1.0 AS val FROM w)";
    let a = format!("{with} SELECT l FROM jvp(loss, w.val, v)");
    let b = format!("{with} SELECT l_tangent FROM jvp(loss, w.val, v)");
    let frames = ad::sql_all(&ctx, &[&a, &b]).await.unwrap();
    let mut out = Vec::new();
    for f in frames {
        out.push(rows(&f.collect().await.unwrap()));
    }
    assert!(close(out[0][0]["l"], 5.25));
    assert!(close(out[1][0]["l_tangent"], 2.0 * 1.5));
}
