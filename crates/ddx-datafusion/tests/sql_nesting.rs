// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Calls in SQL nest, as JAX's transformations do: a call's `f`, or its
//! cotangent or tangent, may read another call. Each is checked against the
//! closed form, and `H·v` against the same product built from programs.

use datafusion::arrow::array::{Array, AsArray, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Float64Type, Int64Type};
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, ColumnRef};

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// The rows of `batches`, each column as numbers.
fn rows(batches: &[RecordBatch]) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|a| match a.data_type() {
                        DataType::Float64 => a.as_primitive::<Float64Type>().value(r),
                        DataType::Int64 => a.as_primitive::<Int64Type>().value(r) as f64,
                        other => panic!("a column of type {other}"),
                    })
                    .collect(),
            );
        }
    }
    out
}

async fn sql_rows(ctx: &SessionContext, sql: &str) -> Vec<Vec<f64>> {
    let df = ad::sql(ctx, sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    rows(&df.collect().await.unwrap())
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * b.abs().max(1.0)
}

const VALS: [f64; 3] = [0.5, -1.0, 2.0];

async fn weights() -> SessionContext {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 0.5), (1, -1.0), (2, 2.0)",
    )
    .await;
    ctx
}

/// l = Σ val³, so ∇l = 3 val² and H = diag(6 val); v = i + 1.
const LOSS: &str = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), \
                         v AS (SELECT i, CAST(i + 1 AS DOUBLE) AS val FROM w)";

fn v(i: usize) -> f64 {
    i as f64 + 1.0
}

#[tokio::test]
async fn jvp_of_a_query_over_grad_is_a_hessian_vector_product() {
    let ctx = weights().await;
    let got = sql_rows(
        &ctx,
        &format!(
            "{LOSS}, g AS (SELECT * FROM grad(loss, w.val)) \
             SELECT i, val, val_tangent FROM jvp(g, w.val, v) ORDER BY i"
        ),
    )
    .await;
    assert_eq!(got.len(), 3);
    for (i, row) in got.iter().enumerate() {
        assert!(close(row[1], 3.0 * VALS[i] * VALS[i]), "{got:?}");
        assert!(close(row[2], 6.0 * VALS[i] * v(i)), "{got:?}");
    }

    // The same product from programs: jvp of the grad program.
    let wrt = [ColumnRef::new("w", "val")];
    let g = ad::grad(&ctx, "SELECT SUM(val * val * val) AS l FROM w", &wrt)
        .await
        .unwrap();
    let hvp = ad::jvp(&ctx, &g, &wrt).await.unwrap();
    for input in &hvp.inputs {
        exec(
            &ctx,
            &format!(
                "CREATE TABLE {} AS SELECT i, CAST(i + 1 AS DOUBLE) AS val FROM w",
                input.name
            ),
        )
        .await;
    }
    ad::run(&ctx, &hvp).await.unwrap();
    let t = &hvp.gradients[0].tangents[0];
    let want = rows(
        &ctx.sql(&format!(
            "SELECT {} FROM {} ORDER BY i",
            t.tangent, hvp.gradients[0].step
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap(),
    );
    for (g, w) in got.iter().zip(&want) {
        assert!(close(g[2], w[0]), "{g:?} vs {w:?}");
    }
}

#[tokio::test]
async fn grad_of_a_query_over_jvp_is_reverse_over_forward() {
    // ∇(∇l·v) = H·v.
    let ctx = weights().await;
    let got = sql_rows(
        &ctx,
        &format!(
            "{LOSS}, d AS (SELECT l_tangent AS d FROM jvp(loss, w.val, v)) \
             SELECT i, val FROM grad(d, w.val) ORDER BY i"
        ),
    )
    .await;
    assert_eq!(got.len(), 3);
    for (i, row) in got.iter().enumerate() {
        assert!(close(row[1], 6.0 * VALS[i] * v(i)), "{got:?}");
    }
}

#[tokio::test]
async fn a_tangent_can_be_another_call() {
    // ∇l·∇l = Σ (3 val²)², and nothing is differentiated through the tangent.
    let ctx = weights().await;
    let got = sql_rows(
        &ctx,
        "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), \
              g AS (SELECT * FROM grad(loss, w.val)) \
         SELECT l_tangent FROM jvp(loss, w.val, g)",
    )
    .await;
    let want: f64 = VALS.iter().map(|x| (3.0 * x * x).powi(2)).sum();
    assert!(close(got[0][0], want), "{got:?}");
}

#[tokio::test]
async fn a_query_over_a_call_can_compute_before_it_is_differentiated() {
    // m = Σ (∇l)² / 2 = Σ 4.5 val⁴, so its jvp along v is Σ 18 val³ v.
    let ctx = weights().await;
    let got = sql_rows(
        &ctx,
        &format!(
            "{LOSS}, m AS (SELECT SUM(g.val * g.val) / 2 AS m FROM grad(loss, w.val) g) \
             SELECT m, m_tangent FROM jvp(m, w.val, v)"
        ),
    )
    .await;
    let m: f64 = VALS.iter().map(|x| 4.5 * x.powi(4)).sum();
    let dm: f64 = VALS
        .iter()
        .enumerate()
        .map(|(i, x)| 18.0 * x.powi(3) * v(i))
        .sum();
    assert!(close(got[0][0], m), "{got:?}");
    assert!(close(got[0][1], dm), "{got:?}");
}

#[tokio::test]
async fn reverse_over_reverse_is_refused_by_ddx_and_leaves_nothing_behind() {
    let ctx = weights().await;
    let e = ad::sql(
        &ctx,
        "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), \
              m AS (SELECT SUM(g.val) AS m FROM grad(loss, w.val) g) \
         SELECT * FROM grad(m, w.val)",
    )
    .await
    .unwrap_err();
    // Reverse mode's own refusal (it cannot yet differentiate through the
    // outer join a gradient table is built with), not the engine's unknown
    // table function.
    assert!(e.to_string().contains("not supported by ddx-ad yet"), "{e}");
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
async fn a_subquery_over_a_call_is_refused_and_a_join_over_it_differentiates() {
    // q = Σ val · Σ ∇l, with ∇l = 3 val². Along 1: 3 Σ ∇l + Σ val · Σ 6 val.
    // A subquery reading g left reading the table g computed would be data,
    // and the tangent through ∇l lost.
    let ctx = weights().await;
    let with = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), \
                g AS (SELECT * FROM grad(loss, w.val)), \
                v AS (SELECT i, 1.0 AS val FROM w)";
    let e = ad::sql(
        &ctx,
        &format!(
            "{with}, q AS (SELECT SUM(val) * (SELECT SUM(val) FROM g) AS s FROM w) \
             SELECT * FROM jvp(q, w.val, v)"
        ),
    )
    .await
    .unwrap_err();
    assert!(e.to_string().contains("write it as a join"), "{e}");

    let got = sql_rows(
        &ctx,
        &format!(
            "{with}, t AS (SELECT SUM(val) AS t FROM g), \
             q AS (SELECT SUM(w.val) * MAX(t.t) AS s FROM w CROSS JOIN t) \
             SELECT s, s_tangent FROM jvp(q, w.val, v)"
        ),
    )
    .await;
    let sum: f64 = VALS.iter().sum();
    let g: f64 = VALS.iter().map(|v| 3.0 * v * v).sum();
    let h: f64 = VALS.iter().map(|v| 6.0 * v).sum();
    assert!(close(got[0][0], sum * g), "{got:?}");
    assert!(close(got[0][1], 3.0 * g + sum * h), "{got:?}");
}
