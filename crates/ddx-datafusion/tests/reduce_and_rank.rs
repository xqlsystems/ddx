// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The reduce rules beyond SUM (MAX, MIN, AVG), rank-select, and
//! `ddx_stop_gradient`: nn.py's softmax cross-entropy and max-pooling.

mod common;

use common::ad::{check_gradients, rows, run, Table};
use common::substrait_of;
use datafusion::prelude::SessionContext;
use ddx_ad::{grad, AdError, ColumnRef};

fn ctx() -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    ctx
}

/// Logits `z(sample, out)` and one-hot labels `y(sample, out)`.
fn logits() -> Vec<Table> {
    let z = [
        [0.2, -1.3, 0.8],
        [1.1, 0.4, -0.2],
        [-0.5, 0.9, 0.3],
        [0.0, 0.1, 2.0],
    ];
    let mut zr = Vec::new();
    let mut yr = Vec::new();
    for (s, row) in z.iter().enumerate() {
        for (o, v) in row.iter().enumerate() {
            zr.push(vec![s as f64, o as f64, *v]);
            yr.push(vec![
                s as f64,
                o as f64,
                if (s + 1) % 3 == o { 1.0 } else { 0.0 },
            ]);
        }
    }
    vec![
        Table {
            name: "z",
            columns: vec![("sample", "BIGINT"), ("out", "BIGINT"), ("val", "DOUBLE")],
            rows: zr,
        },
        Table {
            name: "y",
            columns: vec![("sample", "BIGINT"), ("out", "BIGINT"), ("val", "DOUBLE")],
            rows: yr,
        },
    ]
}

/// nn.py's loss, mean softmax cross-entropy, as nn.py writes it: the max is a
/// numerical shift, `shift` says how it is subtracted.
fn cross_entropy(shift: &str) -> String {
    format!(
        "WITH m AS (SELECT sample, MAX(val) AS m FROM z GROUP BY sample), \
              e AS (SELECT z.sample, z.out, exp(z.val - {shift}) AS e \
                    FROM z JOIN m ON z.sample = m.sample), \
              s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample) \
         SELECT -AVG(ln(e.e / s.s) * y.val) * 3.0 AS loss \
         FROM e JOIN s ON e.sample = s.sample \
                JOIN y ON y.sample = e.sample AND y.out = e.out"
    )
}

/// `(softmax(z) - onehot) / N`, the gradient nn.py derived by hand.
fn softmax_minus_onehot(z: &Table) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for s in 0..4 {
        let zs: Vec<f64> = (0..3)
            .map(|o| {
                z.rows
                    .iter()
                    .find(|r| r[0] == s as f64 && r[1] == o as f64)
                    .unwrap()[2]
            })
            .collect();
        let total: f64 = zs.iter().map(|v| v.exp()).sum();
        for (o, zo) in zs.iter().enumerate() {
            let onehot = if (s + 1) % 3 == o { 1.0 } else { 0.0 };
            out.push(vec![s as f64, o as f64, (zo.exp() / total - onehot) / 4.0]);
        }
    }
    out
}

fn assert_close(got: &[Vec<f64>], want: &[Vec<f64>]) {
    for w in want {
        let g = got.iter().find(|g| g[..2] == w[..2]).unwrap();
        assert!(
            (g[2] - w[2]).abs() < 1e-12,
            "{:?}: {} vs {}",
            &w[..2],
            g[2],
            w[2]
        );
    }
}

#[tokio::test]
async fn softmax_cross_entropy_as_nn_py_writes_it() {
    // A plain MAX: its rule routes gradient to the max, and the shift cancels,
    // so the result is the same as with the shift stopped.
    let tables = logits();
    let grads = check_gradients(
        &ctx(),
        &cross_entropy("m.m"),
        &tables,
        &[ColumnRef::new("z", "val")],
    )
    .await;
    assert_close(&grads["z"], &softmax_minus_onehot(&tables[0]));
}

#[tokio::test]
async fn softmax_cross_entropy_with_the_shift_stopped() {
    let tables = logits();
    let grads = check_gradients(
        &ctx(),
        &cross_entropy("ddx_stop_gradient(m.m)"),
        &tables,
        &[ColumnRef::new("z", "val")],
    )
    .await;
    assert_close(&grads["z"], &softmax_minus_onehot(&tables[0]));
}

#[tokio::test]
async fn a_stop_gradient_changes_the_gradient_it_stops() {
    // loss = Σ val · stop(val): the gradient is val, not the 2·val of the
    // unstopped function. Finite differences would say 2·val, so this checks
    // against the value directly.
    let ctx = ctx();
    let tables = logits();
    tables[0].create(&ctx).await;
    let sql = "SELECT SUM(val * ddx_stop_gradient(val)) AS loss FROM z";
    let plan = substrait_of(&ctx, sql, true).await;
    let program = grad(&plan, &[ColumnRef::new("z", "val")]).unwrap();
    run(&ctx, &program).await;
    let got = rows(
        &ctx,
        &format!("SELECT sample, out, val FROM {}", program.gradients[0].step),
    )
    .await;
    for r in &tables[0].rows {
        let g = got.iter().find(|x| x[0] == r[0] && x[1] == r[1]).unwrap()[2];
        assert_eq!(g, r[2]);
    }
}

#[tokio::test]
async fn avg_is_a_sum_over_a_count() {
    check_gradients(
        &ctx(),
        "WITH m AS (SELECT sample, AVG(val * val) AS a FROM z GROUP BY sample) \
         SELECT SUM(a * a) AS loss FROM m",
        &logits(),
        &[ColumnRef::new("z", "val")],
    )
    .await;
}

/// `x(g, item, val)`: four groups of five, and a weight per group.
fn pool_tables(tie: bool) -> Vec<Table> {
    let mut xr = Vec::new();
    for g in 0..4 {
        for item in 0..5 {
            let v = ((g * 5 + item) * 37 % 23) as f64 / 10.0 - 1.0;
            xr.push(vec![g as f64, item as f64, v]);
        }
    }
    if tie {
        // Items 1 and 3 of group 0 tie for its max.
        xr[1][2] = 5.0;
        xr[3][2] = 5.0;
    }
    vec![
        Table {
            name: "x",
            columns: vec![("g", "BIGINT"), ("item", "BIGINT"), ("val", "DOUBLE")],
            rows: xr,
        },
        Table {
            name: "wt",
            columns: vec![("g", "BIGINT"), ("w", "DOUBLE")],
            rows: vec![
                vec![0.0, 1.5],
                vec![1.0, -0.5],
                vec![2.0, 2.0],
                vec![3.0, 0.25],
            ],
        },
    ]
}

/// A max-pool as an aggregate: the largest `val` per group, weighted.
fn max_pool(f: &str) -> String {
    format!(
        "WITH p AS (SELECT g, {f}(val) AS m FROM x GROUP BY g) \
         SELECT SUM(p.m * wt.w) AS loss FROM p JOIN wt ON p.g = wt.g"
    )
}

/// The same max-pool as a ranking, the idiom nn.py uses for accuracy: rank
/// by value, keep the first, with the tie broken by item.
const RANK_POOL: &str = "\
WITH r AS ( \
  SELECT g, item, val, \
         ROW_NUMBER() OVER (PARTITION BY g ORDER BY val DESC, item) AS rk \
  FROM x) \
SELECT SUM(r.val * wt.w) AS loss \
FROM r JOIN wt ON r.g = wt.g WHERE r.rk = 1";

async fn group0(sql: &str, tables: &[Table]) -> Vec<f64> {
    let ctx = ctx();
    for t in tables {
        t.create(&ctx).await;
    }
    let plan = substrait_of(&ctx, sql, true).await;
    let program = grad(&plan, &[ColumnRef::new("x", "val")]).unwrap();
    run(&ctx, &program).await;
    let sql = format!(
        "SELECT g, item, val FROM {} WHERE g = 0 ORDER BY item",
        program.gradients[0].step
    );
    rows(&ctx, &sql).await.iter().map(|r| r[2]).collect()
}

#[tokio::test]
async fn max_and_min_send_the_cotangent_to_the_extreme() {
    for f in ["MAX", "MIN"] {
        let grads = check_gradients(
            &ctx(),
            &max_pool(f),
            &pool_tables(false),
            &[ColumnRef::new("x", "val")],
        )
        .await;
        assert_eq!(grads["x"].iter().filter(|r| r[2] != 0.0).count(), 4, "{f}");
    }
}

#[tokio::test]
async fn at_a_tie_max_shares_the_cotangent_like_jax() {
    // jax.grad(jnp.max) splits the cotangent evenly across tied maxima.
    assert_eq!(
        group0(&max_pool("MAX"), &pool_tables(true)).await,
        vec![0.0, 0.75, 0.0, 0.75, 0.0]
    );
}

#[tokio::test]
async fn a_rank_filter_sends_the_cotangent_to_the_kept_row() {
    let grads = check_gradients(
        &ctx(),
        RANK_POOL,
        &pool_tables(false),
        &[ColumnRef::new("x", "val")],
    )
    .await;
    assert_eq!(grads["x"].iter().filter(|r| r[2] != 0.0).count(), 4);
}

#[tokio::test]
async fn at_a_tie_a_rank_filter_sends_it_all_to_the_row_it_kept() {
    // The ORDER BY's tie-break keeps one row, and it gets everything
    // (route_ad_spike.py's convention). Unlike MAX, the query picked a
    // winner, so there is nothing to share.
    assert_eq!(
        group0(RANK_POOL, &pool_tables(true)).await,
        vec![0.0, 1.5, 0.0, 0.0, 0.0]
    );
}

#[tokio::test]
async fn a_rank_used_as_a_value_is_refused() {
    let ctx = ctx();
    for t in pool_tables(false) {
        t.create(&ctx).await;
    }
    let sql = "WITH r AS (SELECT g, val, RANK() OVER (PARTITION BY g ORDER BY val) AS rk FROM x) \
               SELECT SUM(val * rk) AS loss FROM r";
    let plan = substrait_of(&ctx, sql, true).await;
    let err = grad(&plan, &[ColumnRef::new("x", "val")]).unwrap_err();
    assert!(matches!(err, AdError::NotImplemented(_)), "{err}");
    assert!(err.to_string().contains("rank"), "{err}");
}
