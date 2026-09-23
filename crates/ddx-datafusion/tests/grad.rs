// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad` and `vjp` end to end on DataFusion: the gradient of a loss query,
//! checked entry by entry against finite differences of the same query.

mod common;

use common::ad::{check_gradients, rows, run, Table};
use common::substrait_of;
use datafusion::prelude::SessionContext;
use ddx_ad::{grad, vjp, AdError, ColumnRef, COTANGENT};

fn w() -> Table {
    Table {
        name: "w",
        columns: vec![("i", "BIGINT"), ("o", "BIGINT"), ("val", "DOUBLE")],
        rows: vec![
            vec![0.0, 0.0, 0.3],
            vec![0.0, 1.0, -0.2],
            vec![1.0, 0.0, 0.7],
            vec![1.0, 1.0, 0.1],
            vec![2.0, 0.0, -0.4],
            vec![2.0, 1.0, 0.5],
        ],
    }
}

fn b() -> Table {
    Table {
        name: "b",
        columns: vec![("o", "BIGINT"), ("val", "DOUBLE")],
        rows: vec![vec![0.0, 0.25], vec![1.0, -0.6]],
    }
}

fn x() -> Table {
    Table {
        name: "x",
        columns: vec![("i", "BIGINT"), ("v", "DOUBLE")],
        rows: vec![vec![0.0, 1.5], vec![1.0, -0.5], vec![2.0, 2.0]],
    }
}

fn ctx() -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    ctx
}

fn wrt(t: &str, c: &str) -> ColumnRef {
    ColumnRef::new(t, c)
}

#[tokio::test]
async fn an_elementwise_loss_over_one_table() {
    check_gradients(
        &ctx(),
        "SELECT SUM(tanh(val) * exp(val / 2.0)) AS loss FROM w",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_reduction_joined_to_data() {
    check_gradients(
        &ctx(),
        "SELECT SUM(w.val * w.val * x.v) AS loss \
         FROM w JOIN x ON w.i = x.i",
        &[w(), x()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_grouped_sum_feeds_a_second_aggregate() {
    // Two nodes: the per-i sums, then the loss over them.
    check_gradients(
        &ctx(),
        "WITH s AS (SELECT i, SUM(val) AS t FROM w GROUP BY i) \
         SELECT SUM(t * t * t) AS loss FROM s",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_bias_is_broadcast_by_a_join_and_summed_back() {
    // b.val is added to every w row with the same o, so its gradient sums
    // over i.
    check_gradients(
        &ctx(),
        "SELECT SUM(power(w.val + b.val, 2)) AS loss \
         FROM w JOIN b ON w.o = b.o",
        &[w(), b()],
        &[wrt("w", "val"), wrt("b", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_table_read_twice_gets_both_contributions() {
    // w joined to itself: each value is read once as `p` and once as `q`.
    check_gradients(
        &ctx(),
        "SELECT SUM(p.val * sin(q.val)) AS loss \
         FROM w p JOIN w q ON p.i = q.i AND p.o <> q.o",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_loss_computed_above_its_aggregates() {
    // SUM / COUNT: the division is elementwise in the root segment, and
    // COUNT has no gradient.
    check_gradients(
        &ctx(),
        "SELECT SUM(val * val) / COUNT(val) - 1.0 AS loss FROM w",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_loss_may_divide_by_a_one_row_constant() {
    // An ungrouped aggregate over data is one row, so the loss still is.
    check_gradients(
        &ctx(),
        "SELECT s.t / c.n AS loss \
         FROM (SELECT SUM(val * val) AS t FROM w) s CROSS JOIN (SELECT COUNT(*) AS n FROM x) c",
        &[w(), x()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_wrt_table_the_loss_does_not_depend_on_gets_zeros() {
    let grads = check_gradients(
        &ctx(),
        "SELECT SUM(w.val * w.val) AS loss FROM w JOIN b ON w.o = b.o",
        &[w(), b()],
        &[wrt("w", "val"), wrt("b", "val")],
    )
    .await;
    assert!(grads["b"].iter().all(|r| r[1] == 0.0), "{:?}", grads["b"]);
}

/// The error `grad` returns for `sql` with respect to `w.val`.
async fn refusal(sql: &str) -> AdError {
    let ctx = ctx();
    w().create(&ctx).await;
    x().create(&ctx).await;
    let plan = substrait_of(&ctx, sql, true).await;
    grad(&plan, &[wrt("w", "val")]).unwrap_err()
}

#[tokio::test]
async fn an_aggregate_with_no_rule_is_refused() {
    let err = refusal("SELECT STDDEV(val) AS loss FROM w").await;
    assert!(matches!(err, AdError::NotImplemented(_)), "{err}");
    assert!(
        err.to_string().contains("SUM, AVG, MAX, MIN and COUNT"),
        "{err}"
    );
}

#[tokio::test]
async fn grad_needs_a_loss() {
    // Two columns.
    let err = refusal("SELECT SUM(val) AS a, SUM(val * val) AS b FROM w").await;
    assert!(matches!(err, AdError::NotScalar(_)), "{err}");
    assert!(err.to_string().contains("[\"a\", \"b\"]"), "{err}");
    // A row per dim.
    let err = refusal("SELECT i, SUM(val * val) AS s FROM w GROUP BY i").await;
    assert!(matches!(err, AdError::NotScalar(_)), "{err}");
    let err = refusal("SELECT SUM(val * val) AS s FROM w GROUP BY i").await;
    assert!(matches!(err, AdError::NotScalar(_)), "{err}");
    // One row times a many-row constant table is many rows.
    let err = refusal(
        "WITH s AS (SELECT SUM(val * val) AS t FROM w) SELECT s.t * x.v AS l FROM s CROSS JOIN x",
    )
    .await;
    assert!(matches!(err, AdError::NotScalar(_)), "{err}");
}

#[tokio::test]
async fn vjp_pulls_a_cotangent_back() {
    // out(i) = Σ_o val(i, o)², and the cotangent c(i) on it: the pullback to
    // val(i, o) is c(i) · 2 · val(i, o).
    let ctx = ctx();
    w().create(&ctx).await;
    let sql = "SELECT i, SUM(val * val) AS s FROM w GROUP BY i";
    let plan = substrait_of(&ctx, sql, true).await;
    let program = vjp(&plan, &[wrt("w", "val")]).unwrap();
    assert_eq!(program.cotangent, vec!["i", "s"]);

    Table {
        name: COTANGENT,
        columns: vec![("i", "BIGINT"), ("s", "DOUBLE")],
        rows: vec![vec![0.0, 2.0], vec![1.0, -1.0], vec![2.0, 0.5]],
    }
    .create(&ctx)
    .await;
    run(&ctx, &program).await;
    let got = rows(
        &ctx,
        &format!(
            "SELECT i, o, val FROM {} ORDER BY i, o",
            program.gradients[0].step
        ),
    )
    .await;
    let cot = [2.0, -1.0, 0.5];
    for (g, row) in got.iter().zip(&w().rows) {
        assert_eq!(g[2], cot[row[0] as usize] * 2.0 * row[2], "{g:?}");
    }
}
