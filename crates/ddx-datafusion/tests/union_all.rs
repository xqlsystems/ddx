// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad` through `UNION ALL`, checked against finite differences of the
//! same query. `UNION ALL` concatenates rows, so its transpose sends each
//! row's cotangent back to the input that gave the row, as the transpose of
//! `jnp.concatenate` is a slice.

mod common;

use common::ad::{check_gradients, rows, run, Table};
use common::substrait_of;
use datafusion::prelude::SessionContext;
use ddx_ad::{grad, AdError, ColumnRef};

fn a() -> Table {
    Table {
        name: "a",
        columns: vec![("i", "BIGINT"), ("x", "DOUBLE")],
        rows: vec![vec![0.0, 1.0], vec![1.0, 2.0]],
    }
}

fn b() -> Table {
    Table {
        name: "b",
        columns: vec![("i", "BIGINT"), ("x", "DOUBLE")],
        rows: vec![vec![0.0, 3.0], vec![1.0, -0.5], vec![2.0, 0.75]],
    }
}

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
            vec![3.0, 0.0, 0.9],
            vec![3.0, 1.0, -0.6],
            vec![4.0, 0.0, 0.2],
            vec![4.0, 1.0, 0.4],
        ],
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

/// `rows` in order of their keys: a step's rows come in no order.
fn sorted(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut rows = rows.to_vec();
    rows.sort_by(|a, b| a.partial_cmp(b).expect("no NaN keys"));
    rows
}

#[tokio::test]
async fn a_sum_over_two_tables_concatenated() {
    let grads = check_gradients(
        &ctx(),
        "WITH u AS (SELECT i, x FROM a UNION ALL SELECT i + 2 AS i, x FROM b) \
         SELECT SUM(x * x) AS l FROM u",
        &[a(), b()],
        &[wrt("a", "x"), wrt("b", "x")],
    )
    .await;
    // d/dx Σ x² = 2x, each row's from its own table.
    assert_eq!(sorted(&grads["a"]), vec![vec![0.0, 2.0], vec![1.0, 4.0]]);
}

#[tokio::test]
async fn one_table_read_by_both_inputs_gets_both_contributions() {
    check_gradients(
        &ctx(),
        "WITH u AS (SELECT i, sin(x) AS v FROM a UNION ALL SELECT i, x * x AS v FROM a) \
         SELECT SUM(v * v) AS l FROM u",
        &[a()],
        &[wrt("a", "x")],
    )
    .await;
}

#[tokio::test]
async fn an_input_that_reads_no_wrt_table_is_concatenated_as_data() {
    check_gradients(
        &ctx(),
        "WITH u AS (SELECT i, x FROM a UNION ALL SELECT i, x FROM b) \
         SELECT SUM(x * x * x) AS l FROM u",
        &[a(), b()],
        &[wrt("a", "x")],
    )
    .await;
}

#[tokio::test]
async fn groups_that_take_rows_from_both_inputs() {
    // Each group of i sums one row of a and one of b, where both have it.
    check_gradients(
        &ctx(),
        "WITH u AS (SELECT i, x FROM a UNION ALL SELECT i, 2.0 * x AS x FROM b), \
         s AS (SELECT i, SUM(x) AS t FROM u GROUP BY i) \
         SELECT SUM(t * t) AS l FROM s",
        &[a(), b()],
        &[wrt("a", "x"), wrt("b", "x")],
    )
    .await;
}

#[tokio::test]
async fn a_union_of_aggregates() {
    check_gradients(
        &ctx(),
        "WITH u AS (SELECT i AS k, SUM(val) AS t FROM w GROUP BY i \
                    UNION ALL SELECT o + 10 AS k, SUM(val * val) AS t FROM w GROUP BY o) \
         SELECT SUM(t * t) AS l FROM u",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn features_concatenated_then_multiplied_by_a_weight() {
    // GraphCast builds an MLP's input as the UNION ALL of an edge's, its
    // sender's and its receiver's features, at offset feature indices,
    // and multiplies it by a weight: a contraction over the union.
    let feats = Table {
        name: "f",
        columns: vec![("n", "BIGINT"), ("k", "BIGINT"), ("v", "DOUBLE")],
        rows: vec![
            vec![0.0, 0.0, 0.5],
            vec![0.0, 1.0, -1.0],
            vec![1.0, 0.0, 1.5],
            vec![1.0, 1.0, 0.25],
        ],
    };
    let more = Table {
        name: "g",
        columns: vec![("n", "BIGINT"), ("k", "BIGINT"), ("v", "DOUBLE")],
        rows: vec![
            vec![0.0, 0.0, -0.75],
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 0.1],
            vec![1.0, 1.0, -0.3],
            vec![1.0, 2.0, 0.6],
        ],
    };
    check_gradients(
        &ctx(),
        "WITH x AS (SELECT n, k, v FROM f UNION ALL SELECT n, k + 2 AS k, v FROM g), \
         h AS (SELECT x.n, w.o, SUM(x.v * w.val) AS h FROM x JOIN w ON x.k = w.i \
               GROUP BY x.n, w.o) \
         SELECT SUM(tanh(h) * tanh(h)) AS l FROM h",
        &[feats, more, w()],
        &[wrt("f", "v"), wrt("g", "v"), wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_maximum_tied_across_the_inputs_is_shared() {
    // Both copies of the largest value attain the maximum, and share its
    // gradient: one half each, so the value gets 1 in all.
    let grads = check_gradients(
        &ctx(),
        "WITH u AS (SELECT x FROM a UNION ALL SELECT x FROM a) SELECT MAX(x) AS l FROM u",
        &[a()],
        &[wrt("a", "x")],
    )
    .await;
    assert_eq!(sorted(&grads["a"]), vec![vec![0.0, 0.0], vec![1.0, 1.0]]);
}

#[tokio::test]
async fn a_window_over_both_inputs_rows() {
    // Rows of a and b with the same i share a partition; ddx tells them
    // apart by which input gave them.
    check_gradients(
        &ctx(),
        "WITH u AS (SELECT i, x FROM a UNION ALL SELECT i, x FROM b), \
         s AS (SELECT x, SUM(x) OVER (PARTITION BY i) AS t FROM u) \
         SELECT SUM(x * t) AS l FROM s",
        &[a(), b()],
        &[wrt("a", "x"), wrt("b", "x")],
    )
    .await;
}

#[tokio::test]
async fn a_union_nested_in_a_union() {
    check_gradients(
        &ctx(),
        "WITH u AS (SELECT i, x FROM a \
                    UNION ALL SELECT i, x FROM (SELECT i, exp(x) AS x FROM b \
                                                UNION ALL SELECT i, x * x AS x FROM a) v) \
         SELECT SUM(x * x) AS l FROM u",
        &[a(), b()],
        &[wrt("a", "x"), wrt("b", "x")],
    )
    .await;
}

#[tokio::test]
async fn unions_of_unions_joined_over_columns_named_as_engines_rename() {
    // DataFusion names a union's second `i` `i_1`; here a table already has
    // one, and a union reads a union.
    let c = Table {
        name: "c",
        columns: vec![
            ("i", "BIGINT"),
            ("i_1", "BIGINT"),
            ("x", "DOUBLE"),
            ("x_1", "DOUBLE"),
        ],
        rows: vec![vec![0.0, 2.0, 3.0, 1.0], vec![1.0, 0.0, -0.5, 2.0]],
    };
    check_gradients(
        &ctx(),
        "WITH u AS (SELECT i, x FROM c UNION ALL SELECT i, x FROM b \
                    UNION ALL SELECT i_1 AS i, x_1 AS x FROM c), \
         v AS (SELECT i, x FROM (SELECT i, x FROM a UNION ALL SELECT i, x FROM b) p \
               UNION ALL SELECT i, x FROM (SELECT i, sin(x_1) AS x FROM c \
                                           UNION ALL SELECT i, 2.0 * x AS x FROM a) q) \
         SELECT SUM(u.x * v.x * v.x) AS l FROM u JOIN v ON u.i = v.i",
        &[a(), b(), c],
        &[wrt("a", "x"), wrt("b", "x"), wrt("c", "x"), wrt("c", "x_1")],
    )
    .await;
}

#[tokio::test]
async fn a_null_in_one_input_sends_its_row_no_gradient() {
    let ctx = ctx();
    b().create(&ctx).await;
    ctx.sql("CREATE TABLE an (i BIGINT, x DOUBLE) AS VALUES (0, 1.0), (1, NULL)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let plan = substrait_of(
        &ctx,
        "WITH u AS (SELECT i, x FROM an UNION ALL SELECT i, x FROM b) \
         SELECT SUM(x * x) AS l FROM u",
        true,
    )
    .await;
    let program = grad(&plan, &[wrt("an", "x"), wrt("b", "x")]).unwrap();
    run(&ctx, &program).await;
    let of = |n: usize| format!("SELECT i, x FROM {} ORDER BY i", program.gradients[n].step);
    let got = rows(&ctx, &of(0)).await;
    assert_eq!(got[0], vec![0.0, 2.0]);
    assert!(
        got[1][1].is_nan(),
        "the NULL row's gradient is NULL: {got:?}"
    );
    assert_eq!(
        rows(&ctx, &of(1)).await,
        vec![vec![0.0, 6.0], vec![1.0, -1.0], vec![2.0, 1.5]]
    );
}

#[tokio::test]
async fn a_group_whose_key_is_null_keeps_its_gradient() {
    // The rows b gives the union hold NULL in the columns of s, s's key
    // among them, as does s's own group whose key is NULL: those rows send
    // nothing, and the group gets its own gradient.
    let ctx = ctx();
    b().create(&ctx).await;
    ctx.sql("CREATE TABLE n (i BIGINT, o BIGINT, x DOUBLE) AS VALUES (0, NULL, 1.0), (1, NULL, 2.0), (2, 1, 3.0)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let plan = substrait_of(
        &ctx,
        "WITH s AS (SELECT o, SUM(x) AS t FROM n GROUP BY o), \
         u AS (SELECT o, t FROM s UNION ALL SELECT i AS o, x AS t FROM b) \
         SELECT SUM(t * t) AS l FROM u",
        true,
    )
    .await;
    let program = grad(&plan, &[wrt("n", "x"), wrt("b", "x")]).unwrap();
    run(&ctx, &program).await;
    let of = |n: usize| format!("SELECT i, x FROM {} ORDER BY i", program.gradients[n].step);
    // The NULL group's t is 3 and the other's 3: d(t²)/dx = 6 for each row.
    assert_eq!(
        rows(&ctx, &of(0)).await,
        vec![vec![0.0, 6.0], vec![1.0, 6.0], vec![2.0, 6.0]]
    );
    assert_eq!(
        rows(&ctx, &of(1)).await,
        vec![vec![0.0, 6.0], vec![1.0, -1.0], vec![2.0, 1.5]]
    );
}

#[tokio::test]
async fn a_union_of_many_inputs_reads_each_one_once() {
    // Each input's rows carry every other input's columns as NULLs of no
    // type, which the engine types as it types the union. Joined to empty
    // copies of every other input, a union of n inputs was n² relations:
    // DuckDB refused its SQL from 16 inputs on, for its 1,000 CTEs (#238).
    let n = 32;
    let union = (1..=n)
        .map(|c| format!("SELECT i, {c}.0 * x AS x FROM a"))
        .collect::<Vec<_>>()
        .join(" UNION ALL ");
    let sql = format!("WITH u AS ({union}) SELECT SUM(x * x) AS l FROM u");
    let ctx = ctx();
    let grads = check_gradients(&ctx, &sql, &[a()], &[wrt("a", "x")]).await;
    // d/dx Σ_c (c x)² = 2 x Σ_c c².
    let k: f64 = (1..=n).map(|c| f64::from(c * c)).sum();
    assert_eq!(
        sorted(&grads["a"]),
        vec![vec![0.0, 2.0 * k], vec![1.0, 4.0 * k]]
    );
    let program = grad(&substrait_of(&ctx, &sql, true).await, &[wrt("a", "x")]).unwrap();
    let reads: usize = program
        .steps()
        .map(|s| format!("{:?}", *s.plan).matches(r#"names: ["a"]"#).count())
        .sum();
    assert!(reads <= 3 * n as usize, "{reads} reads of a for {n} inputs");
}

async fn refusal(sql: &str) -> AdError {
    let ctx = ctx();
    a().create(&ctx).await;
    b().create(&ctx).await;
    let plan = substrait_of(&ctx, sql, true).await;
    grad(&plan, &[wrt("a", "x")]).expect_err("refused")
}

#[tokio::test]
async fn a_loss_of_one_row_from_each_input_is_not_one_row() {
    let err =
        refusal("SELECT t FROM (SELECT SUM(x) AS t FROM a UNION ALL SELECT SUM(x) AS t FROM b) u")
            .await;
    assert!(matches!(err, AdError::NotScalar(_)), "{err}");
}

#[tokio::test]
async fn a_distinct_union_is_refused() {
    // DataFusion plans UNION as a GROUP BY of every column over UNION ALL,
    // so the key that depends on x is what is refused.
    let err = refusal(
        "WITH u AS (SELECT i, x FROM a UNION SELECT i, x FROM b) SELECT SUM(x) AS l FROM u",
    )
    .await;
    assert!(matches!(err, AdError::NotImplemented(_)), "{err}");
}
