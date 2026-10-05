// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `jvp` end to end on DataFusion: the tangent of a query's output, checked
//! row by row against a central finite difference of the same query along
//! the same direction, and against reverse mode by the dot-product test
//! ⟨J t, c⟩ = ⟨t, Jᵀ c⟩.

mod common;

use std::collections::BTreeMap;

use common::ad::{rows, run, scalar, Table};
use common::substrait_of;
use datafusion::prelude::SessionContext;
use ddx_ad::{AdError, ColumnRef, ForwardProgram};

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

fn x() -> Table {
    Table {
        name: "x",
        columns: vec![("n", "BIGINT"), ("i", "BIGINT"), ("v", "DOUBLE")],
        rows: vec![
            vec![0.0, 0.0, 1.5],
            vec![0.0, 1.0, -0.5],
            vec![0.0, 2.0, 2.0],
            vec![1.0, 0.0, 0.25],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 2.0, -1.25],
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

fn ctx() -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    ctx
}

fn wrt(t: &str, c: &str) -> ColumnRef {
    ColumnRef::new(t, c)
}

/// A direction for row `r`, value `j`: varied, nonzero, deterministic.
fn direction(r: usize, j: usize) -> f64 {
    0.3 + 0.7 * (1.3 * r as f64 + 0.4 * j as f64 + 0.2).sin()
}

/// The tangent of `table` along [`direction`], as a table named `name` with
/// the columns the program asks for (the dims, then the `wrt` values).
fn tangent_of(table: &Table, name: &str, columns: &[String]) -> Table {
    let cols: Vec<usize> = columns
        .iter()
        .map(|c| table.columns.iter().position(|(n, _)| n == c).unwrap())
        .collect();
    let values: Vec<usize> = cols
        .iter()
        .copied()
        .filter(|&c| table.columns[c].1 == "DOUBLE")
        .collect();
    Table {
        name: Box::leak(name.to_string().into_boxed_str()),
        columns: cols.iter().map(|&c| table.columns[c]).collect(),
        rows: table
            .rows
            .iter()
            .enumerate()
            .map(|(r, row)| {
                cols.iter()
                    .map(|&c| match values.iter().position(|&v| v == c) {
                        Some(j) => direction(r, j),
                        None => row[c],
                    })
                    .collect()
            })
            .collect(),
    }
}

/// `table` moved `h` along [`direction`] in its `wrt` columns.
fn moved(table: &Table, wrt: &[ColumnRef], h: f64) -> Table {
    let values: Vec<usize> = table
        .columns
        .iter()
        .enumerate()
        .filter(|(_, (n, _))| wrt.iter().any(|w| w.table == table.name && w.column == *n))
        .map(|(c, _)| c)
        .collect();
    let mut out = table.clone();
    for (r, row) in out.rows.iter_mut().enumerate() {
        for (j, &c) in values.iter().enumerate() {
            row[c] += h * direction(r, j);
        }
    }
    out
}

/// Every step of `program` writes an emit only on a projection (see
/// `common::emits_off_projections`).
fn assert_emits_on_projections(program: &ForwardProgram) {
    for step in &program.steps {
        let off = common::emits_off_projections(&step.plan);
        assert!(
            off.is_empty(),
            "step {} writes an emit on: {off:?}",
            step.name
        );
    }
}

/// Build and run `sql`'s jvp with every `wrt` table's tangent along
/// [`direction`]. Returns the program and its output's rows.
async fn jvp_of(
    ctx: &SessionContext,
    sql: &str,
    tables: &[Table],
    wrt: &[ColumnRef],
) -> (ForwardProgram, Vec<Vec<f64>>) {
    for t in tables {
        t.create(ctx).await;
    }
    let plan = substrait_of(ctx, sql, true).await;
    let program = ddx_ad::jvp(&plan, wrt).unwrap_or_else(|e| panic!("jvp: {e}"));
    assert_emits_on_projections(&program);
    for tt in &program.tangent_tables {
        let table = tables
            .iter()
            .find(|t| tt.table.last().map(String::as_str) == Some(t.name))
            .expect("every wrt table is given");
        tangent_of(table, &tt.name, &tt.columns).create(ctx).await;
    }
    ddx_datafusion::ad::run(ctx, &program)
        .await
        .unwrap_or_else(|e| panic!("run: {e}"));
    let out = rows(ctx, &format!("SELECT * FROM {}", program.output.step)).await;
    (program, out)
}

/// The rows of `sql`, keyed by their first `keys` columns.
async fn keyed(ctx: &SessionContext, sql: &str, keys: usize) -> BTreeMap<Vec<i64>, Vec<f64>> {
    rows(ctx, sql)
        .await
        .into_iter()
        .map(|r| (r[..keys].iter().map(|&k| k as i64).collect(), r))
        .collect()
}

/// Check every tangent `sql`'s jvp gives against a central finite difference
/// of `sql` along the same direction, row by row (rows keyed by their first
/// `keys` columns). Returns the jvp's output.
async fn check_jvp(
    ctx: &SessionContext,
    sql: &str,
    tables: &[Table],
    wrt: &[ColumnRef],
    keys: usize,
) -> Vec<Vec<f64>> {
    let (program, out) = jvp_of(ctx, sql, tables, wrt).await;
    let h = 1e-5;
    for t in tables {
        moved(t, wrt, h).create(ctx).await;
    }
    let up = keyed(ctx, sql, keys).await;
    for t in tables {
        moved(t, wrt, -h).create(ctx).await;
    }
    let down = keyed(ctx, sql, keys).await;
    for t in tables {
        t.create(ctx).await;
    }
    assert_eq!(out.len(), up.len(), "the jvp has the query's rows");
    for row in &out {
        let key: Vec<i64> = row[..keys].iter().map(|&k| k as i64).collect();
        for tan in &program.output.tangents {
            let c = program
                .output
                .columns
                .iter()
                .position(|n| *n == tan.column)
                .unwrap();
            let t = program
                .output
                .columns
                .iter()
                .position(|n| *n == tan.tangent)
                .unwrap();
            let fd = (up[&key][c] - down[&key][c]) / (2.0 * h);
            assert!(
                (row[t] - fd).abs() <= 1e-6 * fd.abs().max(1.0),
                "tangent of {} at {key:?}: ddx {} vs finite difference {fd}",
                tan.column,
                row[t]
            );
        }
    }
    out
}

#[tokio::test]
async fn an_elementwise_loss() {
    check_jvp(
        &ctx(),
        "SELECT SUM(tanh(val) * exp(val / 2.0)) AS loss FROM w",
        &[w()],
        &[wrt("w", "val")],
        0,
    )
    .await;
}

#[tokio::test]
async fn a_row_per_input_row() {
    check_jvp(
        &ctx(),
        "SELECT i, o, val * val + sin(val) AS y, 2.0 AS k FROM w",
        &[w()],
        &[wrt("w", "val")],
        2,
    )
    .await;
}

#[tokio::test]
async fn a_layer_with_two_wrt_tables() {
    check_jvp(
        &ctx(),
        "SELECT x.n, w.o, tanh(SUM(x.v * w.val) + MAX(b.val)) AS y \
         FROM x JOIN w ON x.i = w.i JOIN b ON w.o = b.o GROUP BY x.n, w.o",
        &[x(), w(), b()],
        &[wrt("x", "v"), wrt("w", "val"), wrt("b", "val")],
        2,
    )
    .await;
}

#[tokio::test]
async fn means_and_extremes() {
    check_jvp(
        &ctx(),
        "SELECT o, AVG(val) AS a, MAX(val) AS m, MIN(val * 2.0) AS n, COUNT(*) AS c \
         FROM w GROUP BY o",
        &[w()],
        &[wrt("w", "val")],
        1,
    )
    .await;
}

#[tokio::test]
async fn a_window_sum_and_max() {
    check_jvp(
        &ctx(),
        "SELECT i, o, val - SUM(val) OVER (PARTITION BY o) AS s, \
                MAX(val) OVER (PARTITION BY i) AS m FROM w",
        &[w()],
        &[wrt("w", "val")],
        2,
    )
    .await;
}

#[tokio::test]
async fn a_union_all_and_a_semi_join() {
    check_jvp(
        &ctx(),
        "WITH u AS (SELECT i, val AS v FROM w WHERE o = 0 \
                    UNION ALL SELECT i, 2.0 * val FROM w WHERE o = 1 \
                    UNION ALL SELECT o, 1.0 FROM b) \
         SELECT SUM(v * v) AS l FROM u WHERE i IN (SELECT o FROM b)",
        &[w(), b()],
        &[wrt("w", "val")],
        0,
    )
    .await;
}

#[tokio::test]
async fn a_stop_gradient_has_no_tangent() {
    let ctx = ctx();
    let (_, out) = jvp_of(
        &ctx,
        "SELECT SUM(val * ddx_stop_gradient(val)) AS l FROM w",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
    // Only the first factor moves: Σ val · t.
    let want: f64 = w()
        .rows
        .iter()
        .enumerate()
        .map(|(r, row)| row[2] * direction(r, 0))
        .sum();
    assert!((out[0][1] - want).abs() < 1e-12, "{} vs {want}", out[0][1]);
}

#[tokio::test]
async fn a_max_at_a_tie_takes_the_mean_tangent() {
    let ctx = ctx();
    let tied = Table {
        name: "p",
        columns: vec![("i", "BIGINT"), ("val", "DOUBLE")],
        rows: vec![vec![0.0, 2.0], vec![1.0, 2.0], vec![2.0, 1.0]],
    };
    let (_, out) = jvp_of(
        &ctx,
        "SELECT MAX(val) AS m FROM p",
        &[tied],
        &[wrt("p", "val")],
    )
    .await;
    // jax.jvp of jnp.max: the mean of the tied rows' tangents.
    let want = (direction(0, 0) + direction(1, 0)) / 2.0;
    assert!((out[0][1] - want).abs() < 1e-12, "{} vs {want}", out[0][1]);
}

#[tokio::test]
async fn a_missing_tangent_row_is_zero_and_a_null_value_has_a_null_tangent() {
    let ctx = ctx();
    ctx.sql(
        "CREATE TABLE q (i BIGINT, val DOUBLE) AS VALUES \
         (0, 1.0), (1, CAST(NULL AS DOUBLE)), (2, 3.0)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let plan = substrait_of(&ctx, "SELECT i, 2.0 * val AS y FROM q", true).await;
    let program = ddx_ad::jvp(&plan, &[wrt("q", "val")]).unwrap();
    let tt = &program.tangent_tables[0];
    assert_eq!(tt.columns, vec!["i", "val"]);
    // A tangent for rows 0 and 1 only.
    ctx.sql(&format!(
        "CREATE TABLE {} AS VALUES (0, 0.5), (1, 0.5)",
        tt.name
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ctx.sql(&format!(
        "CREATE OR REPLACE TABLE {0} AS SELECT column1 AS i, column2 AS val FROM {0}",
        tt.name
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ddx_datafusion::ad::run(&ctx, &program).await.unwrap();
    let out = rows(
        &ctx,
        &format!("SELECT * FROM {} ORDER BY i", program.output.step),
    )
    .await;
    assert_eq!(out[0][2], 1.0);
    assert!(out[1][2].is_nan(), "a NULL value's tangent is NULL");
    assert_eq!(out[2][2], 0.0, "a row the tangent lacks has tangent 0");
}

/// ⟨∇L, t⟩, from `grad`, equals the jvp of `L` along `t`.
#[tokio::test]
async fn forward_mode_agrees_with_grad() {
    let ctx = ctx();
    let loss = "WITH h AS (SELECT x.n, w.o, tanh(SUM(x.v * w.val) + MAX(b.val)) AS y \
                           FROM x JOIN w ON x.i = w.i JOIN b ON w.o = b.o GROUP BY x.n, w.o) \
                SELECT SUM(y * y) / COUNT(*) + MAX(y) AS loss FROM h";
    let tables = [x(), w(), b()];
    let wrt = [wrt("x", "v"), wrt("w", "val"), wrt("b", "val")];
    let (_, out) = jvp_of(&ctx, loss, &tables, &wrt).await;
    let plan = substrait_of(&ctx, loss, true).await;
    let program = ddx_ad::grad(&plan, &wrt).unwrap();
    run(&ctx, &program).await;
    let mut dot = 0.0;
    for g in &program.gradients {
        let table = tables
            .iter()
            .find(|t| g.table.last().map(String::as_str) == Some(t.name))
            .unwrap();
        let val = table
            .columns
            .iter()
            .position(|(_, t)| *t == "DOUBLE")
            .unwrap();
        let grads = rows(&ctx, &format!("SELECT * FROM {}", g.step)).await;
        for (r, row) in table.rows.iter().enumerate() {
            let key: Vec<f64> = (0..table.columns.len())
                .filter(|&c| c != val)
                .map(|c| row[c])
                .collect();
            let gr = grads.iter().find(|gr| gr[..key.len()] == key[..]).unwrap();
            dot += gr[key.len()] * direction(r, 0);
        }
    }
    let jvp = out[0][1];
    assert!(
        (jvp - dot).abs() <= 1e-12 * dot.abs().max(1.0),
        "jvp {jvp} vs ⟨∇L, t⟩ {dot}"
    );
}

/// ⟨J t, c⟩ from `jvp` equals ⟨t, Jᵀ c⟩ from `vjp`, for an output with
/// rows.
#[tokio::test]
async fn the_dot_product_test_against_vjp() {
    let ctx = ctx();
    let sql = "SELECT x.n, w.o, tanh(SUM(x.v * w.val)) AS y \
               FROM x JOIN w ON x.i = w.i GROUP BY x.n, w.o";
    let (program, out) = jvp_of(&ctx, sql, &[x(), w()], &[wrt("w", "val")]).await;
    let t = program
        .output
        .columns
        .iter()
        .position(|c| *c == program.output.tangents[0].tangent)
        .unwrap();
    let cot = |n: f64, o: f64| 0.4 - 0.9 * (2.0 * n + o).cos();
    let jt_c: f64 = out.iter().map(|r| r[t] * cot(r[0], r[1])).sum();

    let plan = substrait_of(&ctx, sql, true).await;
    let back = ddx_ad::vjp(&plan, &[wrt("w", "val")]).unwrap();
    assert_eq!(back.cotangent, vec!["n", "o", "y"]);
    let values: Vec<String> = out
        .iter()
        .map(|r| format!("({}, {}, {:e})", r[0], r[1], cot(r[0], r[1])))
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE {} (n BIGINT, o BIGINT, y DOUBLE) AS VALUES {}",
        back.cotangent_table,
        values.join(", ")
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    run(&ctx, &back).await;
    let grads = rows(&ctx, &format!("SELECT * FROM {}", back.gradients[0].step)).await;
    let t_jc: f64 = w()
        .rows
        .iter()
        .enumerate()
        .map(|(r, row)| {
            let g = grads
                .iter()
                .find(|g| g[0] == row[0] && g[1] == row[1])
                .unwrap();
            g[2] * direction(r, 0)
        })
        .sum();
    assert!(
        (jt_c - t_jc).abs() <= 1e-12 * t_jc.abs().max(1.0),
        "⟨J t, c⟩ {jt_c} vs ⟨t, Jᵀ c⟩ {t_jc}"
    );
}

#[tokio::test]
async fn the_value_columns_are_the_query_s_own() {
    let ctx = ctx();
    let sql = "SELECT o, SUM(val * val) AS s FROM w GROUP BY o";
    let (program, out) = jvp_of(&ctx, sql, &[w()], &[wrt("w", "val")]).await;
    assert_eq!(program.output.columns, vec!["o", "s", "__ddx_tangent_1"]);
    let want = rows(&ctx, &format!("{sql} ORDER BY o")).await;
    let mut got: Vec<Vec<f64>> = out.iter().map(|r| r[..2].to_vec()).collect();
    got.sort_by(|a, b| a[0].total_cmp(&b[0]));
    assert_eq!(got, want);
    let total = scalar(
        &ctx,
        &format!("SELECT SUM(__ddx_tangent_1) FROM {}", program.output.step),
    )
    .await;
    let want: f64 = w()
        .rows
        .iter()
        .enumerate()
        .map(|(r, row)| 2.0 * row[2] * direction(r, 0))
        .sum();
    assert!((total - want).abs() < 1e-12);
}

async fn refusal(sql: &str, tables: &[Table], wrt: &[ColumnRef]) -> AdError {
    let ctx = ctx();
    for t in tables {
        t.create(&ctx).await;
    }
    let plan = substrait_of(&ctx, sql, true).await;
    ddx_ad::jvp(&plan, wrt).expect_err("refused")
}

#[tokio::test]
async fn a_varied_group_by_key_used_as_a_number_is_refused() {
    let e = refusal(
        "SELECT SUM(k) AS l FROM (SELECT val AS k, COUNT(*) AS c FROM w GROUP BY val)",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
    assert!(e.to_string().contains("GROUP BY key"), "{e}");
}

#[tokio::test]
async fn a_rank_used_as_a_number_is_refused() {
    let e = refusal(
        "SELECT SUM(r) AS l FROM (SELECT RANK() OVER (ORDER BY val) * 1.0 AS r FROM w)",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
    assert!(e.to_string().contains("window function"), "{e}");
}

#[tokio::test]
async fn an_aggregate_with_no_rule_is_refused() {
    let e = refusal("SELECT stddev(val) AS l FROM w", &[w()], &[wrt("w", "val")]).await;
    assert!(e.to_string().contains("no forward-mode rule"), "{e}");
}

#[tokio::test]
async fn an_output_that_does_not_move_is_refused() {
    let e = refusal("SELECT COUNT(*) AS c FROM w", &[w()], &[wrt("w", "val")]).await;
    assert!(matches!(e, AdError::NotScalar(_)), "{e}");
}

#[tokio::test]
async fn a_tangent_table_whose_dims_repeat_is_refused_before_it_runs() {
    let ctx = ctx();
    w().create(&ctx).await;
    let plan = substrait_of(&ctx, "SELECT SUM(val) AS l FROM w", true).await;
    let program = ddx_ad::jvp(&plan, &[wrt("w", "val")]).unwrap();
    let tt = &program.tangent_tables[0];
    ctx.sql(&format!(
        "CREATE TABLE {} (i BIGINT, o BIGINT, val DOUBLE) AS VALUES (0, 0, 1.0), (0, 0, 2.0)",
        tt.name
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let e = ddx_datafusion::ad::run(&ctx, &program)
        .await
        .expect_err("refused");
    assert!(e.to_string().contains("share their dims"), "{e}");
    assert!(
        ctx.table(program.output.step.as_str()).await.is_err(),
        "a refused run leaves no output"
    );
}

/// Each wrt table's gradient of `loss`, keyed by its dims: a fresh `grad`
/// program run on the tables as they are now.
async fn gradients(
    ctx: &SessionContext,
    loss: &str,
    wrt: &[ColumnRef],
) -> BTreeMap<String, BTreeMap<Vec<i64>, Vec<f64>>> {
    let plan = substrait_of(ctx, loss, true).await;
    let program = ddx_ad::grad(&plan, wrt).unwrap();
    run(ctx, &program).await;
    let mut out = BTreeMap::new();
    for g in &program.gradients {
        let dims = g
            .columns
            .iter()
            .filter(|c| !wrt.iter().any(|w| w.column == **c))
            .count();
        let table = g.table.last().unwrap().clone();
        out.insert(
            table,
            keyed(ctx, &format!("SELECT * FROM {}", g.step), dims).await,
        );
    }
    ddx_datafusion::ad::release(ctx, &program).unwrap();
    out
}

/// Forward over reverse: `jvp_of_program` of `grad(loss)` along the tangent
/// gives `H·v` beside each gradient, checked against a central finite
/// difference of the gradient along `v`.
async fn check_hvp(ctx: &SessionContext, loss: &str, tables: &[Table], wrt: &[ColumnRef]) {
    for t in tables {
        t.create(ctx).await;
    }
    let plan = substrait_of(ctx, loss, true).await;
    let program = ddx_ad::grad(&plan, wrt).unwrap();
    let hvp =
        ddx_ad::jvp_of_program(&program, wrt).unwrap_or_else(|e| panic!("jvp_of_program: {e}"));
    assert_emits_on_projections(&hvp);
    assert_eq!(hvp.gradients.len(), program.gradients.len());
    for tt in &hvp.tangent_tables {
        let table = tables
            .iter()
            .find(|t| tt.table.last().map(String::as_str) == Some(t.name))
            .unwrap();
        tangent_of(table, &tt.name, &tt.columns).create(ctx).await;
    }
    ddx_datafusion::ad::run(ctx, &hvp)
        .await
        .unwrap_or_else(|e| panic!("run: {e}"));

    let h = 1e-5;
    for t in tables {
        moved(t, wrt, h).create(ctx).await;
    }
    let up = gradients(ctx, loss, wrt).await;
    for t in tables {
        moved(t, wrt, -h).create(ctx).await;
    }
    let down = gradients(ctx, loss, wrt).await;
    for t in tables {
        t.create(ctx).await;
    }
    for (g, out) in program.gradients.iter().zip(&hvp.gradients) {
        let table = g.table.last().unwrap();
        let values = wrt.iter().filter(|w| w.table == *table).count();
        assert_eq!(
            out.tangents.len(),
            values,
            "H·v has a tangent for each of {table}'s wrt columns"
        );
        let dims = g.columns.len() - values;
        let got = keyed(ctx, &format!("SELECT * FROM {}", out.step), dims).await;
        assert_eq!(got.len(), up[table].len(), "H·v has the gradient's rows");
        for (key, row) in &got {
            for tan in &out.tangents {
                let c = out.columns.iter().position(|n| *n == tan.column).unwrap();
                let t = out.columns.iter().position(|n| *n == tan.tangent).unwrap();
                let fd = (up[table][key][c] - down[table][key][c]) / (2.0 * h);
                assert!(
                    (row[t] - fd).abs() <= 1e-5 * fd.abs().max(1.0),
                    "(H·v) of {table}.{} at {key:?}: ddx {} vs finite difference {fd}",
                    tan.column,
                    row[t]
                );
            }
        }
    }
}

#[tokio::test]
async fn a_hessian_vector_product_of_an_elementwise_loss() {
    check_hvp(
        &ctx(),
        "SELECT SUM(tanh(val) * exp(val / 2.0) + val * val * val) AS loss FROM w",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_hessian_vector_product_of_a_layer() {
    check_hvp(
        &ctx(),
        "WITH h AS (SELECT x.n, w.o, tanh(SUM(x.v * w.val) + MAX(b.val)) AS y \
                    FROM x JOIN w ON x.i = w.i JOIN b ON w.o = b.o GROUP BY x.n, w.o) \
         SELECT SUM((y - 0.1) * (y - 0.1)) / COUNT(*) AS loss FROM h",
        &[x(), w(), b()],
        &[wrt("x", "v"), wrt("w", "val"), wrt("b", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_hessian_vector_product_through_a_max_and_a_mean() {
    check_hvp(
        &ctx(),
        "WITH g AS (SELECT o, MAX(val * val) AS m, AVG(sin(val)) AS a FROM w GROUP BY o) \
         SELECT SUM(m * a + a * a) AS loss FROM g",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_refused_tangent_no_later_step_reads_does_not_refuse_the_program() {
    // The saved aggregate's key, floor(val * 2 + 0.25), has no tangent, but
    // nothing after it reads the key: H·v is still defined, and is computed.
    // (No val sits where the key jumps, so the loss is smooth there.)
    check_hvp(
        &ctx(),
        "WITH g AS (SELECT floor(val * 2.0 + 0.25) AS k, SUM(val * val) AS s FROM w \
                    GROUP BY floor(val * 2.0 + 0.25)) \
         SELECT SUM(s * s) AS loss FROM g",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_null_input_adds_nothing_to_a_tangent_of_two_terms() {
    // Row 0: y = coalesce(NULL, 0) + b, whose tangent is ḃ. The a-term is
    // NULL there (a is), and a plain sum of the terms lost ḃ with it.
    let ctx = ctx();
    ctx.sql(
        "CREATE TABLE p (i BIGINT, a DOUBLE, b DOUBLE) AS VALUES \
         (0, CAST(NULL AS DOUBLE), 2.0), (1, 1.0, 3.0)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let plan = substrait_of(&ctx, "SELECT i, coalesce(a, 0.0) + b AS y FROM p", true).await;
    let program = ddx_ad::jvp(&plan, &[wrt("p", "a"), wrt("p", "b")]).unwrap();
    let tt = &program.tangent_tables[0];
    assert_eq!(tt.columns, vec!["i", "a", "b"]);
    ctx.sql(&format!(
        "CREATE TABLE {} (i BIGINT, a DOUBLE, b DOUBLE) AS VALUES (0, 1.0, 10.0), (1, 1.0, 10.0)",
        tt.name
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ddx_datafusion::ad::run(&ctx, &program).await.unwrap();
    let out = rows(
        &ctx,
        &format!("SELECT * FROM {} ORDER BY i", program.output.step),
    )
    .await;
    assert_eq!(out[0][2], 10.0, "ḃ alone where a is NULL");
    assert_eq!(out[1][2], 11.0, "ȧ + ḃ elsewhere");
}

/// A loss whose Substrait plan repeats a CTE read twice per layer: its
/// grad program's region steps nest projections hundreds deep.
async fn reused_cte_plan(depth: usize) -> ddx_ad::substrait::proto::Plan {
    let ctx = SessionContext::new();
    ctx.sql("CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 0.3), (1, 0.5)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut ctes = vec!["c0 AS (SELECT i, val AS v FROM p)".to_string()];
    for k in 1..depth {
        ctes.push(format!(
            "c{k} AS (SELECT a.i, a.v * b.v AS v FROM c{m} a JOIN c{m} b ON a.i = b.i)",
            m = k - 1
        ));
    }
    let loss = format!(
        "WITH {} SELECT SUM(v) AS l FROM c{}",
        ctes.join(", "),
        depth - 1
    );
    substrait_of(&ctx, &loss, true).await
}

#[test]
fn jvp_program_does_not_overflow_a_worker_threads_stack() {
    // A tokio worker thread has a 2 MB stack. The rewrite recursed once per
    // relation, and a program's region step is hundreds of projections
    // deep: the overflow aborts the process, so this runs in a child.
    if std::env::var("DDX_JVP_CHILD").is_ok() {
        let plan = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(reused_cte_plan(9));
        let t = std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(move || {
                let wrt = [wrt("p", "val")];
                ddx_ad::jvp(&plan, &wrt)?;
                ddx_ad::jvp_of_program(&ddx_ad::grad(&plan, &wrt)?, &wrt).map(|_| ())
            })
            .unwrap();
        t.join().unwrap().unwrap();
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "jvp_program_does_not_overflow_a_worker_threads_stack",
            "--nocapture",
        ])
        .env("DDX_JVP_CHILD", "1")
        .status()
        .unwrap();
    assert!(status.success(), "jvp_of_program on a 2 MB stack: {status}");
}

/// The jvp of the loss `sql` over `p(i, val) = (0, 0.5), (1, NULL), (2,
/// 0.3), (3, 0.8)` along `t = (1, 2, 3, 4)`.
async fn jvp_over_a_null(sql: &str) -> f64 {
    let ctx = ctx();
    ctx.sql(
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES \
         (0, 0.5), (1, CAST(NULL AS DOUBLE)), (2, 0.3), (3, 0.8)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let plan = substrait_of(&ctx, sql, true).await;
    let program = ddx_ad::jvp(&plan, &[wrt("p", "val")]).unwrap();
    ctx.sql(&format!(
        "CREATE TABLE {} (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)",
        program.tangent_tables[0].name
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    ddx_datafusion::ad::run(&ctx, &program).await.unwrap();
    rows(&ctx, &format!("SELECT * FROM {}", program.output.step)).await[0][1]
}

#[tokio::test]
async fn a_null_input_beside_another_through_greatest_and_case() {
    // Found by the fuzzer before the NULL-term fix: greatest(a, NULL) is a,
    // whose tangent ȧ + NULL was NULL, read as 0 (4.0 for 8.0).
    let got = jvp_over_a_null(
        "SELECT SUM(greatest(a.val, b.val)) AS l FROM p a JOIN p b ON b.i = a.i + 1",
    )
    .await;
    assert_eq!(got, 1.0 + 3.0 + 4.0);
    let got = jvp_over_a_null(
        "SELECT SUM(CASE WHEN b.val IS NULL THEN a.val ELSE a.val * b.val END) AS l \
         FROM p a JOIN p b ON b.i = a.i + 1",
    )
    .await;
    // Rows (0, 1): ȧ = 1. (1, 2): NULL, skipped. (2, 3): ȧ·b + a·ḃ = 3·0.8 + 0.3·4.
    assert!((got - (1.0 + 3.0 * 0.8 + 0.3 * 4.0)).abs() < 1e-12, "{got}");
}

#[tokio::test]
async fn a_loss_that_reads_wrt_only_through_conditions_has_a_zero_tangent() {
    // Piecewise constant in val: grad gives zeros, and jax.jvp a zero
    // tangent. jvp refused it as not depending on wrt.
    for loss in [
        "SELECT SUM(CASE WHEN val > 0 THEN 1.0 ELSE 0.0 END) AS l FROM w",
        "SELECT COUNT(val) * 1.0 AS l FROM w",
    ] {
        let ctx = ctx();
        let (program, out) = jvp_of(&ctx, loss, &[w()], &[wrt("w", "val")]).await;
        assert_eq!(program.output.tangents.len(), 1, "{loss}");
        assert_eq!(out[0][1], 0.0, "{loss}");
        let plan = substrait_of(&ctx, loss, true).await;
        let g = ddx_ad::grad(&plan, &[wrt("w", "val")]).unwrap();
        run(&ctx, &g).await;
        let grads = rows(&ctx, &format!("SELECT * FROM {}", g.gradients[0].step)).await;
        assert!(
            grads.iter().all(|r| r[2] == 0.0),
            "{loss}: grad gives zeros"
        );
    }
}

#[tokio::test]
async fn options_restrict_is_refused_not_ignored() {
    let ctx = ctx();
    w().create(&ctx).await;
    let plan = substrait_of(&ctx, "SELECT SUM(val) AS l FROM w", true).await;
    let select = substrait_of(&ctx, "SELECT * FROM w WHERE i = 0", false).await;
    let options = ddx_ad::Options::new().restrict("w", select);
    let e = ddx_ad::jvp_with(&plan, &[wrt("w", "val")], &options).expect_err("refused");
    assert!(matches!(e, AdError::InvalidOptions(_)), "{e}");
    let g = ddx_ad::grad(&plan, &[wrt("w", "val")]).unwrap();
    let e = ddx_ad::jvp_of_program_with(&g, &[wrt("w", "val")], &options).expect_err("refused");
    assert!(matches!(e, AdError::InvalidOptions(_)), "{e}");
}

/// The size of the plan DataFusion consumes for the jvp of `n` aggregates
/// stacked on one another.
async fn stacked_aggregates_plan_size(n: usize) -> usize {
    let ctx = ctx();
    w().create(&ctx).await;
    let mut ctes = vec!["h0 AS (SELECT i, o, val AS v FROM w)".to_string()];
    for k in 1..=n {
        ctes.push(format!(
            "h{k} AS (SELECT i, o, SUM(tanh(v) + 0.5 * v) AS v FROM h{} GROUP BY i, o)",
            k - 1
        ));
    }
    let sql = format!("WITH {} SELECT SUM(v) AS l FROM h{n}", ctes.join(", "));
    let (program, _) = jvp_of(&ctx, &sql, &[w()], &[wrt("w", "val")]).await;
    let lp = ddx_datafusion::ad::logical_plan(&ctx, &program.steps[0].plan)
        .await
        .unwrap();
    format!("{}", lp.display_indent()).len()
}

#[tokio::test]
async fn a_tangent_grows_linearly_through_stacked_aggregates() {
    // Found by the performance review: DataFusion names an aggregate by its
    // expression, and a tangent written into a measure read its input's
    // tangent twice, so the plan doubled at every layer (2,047 ms at 16
    // layers, 137× the forward query). Each measure now reads a column.
    let (small, large) = (
        stacked_aggregates_plan_size(4).await,
        stacked_aggregates_plan_size(12).await,
    );
    assert!(
        large < 5 * small,
        "12 layers' plan is {large} bytes, 4 layers' {small}: not linear"
    );
}

#[tokio::test]
async fn jvp_and_hvp_run_without_the_simplifier() {
    // DataFusion 54 runs coalesce only once its simplifier has rewritten it
    // to a CASE (grad found it, soak #90): a tangent's NULL guard must be a
    // CASE, and a context may run no optimizer rules at all.
    use datafusion::execution::SessionStateBuilder;
    let ctx = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_optimizer_rules(vec![])
            .build(),
    );
    ctx.sql("CREATE TABLE ws (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, NULL), (2, 3.0)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let loss = "SELECT SUM(val * val + val) AS l FROM ws";
    let plan = substrait_of(&ctx, loss, true).await;
    let wrt = [wrt("ws", "val")];
    let jvp = ddx_ad::jvp(&plan, &wrt).unwrap();
    let hvp = ddx_ad::jvp_of_program(&ddx_ad::grad(&plan, &wrt).unwrap(), &wrt).unwrap();
    for (program, tangents) in [(&jvp, &jvp.tangent_tables), (&hvp, &hvp.tangent_tables)] {
        for tt in tangents {
            ctx.sql(&format!(
                "CREATE OR REPLACE TABLE {} (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 1.0), (2, 1.0)",
                tt.name
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        }
        ddx_datafusion::ad::run(&ctx, program)
            .await
            .unwrap_or_else(|e| panic!("run without the simplifier: {e}"));
    }
    // d/dv Σ (v² + v) along 1, over the rows whose value is not NULL.
    let got = rows(&ctx, &format!("SELECT * FROM {}", jvp.output.step)).await;
    assert_eq!(got[0][1], (2.0 * 1.0 + 1.0) + (2.0 * 3.0 + 1.0));
}
