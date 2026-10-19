// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ad::sql` builds each program once per context and runs it again on
//! later calls (`ddx_ad::Programs`), until the tables' schemas or the
//! views' definitions it was built from change.
//!
//! Which program ran shows in the table names the rewritten statement
//! reads: each program writes under a prefix of its own, `__ddx_{id}_`, so
//! two calls that read the same prefix ran the same program.

use datafusion::arrow::array::{AsArray, RecordBatch};
use datafusion::arrow::datatypes::{Float64Type, Int64Type};
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad;

const GRAD: &str = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w) \
                    SELECT i, grad(loss, val) FROM w";

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// `w`, with `val` declared `val_type`, holding `vals` at `i = 0, 1, …`.
async fn create_w(ctx: &SessionContext, val_type: &str, vals: &[f64]) {
    exec(ctx, "DROP TABLE IF EXISTS w").await;
    exec(
        ctx,
        &format!("CREATE TABLE w (i BIGINT NOT NULL, val {val_type})"),
    )
    .await;
    let rows: Vec<String> = vals
        .iter()
        .enumerate()
        .map(|(i, v)| format!("({i}, {v:?})"))
        .collect();
    exec(ctx, &format!("INSERT INTO w VALUES {}", rows.join(", "))).await;
}

/// `sql`'s `(i, value)` pairs, sorted, and the prefix of the program
/// tables its rewritten statement reads.
async fn run(ctx: &SessionContext, sql: &str) -> (Vec<(i64, f64)>, String) {
    let df = ad::sql(ctx, sql).await.unwrap();
    let plan = df.logical_plan().display_indent().to_string();
    let batches: Vec<RecordBatch> = df.collect().await.unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let i = b.column(0).as_primitive::<Int64Type>();
        let v = b.column(1).as_primitive::<Float64Type>();
        out.extend((0..b.num_rows()).map(|r| (i.value(r), v.value(r))));
    }
    out.sort_by_key(|p| p.0);
    (out, prefix(&plan))
}

/// The one program prefix, `__ddx_{hex}_{hex}_`, that `plan` reads; the
/// rewrite's own aliases (`__ddx_t`, `__ddx_g0`, …) are not one.
fn prefix(plan: &str) -> String {
    let hex = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit());
    let mut found: Vec<String> = plan
        .match_indices("__ddx_")
        .filter_map(|(at, _)| {
            let mut parts = plan[at + "__ddx_".len()..].splitn(3, '_');
            let (a, b) = (parts.next()?, parts.next()?);
            parts.next()?;
            (hex(a) && hex(b)).then(|| format!("__ddx_{a}_{b}_"))
        })
        .collect();
    found.sort();
    found.dedup();
    assert_eq!(found.len(), 1, "one program's tables in {plan}");
    found.pop().unwrap()
}

fn assert_close(got: &[(i64, f64)], want: &[(i64, f64)]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for ((gi, g), (wi, w)) in got.iter().zip(want) {
        assert_eq!(gi, wi);
        assert!((g - w).abs() < 1e-12, "{got:?} vs {want:?}");
    }
}

/// d/dval of Σ val³ at each row.
fn three_val_squared(vals: &[f64]) -> Vec<(i64, f64)> {
    vals.iter()
        .enumerate()
        .map(|(i, v)| (i as i64, 3.0 * v * v))
        .collect()
}

#[tokio::test]
async fn a_second_call_runs_the_program_the_first_built() {
    let ctx = SessionContext::new();
    create_w(&ctx, "DOUBLE", &[1.0, -2.0, 0.5]).await;
    let (first, built) = run(&ctx, GRAD).await;
    let (second, ran) = run(&ctx, GRAD).await;
    assert_close(&first, &three_val_squared(&[1.0, -2.0, 0.5]));
    assert_close(&second, &first);
    assert_eq!(ran, built, "the second call built its own program");
}

#[tokio::test]
async fn a_kept_program_reads_the_tables_values_of_its_call() {
    let ctx = SessionContext::new();
    create_w(&ctx, "DOUBLE", &[1.0, -2.0, 0.5]).await;
    let (_, built) = run(&ctx, GRAD).await;
    create_w(&ctx, "DOUBLE", &[4.0, 0.0, -3.0]).await;
    let (got, ran) = run(&ctx, GRAD).await;
    assert_eq!(
        ran, built,
        "new values, same schema: the program is the same"
    );
    assert_close(&got, &three_val_squared(&[4.0, 0.0, -3.0]));
}

#[tokio::test]
async fn a_table_of_another_schema_gets_its_program_built_again() {
    // A program states each column's nullability, and binding it to a
    // table whose column may be NULL where the program says it may not is
    // refused: kept under the old schema, it would fail here.
    let ctx = SessionContext::new();
    create_w(&ctx, "DOUBLE NOT NULL", &[1.0, -2.0, 0.5]).await;
    let (_, built) = run(&ctx, GRAD).await;
    create_w(&ctx, "DOUBLE", &[2.0, 1.0, 0.0]).await;
    let (got, ran) = run(&ctx, GRAD).await;
    assert_ne!(
        ran, built,
        "a nullable column ran the program built for NOT NULL"
    );
    assert_close(&got, &three_val_squared(&[2.0, 1.0, 0.0]));
}

#[tokio::test]
async fn a_view_redefined_gets_its_program_built_again() {
    // The view's schema stays (i, v): only its definition changes.
    let ctx = SessionContext::new();
    create_w(&ctx, "DOUBLE", &[1.0, -2.0, 0.5]).await;
    let sql = "WITH loss AS (SELECT SUM(v * v) AS l FROM m) SELECT i, grad(loss, val) FROM w";
    exec(&ctx, "CREATE VIEW m AS SELECT i, 2.0 * val AS v FROM w").await;
    let (got, built) = run(&ctx, sql).await;
    assert_close(&got, &[(0, 8.0), (1, -16.0), (2, 4.0)]);
    exec(
        &ctx,
        "CREATE OR REPLACE VIEW m AS SELECT i, 3.0 * val AS v FROM w",
    )
    .await;
    let (got, ran) = run(&ctx, sql).await;
    assert_ne!(ran, built, "the old view's program ran");
    assert_close(&got, &[(0, 18.0), (1, -36.0), (2, 9.0)]);
}

#[tokio::test]
async fn a_program_that_reads_another_s_result_is_kept_with_it() {
    // A Hessian-vector product: the jvp program reads what the grad program
    // computed, by its table names, so it can be kept only while the grad
    // program is.
    let ctx = SessionContext::new();
    create_w(&ctx, "DOUBLE", &[1.0, -2.0, 0.5]).await;
    let sql = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), \
                    g AS (SELECT i, grad(loss, val) AS val FROM w), \
                    v AS (SELECT i, 1.0 AS val FROM w) \
               SELECT i, jvp(val, w.val, v) AS hv FROM g";
    let (got, built) = run(&ctx, sql).await;
    assert_close(&got, &[(0, 6.0), (1, -12.0), (2, 3.0)]);
    let (again, ran) = run(&ctx, sql).await;
    assert_eq!(ran, built);
    assert_close(&again, &got);
}

#[tokio::test]
async fn statements_that_restrict_a_gradient_differently_keep_a_program_each() {
    // A `WHERE` on the dims restricts the rows the program computes, so
    // each restriction is a program of its own.
    let ctx = SessionContext::new();
    create_w(&ctx, "DOUBLE", &[1.0, -2.0, 0.5]).await;
    let loss = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w)";
    let one = format!("{loss} SELECT i, grad(loss, val) FROM w WHERE i = 1");
    let two = format!("{loss} SELECT i, grad(loss, val) FROM w WHERE i = 2");
    for _ in 0..2 {
        assert_close(&run(&ctx, &one).await.0, &[(1, 12.0)]);
        assert_close(&run(&ctx, &two).await.0, &[(2, 0.75)]);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overlapping_calls_never_run_one_program_at_once() {
    // A program's runs write the same tables, so two at once would read
    // each other's. The first call to find it kept takes it; the others
    // build their own.
    let ctx = SessionContext::new();
    create_w(&ctx, "DOUBLE", &[1.0, -2.0, 0.5]).await;
    run(&ctx, GRAD).await;
    let calls: Vec<_> = (0..4)
        .map(|_| {
            let ctx = ctx.clone();
            tokio::spawn(async move { run(&ctx, GRAD).await })
        })
        .collect();
    for call in calls {
        let (got, _) = call.await.unwrap();
        assert_close(&got, &three_val_squared(&[1.0, -2.0, 0.5]));
    }
}

#[tokio::test]
async fn a_program_is_kept_again_only_once_its_tables_are_released() {
    // Between `rewrite` and `release` the caller plans statements that read
    // the program's tables. Were the program kept already, a call in
    // between would run it, writing those tables, and its release would
    // drop them.
    use datafusion::sql::sqlparser::dialect::GenericDialect;
    use ddx_datafusion::engine::DataFusion;

    let ctx = SessionContext::new();
    create_w(&ctx, "DOUBLE", &[1.0, -2.0, 0.5]).await;
    let programs = ddx_ad::Programs::new();
    let mut engine = DataFusion(ctx.clone());
    // One engine per call, as `ad::sql` makes them.
    let rewrite = || async {
        let mut engine = DataFusion(ctx.clone());
        ddx_ad::rewrite(&mut engine, &programs, &[GRAD], &GenericDialect {})
            .await
            .unwrap()
    };
    let warm = rewrite().await;
    warm.release(&mut engine, &programs).await.unwrap();

    let first = rewrite().await;
    let second = rewrite().await;
    let (a, b) = (prefix(&first.statements[0]), prefix(&second.statements[0]));
    assert_ne!(
        a, b,
        "the second call ran the program the first had not released"
    );
    second.release(&mut engine, &programs).await.unwrap();
    // The first call's tables are still there to read.
    let df = ctx.sql(&first.statements[0]).await.unwrap();
    let batches = df.collect().await.unwrap();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 3);
    first.release(&mut engine, &programs).await.unwrap();
    // And both programs are kept now.
    let third = rewrite().await;
    assert!([&a, &b].contains(&&prefix(&third.statements[0])));
    third.release(&mut engine, &programs).await.unwrap();
}
