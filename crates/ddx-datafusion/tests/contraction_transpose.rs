// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The contraction rule (`ddx_ad`'s `contraction` module) against the
//! general reduce rule.
//!
//! A saved `SUM(e_X · e_Y)` over one join of two inputs is transposed as a
//! contraction of the cotangent with the other input, without rebuilding the
//! join. Its gradient must equal the general rule's, NULLs, NaNs and
//! infinities included. Adding a filter that reads both sides of the join,
//! and keeps every row (the keys it reads are never negative), sends the
//! same loss down the general rule, as long as it reads a column that is not
//! a grouping key (one that reads only grouping keys is a mask, which the
//! contraction rule takes): the two are compared on random
//! tables with NULLs, NaNs, infinities, missing rows, NULL join keys, and
//! repeated keys in constant data. Each case also checks which rule ran, by
//! whether a gradient step joins the two inputs.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use ddx_core::test_utils::Rng;
use ddx_datafusion::ad::{self, BackwardProgram, ColumnRef};

/// True with probability `p`.
fn chance(rng: &mut Rng, p: f64) -> bool {
    rng.range(0.0, 1.0) < p
}

/// NULL sometimes, NaN or ±∞ rarely, otherwise a multiple of 1/8 in
/// [-2, 2].
fn value(rng: &mut Rng) -> Option<f64> {
    if chance(rng, 0.12) {
        None
    } else if chance(rng, 0.03) {
        Some(f64::NAN)
    } else if chance(rng, 0.02) {
        Some(if chance(rng, 0.5) {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        })
    } else {
        Some((rng.below(33) as f64 - 16.0) / 8.0)
    }
}

/// A table of key columns and value columns, any of them NULL (unlike
/// `common::ad::Table`, whose rows are numbers).
struct Table {
    keys: Vec<(&'static str, Vec<Option<i64>>)>,
    values: Vec<(&'static str, Vec<Option<f64>>)>,
}

/// Rows over the grid of `extents` (one key column per name in `keys`):
/// each present with probability 0.8, repeated with probability `dup`; the
/// key named `null_key`, if any, NULL with probability 0.1; one value column
/// per name in `values`.
fn table(
    rng: &mut Rng,
    keys: &[(&'static str, usize)],
    values: &[&'static str],
    dup: f64,
    null_key: Option<&str>,
) -> Table {
    let mut t = Table {
        keys: keys.iter().map(|&(k, _)| (k, Vec::new())).collect(),
        values: values.iter().map(|&v| (v, Vec::new())).collect(),
    };
    let total: usize = keys.iter().map(|&(_, e)| e).product();
    for cell in 0..total {
        if !chance(rng, 0.8) {
            continue;
        }
        for _ in 0..if chance(rng, dup) { 2 } else { 1 } {
            let mut rest = cell;
            for (i, &(name, extent)) in keys.iter().enumerate().rev() {
                let k = (rest % extent) as i64;
                rest /= extent;
                let null = null_key == Some(name) && chance(rng, 0.1);
                t.keys[i].1.push((!null).then_some(k));
            }
            for v in t.values.iter_mut() {
                v.1.push(value(rng));
            }
        }
    }
    t
}

fn register(ctx: &SessionContext, name: &str, t: Table) {
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();
    for (k, data) in t.keys {
        fields.push(Field::new(k, DataType::Int64, true));
        columns.push(Arc::new(Int64Array::from(data)));
    }
    for (v, data) in t.values {
        fields.push(Field::new(v, DataType::Float64, true));
        columns.push(Arc::new(Float64Array::from(data)));
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    ctx.register_batch(name, batch).unwrap();
}

/// Does any gradient step join rows `left` accepts with rows `right`
/// accepts, that is, rebuild the forward join? Each judges a side of a join
/// by the tables it scans. Runs the steps again first: `ad::run` releases
/// the tables a step's plan reads.
async fn rebuilds(
    ctx: &SessionContext,
    program: &BackwardProgram,
    left: &dyn Fn(&[String]) -> bool,
    right: &dyn Fn(&[String]) -> bool,
) -> bool {
    for step in program.steps() {
        ddx_datafusion::engine::run_step(ctx, step).await.unwrap();
    }
    let mut found = false;
    for step in &program.backward_steps {
        let plan = ddx_datafusion::engine::logical_plan(ctx, &step.plan)
            .await
            .unwrap();
        plan.apply(|node| {
            if let LogicalPlan::Join(j) = node {
                let scans = |p: &LogicalPlan| {
                    let mut names = Vec::new();
                    p.apply(|n| {
                        if let LogicalPlan::TableScan(t) = n {
                            names.push(t.table_name.table().to_string());
                        }
                        Ok(TreeNodeRecursion::Continue)
                    })
                    .unwrap();
                    names
                };
                let (l, r) = (scans(&j.left), scans(&j.right));
                if (left(&l) && right(&r)) || (left(&r) && right(&l)) {
                    found = true;
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
    }
    found
}

/// `(a row's key columns, its gradient)` rows.
type Rows = Vec<(Vec<Option<i64>>, Option<f64>)>;

/// Each gradient of each `wrt` column, its rows sorted by their keys, and
/// rows with the same keys (repeated keys in constant data) by value, every
/// NaN alike: two programs may give NaNs of different signs.
async fn gradients(
    ctx: &SessionContext,
    program: &BackwardProgram,
    wrt: &[ColumnRef],
) -> Vec<Rows> {
    let is_wrt = |c: &str| wrt.iter().any(|w| w.column.eq_ignore_ascii_case(c));
    let mut out = Vec::new();
    for g in &program.gradients {
        // The key columns: the tables here key on integers and hold floats.
        let schema = ctx.table(g.step.as_str()).await.unwrap().schema().clone();
        let dims: Vec<&String> = g
            .columns
            .iter()
            .filter(|c| {
                schema
                    .field_with_unqualified_name(c)
                    .is_ok_and(|f| f.data_type() == &DataType::Int64)
            })
            .collect();
        for col in g.columns.iter().filter(|c| is_wrt(c)) {
            let keys: Vec<String> = dims.iter().map(|d| format!("\"{d}\"")).collect();
            let sql = format!("SELECT {}, \"{col}\" FROM \"{}\"", keys.join(", "), g.step);
            let mut rows = Vec::new();
            for b in ctx.sql(&sql).await.unwrap().collect().await.unwrap() {
                let v = b
                    .column(dims.len())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                for i in 0..b.num_rows() {
                    let key: Vec<Option<i64>> = (0..dims.len())
                        .map(|d| {
                            let a = b.column(d).as_any().downcast_ref::<Int64Array>().unwrap();
                            (!a.is_null(i)).then(|| a.value(i))
                        })
                        .collect();
                    rows.push((key, (!v.is_null(i)).then(|| v.value(i))));
                }
            }
            let bits = |v: Option<f64>| v.map(|v| if v.is_nan() { u64::MAX } else { v.to_bits() });
            rows.sort_by(|x, y| x.0.cmp(&y.0).then(bits(x.1).cmp(&bits(y.1))));
            out.push(rows);
        }
    }
    out
}

/// The same gradients: same rows, same NULLs, NaN where NaN, the same
/// infinities, and values within rounding (the contraction rule sums in a
/// different order).
fn agree(a: &[Rows], b: &[Rows]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("{} gradients vs {}", a.len(), b.len()));
    }
    for (ga, gb) in a.iter().zip(b) {
        if ga.len() != gb.len() {
            return Err(format!("{} rows vs {}", ga.len(), gb.len()));
        }
        for (x, y) in ga.iter().zip(gb) {
            let ok = x.0 == y.0
                && match (x.1, y.1) {
                    (None, None) => true,
                    (Some(p), Some(q)) if p.is_nan() || q.is_nan() => p.is_nan() && q.is_nan(),
                    (Some(p), Some(q)) if p.is_infinite() || q.is_infinite() => p == q,
                    (Some(p), Some(q)) => (p - q).abs() <= 1e-12 * (1.0 + q.abs()),
                    _ => false,
                };
            if !ok {
                return Err(format!("{x:?} vs {y:?}"));
            }
        }
    }
    Ok(())
}

/// Two predicates on the tables a side of a join scans: a join the general
/// rule rebuilds and the contraction rule must not.
type Join<'p> = (&'p dyn Fn(&[String]) -> bool, &'p dyn Fn(&[String]) -> bool);

/// A side that scans table `name` alone.
fn named(name: &'static str) -> impl Fn(&[String]) -> bool {
    move |scans: &[String]| scans.len() == 1 && scans[0] == name
}

/// A side that scans attention's scores, the first saved aggregate, alone
/// or joined to other saved aggregates: the rows its softmax's windows,
/// read as GROUP BYs joined back, make of them.
fn scores(scans: &[String]) -> bool {
    scans.iter().all(|t| t.contains("_saved_")) && scans.iter().any(|t| t.ends_with("saved_0"))
}

/// Runs `loss` and `general` (the same loss, written so the general rule
/// applies) and compares their gradients. Each pair in `joins` is two inputs
/// whose join the general rule rebuilds and the contraction rule must not.
/// `Ok(false)` if ddx refused the tables (repeated keys in a table it
/// differentiates).
async fn compare(
    ctx: &SessionContext,
    loss: &str,
    general: &str,
    wrt: &[ColumnRef],
    joins: &[Join<'_>],
) -> Result<bool, String> {
    let fast = ad::grad(ctx, loss, wrt).await.unwrap();
    let slow = ad::grad(ctx, general, wrt).await.unwrap();
    if ddx_datafusion::engine::execute(ctx, &fast).await.is_err() {
        assert!(
            ddx_datafusion::engine::execute(ctx, &slow).await.is_err(),
            "only the contraction rule's program was refused"
        );
        return Ok(false);
    }
    ddx_datafusion::engine::execute(ctx, &slow).await.unwrap();
    for &(l, r) in joins {
        assert!(
            !rebuilds(ctx, &fast, l, r).await,
            "the contraction rule did not apply"
        );
        assert!(
            rebuilds(ctx, &slow, l, r).await,
            "the comparison did not take the general rule"
        );
    }
    let a = gradients(ctx, &fast, wrt).await;
    let b = gradients(ctx, &slow, wrt).await;
    agree(&a, &b).map(|_| true)
}

/// A filter across the sides of `a JOIN w` that keeps every row and reads a
/// join key, so it sends a loss down the general rule.
const GENERAL: &str = "WHERE a.k IS NULL OR a.k + w.o >= 0";

/// A matrix product's loss, joined `on`, and the same loss with a filter
/// across the sides that keeps every row.
fn matmul(on: &str) -> (String, String) {
    let loss = |filter: &str| {
        format!(
            "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON {on} {filter} \
             GROUP BY a.s, w.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM c"
        )
    };
    (loss(""), loss(GENERAL))
}

/// Runs `cases` random matrix products joined `on`, with join keys NULL
/// sometimes if `null_keys`, and asserts nearly all compared and all agreed.
async fn matrix_products(seed: u64, cases: usize, on: &str, null_keys: bool) {
    let (loss, general) = matmul(on);
    let nulls = null_keys.then_some("k");
    matrix_products_of(seed, cases, on, &loss, &general, nulls).await;
}

/// [`matrix_products`] of a given loss and its general-rule spelling.
async fn matrix_products_of(
    seed: u64,
    cases: usize,
    on: &str,
    loss: &str,
    general: &str,
    nulls: Option<&str>,
) {
    let mut rng = Rng::new(seed);
    let (mut compared, mut failed) = (0, Vec::new());
    for case in 0..cases {
        // In a third of the cases only w is differentiated, so a is constant
        // data and may repeat keys.
        let both = case % 3 != 2;
        let (ns, nk, no) = (
            1 + rng.below(5) as usize,
            1 + rng.below(4) as usize,
            1 + rng.below(4) as usize,
        );
        let null_k = nulls;
        let ctx = SessionContext::new();
        let dup = if both { 0.0 } else { 0.3 };
        register(
            &ctx,
            "a",
            table(&mut rng, &[("s", ns), ("k", nk)], &["val"], dup, null_k),
        );
        register(
            &ctx,
            "w",
            table(&mut rng, &[("k", nk), ("o", no)], &["val"], 0.0, null_k),
        );
        let wrt = if both {
            vec![ColumnRef::new("w", "val"), ColumnRef::new("a", "val")]
        } else {
            vec![ColumnRef::new("w", "val")]
        };
        match compare(&ctx, loss, general, &wrt, &[(&named("a"), &named("w"))]).await {
            Ok(true) => compared += 1,
            Ok(false) => {}
            Err(e) => failed.push(format!("case {case} ({ns}×{nk}×{no}): {e}")),
        }
    }
    assert!(
        failed.is_empty(),
        "{} of {cases} disagree on `{on}`:\n{}",
        failed.len(),
        failed.join("\n")
    );
    assert!(
        compared * 10 >= cases * 7,
        "only {compared} of {cases} cases compared on `{on}`"
    );
}

#[tokio::test]
async fn a_matrix_product_matches_the_general_rule_on_awkward_tables() {
    matrix_products(0x5eed_c0de, 150, "a.k = w.k", false).await;
}

#[tokio::test]
async fn a_masked_matrix_product_matches_the_general_rule() {
    // A conjunct that reads only grouping keys, as causal attention's
    // `q.t >= k.t` does, keeps the contraction rule: it applies to whole
    // groups, so the transpose applies it to the cotangent. A NULL grouping
    // key makes the mask NULL, which drops the row in both.
    for (seed, mask, nulls) in [
        (0x3a5c, "a.s >= w.o", None),
        (0x3a5d, "a.s >= w.o AND w.o <> 1", None),
        (0x3a5e, "a.s >= w.o", Some("s")),
        (0x3a5f, "a.s + w.o <> 2", Some("o")),
    ] {
        let loss = |filter: &str| {
            format!(
                "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k \
                 WHERE {mask}{filter} GROUP BY a.s, w.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM c"
            )
        };
        let general = loss(" AND (a.k IS NULL OR a.k + w.o >= 0)");
        matrix_products_of(seed, 90, mask, &loss(""), &general, nulls).await;
    }
}

#[tokio::test]
async fn a_count_beside_the_sum_matches_the_general_rule() {
    // COUNT moves with no wrt column, so it has no cotangent, and the
    // aggregate is still one SUM of a product for the contraction rule.
    let loss = |filter: &str| {
        format!(
            "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z, COUNT(*) AS n FROM a JOIN w \
             ON a.k = w.k {filter} GROUP BY a.s, w.o) \
             SELECT SUM(tanh(z) * tanh(z) * n) AS l FROM c"
        )
    };
    let (fast, general) = (loss(""), loss(GENERAL));
    matrix_products_of(0xc0_4e7, 90, "a.k = w.k", &fast, &general, None).await;
}

#[tokio::test]
async fn null_join_keys_match_under_each_kind_of_equality() {
    // `=` never matches a NULL key; IS NOT DISTINCT FROM matches NULL to
    // NULL. The contraction rule drops `=`'s NULL keys so that its last join
    // can be null-safe, and keeps them for IS NOT DISTINCT FROM.
    matrix_products(0x6e11, 60, "a.k = w.k", true).await;
    matrix_products(0x6e12, 60, "a.k IS NOT DISTINCT FROM w.k", true).await;
    // The condition written right side first.
    matrix_products(0x6e13, 60, "w.k = a.k", true).await;
}

#[tokio::test]
async fn a_join_on_two_keys_matches_the_general_rule() {
    let on = "a.k = w.k AND a.b = w.b";
    let loss = |filter: &str| {
        format!(
            "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON {on} {filter} \
             GROUP BY a.s, w.o) SELECT SUM(z * z) AS l FROM c"
        )
    };
    let (fast, general) = (loss(""), loss(GENERAL));
    let mut rng = Rng::new(0x2ce7);
    let (mut compared, mut failed) = (0, Vec::new());
    for case in 0..60 {
        let (ns, nk, nb, no) = (
            1 + rng.below(4) as usize,
            1 + rng.below(3) as usize,
            1 + rng.below(3) as usize,
            1 + rng.below(3) as usize,
        );
        let ctx = SessionContext::new();
        register(
            &ctx,
            "a",
            table(
                &mut rng,
                &[("s", ns), ("k", nk), ("b", nb)],
                &["val"],
                0.0,
                Some("b"),
            ),
        );
        register(
            &ctx,
            "w",
            table(
                &mut rng,
                &[("k", nk), ("b", nb), ("o", no)],
                &["val"],
                0.0,
                Some("b"),
            ),
        );
        let wrt = [ColumnRef::new("w", "val"), ColumnRef::new("a", "val")];
        match compare(&ctx, &fast, &general, &wrt, &[(&named("a"), &named("w"))]).await {
            Ok(true) => compared += 1,
            Ok(false) => {}
            Err(e) => failed.push(format!("case {case}: {e}")),
        }
    }
    assert!(
        failed.is_empty(),
        "{} of 60 disagree:\n{}",
        failed.len(),
        failed.join("\n")
    );
    assert!(compared >= 40, "only {compared} of 60 cases compared");
}

#[tokio::test]
async fn a_second_layer_through_tanh_matches_the_general_rule() {
    // The second layer's side is tanh of a saved aggregate: a side with a
    // projection, read from a saved relation. Both layers must take the
    // contraction rule.
    let loss = |filter1: &str, filter2: &str| {
        format!(
            "WITH h AS (SELECT x.s, w1.j, tanh(SUM(x.val * w1.val)) AS v FROM x JOIN w1 ON x.k = w1.k \
             {filter1} GROUP BY x.s, w1.j), y AS (SELECT h.s, w2.o, SUM(h.v * w2.val) AS z FROM h JOIN w2 \
             ON h.j = w2.j {filter2} GROUP BY h.s, w2.o) SELECT SUM(z * z) AS l FROM y"
        )
    };
    let fast = loss("", "");
    let general = loss(
        "WHERE x.k IS NULL OR x.k + w1.j >= 0",
        "WHERE h.j IS NULL OR h.j + w2.o >= 0",
    );
    let saved = |scans: &[String]| scans.len() == 1 && scans[0].ends_with("saved_0");
    let mut rng = Rng::new(0x7a17);
    let (mut compared, mut failed) = (0, Vec::new());
    for case in 0..60 {
        let ctx = SessionContext::new();
        let (n, k, j, o) = (
            1 + rng.below(4) as usize,
            1 + rng.below(3) as usize,
            1 + rng.below(3) as usize,
            1 + rng.below(3) as usize,
        );
        register(
            &ctx,
            "x",
            table(&mut rng, &[("s", n), ("k", k)], &["val"], 0.2, None),
        );
        register(
            &ctx,
            "w1",
            table(&mut rng, &[("k", k), ("j", j)], &["val"], 0.0, None),
        );
        register(
            &ctx,
            "w2",
            table(&mut rng, &[("j", j), ("o", o)], &["val"], 0.0, None),
        );
        let wrt = [ColumnRef::new("w1", "val"), ColumnRef::new("w2", "val")];
        let joins: [Join; 2] = [(&named("x"), &named("w1")), (&saved, &named("w2"))];
        match compare(&ctx, &fast, &general, &wrt, &joins).await {
            Ok(true) => compared += 1,
            Ok(false) => {}
            Err(e) => failed.push(format!("case {case}: {e}")),
        }
    }
    assert!(
        failed.is_empty(),
        "{} of 60 disagree:\n{}",
        failed.len(),
        failed.join("\n")
    );
    assert!(compared >= 50, "only {compared} of 60 cases compared");
}

#[tokio::test]
async fn a_factor_of_two_columns_respects_a_null_in_either() {
    // e_X = a.val + a.bias: where one is NULL the product is NULL, SUM skips
    // the row, and the other gets nothing although its partial (1) is not
    // NULL; only the guard on e_X sees it. Differentiated in both, the two
    // columns have the same partial, and must still be two gradients.
    let loss = |filter: &str| {
        format!(
            "WITH c AS (SELECT a.s, w.o, SUM((a.val + a.bias) * w.val) AS z FROM a JOIN w ON a.k = w.k \
             {filter} GROUP BY a.s, w.o) SELECT SUM(z * z) AS l FROM c"
        )
    };
    let (fast, general) = (loss(""), loss(GENERAL));
    for wrt in [
        vec![ColumnRef::new("a", "val")],
        vec![ColumnRef::new("a", "val"), ColumnRef::new("a", "bias")],
    ] {
        let mut rng = Rng::new(0xb1a5);
        let (mut compared, mut failed) = (0, Vec::new());
        for case in 0..60 {
            let (ns, nk, no) = (
                1 + rng.below(4) as usize,
                1 + rng.below(3) as usize,
                1 + rng.below(3) as usize,
            );
            let ctx = SessionContext::new();
            register(
                &ctx,
                "a",
                table(
                    &mut rng,
                    &[("s", ns), ("k", nk)],
                    &["val", "bias"],
                    0.0,
                    None,
                ),
            );
            register(
                &ctx,
                "w",
                table(&mut rng, &[("k", nk), ("o", no)], &["val"], 0.0, None),
            );
            match compare(&ctx, &fast, &general, &wrt, &[(&named("a"), &named("w"))]).await {
                Ok(true) => compared += 1,
                Ok(false) => {}
                Err(e) => failed.push(format!("case {case}: {e}")),
            }
        }
        assert!(
            failed.is_empty(),
            "{} of 60 disagree for {wrt:?}:\n{}",
            failed.len(),
            failed.join("\n")
        );
        assert!(
            compared >= 50,
            "only {compared} of 60 cases compared for {wrt:?}"
        );
    }
}

#[tokio::test]
async fn a_sum_of_a_sum_keeps_the_general_rule() {
    // SUM(a.val + w.val) is not a product: a NULL w.val must still stop a's
    // gradient, which only the general rule's guard does.
    let ctx = SessionContext::new();
    register(
        &ctx,
        "a",
        Table {
            keys: vec![("s", vec![Some(0), Some(0)]), ("k", vec![Some(0), Some(1)])],
            values: vec![("val", vec![Some(1.0), Some(2.0)])],
        },
    );
    register(
        &ctx,
        "w",
        Table {
            keys: vec![("k", vec![Some(0), Some(1)]), ("o", vec![Some(0), Some(0)])],
            values: vec![("val", vec![None, Some(3.0)])],
        },
    );
    let loss = "SELECT SUM(z) AS l FROM (SELECT a.s, w.o, SUM(a.val + w.val) AS z FROM a JOIN w \
                ON a.k = w.k GROUP BY a.s, w.o)";
    let wrt = [ColumnRef::new("a", "val")];
    let program = ad::grad(&ctx, loss, &wrt).await.unwrap();
    ddx_datafusion::engine::execute(&ctx, &program)
        .await
        .unwrap();
    assert!(rebuilds(&ctx, &program, &named("a"), &named("w")).await);
    let g = gradients(&ctx, &program, &wrt).await;
    // a(0, 0) meets w(0, 0), whose value is NULL: SUM skipped the row.
    assert_eq!(
        g[0],
        vec![
            (vec![Some(0), Some(0)], Some(0.0)),
            (vec![Some(0), Some(1)], Some(1.0))
        ]
    );
}

#[tokio::test]
async fn a_side_that_doubles_at_every_level_falls_back_in_time() {
    // Each level reads the one below twice, so inlining the side's factor
    // into the product would build 2^40 nodes. The contraction rule must give
    // up on it early, and the general rule computes each level once.
    let depth = 40;
    let mut ctes = vec!["a0 AS (SELECT s, k, val FROM a)".to_string()];
    for i in 1..=depth {
        ctes.push(format!(
            "a{i} AS (SELECT s, k, sin(val) + cos(val) AS val FROM a{})",
            i - 1
        ));
    }
    let loss = format!(
        "WITH {}, c AS (SELECT x.s, w.o, SUM(x.val * w.val) AS z FROM a{depth} x JOIN w \
         ON x.k = w.k GROUP BY x.s, w.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM c",
        ctes.join(", ")
    );
    let ctx = SessionContext::new();
    let mut rng = Rng::new(0xd0_b1e);
    register(
        &ctx,
        "a",
        table(&mut rng, &[("s", 3), ("k", 2)], &["val"], 0.0, None),
    );
    register(
        &ctx,
        "w",
        table(&mut rng, &[("k", 2), ("o", 2)], &["val"], 0.0, None),
    );
    let wrt = [ColumnRef::new("w", "val"), ColumnRef::new("a", "val")];
    let t = std::time::Instant::now();
    let program = ad::grad(&ctx, loss.as_str(), &wrt).await.unwrap();
    assert!(
        t.elapsed() < std::time::Duration::from_secs(30),
        "building took {:?}",
        t.elapsed()
    );
    ddx_datafusion::engine::execute(&ctx, &program)
        .await
        .unwrap();
}

/// Runs `cases` random cases of `loss` against `general` over the tables
/// `make` registers, differentiated in `wrt`, and asserts nearly all
/// compared and all agreed.
async fn cases_of(
    seed: u64,
    cases: usize,
    loss: &str,
    general: &str,
    wrt: &[ColumnRef],
    joins: &[Join<'_>],
    make: &dyn Fn(&mut Rng, &SessionContext),
) {
    let mut rng = Rng::new(seed);
    let (mut compared, mut failed) = (0, Vec::new());
    for case in 0..cases {
        let ctx = SessionContext::new();
        make(&mut rng, &ctx);
        match compare(&ctx, loss, general, wrt, joins).await {
            Ok(true) => compared += 1,
            Ok(false) => {}
            Err(e) => failed.push(format!("case {case}: {e}")),
        }
    }
    assert!(
        failed.is_empty(),
        "{} of {cases} disagree on `{loss}`:\n{}",
        failed.len(),
        failed.join("\n")
    );
    assert!(
        compared * 10 >= cases * 7,
        "only {compared} of {cases} cases compared on `{loss}`"
    );
}

#[tokio::test]
async fn a_divisor_constant_within_a_group_matches_the_general_rule() {
    // Attention's output, SUM(w / z * v) over weights, values and the
    // softmax's normalizer: z is joined on a grouping key, so the sides are
    // {p, z} (one row per score) and {v}, whatever order the joins are
    // written in. A table no factor reads (m) goes with the side its join
    // ties it to, and its repeated keys repeat rows on that side. (Written
    // `p JOIN z … JOIN v`, no step of either rule joins p to v directly, so
    // that spelling has no check of which rule ran.)
    let froms = [
        "FROM p JOIN v ON p.j = v.t JOIN z ON p.i = z.i",
        "FROM p JOIN z ON p.i = z.i JOIN v ON p.j = v.t",
        "FROM v JOIN p ON p.j = v.t JOIN z ON z.i = p.i",
        "FROM p JOIN v ON p.j = v.t JOIN z ON p.i = z.i JOIN m ON m.i = p.i",
    ];
    let (p, v) = (named("p"), named("v"));
    let checked: [Join; 1] = [(&p, &v)];
    for (n, from) in froms.into_iter().enumerate() {
        let loss = |filter: &str| {
            format!(
                "WITH o AS (SELECT p.i, v.d, SUM(p.val / z.val * v.val) AS x {from} {filter} \
                 GROUP BY p.i, v.d) SELECT SUM(tanh(x) * tanh(x)) AS l FROM o"
            )
        };
        let wrt = [
            ColumnRef::new("p", "val"),
            ColumnRef::new("v", "val"),
            ColumnRef::new("z", "val"),
        ];
        let make = |rng: &mut Rng, ctx: &SessionContext| {
            let (ni, nj, nd) = (
                1 + rng.below(4) as usize,
                1 + rng.below(3) as usize,
                1 + rng.below(3) as usize,
            );
            register(
                ctx,
                "p",
                table(rng, &[("i", ni), ("j", nj)], &["val"], 0.0, None),
            );
            register(
                ctx,
                "v",
                table(rng, &[("t", nj), ("d", nd)], &["val"], 0.0, None),
            );
            register(ctx, "z", table(rng, &[("i", ni)], &["val"], 0.0, None));
            register(ctx, "m", table(rng, &[("i", ni)], &[], 0.3, None));
        };
        cases_of(
            0x0d10 + n as u64,
            60,
            &loss(""),
            &loss("WHERE v.t IS NULL OR v.t + z.i >= 0"),
            &wrt,
            if n == 1 { &[] } else { &checked },
            &make,
        )
        .await;
    }
}

#[tokio::test]
async fn a_saved_normalizer_matches_the_general_rule() {
    // A normalizer that is itself saved, n.z = SUM(p.val) by i: where each
    // group joins one row of it (grouped by i, or n has no keys at all),
    // its cotangent comes from the output and the output's cotangent; where
    // a group joins several (grouped by j and d), from the sum over the
    // region.
    let cases = [
        (
            "GROUP BY i",
            "JOIN n ON p.i = n.i",
            "GROUP BY p.i, v.d",
            true,
        ),
        (
            "GROUP BY i",
            "JOIN n ON n.i = p.i",
            "GROUP BY p.j, v.d",
            true,
        ),
        ("", "CROSS JOIN n", "GROUP BY p.i, v.d", false),
    ];
    let (p, v) = (named("p"), named("v"));
    let checked: [Join; 1] = [(&p, &v)];
    for (c, (norm, join, group, check)) in cases.into_iter().enumerate() {
        let keys = if norm.is_empty() { "" } else { "i, " };
        let loss = |filter: &str| {
            format!(
                "WITH n AS (SELECT {keys}SUM(val) AS z FROM p {norm}), \
                 o AS (SELECT v.d, SUM(p.val / n.z * v.val) AS x \
                 FROM p JOIN v ON p.j = v.t {join} {filter} {group}) \
                 SELECT SUM(tanh(x) * tanh(x)) AS l FROM o"
            )
        };
        let wrt = [ColumnRef::new("p", "val"), ColumnRef::new("v", "val")];
        let make = |rng: &mut Rng, ctx: &SessionContext| {
            let (ni, nj, nd) = (
                1 + rng.below(4) as usize,
                1 + rng.below(3) as usize,
                1 + rng.below(3) as usize,
            );
            register(
                ctx,
                "p",
                table(rng, &[("i", ni), ("j", nj)], &["val"], 0.0, None),
            );
            register(
                ctx,
                "v",
                table(rng, &[("t", nj), ("d", nd)], &["val"], 0.0, None),
            );
        };
        cases_of(
            0x2510 + c as u64,
            60,
            &loss(""),
            &loss("WHERE v.t IS NULL OR v.t + p.i >= 0"),
            &wrt,
            if check { &checked } else { &[] },
            &make,
        )
        .await;
    }
}

#[tokio::test]
async fn causal_attention_matches_the_general_rule() {
    // The query of issue #153: a causal mask on the scores, a softmax
    // through a MAX window, and the normalizer joined on a grouping key.
    // Neither the scores' join nor the output's may be rebuilt.
    let loss = |scores: &str, output: &str| {
        format!(
            "WITH scores AS (SELECT q.t AS i, k.t AS j, SUM(q.val * k.val) / 8.0 AS s \
             FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t {scores} GROUP BY q.t, k.t), \
             weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores), \
             norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i), \
             attention AS (SELECT w.i AS t, v.d, SUM(w.w / n.z * v.val) AS x \
             FROM weights w JOIN v ON w.j = v.t JOIN norm n ON w.i = n.i {output} GROUP BY w.i, v.d) \
             SELECT SUM(x * x) / 2 AS l FROM attention"
        )
    };
    let wrt = [
        ColumnRef::new("q", "val"),
        ColumnRef::new("k", "val"),
        ColumnRef::new("v", "val"),
    ];
    let make = |rng: &mut Rng, ctx: &SessionContext| {
        let (nt, nd) = (1 + rng.below(4) as usize, 1 + rng.below(3) as usize);
        for name in ["q", "k", "v"] {
            register(
                ctx,
                name,
                table(rng, &[("t", nt), ("d", nd)], &["val"], 0.0, None),
            );
        }
    };
    cases_of(
        0xa77e,
        60,
        &loss("", ""),
        &loss(
            "AND (q.d IS NULL OR q.d + k.t >= 0)",
            "WHERE v.t IS NULL OR v.t + w.i >= 0",
        ),
        &wrt,
        &[(&named("q"), &named("k")), (&scores, &named("v"))],
        &make,
    )
    .await;
}

/// Batched, multi-head causal attention: each sequence `s` and head `h`
/// attends within itself.
fn batched_attention(scores: &str, output: &str) -> String {
    format!(
        "WITH scores AS (SELECT q.s, q.h, q.t AS i, k.t AS j, SUM(q.val * k.val) / 2.0 AS e \
         FROM q JOIN k ON q.s = k.s AND q.h = k.h AND q.d = k.d WHERE q.t >= k.t {scores} \
         GROUP BY q.s, q.h, q.t, k.t), \
         weights AS (SELECT s, h, i, j, exp(e - MAX(e) OVER (PARTITION BY s, h, i)) AS w \
         FROM scores), \
         norm AS (SELECT s, h, i, SUM(w) AS z FROM weights GROUP BY s, h, i), \
         attention AS (SELECT a.s, a.i AS t, a.h, v.d, SUM(a.w / n.z * v.val) AS x \
         FROM weights a JOIN v ON a.s = v.s AND a.h = v.h AND a.j = v.t \
         JOIN norm n ON a.s = n.s AND a.h = n.h AND a.i = n.i {output} \
         GROUP BY a.s, a.i, a.h, v.d) \
         SELECT SUM(x * x) / 2 AS l FROM attention"
    )
}

#[tokio::test]
async fn batched_attention_matches_the_general_rule() {
    // The output's join ties v's sequence and head to the cotangent's only
    // through the weights, so t joins the cotangent on keys the query
    // never joined v to it on (issue #217).
    let wrt = [
        ColumnRef::new("q", "val"),
        ColumnRef::new("k", "val"),
        ColumnRef::new("v", "val"),
    ];
    let make = |rng: &mut Rng, ctx: &SessionContext| {
        let (ns, nh) = (1 + rng.below(3) as usize, 1 + rng.below(2) as usize);
        let (nt, nd) = (1 + rng.below(3) as usize, 1 + rng.below(2) as usize);
        for name in ["q", "k", "v"] {
            let null = (name == "v").then_some("s");
            let dims = [("s", ns), ("h", nh), ("t", nt), ("d", nd)];
            register(ctx, name, table(rng, &dims, &["val"], 0.0, null));
        }
    };
    cases_of(
        0xba7c,
        60,
        &batched_attention("", ""),
        &batched_attention(
            "AND (q.d IS NULL OR q.d + k.t >= 0)",
            "WHERE v.t IS NULL OR v.t + a.i >= 0",
        ),
        &wrt,
        &[(&named("q"), &named("k")), (&scores, &named("v"))],
        &make,
    )
    .await;
}

#[tokio::test]
async fn batched_attention_pairs_no_two_sequences_or_heads() {
    // Without the keys the join makes equal, the contraction of v with the
    // output's cotangent has (S·H·T)² rows: 2,304 here (issue #217).
    let (ns, nh, nt, nd) = (3, 2, 8, 4);
    let ctx = SessionContext::new();
    let mut rng = Rng::new(0x217);
    for name in ["q", "k", "v"] {
        let dims = [("s", ns), ("h", nh), ("t", nt), ("d", nd)];
        register(&ctx, name, table(&mut rng, &dims, &["val"], 0.0, None));
    }
    let wrt = [ColumnRef::new("q", "val")];
    let program = ad::grad(&ctx, &batched_attention("", ""), &wrt)
        .await
        .unwrap();
    let mut contractions = 0;
    for step in program.steps() {
        ddx_datafusion::engine::run_step(&ctx, step).await.unwrap();
        if step.name.as_str().contains("contraction_") {
            let rows = ctx.table(step.name.as_str()).await.unwrap().count().await;
            let rows = rows.unwrap();
            assert!(
                rows <= ns * nh * nt * nt,
                "{} has {rows} rows",
                step.name.as_str()
            );
            contractions += 1;
        }
    }
    assert!(contractions > 0, "the contraction rule did not apply");
}

#[tokio::test]
async fn batched_attention_joined_through_v_matches_the_general_rule() {
    // The normalizer's sequence and head are joined through v, as DuckDB's
    // plan joins them, so the side `weights JOIN norm` is rebuilt with keys
    // its own conditions don't state (issue #222): `=`, which drops v's
    // NULL sequences, and IS NOT DISTINCT FROM, which joins them. Under
    // IS NOT DISTINCT FROM every input may have NULL sequences, and the
    // scores join them too, so NULL sequences reach the side rebuilt.
    let wrt = [
        ColumnRef::new("q", "val"),
        ColumnRef::new("k", "val"),
        ColumnRef::new("v", "val"),
    ];
    for (seed, eq) in [(0x222a, "="), (0x222b, "IS NOT DISTINCT FROM")] {
        let make = |rng: &mut Rng, ctx: &SessionContext| {
            let (ns, nh) = (1 + rng.below(3) as usize, 1 + rng.below(2) as usize);
            let (nt, nd) = (1 + rng.below(3) as usize, 1 + rng.below(2) as usize);
            for name in ["q", "k", "v"] {
                let null = (eq != "=" || name != "k").then_some("s");
                let dims = [("s", ns), ("h", nh), ("t", nt), ("d", nd)];
                register(ctx, name, table(rng, &dims, &["val"], 0.0, null));
            }
        };
        let through_v = |q: String| {
            q.replace("JOIN k ON q.s = k.s", &format!("JOIN k ON (q.s {eq} k.s)"))
                .replace("JOIN v ON a.s = v.s", &format!("JOIN v ON (a.s {eq} v.s)"))
                .replace(
                    "JOIN norm n ON a.s = n.s AND a.h = n.h",
                    &format!("JOIN norm n ON (v.s {eq} n.s) AND v.h = n.h"),
                )
        };
        cases_of(
            seed,
            25,
            &through_v(batched_attention("", "")),
            &through_v(batched_attention(
                "AND (q.d IS NULL OR q.d + k.t >= 0)",
                "WHERE v.t IS NULL OR v.t + a.i >= 0",
            )),
            &wrt,
            &[(&named("q"), &named("k")), (&scores, &named("v"))],
            &make,
        )
        .await;
    }
}

#[tokio::test]
async fn batched_attention_s_normalizer_joins_each_score_once() {
    // DuckDB's plan joins the normalizer on the sequence and head through
    // v (`v.s = n.s`), as this query does. Rebuilt without v, the side
    // `weights JOIN norm` joined each score to every sequence's and head's
    // normalizer: S·H times its rows, 1,296 here (issue #222). With one
    // channel, no join the query needs has more than S·H·T² rows.
    let (ns, nh, nt, nd) = (3, 2, 8, 1);
    let ctx = SessionContext::new();
    let mut rng = Rng::new(0x222);
    for name in ["q", "k", "v"] {
        let dims = [("s", ns), ("h", nh), ("t", nt), ("d", nd)];
        register(&ctx, name, table(&mut rng, &dims, &["val"], 0.0, None));
    }
    let loss = batched_attention("", "").replace(
        "JOIN norm n ON a.s = n.s AND a.h = n.h",
        "JOIN norm n ON v.s = n.s AND v.h = n.h",
    );
    let wrt = [
        ColumnRef::new("q", "val"),
        ColumnRef::new("k", "val"),
        ColumnRef::new("v", "val"),
    ];
    let program = ad::grad(&ctx, &loss, &wrt).await.unwrap();
    assert!(
        !rebuilds(&ctx, &program, &scores, &named("v")).await,
        "the contraction rule did not apply"
    );
    let mut most = (0, String::new());
    for step in program.steps() {
        let plan = ddx_datafusion::engine::logical_plan(&ctx, &step.plan)
            .await
            .unwrap();
        let df = ctx.execute_logical_plan(plan).await.unwrap();
        let task = Arc::new(df.task_ctx());
        let physical = df.create_physical_plan().await.unwrap();
        datafusion::physical_plan::collect(physical.clone(), task)
            .await
            .unwrap();
        let rows = largest_join(&physical);
        if rows > most.0 {
            most = (rows, step.name.as_str().to_string());
        }
        ddx_datafusion::engine::run_step(&ctx, step).await.unwrap();
    }
    assert!(
        most.0 <= ns * nh * nt * nt,
        "a join in {} has {} rows",
        most.1,
        most.0
    );
}

/// The most rows any join in `plan` returned when it ran.
fn largest_join(plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>) -> usize {
    let own = if plan.name().contains("Join") {
        plan.metrics().and_then(|m| m.output_rows()).unwrap_or(0)
    } else {
        0
    };
    plan.children()
        .into_iter()
        .map(largest_join)
        .fold(own, usize::max)
}

#[tokio::test]
async fn a_side_that_does_not_move_is_still_checked_for_a_total_order() {
    // w is constant here, but t recomputes it: a LIMIT inside it may keep
    // other rows on a second run, so the contraction rule refuses it as the
    // general rule does.
    let ctx = SessionContext::new();
    let mut rng = Rng::new(0x1171);
    register(
        &ctx,
        "a",
        table(&mut rng, &[("s", 3), ("k", 3)], &["val"], 0.0, None),
    );
    register(
        &ctx,
        "w",
        table(&mut rng, &[("k", 3), ("o", 3)], &["val"], 0.0, None),
    );
    let loss = "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a \
                JOIN (SELECT * FROM w LIMIT 4) w ON a.k = w.k GROUP BY a.s, w.o) \
                SELECT SUM(z * z) AS l FROM c";
    let refused = ad::grad(&ctx, loss, &[ColumnRef::new("a", "val")]).await;
    let why = refused
        .expect_err("a LIMIT in recomputed constant data")
        .to_string();
    assert!(why.contains("LIMIT"), "{why}");
}

#[tokio::test]
async fn a_side_its_conditions_leave_in_pieces_takes_the_general_rule() {
    // The factors put p and z on one side, but no condition joins them: each
    // joins v. Rebuilt alone, that side would be p × z, a cross join the
    // query never formed, so the general rule applies. Its gradient still
    // matches, and its steps rebuild the join of p and v.
    let loss = |filter: &str| {
        format!(
            "WITH o AS (SELECT p.i, v.d, SUM(p.val / z.val * v.val) AS x FROM p JOIN v ON p.j = v.t \
             JOIN z ON v.u = z.i {filter} GROUP BY p.i, v.d) SELECT SUM(tanh(x) * tanh(x)) AS l FROM o"
        )
    };
    let wrt = [ColumnRef::new("p", "val"), ColumnRef::new("z", "val")];
    let mut rng = Rng::new(0xc055);
    for _ in 0..10 {
        let ctx = SessionContext::new();
        register(
            &ctx,
            "p",
            table(&mut rng, &[("i", 3), ("j", 2)], &["val"], 0.0, None),
        );
        register(
            &ctx,
            "v",
            table(
                &mut rng,
                &[("t", 2), ("u", 2), ("d", 2)],
                &["val"],
                0.0,
                None,
            ),
        );
        register(&ctx, "z", table(&mut rng, &[("i", 2)], &["val"], 0.0, None));
        let program = ad::grad(&ctx, &loss(""), &wrt).await.unwrap();
        ddx_datafusion::engine::execute(&ctx, &program)
            .await
            .unwrap();
        assert!(rebuilds(&ctx, &program, &named("p"), &named("v")).await);
        let general = loss("WHERE v.t IS NULL OR v.t + z.i >= 0");
        let slow = ad::grad(&ctx, &general, &wrt).await.unwrap();
        ddx_datafusion::engine::execute(&ctx, &slow).await.unwrap();
        let (a, b) = (
            gradients(&ctx, &program, &wrt).await,
            gradients(&ctx, &slow, &wrt).await,
        );
        agree(&a, &b).unwrap();
    }
}
