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
//! and keeps every row (the keys it reads are never NULL or negative), sends
//! the same loss down the general rule: the two are compared on random
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
use ddx_datafusion::ad::{self, BackwardProgram, ColumnRef};

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }
    /// NULL sometimes, NaN or ±∞ rarely, otherwise a multiple of 1/8 in
    /// [-2, 2].
    fn value(&mut self) -> Option<f64> {
        if self.chance(0.12) {
            None
        } else if self.chance(0.03) {
            Some(f64::NAN)
        } else if self.chance(0.02) {
            Some(if self.chance(0.5) {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            })
        } else {
            Some((self.below(33) as f64 - 16.0) / 8.0)
        }
    }
}

/// A table of key columns and value columns.
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
        if !rng.chance(0.8) {
            continue;
        }
        for _ in 0..if rng.chance(dup) { 2 } else { 1 } {
            let mut rest = cell;
            for (i, &(name, extent)) in keys.iter().enumerate().rev() {
                let k = (rest % extent) as i64;
                rest /= extent;
                let null = null_key == Some(name) && rng.chance(0.1);
                t.keys[i].1.push((!null).then_some(k));
            }
            for v in t.values.iter_mut() {
                v.1.push(rng.value());
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

/// Does any gradient step join a scan whose table `left` accepts directly
/// with one `right` accepts, that is, rebuild the forward join? Runs the
/// steps again first: `ad::run` releases the tables a step's plan reads.
async fn rebuilds(
    ctx: &SessionContext,
    program: &BackwardProgram,
    left: &dyn Fn(&str) -> bool,
    right: &dyn Fn(&str) -> bool,
) -> bool {
    for step in program.steps() {
        ad::run_step(ctx, step).await.unwrap();
    }
    let mut found = false;
    for step in &program.backward_steps {
        let plan = ad::logical_plan(ctx, &step.plan).await.unwrap();
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
                let one =
                    |names: &[String], f: &dyn Fn(&str) -> bool| names.len() == 1 && f(&names[0]);
                if (one(&l, left) && one(&r, right)) || (one(&l, right) && one(&r, left)) {
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

/// `(the first two key columns, a gradient)` rows.
type Rows = Vec<(Option<i64>, Option<i64>, Option<f64>)>;

/// Each gradient of each `wrt` column, its rows sorted. A table's columns
/// that no gradient is taken of are dims too; the first two are enough to
/// tell rows apart in these tests, as rows are compared as multisets.
async fn gradients(
    ctx: &SessionContext,
    program: &BackwardProgram,
    wrt: &[ColumnRef],
) -> Vec<Rows> {
    let mut out = Vec::new();
    for g in &program.gradients {
        for col in g
            .columns
            .iter()
            .filter(|c| wrt.iter().any(|w| w.column.eq_ignore_ascii_case(c)))
        {
            let sql = format!(
                "SELECT \"{}\", \"{}\", \"{col}\" FROM \"{}\"",
                g.columns[0], g.columns[1], g.step
            );
            let mut rows = Vec::new();
            for b in ctx.sql(&sql).await.unwrap().collect().await.unwrap() {
                let k0 = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                let k1 = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                let v = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
                for i in 0..b.num_rows() {
                    let get = |a: &Int64Array| (!a.is_null(i)).then(|| a.value(i));
                    rows.push((get(k0), get(k1), (!v.is_null(i)).then(|| v.value(i))));
                }
            }
            rows.sort_by(|x, y| {
                (x.0, x.1)
                    .cmp(&(y.0, y.1))
                    .then(x.2.map(f64::to_bits).cmp(&y.2.map(f64::to_bits)))
            });
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
            let ok = (x.0, x.1) == (y.0, y.1)
                && match (x.2, y.2) {
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

/// Two predicates on table names: a join the general rule rebuilds and the
/// contraction rule must not.
type Join<'p> = (&'p dyn Fn(&str) -> bool, &'p dyn Fn(&str) -> bool);

fn named(name: &'static str) -> impl Fn(&str) -> bool {
    move |t: &str| t == name
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
    if ad::run(ctx, &fast).await.is_err() {
        assert!(
            ad::run(ctx, &slow).await.is_err(),
            "only the contraction rule's program was refused"
        );
        return Ok(false);
    }
    ad::run(ctx, &slow).await.unwrap();
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

/// A matrix product's loss, joined `on`, and the same loss with a filter
/// across the sides that keeps every row.
fn matmul(on: &str) -> (String, String) {
    let loss = |filter: &str| {
        format!(
            "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON {on} {filter} \
             GROUP BY a.s, w.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM c"
        )
    };
    (loss(""), loss("WHERE a.s + w.o >= 0"))
}

/// Runs `cases` random matrix products joined `on`, with join keys NULL
/// sometimes if `null_keys`, and asserts nearly all compared and all agreed.
async fn matrix_products(seed: u64, cases: usize, on: &str, null_keys: bool) {
    let (loss, general) = matmul(on);
    let mut rng = Rng(seed);
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
        let null_k = null_keys.then_some("k");
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
        match compare(&ctx, &loss, &general, &wrt, &[(&named("a"), &named("w"))]).await {
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
    let (fast, general) = (loss(""), loss("WHERE a.s + w.o >= 0"));
    let mut rng = Rng(0x2ce7);
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
    let general = loss("WHERE x.s + w1.j >= 0", "WHERE h.s + w2.o >= 0");
    let saved = |t: &str| t.ends_with("saved_0");
    let mut rng = Rng(0x7a17);
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
    let (fast, general) = (loss(""), loss("WHERE a.s + w.o >= 0"));
    for wrt in [
        vec![ColumnRef::new("a", "val")],
        vec![ColumnRef::new("a", "val"), ColumnRef::new("a", "bias")],
    ] {
        let mut rng = Rng(0xb1a5);
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
    ad::run(&ctx, &program).await.unwrap();
    assert!(rebuilds(&ctx, &program, &named("a"), &named("w")).await);
    let g = gradients(&ctx, &program, &wrt).await;
    // a(0, 0) meets w(0, 0), whose value is NULL: SUM skipped the row.
    assert_eq!(
        g[0],
        vec![(Some(0), Some(0), Some(0.0)), (Some(0), Some(1), Some(1.0))]
    );
}
