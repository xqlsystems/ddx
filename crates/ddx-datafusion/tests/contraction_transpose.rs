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
//! and keeps every row, sends the same loss down the general rule: the two
//! are compared on random tables with NULLs, NaNs, infinities, missing rows,
//! and repeated keys in constant data. Each case also checks which rule ran,
//! by whether a gradient step joins the two inputs.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Float64Array, Int64Array};
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
}

type Rows = (Vec<i64>, Vec<i64>, Vec<Option<f64>>);

/// Rows over `0..e0 × 0..e1`: each present with probability 0.8, repeated
/// with probability `dup`; values NULL, NaN or infinite sometimes, otherwise
/// multiples of 1/8 in [-2, 2].
fn rows(rng: &mut Rng, e0: usize, e1: usize, dup: f64) -> Rows {
    let (mut c0, mut c1, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..e0 {
        for j in 0..e1 {
            if !rng.chance(0.8) {
                continue;
            }
            for _ in 0..if rng.chance(dup) { 2 } else { 1 } {
                c0.push(i as i64);
                c1.push(j as i64);
                v.push(if rng.chance(0.12) {
                    None
                } else if rng.chance(0.03) {
                    Some(f64::NAN)
                } else if rng.chance(0.02) {
                    Some(if rng.chance(0.5) {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    })
                } else {
                    Some((rng.below(33) as f64 - 16.0) / 8.0)
                });
            }
        }
    }
    (c0, c1, v)
}

fn register(ctx: &SessionContext, name: &str, cols: (&str, &str), data: Rows) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(cols.0, DataType::Int64, false),
        Field::new(cols.1, DataType::Int64, false),
        Field::new("val", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(data.0)),
            Arc::new(Int64Array::from(data.1)),
            Arc::new(Float64Array::from(data.2)),
        ],
    )
    .unwrap();
    ctx.register_batch(name, batch).unwrap();
}

/// Does any gradient step join a scan of `left` directly with a scan of
/// `right`, that is, rebuild the forward join? Runs the steps again first:
/// `ad::run` releases the tables a step's plan reads.
async fn rebuilds(
    ctx: &SessionContext,
    program: &BackwardProgram,
    left: &str,
    right: &str,
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
                if (l == [left] && r == [right]) || (l == [right] && r == [left]) {
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

/// Each gradient table's rows, sorted, as `(first two dims, gradient of val)`.
async fn gradients(
    ctx: &SessionContext,
    program: &BackwardProgram,
) -> Vec<Vec<(i64, i64, Option<f64>)>> {
    let mut out = Vec::new();
    for g in &program.gradients {
        // The first two dims, and the gradient of `val`: a table's other
        // columns that no gradient is taken of are dims too.
        let sql = format!(
            "SELECT \"{}\", \"{}\", val FROM \"{}\"",
            g.columns[0], g.columns[1], g.step
        );
        let mut rows = Vec::new();
        for b in ctx.sql(&sql).await.unwrap().collect().await.unwrap() {
            let k0 = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let k1 = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            let v = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
            for i in 0..b.num_rows() {
                rows.push((
                    k0.value(i),
                    k1.value(i),
                    (!v.is_null(i)).then(|| v.value(i)),
                ));
            }
        }
        rows.sort_by(|x, y| {
            (x.0, x.1)
                .cmp(&(y.0, y.1))
                .then(x.2.map(f64::to_bits).cmp(&y.2.map(f64::to_bits)))
        });
        out.push(rows);
    }
    out
}

/// The same gradients: same rows, same NULLs, NaN where NaN, the same
/// infinities, and values within rounding (the contraction rule sums in a
/// different order).
fn agree(
    a: &[Vec<(i64, i64, Option<f64>)>],
    b: &[Vec<(i64, i64, Option<f64>)>],
) -> Result<(), String> {
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

/// Runs `loss` and `general` (the same loss, written so the general rule
/// applies) and compares their gradients. `Ok(false)` if ddx refused the
/// tables (repeated keys in a table it differentiates).
async fn compare(
    ctx: &SessionContext,
    loss: &str,
    general: &str,
    wrt: &[ColumnRef],
    pair: (&str, &str),
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
    assert!(
        !rebuilds(ctx, &fast, pair.0, pair.1).await,
        "the contraction rule did not apply"
    );
    assert!(
        rebuilds(ctx, &slow, pair.0, pair.1).await,
        "the comparison did not take the general rule"
    );
    let (a, b) = (gradients(ctx, &fast).await, gradients(ctx, &slow).await);
    agree(&a, &b).map(|_| true)
}

const MATMUL: &str =
    "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k \
                      GROUP BY a.s, w.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM c";
// The same loss, with a filter that reads both sides and keeps every row
// (keys are never negative), which the contraction rule declines.
const MATMUL_GENERAL: &str = "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k \
                              WHERE a.s + w.o >= 0 GROUP BY a.s, w.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM c";

#[tokio::test]
async fn a_matrix_product_matches_the_general_rule_on_awkward_tables() {
    let mut rng = Rng(0x5eed_c0de);
    let (mut compared, mut failed) = (0, Vec::new());
    for case in 0..150 {
        // In a third of the cases only w is differentiated, so a is constant
        // data and may repeat keys.
        let both = case % 3 != 2;
        let (ns, nk, no) = (
            1 + rng.below(5) as usize,
            1 + rng.below(4) as usize,
            1 + rng.below(4) as usize,
        );
        let ctx = SessionContext::new();
        register(
            &ctx,
            "a",
            ("s", "k"),
            rows(&mut rng, ns, nk, if both { 0.0 } else { 0.3 }),
        );
        register(&ctx, "w", ("k", "o"), rows(&mut rng, nk, no, 0.0));
        let wrt = if both {
            vec![ColumnRef::new("w", "val"), ColumnRef::new("a", "val")]
        } else {
            vec![ColumnRef::new("w", "val")]
        };
        match compare(&ctx, MATMUL, MATMUL_GENERAL, &wrt, ("a", "w")).await {
            Ok(true) => compared += 1,
            Ok(false) => {}
            Err(e) => failed.push(format!("case {case} ({ns}×{nk}×{no}): {e}")),
        }
    }
    assert!(
        failed.is_empty(),
        "{} of 150 disagree:\n{}",
        failed.len(),
        failed.join("\n")
    );
    assert!(compared >= 140, "only {compared} cases compared");
}

#[tokio::test]
async fn a_second_layer_through_tanh_matches_the_general_rule() {
    // The second layer's side is tanh of a saved aggregate: a side with a
    // projection, read from a saved relation.
    let loss =
        "WITH h AS (SELECT x.s, w1.j, tanh(SUM(x.val * w1.val)) AS v FROM x JOIN w1 ON x.k = w1.k \
                GROUP BY x.s, w1.j), y AS (SELECT h.s, w2.o, SUM(h.v * w2.val) AS z FROM h JOIN w2 \
                ON h.j = w2.j GROUP BY h.s, w2.o) SELECT SUM(z * z) AS l FROM y";
    let general = "WITH h AS (SELECT x.s, w1.j, tanh(SUM(x.val * w1.val)) AS v FROM x JOIN w1 ON x.k = w1.k \
                   WHERE x.s + w1.j >= 0 GROUP BY x.s, w1.j), y AS (SELECT h.s, w2.o, SUM(h.v * w2.val) AS z \
                   FROM h JOIN w2 ON h.j = w2.j WHERE h.s + w2.o >= 0 GROUP BY h.s, w2.o) SELECT SUM(z * z) AS l FROM y";
    let mut rng = Rng(0x7a17);
    let mut failed = Vec::new();
    for case in 0..60 {
        let ctx = SessionContext::new();
        let (n, k, j, o) = (
            1 + rng.below(4) as usize,
            1 + rng.below(3) as usize,
            1 + rng.below(3) as usize,
            1 + rng.below(3) as usize,
        );
        register(&ctx, "x", ("s", "k"), rows(&mut rng, n, k, 0.2));
        register(&ctx, "w1", ("k", "j"), rows(&mut rng, k, j, 0.0));
        register(&ctx, "w2", ("j", "o"), rows(&mut rng, j, o, 0.0));
        let wrt = [ColumnRef::new("w1", "val"), ColumnRef::new("w2", "val")];
        if let Err(e) = compare(&ctx, loss, general, &wrt, ("x", "w1")).await {
            failed.push(format!("case {case}: {e}"));
        }
    }
    assert!(
        failed.is_empty(),
        "{} of 60 disagree:\n{}",
        failed.len(),
        failed.join("\n")
    );
}

#[tokio::test]
async fn a_factor_of_two_columns_respects_a_null_in_either() {
    // e_X = a.val + a.bias, differentiated in a.val only: where a.bias is
    // NULL the product is NULL, SUM skips the row, and a.val gets nothing,
    // although its partial (1) is not NULL. Only the guard on e_X sees it.
    let loss = "WITH c AS (SELECT a.s, w.o, SUM((a.val + a.bias) * w.val) AS z FROM a JOIN w ON a.k = w.k \
                GROUP BY a.s, w.o) SELECT SUM(z * z) AS l FROM c";
    let general = "WITH c AS (SELECT a.s, w.o, SUM((a.val + a.bias) * w.val) AS z FROM a JOIN w ON a.k = w.k \
                   WHERE a.s + w.o >= 0 GROUP BY a.s, w.o) SELECT SUM(z * z) AS l FROM c";
    let mut rng = Rng(0xb1a5);
    let mut failed = Vec::new();
    for case in 0..60 {
        let (ns, nk, no) = (
            1 + rng.below(4) as usize,
            1 + rng.below(3) as usize,
            1 + rng.below(3) as usize,
        );
        let (s, k, val) = rows(&mut rng, ns, nk, 0.0);
        let bias: Vec<Option<f64>> = (0..s.len())
            .map(|_| (!rng.chance(0.3)).then(|| (rng.below(9) as f64 - 4.0) / 4.0))
            .collect();
        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Int64, false),
            Field::new("k", DataType::Int64, false),
            Field::new("val", DataType::Float64, true),
            Field::new("bias", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(s)),
                Arc::new(Int64Array::from(k)),
                Arc::new(Float64Array::from(val)),
                Arc::new(Float64Array::from(bias)),
            ],
        )
        .unwrap();
        ctx.register_batch("a", batch).unwrap();
        register(&ctx, "w", ("k", "o"), rows(&mut rng, nk, no, 0.0));
        let wrt = [ColumnRef::new("a", "val")];
        if let Err(e) = compare(&ctx, loss, general, &wrt, ("a", "w")).await {
            failed.push(format!("case {case}: {e}"));
        }
    }
    assert!(
        failed.is_empty(),
        "{} of 60 disagree:\n{}",
        failed.len(),
        failed.join("\n")
    );
}

#[tokio::test]
async fn a_sum_of_a_sum_keeps_the_general_rule() {
    // SUM(a.val + w.val) is not a product: a NULL w.val must still stop a's
    // gradient, which only the general rule's guard does.
    let ctx = SessionContext::new();
    register(
        &ctx,
        "a",
        ("s", "k"),
        (vec![0, 0], vec![0, 1], vec![Some(1.0), Some(2.0)]),
    );
    register(
        &ctx,
        "w",
        ("k", "o"),
        (vec![0, 1], vec![0, 0], vec![None, Some(3.0)]),
    );
    let loss = "SELECT SUM(z) AS l FROM (SELECT a.s, w.o, SUM(a.val + w.val) AS z FROM a JOIN w \
                ON a.k = w.k GROUP BY a.s, w.o)";
    let program = ad::grad(&ctx, loss, &[ColumnRef::new("a", "val")])
        .await
        .unwrap();
    ad::run(&ctx, &program).await.unwrap();
    assert!(rebuilds(&ctx, &program, "a", "w").await);
    let g = gradients(&ctx, &program).await;
    // a(0, 0) meets w(0, 0), whose value is NULL: SUM skipped the row.
    assert_eq!(g[0], vec![(0, 0, Some(0.0)), (0, 1, Some(1.0))]);
}
