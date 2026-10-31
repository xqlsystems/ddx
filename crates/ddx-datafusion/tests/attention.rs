// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Single-head attention, the fixture of `attention_ad_spike.py`, as one
//! query: Q/K/V projections, QKᵀ scaled, softmax over the key axis,
//! A·V, and a squared-error loss.

mod common;

use common::ad::{check_gradients, Table};
use datafusion::prelude::SessionContext;
use ddx_ad::ColumnRef;

fn value(seed: usize) -> f64 {
    ((seed * 7919 + 13) % 1000) as f64 / 1000.0 - 0.5
}

fn matrix(name: &'static str, dims: [&'static str; 2], shape: [usize; 2], seed: usize) -> Table {
    let mut rows = Vec::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            rows.push(vec![i as f64, j as f64, value(seed + i * shape[1] + j)]);
        }
    }
    Table {
        name,
        columns: vec![(dims[0], "BIGINT"), (dims[1], "BIGINT"), ("val", "DOUBLE")],
        rows,
    }
}

/// 3 positions, d_model 3, d_head 2.
fn tables() -> Vec<Table> {
    vec![
        matrix("x", ["t", "d"], [3, 3], 1),
        matrix("wq", ["d", "e"], [3, 2], 20),
        matrix("wk", ["d", "e"], [3, 2], 40),
        matrix("wv", ["d", "e"], [3, 2], 60),
        matrix("tgt", ["t", "e"], [3, 2], 80),
    ]
}

/// Every intermediate is a CTE, several read twice: `x` by all three
/// projections, `s` by the max and the exponent, `ex` by the normalizer and
/// the division.
pub const ATTENTION: &str = "\
WITH q AS (SELECT x.t, w.e, SUM(x.val * w.val) AS val \
           FROM x JOIN wq w ON x.d = w.d GROUP BY x.t, w.e), \
     k AS (SELECT x.t, w.e, SUM(x.val * w.val) AS val \
           FROM x JOIN wk w ON x.d = w.d GROUP BY x.t, w.e), \
     v AS (SELECT x.t, w.e, SUM(x.val * w.val) AS val \
           FROM x JOIN wv w ON x.d = w.d GROUP BY x.t, w.e), \
     s AS (SELECT q.t, k.t AS u, SUM(q.val * k.val) * 0.7071067811865476 AS val \
           FROM q JOIN k ON q.e = k.e GROUP BY q.t, k.t), \
     m AS (SELECT t, MAX(val) AS m FROM s GROUP BY t), \
     ex AS (SELECT s.t, s.u, exp(s.val - m.m) AS val \
            FROM s JOIN m ON s.t = m.t), \
     z AS (SELECT t, SUM(val) AS val FROM ex GROUP BY t), \
     a AS (SELECT ex.t, ex.u, ex.val / z.val AS val FROM ex JOIN z ON ex.t = z.t), \
     o AS (SELECT a.t, v.e, SUM(a.val * v.val) AS val \
           FROM a JOIN v ON a.u = v.t GROUP BY a.t, v.e) \
SELECT SUM(0.5 * power(o.val - tgt.val, 2)) AS loss \
FROM o JOIN tgt ON o.t = tgt.t AND o.e = tgt.e";

#[tokio::test]
async fn attention_with_respect_to_every_weight_and_the_input() {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    check_gradients(
        &ctx,
        ATTENTION,
        &tables(),
        &[
            ColumnRef::new("wq", "val"),
            ColumnRef::new("wk", "val"),
            ColumnRef::new("wv", "val"),
            ColumnRef::new("x", "val"),
        ],
    )
    .await;
}

#[tokio::test]
async fn a_cte_read_twice_is_saved_once() {
    // Eight aggregates are written: q, k, v, s, the max, the normalizer, o and
    // the loss. The producer inlines each CTE at every read, so the plan
    // holds eighteen; the ones that are the same aggregate are saved once and
    // share a cotangent.
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    for t in tables() {
        t.create(&ctx).await;
    }
    for optimized in [false, true] {
        let plan = common::substrait_of(&ctx, ATTENTION, optimized).await;
        let g = ddx_ad::Forward::new(&plan, &[ColumnRef::new("x", "val")]).unwrap();
        assert_eq!(g.saved.len(), 8, "optimized: {optimized}");
        let program = ddx_ad::grad(&plan, &[ColumnRef::new("x", "val")]).unwrap();
        assert_eq!(program.forward_steps.len(), 8 + 1);
    }
}

#[tokio::test]
async fn the_normalizers_cotangent_reads_no_score() {
    // The normalizer z divides every term of o's sum, so its cotangent is
    // -Σ ō·o / z (FlashAttention's D), from o and o's cotangent, never a
    // sum over the scores. So every table it reads, directly or through
    // other backward steps, has at most one row per position and feature
    // (6), where the scores have one per pair of positions (9).
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    for t in tables() {
        t.create(&ctx).await;
    }
    let wrt = [ColumnRef::new("x", "val")];
    let plan = common::substrait_of(&ctx, ATTENTION, true).await;
    let f = ddx_ad::Forward::new(&plan, &wrt).unwrap();
    // The normalizer: the one SUM grouped by one key.
    let z = (0..f.saved.len())
        .find(|&n| {
            let s = &f.saved[n];
            s.dims().len() == 1
                && f.functions.name(s.measures[0].function_reference).unwrap() == "sum"
        })
        .unwrap();
    let program = ddx_ad::grad(&plan, &wrt).unwrap();
    for step in program.steps() {
        ddx_datafusion::engine::run_step(&ctx, step).await.unwrap();
    }
    let backward = &program.backward_steps;
    let start = backward
        .iter()
        .find(|s| s.name.ends_with(&format!("cotangent_{z}")))
        .expect("the normalizer has a cotangent");
    let mut pending = vec![start];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(step) = pending.pop() {
        if !seen.insert(step.name.clone()) {
            continue;
        }
        for read in ddx_ad::tables_read(&step.plan) {
            let rows = ctx
                .table(format!("\"{read}\""))
                .await
                .unwrap()
                .count()
                .await
                .unwrap();
            assert!(rows <= 6, "{} reads {read}, of {rows} rows", step.name);
            pending.extend(backward.iter().find(|s| s.name == read));
        }
    }
}
