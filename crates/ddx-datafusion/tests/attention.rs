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
async fn a_softmax_written_with_windows_gets_the_group_by_program() {
    // Each window over a whole partition is read as its GROUP BY joined
    // back, so both spellings of causal attention save the same aggregates
    // and have the same steps.
    let ctx = SessionContext::new();
    for (name, seed) in [("q", 1), ("k", 30), ("v", 60)] {
        matrix(name, ["t", "d"], [3, 2], seed).create(&ctx).await;
    }
    let attention = |softmax: &str| {
        format!(
            "WITH s AS (SELECT q.t AS i, k.t AS j, SUM(q.val * k.val) AS s \
                        FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t), \
                  {softmax}, \
                  o AS (SELECT p.i AS t, v.d, SUM(p.p * v.val) AS x \
                        FROM p JOIN v ON p.j = v.t GROUP BY p.i, v.d) \
             SELECT SUM(x * x) AS l FROM o"
        )
    };
    let windows = attention(
        "w AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS e FROM s), \
         p AS (SELECT i, j, e / SUM(e) OVER (PARTITION BY i) AS p FROM w)",
    );
    let grouped = attention(
        "m AS (SELECT i, MAX(s) AS m FROM s GROUP BY i), \
         e AS (SELECT s.i, s.j, exp(s.s - m.m) AS e FROM s JOIN m ON s.i = m.i), \
         z AS (SELECT i, SUM(e) AS z FROM e GROUP BY i), \
         p AS (SELECT e.i, e.j, e.e / z.z AS p FROM e JOIN z ON e.i = z.i)",
    );
    let wrt = [
        ColumnRef::new("q", "val"),
        ColumnRef::new("k", "val"),
        ColumnRef::new("v", "val"),
    ];
    let mut shapes = Vec::new();
    for sql in [&windows, &grouped] {
        let plan = common::substrait_of(&ctx, sql, true).await;
        let options = ddx_ad::Options::new().namespace("__ddx_softmax_");
        let program = ddx_ad::grad_with(&plan, &wrt, &options).unwrap();
        shapes.push(program.steps().map(|s| s.name.clone()).collect::<Vec<_>>());
    }
    assert_eq!(shapes[0], shapes[1]);
}

/// Causal attention over `q`, `k` and `v` (4 positions, 2 features), its
/// softmax written as `softmax`.
fn causal(softmax: &str) -> String {
    format!(
        "WITH s AS (SELECT q.t AS i, k.t AS j, SUM(q.val * k.val) AS s \
                    FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t), \
              {softmax}, \
              o AS (SELECT p.i AS t, v.d, SUM(p.p * v.val) AS x \
                    FROM p JOIN v ON p.j = v.t GROUP BY p.i, v.d) \
         SELECT SUM(x * x) AS l FROM o"
    )
}

const WINDOWS: &str = "w AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS e FROM s), \
                       p AS (SELECT i, j, e / SUM(e) OVER (PARTITION BY i) AS p FROM w)";

const GROUPED: &str = "m AS (SELECT i, MAX(s) AS m FROM s GROUP BY i), \
                       e AS (SELECT s.i, s.j, exp(s.s - m.m) AS e FROM s JOIN m ON s.i = m.i), \
                       z AS (SELECT i, SUM(e) AS z FROM e GROUP BY i), \
                       p AS (SELECT e.i, e.j, e.e / z.z AS p FROM e JOIN z ON e.i = z.i)";

fn qkv() -> Vec<Table> {
    [("q", 1), ("k", 30), ("v", 60)]
        .into_iter()
        .map(|(name, seed)| matrix(name, ["t", "d"], [4, 2], seed))
        .collect()
}

#[tokio::test]
async fn attentions_backward_writes_one_table_of_scores() {
    // The normalizer's cotangent comes from o (FlashAttention's D), and the
    // max, which cancels, is held constant: so the one backward step with
    // a row per score is dS. Every other has at most a row per position and
    // feature (8), where the causal scores have 10.
    let wrt = [
        ColumnRef::new("q", "val"),
        ColumnRef::new("k", "val"),
        ColumnRef::new("v", "val"),
    ];
    for softmax in [WINDOWS, GROUPED] {
        let ctx = SessionContext::new();
        for t in qkv() {
            t.create(&ctx).await;
        }
        let plan = common::substrait_of(&ctx, &causal(softmax), true).await;
        let program = ddx_ad::grad(&plan, &wrt).unwrap();
        let mut large = Vec::new();
        for step in program.steps() {
            ddx_datafusion::engine::run_step(&ctx, step).await.unwrap();
        }
        for step in &program.backward_steps {
            let name = format!("\"{}\"", step.name);
            if ctx.table(name).await.unwrap().count().await.unwrap() > 8 {
                large.push(step.name.clone());
            }
        }
        assert_eq!(large.len(), 1, "{large:?}");
    }
}

#[tokio::test]
async fn a_max_used_beyond_its_softmax_keeps_its_gradient() {
    // The loss reads the max itself, so it doesn't cancel, and its gradient
    // goes through the max as before.
    let ctx = SessionContext::new();
    let loss = causal(GROUPED).replace(
        "SELECT SUM(x * x) AS l FROM o",
        "SELECT SUM(x * x + m.m) AS l FROM o JOIN m ON o.t = m.i",
    );
    check_gradients(
        &ctx,
        &loss,
        &qkv(),
        &[ColumnRef::new("q", "val"), ColumnRef::new("k", "val")],
    )
    .await;
}
