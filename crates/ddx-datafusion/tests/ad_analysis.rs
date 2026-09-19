// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-ad`'s analysis, run on the Substrait plans DataFusion actually produces.
//!
//! Hand-built plans (ddx-ad's unit tests) show the analysis does what it says;
//! these show it survives a real producer — the `Project`/`emit` layers, masked
//! reads and function naming DataFusion uses. The queries are `nn.py`'s
//! (xarray-sql#196), changed only by the markers.

mod common;

use datafusion::arrow::array::Float64Array;
use datafusion::prelude::SessionContext;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;

use ddx_ad::substrait::proto::Plan;
use ddx_ad::{AdError, Analysis, Marker, Param, RelRef};

/// A context with `nn.py`'s tables — 2×2 images, one hidden layer of width 2 —
/// and the v2 markers registered.
async fn nn_context() -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_ad_markers(&ctx);
    for sql in [
        "CREATE TABLE pixels (sample BIGINT, height BIGINT, width BIGINT, images DOUBLE)",
        "INSERT INTO pixels VALUES
           (0, 0, 0, 0.5), (0, 0, 1, -1.0), (0, 1, 0, 2.0), (0, 1, 1, 0.25),
           (1, 0, 0, 1.5), (1, 0, 1, 0.0),  (1, 1, 0, -0.5), (1, 1, 1, 1.0)",
        "CREATE TABLE weight (layer BIGINT, inp BIGINT, out BIGINT, val DOUBLE)",
        "INSERT INTO weight VALUES
           (0, 0, 0, 0.1), (0, 0, 1, -0.2), (0, 1, 0, 0.3), (0, 1, 1, 0.4),
           (0, 2, 0, -0.5), (0, 2, 1, 0.6), (0, 3, 0, 0.7), (0, 3, 1, -0.8),
           (1, 0, 0, 0.9), (1, 0, 1, -1.0), (1, 1, 0, 1.1), (1, 1, 1, 1.2)",
        "CREATE TABLE bias (layer BIGINT, out BIGINT, val DOUBLE)",
        "INSERT INTO bias VALUES (0, 0, 0.01), (0, 1, -0.02), (1, 0, 0.03), (1, 1, 0.04)",
    ] {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }
    ctx
}

/// `nn.py`'s first layer, with the contraction tagged.
const FWD0: &str = "
WITH c AS (
  SELECT a.sample, w.out AS out, SUM(ddx_contract_mark(a.val * w.val)) AS z
  FROM (SELECT sample, height * 2 + width AS inp, images AS val FROM pixels) a
  JOIN weight w ON a.inp = w.inp AND w.layer = 0
  GROUP BY a.sample, w.out
)
SELECT c.sample, c.out AS out, c.z + b.val AS z, tanh(c.z + b.val) AS val
FROM c JOIN bias b ON c.out = b.out AND b.layer = 0";

/// The Substrait plan DataFusion produces for `sql`, after optimization — the
/// plan a user would hand to ddx.
async fn substrait(ctx: &SessionContext, sql: &str) -> Plan {
    let plan = ctx.sql(sql).await.unwrap().into_optimized_plan().unwrap();
    *to_substrait_plan(&plan, &ctx.state()).unwrap()
}

fn params() -> Vec<Param> {
    vec![
        Param::new(RelRef::new(["weight"]), "val"),
        Param::new(RelRef::new(["bias"]), "val"),
    ]
}

/// The names of the root's gradient-carrying columns.
fn active_outputs(a: &Analysis<'_>) -> Vec<String> {
    let names = a.index.root_names();
    a.activity
        .active(a.index.root())
        .into_iter()
        .map(|c| names[c].clone())
        .collect()
}

/// Every gradient-carrying measure's tag, across the whole plan.
fn tags(a: &Analysis<'_>) -> Vec<Marker> {
    let mut out = Vec::new();
    for n in a.index.nodes() {
        for c in 0..a.columns.of(n.id).len() {
            out.extend(a.measure_marker(n.id, c));
        }
    }
    out
}

fn refusal(plan: &Plan, wrt: &[Param]) -> AdError {
    Analysis::new(plan, wrt).map(|_| ()).unwrap_err()
}

#[test]
fn the_marker_names_are_ddx_ads() {
    let ad: Vec<&str> = Marker::ALL.iter().map(|m| m.name()).collect();
    assert_eq!(ddx_datafusion::AD_MARKERS.to_vec(), ad);
}

/// The markers are identities: tagging a query doesn't change its answer.
#[tokio::test]
async fn a_marked_forward_query_returns_the_unmarked_answer() {
    let ctx = nn_context().await;
    let run = |sql: String| {
        let ctx = ctx.clone();
        async move {
            let sorted = format!("SELECT val FROM ({sql}) ORDER BY sample, out");
            let batches = ctx.sql(&sorted).await.unwrap().collect().await.unwrap();
            let mut out = Vec::new();
            for b in &batches {
                let col = b.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
                out.extend(col.values().iter().copied());
            }
            out
        }
    };
    let marked = run(FWD0.to_string()).await;
    let unmarked = run(FWD0.replace("ddx_contract_mark(a.val * w.val)", "a.val * w.val")).await;
    assert_eq!(marked.len(), 4);
    assert_eq!(marked, unmarked);
}

/// The first layer: `z` and `val` carry gradient, and nothing else does —
/// including `inp`, which is computed, and `images`, which is data.
#[tokio::test]
async fn nn_first_layer() {
    let ctx = nn_context().await;
    let plan = substrait(&ctx, FWD0).await;
    let a = Analysis::new(&plan, &params()).unwrap();
    assert_eq!(a.index.root_names(), ["sample", "out", "z", "val"]);
    assert_eq!(active_outputs(&a), ["z", "val"]);
    assert_eq!(tags(&a), [Marker::Contract]);
}

/// With respect to the bias alone, the same columns carry gradient — through the
/// `+ b.val`, not the contraction, which no longer carries any.
#[tokio::test]
async fn nn_first_layer_wrt_bias_only() {
    let ctx = nn_context().await;
    let plan = substrait(&ctx, FWD0).await;
    let a = Analysis::new(&plan, &params()[1..]).unwrap();
    assert_eq!(active_outputs(&a), ["z", "val"]);
    assert_eq!(tags(&a), []);
}

/// Two layers in one query read `weight` twice — fan-in — and a scalar loss
/// reduces the output.
#[tokio::test]
async fn nn_two_layers_and_a_loss() {
    let ctx = nn_context().await;
    let sql = format!(
        "WITH fwd0 AS ({FWD0}),
         c1 AS (
           SELECT a.sample, w.out AS out, SUM(ddx_contract_mark(a.val * w.val)) AS z
           FROM (SELECT sample, out AS inp, val FROM fwd0) a
           JOIN weight w ON a.inp = w.inp AND w.layer = 1
           GROUP BY a.sample, w.out
         )
         SELECT SUM(ddx_reduce_mark(c1.z * c1.z)) AS loss FROM c1"
    );
    let plan = substrait(&ctx, &sql).await;
    let a = Analysis::new(&plan, &params()).unwrap();
    assert_eq!(active_outputs(&a), ["loss"]);
    assert_eq!(a.index.sources()[&RelRef::new(["weight"])].len(), 2);
    let mut t = tags(&a);
    t.sort_by_key(|m| m.name());
    assert_eq!(t, [Marker::Contract, Marker::Contract, Marker::Reduce]);
}

/// The whole point of tagging: an untagged SUM over a parameter is refused, not
/// guessed at.
#[tokio::test]
async fn an_untagged_contraction_is_refused() {
    let ctx = nn_context().await;
    let plan = substrait(
        &ctx,
        &FWD0.replace("ddx_contract_mark(a.val * w.val)", "a.val * w.val"),
    )
    .await;
    let err = refusal(&plan, &params());
    assert!(
        matches!(err, AdError::Marker(ref m) if m.contains("ddx_contract_mark")),
        "{err}"
    );
}

/// `nn.py`'s softmax shift: `exp(z - max(z))`. With the max stopped it needs no
/// rule; without, it is refused and the error says what to do.
#[tokio::test]
async fn softmax_shift_needs_stop_gradient() {
    let ctx = nn_context().await;
    let softmax = |shift: &str| {
        format!(
            "WITH logits AS ({FWD0}),
             m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample)
             SELECT l.sample, l.out, exp(l.z - {shift}) AS e
             FROM logits l JOIN m ON l.sample = m.sample"
        )
    };
    let plan = substrait(&ctx, &softmax("ddx_stop_gradient(m.m)")).await;
    let a = Analysis::new(&plan, &params()).unwrap();
    assert_eq!(active_outputs(&a), ["e"]);

    let plan = substrait(&ctx, &softmax("m.m")).await;
    let err = refusal(&plan, &params());
    assert!(
        matches!(err, AdError::Marker(ref m) if m.contains("ddx_stop_gradient")),
        "{err}"
    );
}

/// A typo in `wrt` is an error naming what the plan does have, never a gradient
/// of zero.
#[tokio::test]
async fn a_wrt_typo_is_refused() {
    let ctx = nn_context().await;
    let plan = substrait(&ctx, FWD0).await;
    let err = refusal(&plan, &[Param::new(RelRef::new(["weights"]), "val")]);
    assert!(
        matches!(err, AdError::InvalidWrt(ref m) if m.contains("bias, pixels, weight")),
        "{err}"
    );
    let err = refusal(&plan, &[Param::new(RelRef::new(["weight"]), "value")]);
    assert!(
        matches!(err, AdError::InvalidWrt(ref m) if m.contains("layer, inp, out, val")),
        "{err}"
    );
}

/// Route, design.md §4.3's idiom: keep each sample's top unit. The row number
/// depends on `val` only through its ORDER BY, so it isn't active; the routed
/// `val` is.
#[tokio::test]
async fn route_idiom() {
    let ctx = nn_context().await;
    let sql = format!(
        "WITH ranked AS (
           SELECT sample, out, ddx_route_mark(val) AS val,
                  ROW_NUMBER() OVER (PARTITION BY sample ORDER BY val DESC) AS rk
           FROM ({FWD0}) f
         )
         SELECT sample, out, val FROM ranked WHERE rk = 1"
    );
    let plan = substrait(&ctx, &sql).await;
    let a = Analysis::new(&plan, &params()).unwrap();
    assert_eq!(active_outputs(&a), ["val"]);
}

/// `nn.py`'s `delta2` restricts to the train split with `WHERE sample IN (…)`,
/// which DataFusion plans as a semi-join: a mask over one side.
#[tokio::test]
async fn an_in_subquery_is_a_mask() {
    let ctx = nn_context().await;
    let sql = format!(
        "SELECT sample, out, val FROM ({FWD0}) f
         WHERE sample IN (SELECT sample FROM pixels WHERE height = 0 AND width = 0)"
    );
    let plan = substrait(&ctx, &sql).await;
    let a = Analysis::new(&plan, &params()).unwrap();
    assert_eq!(active_outputs(&a), ["val"]);
}
