// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-ad`'s analysis, run on the Substrait plans DataFusion actually produces.
//!
//! Hand-built plans (ddx-ad's unit tests) show the analysis does what it says;
//! these show it survives a real producer — the `Project`/`emit` layers, masked
//! reads and function naming DataFusion uses.
//!
//! The queries are a small MLP's forward pass over long/tidy tables — the shape
//! design.md §4.5 works through — with one marker added per contraction and
//! nothing else changed:
//!
//! - `pixels(sample, height, width, images)` is the input, one row per pixel;
//! - `weight(layer, inp, out, val)` and `bias(layer, out, val)` are the
//!   parameters, one row per matrix entry, all layers in one table;
//! - a layer is a contraction (`JOIN` on the shared index + grouped `SUM`), then
//!   a bias term (`JOIN` on `out`), then an activation.

mod common;

use datafusion::arrow::array::Float64Array;
use datafusion::prelude::SessionContext;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;

use ddx_ad::substrait::proto::Plan;
use ddx_ad::{AdError, Analysis, ColumnRef, Marker, TableRef};

/// A context with the tables above — 2×2 images, two units per layer — and the
/// v2 markers registered.
async fn nn_context() -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_ad_markers(&ctx);
    for sql in [
        "CREATE TABLE pixels (sample BIGINT, height BIGINT, width BIGINT, images DOUBLE)",
        "INSERT INTO pixels VALUES
           (0, 0, 0, 0.5), (0, 0, 1, -1.0), (0, 1, 0, 2.0), (0, 1, 1, 0.25),
           (1, 0, 0, 1.5), (1, 0, 1, 0.0),  (1, 1, 0, -0.5), (1, 1, 1, 1.0)",
        // `fval` is deliberately REAL: summing it makes DataFusion coerce, which
        // is how a cast ends up wrapped around a marker.
        "CREATE TABLE weight (layer BIGINT, inp BIGINT, out BIGINT, val DOUBLE, fval REAL)",
        "INSERT INTO weight VALUES
           (0, 0, 0, 0.1, 0.1), (0, 0, 1, -0.2, -0.2), (0, 1, 0, 0.3, 0.3),
           (0, 1, 1, 0.4, 0.4), (0, 2, 0, -0.5, -0.5), (0, 2, 1, 0.6, 0.6),
           (0, 3, 0, 0.7, 0.7), (0, 3, 1, -0.8, -0.8), (1, 0, 0, 0.9, 0.9),
           (1, 0, 1, -1.0, -1.0), (1, 1, 0, 1.1, 1.1), (1, 1, 1, 1.2, 1.2)",
        "CREATE TABLE bias (layer BIGINT, out BIGINT, val DOUBLE)",
        "INSERT INTO bias VALUES (0, 0, 0.01), (0, 1, -0.02), (1, 0, 0.03), (1, 1, 0.04)",
    ] {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }
    ctx
}

/// The first layer, with its contraction tagged.
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

/// The parameters: the weight and bias value columns.
fn wrt() -> Vec<ColumnRef> {
    vec![
        ColumnRef::new(TableRef::new(["weight"]), "val"),
        ColumnRef::new(TableRef::new(["bias"]), "val"),
    ]
}

/// The names of the root's gradient-carrying columns.
fn active_outputs(a: &Analysis<'_>) -> Vec<String> {
    let names = a.index.root_names();
    a.activity
        .active(a.index.root())
        .into_iter()
        .map(|c| names[c.index()].clone())
        .collect()
}

/// Every gradient-carrying measure's tag, across the whole plan.
fn tags(a: &Analysis<'_>) -> Vec<Marker> {
    let mut out = Vec::new();
    for n in a.index.nodes() {
        for c in a.columns.cols(n.id) {
            out.extend(a.measure_marker(n.id, c));
        }
    }
    out
}

fn refusal(plan: &Plan, wrt: &[ColumnRef]) -> AdError {
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
    let a = Analysis::new(&plan, &wrt()).unwrap();
    assert_eq!(a.index.root_names(), ["sample", "out", "z", "val"]);
    assert_eq!(active_outputs(&a), ["z", "val"]);
    assert_eq!(tags(&a), [Marker::Contraction]);
}

/// With respect to the bias alone, the same columns carry gradient — through the
/// `+ b.val`, not the contraction, which no longer carries any.
#[tokio::test]
async fn nn_first_layer_wrt_bias_only() {
    let ctx = nn_context().await;
    let plan = substrait(&ctx, FWD0).await;
    let a = Analysis::new(&plan, &wrt()[1..]).unwrap();
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
    let a = Analysis::new(&plan, &wrt()).unwrap();
    assert_eq!(active_outputs(&a), ["loss"]);
    assert_eq!(a.index.sources()[&TableRef::new(["weight"])].len(), 2);
    let mut t = tags(&a);
    t.sort_by_key(|m| m.name());
    assert_eq!(
        t,
        [Marker::Contraction, Marker::Contraction, Marker::Reduce]
    );
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
    let err = refusal(&plan, &wrt());
    assert!(
        matches!(err, AdError::Untagged(ref m) if m.contains("ddx_contract_mark")),
        "{err}"
    );
}

/// The softmax stability shift, `exp(z - max(z))`, which is a no-op for the
/// gradient. With the max stopped it needs no rule; without, it is refused and
/// the error says what to do.
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
    let a = Analysis::new(&plan, &wrt()).unwrap();
    assert_eq!(active_outputs(&a), ["e"]);

    let plan = substrait(&ctx, &softmax("m.m")).await;
    let err = refusal(&plan, &wrt());
    assert!(
        matches!(err, AdError::Untagged(ref m) if m.contains("ddx_stop_gradient")),
        "{err}"
    );
}

/// A typo in `wrt` is an error naming what the plan does have, never a gradient
/// of zero.
#[tokio::test]
async fn a_wrt_typo_is_refused() {
    let ctx = nn_context().await;
    let plan = substrait(&ctx, FWD0).await;
    let err = refusal(&plan, &[ColumnRef::new(TableRef::new(["weights"]), "val")]);
    assert!(
        matches!(err, AdError::InvalidWrt(ref m) if m.contains("bias, pixels, weight")),
        "{err}"
    );
    let err = refusal(&plan, &[ColumnRef::new(TableRef::new(["weight"]), "value")]);
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
    let a = Analysis::new(&plan, &wrt()).unwrap();
    assert_eq!(active_outputs(&a), ["val"]);
}

/// Restricting the backward pass to a subset of rows — a train/test split, say —
/// reads as `WHERE sample IN (…)`, which DataFusion plans as a semi-join: a mask
/// over one side.
#[tokio::test]
async fn an_in_subquery_is_a_mask() {
    let ctx = nn_context().await;
    let sql = format!(
        "SELECT sample, out, val FROM ({FWD0}) f
         WHERE sample IN (SELECT sample FROM pixels WHERE height = 0 AND width = 0)"
    );
    let plan = substrait(&ctx, &sql).await;
    let a = Analysis::new(&plan, &wrt()).unwrap();
    assert_eq!(active_outputs(&a), ["val"]);
}

/// A `REAL` value column makes DataFusion coerce the sum to `Float64`, wrapping
/// the marker in a cast — `sum(CAST(ddx_reduce_mark(fval) AS Float64))`. The
/// marker is then not the syntactic root of the measure's argument even though
/// the user wrote it there, so placement must see through the cast.
#[tokio::test]
async fn a_coerced_marker_is_still_the_whole_argument() {
    let ctx = nn_context().await;
    let plan = substrait(
        &ctx,
        "SELECT layer, SUM(ddx_reduce_mark(fval)) AS val FROM weight GROUP BY layer",
    )
    .await;
    let a = Analysis::new(&plan, &[ColumnRef::new(TableRef::new(["weight"]), "fval")]).unwrap();
    assert_eq!(active_outputs(&a), ["val"]);
    assert_eq!(tags(&a), [Marker::Reduce]);
}

/// A value that reaches the output *only* through a window's `ORDER BY` is not
/// useful by §4.4's definition, so it carries no gradient and needs no tag. This
/// is the expression form of a window, which is what DataFusion emits.
#[tokio::test]
async fn a_value_used_only_as_a_sort_key_is_not_active() {
    let ctx = nn_context().await;
    let plan = substrait(
        &ctx,
        "WITH s AS (SELECT inp, SUM(val) AS total FROM weight GROUP BY inp)
         SELECT inp, ROW_NUMBER() OVER (ORDER BY total) AS rk FROM s",
    )
    .await;
    let a = Analysis::new(&plan, &[ColumnRef::new(TableRef::new(["weight"]), "val")]);
    // `total` orders rows and nothing else, so the SUM needs no marker — but no
    // gradient reaches the output either, which is what is refused.
    let err = a.map(|_| ()).unwrap_err();
    assert!(
        matches!(err, AdError::InvalidWrt(ref m) if m.contains("no gradient can reach")),
        "{err}"
    );
}

/// `COUNT` counts rows, so its derivative is zero and it needs no tag — which is
/// what makes §4.3's "SUM then divide by the count" recipe expressible.
#[tokio::test]
async fn a_count_needs_no_tag() {
    let ctx = nn_context().await;
    let plan = substrait(
        &ctx,
        "SELECT SUM(ddx_reduce_mark(val)) / COUNT(val) AS mean FROM weight",
    )
    .await;
    let a = Analysis::new(&plan, &[ColumnRef::new(TableRef::new(["weight"]), "val")]).unwrap();
    assert_eq!(active_outputs(&a), ["mean"]);
    assert_eq!(tags(&a), [Marker::Reduce]);
}

/// An outer join NULL-extends unmatched rows, and a missing cotangent row means
/// *zero*, not NULL. No transpose rule covers the difference, so a gradient
/// through the nullable side is refused rather than assumed.
#[tokio::test]
async fn an_outer_join_over_a_gradient_carrying_column_is_refused() {
    let ctx = nn_context().await;
    let plan = substrait(
        &ctx,
        "SELECT b.out, SUM(ddx_reduce_mark(w.val)) AS val
         FROM bias b LEFT JOIN weight w ON b.out = w.out GROUP BY b.out",
    )
    .await;
    let err = refusal(&plan, &[ColumnRef::new(TableRef::new(["weight"]), "val")]);
    assert!(
        matches!(err, AdError::NotImplemented(ref m) if m.contains("unmatched row")),
        "{err}"
    );
}

/// A stop-gradient in a condition or key cuts nothing, because no cotangent
/// flows there. Silently doing nothing is what this marker exists to prevent.
#[tokio::test]
async fn a_stop_gradient_that_cuts_nothing_is_refused() {
    let ctx = nn_context().await;
    let plan = substrait(
        &ctx,
        "SELECT SUM(ddx_reduce_mark(val)) AS val FROM weight
         WHERE ddx_stop_gradient(val) > 0",
    )
    .await;
    let err = refusal(&plan, &[ColumnRef::new(TableRef::new(["weight"]), "val")]);
    assert!(
        matches!(err, AdError::InvalidMarker(ref m) if m.contains("ddx_stop_gradient")),
        "{err}"
    );
}
