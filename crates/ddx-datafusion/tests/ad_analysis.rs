// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The analysis of `ddx-ad`, run on the Substrait plans that DataFusion
//! produces.
//!
//! The unit tests of `ddx-ad` use plans built by hand, and they show that the
//! analysis does what it claims. These tests show that the analysis survives a
//! real producer. DataFusion inserts `Project` and `emit` layers, masks the
//! columns of a read, and chooses its own function names.
//!
//! The queries are the forward pass of a small MLP over long and tidy tables,
//! which is the shape that design.md §4.5 works through. Each contraction
//! carries one marker, and nothing else changes:
//!
//! - `pixels(sample, height, width, images)` is the input, with one row for
//!   each pixel.
//! - `weight(layer, inp, out, val)` and `bias(layer, out, val)` are the
//!   parameters, with one row for each matrix entry and every layer in one
//!   table.
//! - A layer is a contraction, which is a `JOIN` on the shared index and a
//!   grouped `SUM`. A bias term follows, which is a `JOIN` on `out`, and then an
//!   activation.

mod common;

use datafusion::arrow::array::Float64Array;
use datafusion::prelude::SessionContext;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;

use ddx_ad::substrait::proto::Plan;
use ddx_ad::{AdError, Analysis, ColumnRef, Marker, TableRef};

/// A context that holds the tables named above, with 2×2 images and two units
/// for each layer, and that has the v2 markers registered.
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

/// The Substrait plan that DataFusion produces for `sql` after optimization.
/// This is the plan that a user hands to ddx.
async fn substrait(ctx: &SessionContext, sql: &str) -> Plan {
    let plan = ctx.sql(sql).await.unwrap().into_optimized_plan().unwrap();
    *to_substrait_plan(&plan, &ctx.state()).unwrap()
}

/// The parameters, which are the value columns of `weight` and `bias`.
fn wrt() -> Vec<ColumnRef> {
    vec![
        ColumnRef::new(TableRef::new(["weight"]), "val"),
        ColumnRef::new(TableRef::new(["bias"]), "val"),
    ]
}

/// The names of the columns of the root that carry gradient.
fn active_outputs(a: &Analysis<'_>) -> Vec<String> {
    let names = a.index.root_names();
    a.activity
        .active(a.index.root())
        .into_iter()
        .map(|c| names[c.index()].clone())
        .collect()
}

/// The tag on every measure that carries gradient, across the whole plan.
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

/// The markers are identity functions, so a tag does not change the answer of a
/// query.
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

/// The first layer. Here `z` and `val` carry gradient and no other column does.
/// That includes `inp`, which arithmetic computes, and `images`, which is
/// data.
#[tokio::test]
async fn nn_first_layer() {
    let ctx = nn_context().await;
    let plan = substrait(&ctx, FWD0).await;
    let a = Analysis::new(&plan, &wrt()).unwrap();
    assert_eq!(a.index.root_names(), ["sample", "out", "z", "val"]);
    assert_eq!(active_outputs(&a), ["z", "val"]);
    assert_eq!(tags(&a), [Marker::Contraction]);
}

/// With respect to the bias alone, the same columns carry gradient. The gradient
/// arrives through the `+ b.val` term. The contraction carries none.
#[tokio::test]
async fn nn_first_layer_wrt_bias_only() {
    let ctx = nn_context().await;
    let plan = substrait(&ctx, FWD0).await;
    let a = Analysis::new(&plan, &wrt()[1..]).unwrap();
    assert_eq!(active_outputs(&a), ["z", "val"]);
    assert_eq!(tags(&a), []);
}

/// Two layers in one query read `weight` twice, which is fan-in. A scalar loss
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

/// An untagged `SUM` over a parameter is refused and never guessed at, which is
/// what tagging is for.
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

/// The softmax stability shift, `exp(z - max(z))`, which leaves the gradient
/// unchanged. With a stop-gradient on the max it needs no rule. Without one, ddx
/// refuses it and the error states the remedy.
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

/// A mistyped `wrt` column produces an error that names what the plan does
/// hold, and never a gradient of zero.
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

/// The Route idiom of design.md §4.3, which keeps the top unit of each sample.
/// The row number depends on `val` only through the `ORDER BY`, so the row
/// number is not active. The routed `val` is active.
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

/// A restriction of the backward pass to a subset of rows, such as a train and
/// test split, reads as `WHERE sample IN (…)`. DataFusion plans it as a
/// semi-join, which is a mask over one side.
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

/// A `REAL` value column makes DataFusion coerce the sum to `Float64`, and the
/// cast wraps the marker: `sum(CAST(ddx_reduce_mark(fval) AS Float64))`. The
/// marker is then not the root of the argument of the measure, although the user
/// wrote it there. The placement check must therefore see through the cast.
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

/// A value that reaches the output only through the `ORDER BY` of a window is
/// not useful, by the definition in design.md §4.4. It carries no gradient and
/// needs no tag. This test uses the expression form of a window, which is the
/// form that DataFusion emits.
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
    // `total` orders rows and nothing else, so the SUM needs no marker. No
    // gradient reaches the output either, and that is what ddx refuses.
    let err = a.map(|_| ()).unwrap_err();
    assert!(
        matches!(err, AdError::InvalidWrt(ref m) if m.contains("no gradient can reach")),
        "{err}"
    );
}

/// `COUNT` counts rows, so its derivative is zero and it needs no tag. This is
/// what lets a user write the recipe in design.md §4.3: take a `SUM`, then
/// divide by the count.
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

/// An outer join holds NULL in an unmatched row, and a missing cotangent row
/// means zero rather than NULL. No transpose rule covers that difference, so ddx
/// refuses a gradient through the nullable side.
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

/// A stop-gradient in a condition or a key cuts nothing, because no cotangent
/// flows there. This marker exists to stop a cotangent, so a placement where it
/// does nothing is refused.
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
