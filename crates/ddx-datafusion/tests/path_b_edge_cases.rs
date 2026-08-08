// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Path B across awkward plan shapes and binding edges.
//!
//! Each test here started as a claim the crate made about itself that a live
//! engine did not honour. They assert on *executed numbers* rather than on
//! rewritten SQL, because every bug this rewrite has had was invisible to
//! reading and only surfaced when a query actually ran.
//!
//! The recurring theme, and the thing to keep in mind when editing the analyzer:
//! an `AnalyzerRule` runs *after* the planner has bound columns, coerced types,
//! and cached schemas, so a rewrite must hand back an expression consistent with
//! all three — and it must reach every corner of the plan the planner filled in,
//! including the ones that live inside expressions rather than under `inputs()`.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::{ColumnarValue, ScalarUDF, Volatility};
use datafusion::prelude::{create_udf, SessionContext};

mod common;

/// A context with ddx installed and `t(x, y)` = {(1,4), (2,5), (3,6)}.
async fn ctx() -> Result<SessionContext> {
    let ctx = SessionContext::new();
    ddx_datafusion::install(&ctx);
    ctx.sql("CREATE TABLE t AS SELECT * FROM (VALUES (1.0,4.0),(2.0,5.0),(3.0,6.0)) AS v(x,y)")
        .await?
        .collect()
        .await?;
    Ok(ctx)
}

/// Run `sql`, returning the first column as `f64`s.
async fn col(ctx: &SessionContext, sql: &str) -> Result<Vec<f64>> {
    let batches = ctx.sql(sql).await?.collect().await?;
    Ok(common::f64_column(&batches))
}

// ---------------------------------------------------------------------------
// Subqueries embedded in expressions, including the quantified forms.
// ---------------------------------------------------------------------------

/// A marker inside a quantified-comparison subquery — `> ALL (…)`, `= ANY (…)`,
/// `SOME (…)`.
///
/// This once reached execution. The walk hand-rolled its own recursion over the
/// expression variants that can carry a plan, listing `ScalarSubquery`,
/// `InSubquery` and `Exists` — three of the four DataFusion actually has. The
/// missing `Expr::SetComparison` blinded *both* the pre-gate (which then skipped
/// the rewrite for the whole plan) and the walk itself. Path A handled the same
/// query fine, which was the tell: the limitation was in the recursion, not the
/// query shape. Both now delegate to DataFusion's own traversal.
#[tokio::test]
async fn set_comparison_subquery_markers_are_rewritten() -> Result<()> {
    let ctx = ctx().await?;

    // Path A is the oracle. AVG-free: grad(y*y, y) = 2y = [8,10,12], /100 =
    // [0.08,0.10,0.12], so `x > ALL (...)` holds for all three rows.
    let rewritten = ddx_datafusion::rewrite_sql(
        "SELECT x FROM t WHERE x > ALL (SELECT grad(y*y,y)/100 FROM t)",
    )?;
    assert_eq!(
        rewritten, "SELECT x FROM t WHERE x > ALL (SELECT (y + y)/100 FROM t)",
        "Path A handles this shape fine"
    );

    assert_eq!(
        col(
            &ctx,
            "SELECT x FROM t WHERE x > ALL (SELECT grad(y*y,y)/100 FROM t) ORDER BY x"
        )
        .await?,
        vec![1.0, 2.0, 3.0],
    );
    Ok(())
}

/// The `ANY` spelling of the same hole, so a fix that special-cases `ALL`
/// doesn't look green.
#[tokio::test]
async fn any_subquery_markers_are_rewritten() -> Result<()> {
    let ctx = ctx().await?;
    assert_eq!(
        col(
            &ctx,
            "SELECT x FROM t WHERE x > ANY (SELECT grad(y*y,y)/100 FROM t) ORDER BY x"
        )
        .await?,
        vec![1.0, 2.0, 3.0],
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Nodes that derive their output field names from their expressions.
// ---------------------------------------------------------------------------

/// `SELECT DISTINCT ON (…) grad(…)` must not rename the output field, or the
/// parent that refers to it by name dangles.
///
/// The analyzer aliases a rewritten expression back to its original
/// `schema_name` so a parent's column reference keeps resolving. That decision
/// was once made from a list of node variants — `Projection | Aggregate |
/// Window` — which omitted `Distinct::On`, whose schema is also derived from its
/// expressions. The field silently became `t.x + t.x` and the enclosing
/// projection dangled: the identical failure mode the alias-back exists to
/// prevent, one variant later. It is now derived from the node's own schema, so
/// a node kind nobody anticipated is handled by construction.
#[tokio::test]
async fn distinct_on_preserves_derived_field_names() -> Result<()> {
    let ctx = ctx().await?;

    // Standalone: the user-visible column name should still be what they wrote.
    let batches = ctx
        .sql("SELECT DISTINCT ON (x) grad(x*x, x) FROM t")
        .await?
        .collect()
        .await?;
    assert_eq!(
        batches[0].schema().field(0).name(),
        "grad(t.x * t.x,t.x)",
        "the rewrite must not rename the field it replaces"
    );

    // Nested: the parent refers to the field by name, so a rename is fatal —
    // it surfaced as `Schema error: No field named "grad(t.x * t.x,t.x)"`.
    let mut got = col(
        &ctx,
        "SELECT * FROM (SELECT DISTINCT ON (x) grad(x*x, x) FROM t)",
    )
    .await?;
    got.sort_by(f64::total_cmp);
    assert_eq!(got, vec![2.0, 4.0, 6.0]);
    Ok(())
}

// ---------------------------------------------------------------------------
// A user's own UDF surviving into the derivative.
// ---------------------------------------------------------------------------

fn double_it() -> ScalarUDF {
    create_udf(
        "double_it",
        vec![DataType::Float64],
        DataType::Float64,
        Volatility::Immutable,
        Arc::new(|args: &[ColumnarValue]| {
            let a = match &args[0] {
                ColumnarValue::Array(a) => Arc::clone(a),
                ColumnarValue::Scalar(s) => s.to_array()?,
            };
            let a = a.as_any().downcast_ref::<Float64Array>().unwrap();
            let out: Float64Array = a.iter().map(|v| v.map(|v| v * 2.0)).collect();
            Ok(ColumnarValue::Array(Arc::new(out) as ArrayRef))
        }),
    )
}

/// A UDF the user registered on their own `SessionContext`, appearing in the
/// *body* of a marker as a constant coefficient.
///
/// `d/dx [f(y) · x] = f(y)`, so `double_it(y)` survives verbatim into the
/// derivative. The re-plan registry was once seeded only from DataFusion's
/// built-ins, so `SqlToRel` could not resolve it and planning failed with
/// `Invalid function 'double_it'` plus a "did you mean" guess at an unrelated
/// built-in — naming neither the cause nor the remedy. Fixed by harvesting the
/// UDFs from the marker's own arguments, which needs no registration step and
/// works whenever the function was registered.
///
/// `DdxAnalyzer::with_engine_and_functions` remains the escape hatch, but only
/// for a function that appears in the derivative *without* appearing in the
/// body — which means a UDF emitted by a custom differentiation rule. Nothing
/// the user merely calls needs declaring.
#[tokio::test]
async fn a_session_registered_udf_survives_into_the_derivative() -> Result<()> {
    let ctx = ctx().await?;
    ctx.register_udf(double_it());

    // Sanity: the UDF itself works, and is exactly the expected derivative.
    assert_eq!(
        col(&ctx, "SELECT double_it(y) AS d FROM t").await?,
        vec![8.0, 10.0, 12.0]
    );

    assert_eq!(
        col(&ctx, "SELECT grad(double_it(y) * x, x) AS d FROM t").await?,
        vec![8.0, 10.0, 12.0],
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Recursive CTEs — a capability the docs once denied Path B had.
// ---------------------------------------------------------------------------

/// `lib.rs` tells users to "Reach for [`ddx_sql`] when the marker sits
/// somewhere a bound plan can't carry it — **most importantly inside a
/// recursive CTE**, which is exactly where a whole training loop lives."
///
/// Path B handles it. `LogicalPlan::RecursiveQuery` is in `inputs()`, so
/// `transform_up` walks both terms like any other node. This test passes today
/// and pins that, so the sole concrete justification offered for Path A can be
/// corrected rather than repeated.
///
/// (The shape that *looks* like a failure — `WITH RECURSIVE r(n) AS (SELECT
/// 1.0 UNION ALL …)` — fails identically with the derivative written out by
/// hand on a context with no ddx installed. It is DataFusion's own column
/// naming for an unaliased literal in a recursive term, nothing to do with ddx.)
#[tokio::test]
async fn path_b_carries_a_marker_inside_a_recursive_cte() -> Result<()> {
    let ctx = ctx().await?;
    // n = 1 → grad(n², n) = 2n = 2 → 4; the filter stops it there.
    assert_eq!(
        col(
            &ctx,
            "WITH RECURSIVE r(n) AS (\
                 SELECT CAST(1.0 AS DOUBLE) AS n \
                 UNION ALL \
                 SELECT grad(n*n, n) FROM r WHERE n < 3\
             ) SELECT n FROM r"
        )
        .await?,
        vec![1.0, 2.0, 4.0],
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Where Path A and Path B legitimately disagree about a valid `wrt`.
// ---------------------------------------------------------------------------

/// `grad(sum(x)*sum(x), sum(x))`: Path B answers `12`, Path A refuses.
///
/// Neither behaviour is obviously wrong — by the time Path B sees the plan the
/// aggregate has already become the bound column `sum(t.x)`, so `2·sum(x)` is a
/// perfectly good answer to a question Path A cannot even parse as legal. What
/// *is* wrong is that nothing says so. `lib.rs` presents the two as "the same
/// rewrite by two routes" and lists their differences in a table that does not
/// mention this one, and `regressions.rs` installs
/// `path_a_and_path_b_agree_on_every_regression_case` as a standing guard
/// against exactly this drift.
///
/// The divergence is deliberate and documented rather than reconciled. Forcing
/// the two paths to agree would be the wrong fix:
///
/// * Path B's answer is correct. `sum(x)` over `[1,2,3]` is 6, and `d/ds(s·s)`
///   at `s = 6` is 12.
/// * Rejecting it would contradict a case ddx already accepts: differentiating
///   with respect to a *computed alias* is supported and correct — `grad(s*s, s)`
///   is `2s`. An aggregate output as the `wrt` is that same shape one level down.
/// * Detecting "this column is planner-derived" in order to refuse it would be
///   fragile, and would take the computed-alias case down with it.
///
/// So the rule is stated instead: Path B's `wrt` is any column of the node's
/// input schema, including planner-derived ones. See the "Where the two paths
/// genuinely differ" section of `lib.rs`.
#[tokio::test]
async fn the_aggregate_wrt_divergence_is_pinned_and_documented() -> Result<()> {
    let sql = "SELECT grad(sum(x) * sum(x), sum(x)) AS d FROM t";
    let ctx = ctx().await?;

    // Path A refuses: syntactically `sum(x)` is not a bare column.
    let path_a = ddx_datafusion::rewrite_sql(sql);
    let err = path_a.expect_err("Path A must still refuse an aggregate as the wrt");
    assert!(
        err.to_string().contains("must be a bare column"),
        "unexpected Path A error: {err}"
    );

    // Path B accepts, and is right: sum(x) = 6, d/ds(s*s) = 2s = 12.
    assert_eq!(col(&ctx, sql).await?, vec![12.0]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Correlated outer references: the one shape Path B genuinely cannot carry.
// ---------------------------------------------------------------------------

/// The bridge unparses a bound `Expr` to text and re-plans it against the
/// node's input schema. `Expr::OuterReferenceColumn` does not survive that
/// round trip: it unparses to an ordinary qualified column, which by
/// construction is not in the inner schema, so the user is told their column
/// does not exist.
///
/// Today: `ddx_markers caused by Schema error: No field named t.x. Valid fields
/// are u.x, u.y.` — a true statement about the wrong thing. Failing loudly is
/// right — ddx never guesses; blaming the user's column is not. The message
/// must name the real constraint, the way `get_table_source` in `replan.rs`
/// already does for its own unreachable case.
#[tokio::test]
async fn a_correlated_outer_reference_is_diagnosed_as_such() -> Result<()> {
    let ctx = ctx().await?;
    let err = ctx
        .sql("SELECT (SELECT AVG(grad(u.y * t.x, u.y)) FROM t u) AS d FROM t")
        .await?
        .collect()
        .await
        .expect_err("Path B cannot carry an outer reference through the bridge");

    let msg = err.to_string();
    assert!(
        msg.contains("correlat") || msg.contains("outer"),
        "the error must name the real cause — an outer reference the bridge \
         cannot carry — not accuse the user's column of not existing: {msg}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// A fact-check on a load-bearing code comment.
// ---------------------------------------------------------------------------

/// `replan.rs` justifies building — and throwing away — an entire
/// `SessionState` with: "`SessionStateDefaults::default_expr_planners()` would
/// say this directly but is private, so the list is borrowed from a throwaway
/// default state."
///
/// It is public. This test compiles, which is the proof. The throwaway
/// `SessionState` (catalog list, runtime env, object-store registry, every
/// function registry) exists to obtain a `Vec<Arc<dyn ExprPlanner>>` that one
/// public call returns.
///
/// This one passes today. It fails the day DataFusion actually makes the
/// function private — which is the signal that would justify the workaround.
#[test]
fn default_expr_planners_is_public() {
    use datafusion::execution::SessionStateDefaults;
    assert!(
        !SessionStateDefaults::default_expr_planners().is_empty(),
        "if this ever goes private, replan.rs's throwaway-SessionState comment \
         becomes true and this test should be deleted"
    );
}
