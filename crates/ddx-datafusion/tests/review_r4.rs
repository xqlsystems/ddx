// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Round-4 adversarial review of PR #49 (M2, `ddx-datafusion`).
//!
//! **Most of these tests currently FAIL.** They are the executable half of the
//! review: each one is a claim the crate makes about itself that a live engine
//! does not honour. They are deliberately checked in red rather than described
//! in prose, because the whole lesson of #77 is that this rewrite's bugs are
//! invisible to reading and only surface when a query actually runs.
//!
//! One test (`path_b_carries_a_marker_inside_a_recursive_cte`) passes: it pins
//! a capability the crate documentation says Path B does *not* have, so the
//! documentation — not the code — is what needs to change.
//!
//! Chainlink issues: #79 (ANY/ALL subqueries), #80 (`DISTINCT ON`),
//! #81 (session UDFs), #82 (recursive-CTE claim), #83 (path divergence),
//! #84 (correlated outer references).

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::{ColumnarValue, ScalarUDF, Volatility};
use datafusion::prelude::{create_udf, SessionContext};

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
    let mut out = Vec::new();
    for b in &batches {
        let a = b
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("derivatives are always Float64 (design.md §3.2, F4)");
        out.extend((0..a.len()).map(|i| a.value(i)));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// #79 — the subquery fix of #77/F5 is incomplete.
// ---------------------------------------------------------------------------

/// `x > ALL (SELECT grad(...))` — a marker inside a quantified-comparison
/// subquery reaches execution.
///
/// `analyzer.rs::subquery_plan` enumerates `ScalarSubquery`, `InSubquery` and
/// `Exists`. DataFusion's own `LogicalPlan::map_subqueries` enumerates those
/// three **plus `Expr::SetComparison`** — `= ANY (…)`, `> ALL (…)`, `SOME (…)`.
/// The missing arm blinds *both* the pre-gate (`plan_has_marker`, which then
/// returns `false` and skips the rewrite entirely) and the walk.
///
/// Path A rewrites this query without complaint, which is the tell: the
/// limitation is not in the query shape, it is in the hand-rolled recursion.
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
// #80 — the `names_fields` list is incomplete.
// ---------------------------------------------------------------------------

/// `SELECT DISTINCT ON (…) grad(…)` renames the output field, breaking the
/// parent that refers to it by name.
///
/// `analyzer.rs` aliases a rewritten expression back to its original
/// `schema_name` for `Projection | Aggregate | Window`, precisely so a parent's
/// column reference keeps resolving. `LogicalPlan::Distinct(Distinct::On(..))`
/// also names output fields — from its `select_expr` — and is missing from that
/// list, so the derived field silently becomes `t.x + t.x` and the enclosing
/// projection dangles. This is the *identical* failure mode the alias-back was
/// introduced to fix (#11, bug 1), one node variant later.
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

    // Nested: the parent refers to the field by name, so the rename is fatal.
    // Currently: `Schema error: No field named "grad(t.x * t.x,t.x)".`
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
// #81 — the re-planner cannot see the session's own functions.
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
/// *body* of a marker as a constant coefficient, breaks the re-plan.
///
/// `d/dx [f(y) · x] = f(y)`, so `double_it(y)` survives verbatim into the
/// derivative — and `replan::ExprContext` is seeded only from
/// `datafusion::functions::all_default_functions()`, so `SqlToRel` cannot
/// resolve it. The failure is `Error during planning: Invalid function
/// 'double_it'.` plus a "did you mean" guess at some unrelated built-in —
/// attributed to the `ddx_markers` rule, naming neither the real cause nor the
/// remedy.
///
/// The escape hatch exists (`DdxAnalyzer::with_engine_and_functions`) but its
/// doc comment says it is "Needed only when a custom rule emits a call to a UDF
/// of your own", which this query shows is not true: no custom rule is
/// involved. `install(&ctx)` holds the `SessionContext` and could snapshot
/// `ctx.state().scalar_functions()` for free.
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
// #82 — Path B's documented limitation is not real.
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
// #83 — the two paths do not agree on what a valid `wrt` is.
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
/// This test asserts only the weak invariant: the two paths must both accept or
/// both reject. Either resolution is fine; silence is not.
#[tokio::test]
async fn both_paths_agree_on_whether_an_aggregate_may_be_the_wrt() -> Result<()> {
    let sql = "SELECT grad(sum(x) * sum(x), sum(x)) AS d FROM t";
    let ctx = ctx().await?;

    let path_a = ddx_datafusion::rewrite_sql(sql);
    let path_b = ctx.sql(sql).await?.collect().await;

    assert_eq!(
        path_a.is_ok(),
        path_b.is_ok(),
        "path A: {path_a:?}\npath B: {path_b:?}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// #84 — a correlated outer reference is diagnosed as a missing column.
// ---------------------------------------------------------------------------

/// The bridge unparses a bound `Expr` to text and re-plans it against the
/// node's input schema. `Expr::OuterReferenceColumn` does not survive that
/// round trip: it unparses to an ordinary qualified column, which by
/// construction is not in the inner schema, so the user is told their column
/// does not exist.
///
/// Today: `ddx_markers caused by Schema error: No field named t.x. Valid fields
/// are u.x, u.y.` — a true statement about the wrong thing. Failing loudly is
/// right (design principle 5); blaming the user's column is not. The message
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
