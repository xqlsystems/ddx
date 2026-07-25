// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! End-to-end `rewrite_sql` tests: the marker path, span splicing, the guards,
//! and identifier folding — the M0 deliverables of design.md §3.2 / §8.

use ddx_core::sqlparser::dialect::{DuckDbDialect, GenericDialect};
use ddx_core::{Ddx, DiffError};

fn rw(sql: &str) -> String {
    Ddx::new()
        .rewrite_sql(sql, &GenericDialect {})
        .unwrap_or_else(|e| panic!("rewrite_sql({sql}) failed: {e}"))
}

// ---------------------------------------------------------------------------
// The basic marker path
// ---------------------------------------------------------------------------

#[test]
fn replaces_a_grad_call_in_place() {
    // grad(sin(x), x) -> (cos(x)); the surrounding SELECT is preserved.
    assert_eq!(
        rw("SELECT grad(sin(x), x) AS d FROM t"),
        "SELECT (cos(x)) AS d FROM t"
    );
}

#[test]
fn full_gradient_as_tidy_columns() {
    assert_eq!(
        rw("SELECT grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g"),
        "SELECT (y) AS dfdx, (x) AS dfdy FROM g"
    );
}

#[test]
fn marker_free_query_is_untouched() {
    // Never parsed (parse-free pre-gate), returned byte-identical — even with
    // formatting sqlparser would not reproduce.
    let sql = "SELECT   a+b  AS  s   FROM t  -- comment";
    assert_eq!(rw(sql), sql);
}

#[test]
fn marker_free_query_mentioning_grad_in_a_string_is_returned_verbatim() {
    // The pre-gate matches the `grad(` substring even inside a string literal,
    // so this statement *is* parsed (unlike the case above). With no real
    // marker it must still return byte-identical: a pre-gate false positive
    // costs only a parse that finds nothing, never a wrong rewrite (review #45,
    // finding D).
    let sql = "SELECT 'grad(' AS label FROM t";
    assert_eq!(rw(sql), sql);
}

#[test]
fn nested_higher_order_grad() {
    // grad(grad(power(x,3), x), x) = d2/dx2 x^3 = 6x; inner differentiated first.
    let out = rw("SELECT grad(grad(power(x, 3), x), x) AS d FROM t");
    assert!(
        !out.to_lowercase().contains("grad("),
        "marker left behind: {out}"
    );
    // 3*power(x,2) -> d/dx -> 3 * (2 * power(x,1)) = 6*power(x,1)
    assert!(out.contains("power(x, 1.0)"), "unexpected rewrite: {out}");
}

#[test]
fn fires_inside_recursive_cte() {
    // A whole Newton-step loop in one query: d/dx(x*x - 2) = x + x.
    let out = rw("WITH RECURSIVE r AS (SELECT 1.0 AS x UNION ALL \
         SELECT x - grad(x * x - 2, x) FROM r WHERE x < 10) SELECT x FROM r");
    assert!(out.contains("(x + x)"), "unexpected rewrite: {out}");
    assert!(
        !out.to_lowercase().contains("grad("),
        "marker left behind: {out}"
    );
}

#[test]
fn splice_end_ignores_parens_inside_strings_and_comments() {
    // The splice finds the call's matching `)` over the token stream, so a `)`
    // that appears only inside a string literal or comment in a (jvp) tangent is
    // not miscounted (#57). The tangent is spliced verbatim.
    assert_eq!(
        rw("SELECT jvp(x, x, CASE WHEN a = ')' THEN y ELSE z END) FROM t"),
        "SELECT (CASE WHEN a = ')' THEN y ELSE z END) FROM t"
    );
    assert_eq!(
        rw("SELECT jvp(x, x, y /* ) ) */) FROM t"),
        "SELECT (y) FROM t"
    );
}

#[test]
fn dml_update_rule_is_rewritten() {
    let out = rw("INSERT INTO p SELECT theta - 0.1 * grad(x * theta, theta) FROM t");
    assert!(
        !out.to_lowercase().contains("grad("),
        "marker left behind: {out}"
    );
    assert!(out.contains("(x)"), "unexpected rewrite: {out}");
}

// ---------------------------------------------------------------------------
// Span splicing (G3 / F5): byte-identity outside the marker
// ---------------------------------------------------------------------------

#[test]
fn splice_preserves_multibyte_prefix() {
    // A multibyte character before the marker shifts its byte offset past its
    // character column; naive column-as-byte splicing would corrupt this.
    let out = rw("SELECT 'héllo', grad(sin(x), x) AS d FROM t");
    assert_eq!(out, "SELECT 'héllo', (cos(x)) AS d FROM t");
}

#[test]
fn splice_multiple_markers_on_one_line() {
    // Two markers spliced independently, in reverse source order.
    let out = rw("SELECT grad(sin(x), x), grad(cos(y), y) FROM t");
    assert_eq!(out, "SELECT (cos(x)), (-sin(y)) FROM t");
}

#[test]
fn splice_preserves_exact_surrounding_bytes() {
    // Odd but valid spacing/casing around the marker is preserved verbatim.
    let out = rw("SELECT    GRAD(x*x,x)  ,   y   FROM t");
    assert_eq!(out, "SELECT    (x + x)  ,   y   FROM t");
}

// ---------------------------------------------------------------------------
// Pre-gate coverage: comment-separated markers (#52, fixed)
// ---------------------------------------------------------------------------

#[test]
fn pre_gate_must_not_miss_a_comment_separated_marker() {
    // A SQL comment is lexical whitespace, so `grad /* c */ (x, x)` parses as a
    // genuine `Function(name=grad)` marker call — the *same* AST as plain
    // `grad(x, x)`. The whitespace-separated forms below ARE rewritten, proving
    // the marker is real and reachable; the comment-separated forms must be too.
    //
    // But `pre_gate_hit` requires the next *non-whitespace* char after the name
    // to be `(` and does not skip comments, so these slip past the parse-free
    // gate and `rewrite_sql` returns them VERBATIM — the marker survives to
    // execution un-rewritten. This is a false-NEGATIVE missing a REAL marker,
    // contradicting design.md §3.2's guarantee that the gate only ever produces
    // harmless false-positives.

    // Baseline: whitespace-separated markers are correctly rewritten to (1.0).
    assert_eq!(rw("SELECT grad\n(x, x) FROM t"), "SELECT (1.0) FROM t");
    assert_eq!(rw("SELECT grad\t(x, x) FROM t"), "SELECT (1.0) FROM t");

    // The bug: a comment between the name and its argument list.
    assert_eq!(
        rw("SELECT grad /* c */ (x, x) FROM t"),
        "SELECT (1.0) FROM t",
        "block-comment-separated marker was not rewritten"
    );
    assert_eq!(
        rw("SELECT grad-- c\n(x, x) FROM t"),
        "SELECT (1.0) FROM t",
        "line-comment-separated marker was not rewritten"
    );
}

// ---------------------------------------------------------------------------
// Reserved names (F8) and the cut of scalar vjp (Q7)
// ---------------------------------------------------------------------------

#[test]
fn qualified_grad_is_left_alone() {
    // Only unqualified grad/jvp are markers; myschema.grad(...) is a user fn.
    let sql = "SELECT myschema.grad(x, x) AS d FROM t";
    assert_eq!(rw(sql), sql);
}

#[test]
fn scalar_vjp_is_not_a_marker() {
    // vjp is reserved for query-level reverse-mode AD (Q7); as scalar SQL it is
    // an ordinary (unknown) function, left untouched.
    let sql = "SELECT vjp(sin(x), x, w) AS v FROM t";
    assert_eq!(rw(sql), sql);
}

// ---------------------------------------------------------------------------
// Identifier folding (F1)
// ---------------------------------------------------------------------------

#[test]
fn unquoted_identifiers_fold_case() {
    // grad(Temp*Temp, temp) must match despite the case difference, and keep
    // the original spelling in the output.
    let out = Ddx::new()
        .differentiate_sql("Temp * Temp", "temp", &GenericDialect {})
        .unwrap();
    assert_eq!(out, "Temp + Temp");
}

#[test]
fn duckdb_folds_quoted_identifiers_too() {
    // DuckDB is fully case-insensitive: "Temp" and temp are the same column.
    let out = Ddx::for_duckdb()
        .differentiate_sql(r#""Temp" * "Temp""#, "temp", &DuckDbDialect {})
        .unwrap();
    // Original (quoted) spelling preserved.
    assert_eq!(out, r#""Temp" + "Temp""#);
}

#[test]
fn datafusion_keeps_quoted_identifiers_case_sensitive() {
    // Under the FoldUnquoted policy, quoted "Temp" is a different column from
    // unquoted temp, so the derivative is 0.
    let out = Ddx::for_datafusion()
        .differentiate_sql(r#""Temp" * "Temp""#, "temp", &GenericDialect {})
        .unwrap();
    assert_eq!(out, "0.0");
}

// ---------------------------------------------------------------------------
// Ambiguity guard (F2)
// ---------------------------------------------------------------------------

#[test]
fn qualified_wrt_across_a_join_is_accepted() {
    let out = rw("SELECT grad(a.v * b.w, a.v) AS d FROM t a JOIN u b ON a.k = b.k");
    assert_eq!(out, "SELECT (b.w) AS d FROM t a JOIN u b ON a.k = b.k");
}

#[test]
fn fully_qualified_same_name_across_join_is_accepted() {
    // grad(a.x * b.x, a.x): both occurrences qualified — unambiguous, = b.x.
    let out = rw("SELECT grad(a.x * b.x, a.x) AS d FROM t a JOIN u b ON a.k = b.k");
    assert_eq!(out, "SELECT (b.x) AS d FROM t a JOIN u b ON a.k = b.k");
}

#[test]
fn bare_occurrence_with_qualified_wrt_errors() {
    // grad(x * a.x, a.x): bare x might be a.x — demand qualification.
    let err = Ddx::new()
        .rewrite_sql("SELECT grad(x * a.x, a.x) FROM t a", &GenericDialect {})
        .unwrap_err();
    assert!(matches!(err, DiffError::AmbiguousColumn(_)), "got {err:?}");
}

#[test]
fn qualified_occurrence_with_bare_wrt_errors() {
    // grad(a.x * b.x, x): bare wrt, qualified occurrences — ambiguous.
    let err = Ddx::new()
        .rewrite_sql(
            "SELECT grad(a.x * b.x, x) FROM t a JOIN u b ON a.k = b.k",
            &GenericDialect {},
        )
        .unwrap_err();
    assert!(matches!(err, DiffError::AmbiguousColumn(_)), "got {err:?}");
}

#[test]
fn wrt_must_be_a_bare_column() {
    let err = Ddx::new()
        .rewrite_sql("SELECT grad(x * y, x + y) FROM t", &GenericDialect {})
        .unwrap_err();
    assert!(matches!(err, DiffError::InvalidMarker(_)), "got {err:?}");
}

// ---------------------------------------------------------------------------
// Projection-boundary guard (F3 / G4)
// ---------------------------------------------------------------------------

#[test]
fn computed_cte_alias_as_non_wrt_term_errors() {
    // s = sin(x) is a computed CTE alias; grad(s*x, x) would silently drop ds/dx.
    let err = Ddx::new()
        .rewrite_sql(
            "WITH v AS (SELECT x, sin(x) AS s FROM t) SELECT grad(s * x, x) FROM v",
            &GenericDialect {},
        )
        .unwrap_err();
    assert!(
        matches!(err, DiffError::ProjectionBoundary(_)),
        "got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Through-aggregate differentiation by linearity (design.md §3.1)
// ---------------------------------------------------------------------------

#[test]
fn marker_inside_an_aggregate_is_rewritten() {
    // One gradient-descent step: the marker goes *inside* the aggregate, and
    // d/dθ Σ f = Σ ∂f/∂θ is just linearity.
    let out = rw("SELECT AVG(grad(x * theta, theta)) AS step FROM batch");
    assert_eq!(out, "SELECT AVG((x)) AS step FROM batch");
}

#[test]
fn qualified_base_column_colliding_with_unrelated_cte_alias_is_accepted() {
    // Review #42: w.s is table u's own base column, explicitly qualified — it
    // cannot be v's computed alias `s`, so the guard must NOT fire. Preserves
    // the qualifier-awareness of the F2 ambiguity guard.
    let out = rw("WITH v AS (SELECT sin(x) AS s FROM t) \
         SELECT grad(w.s * x, x) AS d FROM u w JOIN v ON w.k = v.k");
    assert_eq!(
        out,
        "WITH v AS (SELECT sin(x) AS s FROM t) \
         SELECT (w.s) AS d FROM u w JOIN v ON w.k = v.k"
    );
}

#[test]
fn qualified_reference_to_the_owning_cte_alias_still_errors() {
    // But v.s IS v's computed alias, so differentiating it as a non-wrt term
    // still crosses the projection boundary and must error.
    let err = Ddx::new()
        .rewrite_sql(
            "WITH v AS (SELECT sin(x) AS s FROM t) \
             SELECT grad(v.s * x, x) AS d FROM v",
            &GenericDialect {},
        )
        .unwrap_err();
    assert!(
        matches!(err, DiffError::ProjectionBoundary(_)),
        "got {err:?}"
    );
}

#[test]
fn differentiating_wrt_a_computed_alias_is_allowed() {
    // Carve-out (G4): when the computed alias IS the wrt, every occurrence is
    // the leaf, so d/ds (s*s) = 2s is exactly right — no error.
    let out = rw("SELECT a + b AS s, grad(s * s, s) AS d FROM t");
    assert_eq!(out, "SELECT a + b AS s, (s + s) AS d FROM t");
}

// ---------------------------------------------------------------------------
// explain(): inspect a rewrite before running it (PR #3 feedback)
// ---------------------------------------------------------------------------

#[test]
fn explain_reports_each_marker_and_the_rewrite() {
    let ex = Ddx::new()
        .explain(
            "SELECT grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g",
            &GenericDialect {},
        )
        .unwrap();
    assert_eq!(
        ex.original,
        "SELECT grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g"
    );
    assert_eq!(ex.rewritten, "SELECT (y) AS dfdx, (x) AS dfdy FROM g");
    assert_eq!(ex.steps.len(), 2);
    assert_eq!(ex.steps[0].function, "grad");
    assert_eq!(ex.steps[0].marker, "grad(x * y, x)");
    assert_eq!(ex.steps[0].derivative, "(y)");
    assert_eq!(ex.steps[1].marker, "grad(x * y, y)");
    assert_eq!(ex.steps[1].derivative, "(x)");
    // The rewrite explain reports matches rewrite_sql exactly.
    assert_eq!(ex.rewritten, rw(&ex.original));
}

#[test]
fn explain_display_is_readable() {
    let ex = Ddx::new()
        .explain("SELECT jvp(sin(x), x, dx) AS t FROM g", &GenericDialect {})
        .unwrap();
    let shown = format!("{ex}");
    assert_eq!(
        shown,
        "ddx rewrites 1 marker:\n  \
         • jvp(sin(x), x, dx) → (cos(x) * dx)\n\n  \
         from: SELECT jvp(sin(x), x, dx) AS t FROM g\n  \
         into: SELECT (cos(x) * dx) AS t FROM g"
    );
}

#[test]
fn explain_on_marker_free_sql_has_no_steps() {
    let sql = "SELECT a + b FROM t";
    let ex = Ddx::new().explain(sql, &GenericDialect {}).unwrap();
    assert!(ex.steps.is_empty());
    assert_eq!(ex.rewritten, sql);
    assert!(format!("{ex}").contains("unchanged"));
}

#[test]
fn explain_surfaces_errors_like_rewrite() {
    // An unsupported construct errors in explain just as in rewrite_sql — you
    // find out before running, which is the point.
    let err = Ddx::new()
        .explain("SELECT grad(atan2(x, y), x) FROM t", &GenericDialect {})
        .unwrap_err();
    assert!(matches!(err, DiffError::NotImplemented(_)), "got {err:?}");
}
