// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The prototype's rule unit tests, ported to `ddx-core` (design.md §3.2, M0:
//! "port the prototype's 15 rule tests — they pin the math unchanged").
//!
//! The *math* is identical to xarray-sql#192's `src/autograd.rs`; the rendered
//! output differs in two documented, deliberate ways introduced by the design:
//! `div()` casts its numerator to `DOUBLE` (F4), and composite operands are
//! `Nested`-wrapped for precedence safety (G1). Both are asserted here.

use ddx_core::sqlparser::dialect::GenericDialect;
use ddx_core::{Ddx, DiffError};

fn d(expr: &str, wrt: &str) -> String {
    Ddx::new()
        .differentiate_sql(expr, wrt, &GenericDialect {})
        .unwrap_or_else(|e| panic!("differentiate_sql({expr}, {wrt}) failed: {e}"))
}

#[test]
fn constant_has_zero_derivative() {
    assert_eq!(d("3.0", "x"), "0.0");
}

#[test]
fn variable_has_unit_derivative() {
    assert_eq!(d("x", "x"), "1.0");
}

#[test]
fn other_variable_has_zero_derivative() {
    assert_eq!(d("y", "x"), "0.0");
}

#[test]
fn sum_rule_folds_constants() {
    // d/dx (x + y) = 1 + 0 = 1
    assert_eq!(d("x + y", "x"), "1.0");
}

#[test]
fn product_rule() {
    // d/dx (x * x) = 1*x + x*1 = x + x
    assert_eq!(d("x * x", "x"), "x + x");
}

#[test]
fn quotient_rule() {
    // d/dx (x / y) = (1*y - x*0) / (y*y) = y / (y*y), numerator cast to DOUBLE.
    assert_eq!(d("x / y", "x"), "CAST(y AS DOUBLE) / (y * y)");
}

#[test]
fn chain_rule_sin() {
    // d/dx sin(x) = cos(x) * 1 = cos(x)
    assert_eq!(d("sin(x)", "x"), "cos(x)");
}

#[test]
fn composite_sin_times_x() {
    // d/dx (sin(x) * x) = cos(x)*x + sin(x)
    assert_eq!(d("sin(x) * x", "x"), "cos(x) * x + sin(x)");
}

#[test]
fn power_constant_exponent() {
    // d/dx power(x, 2) = 2 * power(x, 1) * 1 = 2 * power(x, 1)
    assert_eq!(d("power(x, 2)", "x"), "2.0 * power(x, 1.0)");
}

#[test]
fn higher_order_derivative() {
    // Differentiation composes: d2/dx2 sin(x) = -sin(x).
    let d1 = d("sin(x)", "x");
    assert_eq!(d(&d1, "x"), "-sin(x)");
}

#[test]
fn unsupported_operator_errors() {
    assert!(Ddx::new()
        .differentiate_sql("x % y", "x", &GenericDialect {})
        .is_err());
}

#[test]
fn unsupported_function_errors() {
    // atan2 is binary and has no rule yet.
    assert!(Ddx::new()
        .differentiate_sql("atan2(x, y)", "x", &GenericDialect {})
        .is_err());
}

#[test]
fn power_negative_constant_exponent() {
    // d/dx power(x, -2) = -2 * power(x, -3); a negative constant exponent is
    // inside the stated v1 surface and must not be rejected (review #46).
    assert_eq!(d("power(x, -2)", "x"), "-2.0 * power(x, -3.0)");
}

#[test]
fn power_negative_fractional_exponent() {
    // d/dx power(x, -0.5) = -0.5 * power(x, -1.5); the negative *fractional*
    // case, complementing the -2 integer case above (review #46).
    assert_eq!(d("power(x, -0.5)", "x"), "-0.5 * power(x, -1.5)");
}

#[test]
fn power_fractional_exponent_output_is_reconsumable() {
    // d/dx power(x, 0.5) emits a negative exponent; differentiating that TEXT
    // again must work (the engine must be able to re-consume its own output).
    let once = d("power(x, 0.5)", "x");
    assert!(once.contains("power(x, -0.5)"), "unexpected: {once}");
    // Re-parse and differentiate the emitted text again — no error.
    let twice = Ddx::new().differentiate_sql(&once, "x", &GenericDialect {});
    assert!(twice.is_ok(), "engine rejected its own output: {twice:?}");
}

#[test]
fn power_non_positive_constant_base_errors() {
    // d/dx power(0, x) would need ln(0) = -inf; must fail loud, not emit
    // "power(0, x) * -inf" (review #33).
    let err = Ddx::new()
        .differentiate_sql("power(0, x)", "x", &GenericDialect {})
        .unwrap_err();
    assert!(matches!(err, DiffError::NotImplemented(_)), "got {err:?}");
    let out = Ddx::new().differentiate_sql("power(2, x)", "x", &GenericDialect {});
    assert!(out.is_ok(), "positive base should differentiate: {out:?}");
    assert!(!out.unwrap().to_lowercase().contains("inf"));
}

#[test]
fn power_non_finite_constant_exponent_errors() {
    // The mirror of the non-positive-base guard, on the exponent side: an
    // out-of-range literal exponent (1e400 overflows f64 to +inf) must fail
    // loud rather than emit an `inf` token (review #33).
    let err = Ddx::new()
        .differentiate_sql("power(x, 1e400)", "x", &GenericDialect {})
        .unwrap_err();
    assert!(matches!(err, DiffError::NotImplemented(_)), "got {err:?}");
}

#[test]
fn non_finite_exponent_still_differentiates_to_zero_when_base_is_constant_in_wrt() {
    // The non-finite constant only blocks emission *when actually emitted*. If
    // the base doesn't depend on the differentiation variable, the derivative is
    // 0 and no non-finite literal is ever built — so this must return 0, not
    // error (review #49/F1: the finiteness check must follow the zero
    // short-circuit, matching the constant-base branch).
    assert_eq!(d("power(x, 1e400)", "y"), "0.0");
}

#[test]
fn schema_qualified_call_does_not_match_builtin_rule() {
    // myschema.sin(x) may be an unrelated user function; it must NOT silently
    // differentiate via the built-in sin rule (review #47).
    assert!(Ddx::new()
        .differentiate_sql("myschema.sin(x)", "x", &GenericDialect {})
        .is_err());
    // The unqualified builtin still works.
    assert_eq!(d("sin(x)", "x"), "cos(x)");
}

#[test]
fn general_power_uv_errors() {
    // power(x, x): both base and exponent vary — not supported yet.
    assert!(Ddx::new()
        .differentiate_sql("power(x, x)", "x", &GenericDialect {})
        .is_err());
}

#[test]
fn double_negation_does_not_render_a_comment() {
    // d/dx(-cos(x)) = sin(x). The intermediate is neg(neg(sin(x))); without
    // folding it renders as `--sin(x)`, a SQL line comment (a silently-wrong
    // result). It must fold to `sin(x)`.
    let out = d("-cos(x)", "x");
    assert!(!out.contains("--"), "rendered a `--` comment: {out}");
    assert_eq!(out, "sin(x)");
    // And nested inside a larger expression, still no stray `--`.
    let nested = d("sin(-cos(x))", "x");
    assert!(!nested.contains("--"), "rendered a `--` comment: {nested}");
    assert_eq!(nested, "cos(-cos(x)) * sin(x)");
}

#[test]
fn cast_to_numeric_type_is_differentiable() {
    // A numeric cast is locally linear: d/dx CAST(x*x AS DOUBLE) = CAST(x+x AS DOUBLE).
    assert_eq!(d("CAST(x * x AS DOUBLE)", "x"), "CAST(x + x AS DOUBLE)");
}

#[test]
fn cast_to_non_numeric_type_errors() {
    // Differentiating through CAST(... AS VARCHAR) would emit a nonsensical
    // CAST(1.0 AS VARCHAR); it must be a typed error instead.
    assert!(Ddx::new()
        .differentiate_sql("CAST(x AS VARCHAR)", "x", &GenericDialect {})
        .is_err());
}

#[test]
fn abs_derivative_is_portable_and_pins_the_kink_at_zero() {
    // d/du |u| = sign(u), emitted as a portable CASE (no engine-specific
    // signum/sign builtin) that pins abs'(0) = 0 on every engine.
    let out = d("abs(x)", "x");
    assert!(
        !out.to_lowercase().contains("signum"),
        "must not emit the non-portable signum builtin: {out}"
    );
    // Every branch of the sign is a stated condition, and `ELSE` means only
    // "none of them answered" — which for a NULL input is the truth, because all
    // three comparisons are NULL. An `ELSE 0.0` would instead report a zero
    // derivative for a row with no value, which no caller can tell apart from a
    // parameter that genuinely does not move. The typed NULL also fixes the
    // CASE's result type at DOUBLE; without it DuckDB reads the bare literals as
    // DECIMAL(2,1) and this one derivative comes back in a type of its own.
    assert_eq!(
        out,
        "CASE WHEN x > 0.0 THEN 1.0 WHEN x < 0.0 THEN -1.0 \
         WHEN x = 0.0 THEN 0.0 ELSE CAST(NULL AS DOUBLE) END"
    );
    // Chain rule still applies: d/dx |x*y| = sign(x*y) * y.
    let chained = d("abs(x * y)", "x");
    assert!(chained.contains("* y"), "chain rule missing: {chained}");
    assert!(!chained.to_lowercase().contains("signum"));
}

#[test]
fn jvp_seeds_a_tangent_on_one_input() {
    // jvp(x*y, x, dx) = product rule with tangent(x)=dx, tangent(y)=0 = dx*y
    let out = Ddx::new()
        .rewrite_sql("SELECT jvp(x * y, x, dx) AS t FROM g", &GenericDialect {})
        .unwrap();
    assert_eq!(out, "SELECT (dx * y) AS t FROM g");
}

#[test]
fn jvp_with_unit_seed_matches_grad() {
    // A one-hot tangent reproduces the partial derivative.
    let jvp = Ddx::new()
        .rewrite_sql("SELECT jvp(sin(x), x, 1.0) AS t FROM g", &GenericDialect {})
        .unwrap();
    let grad = Ddx::new()
        .rewrite_sql("SELECT grad(sin(x), x) AS t FROM g", &GenericDialect {})
        .unwrap();
    assert_eq!(jvp, grad);
}

// ---------------------------------------------------------------------------
// Ergonomic error messages: unsupported constructs state what IS supported and
// how to proceed (PR #3 feedback).
// ---------------------------------------------------------------------------

fn err_msg(expr: &str, wrt: &str) -> String {
    Ddx::new()
        .differentiate_sql(expr, wrt, &GenericDialect {})
        .unwrap_err()
        .to_string()
}

#[test]
fn unsupported_function_error_names_the_supported_set_and_the_escape_hatch() {
    // A unary function with no built-in rule (`cbrt`) surfaces the full guidance.
    let m = err_msg("cbrt(x)", "x");
    assert!(m.contains("cbrt"), "should name the function: {m}");
    assert!(
        m.contains("sin/cos"),
        "should list supported functions: {m}"
    );
    assert!(
        m.contains("Ddx::register"),
        "should point at the custom-rule hatch: {m}"
    );
}

#[test]
fn unsupported_operator_error_lists_supported_operators() {
    let m = err_msg("x % y", "x");
    assert!(
        m.contains("+ - * /"),
        "should list differentiable operators: {m}"
    );
}

#[test]
fn wrt_not_a_column_error_shows_the_right_form() {
    let m = err_msg("x * y", "x + y");
    assert!(
        m.contains("bare column"),
        "should say a bare column is required: {m}"
    );
    assert!(m.contains("x + y"), "should echo the offending wrt: {m}");
}

#[test]
fn general_power_error_suggests_the_constant_side_and_the_exp_ln_rewrite() {
    let m = err_msg("power(x, x)", "x");
    assert!(
        m.contains("power(x, 2)") || m.contains("constant"),
        "should suggest a constant side: {m}"
    );
    assert!(
        m.contains("exp(exponent * ln(base))"),
        "should suggest the u^v rewrite: {m}"
    );
}
