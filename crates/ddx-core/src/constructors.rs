// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Smart constructors for building derivative `sqlparser::ast::Expr` trees.
//!
//! These own three correctness properties, not just algebraic tidiness
//! (design.md §3.2):
//!
//! 1. **0/1-folding** — the JAX-`Zero`-tangent equivalent. Structurally-zero
//!    terms are dropped and dead product branches short-circuit, keeping output
//!    compact. This is a *stated* NULL-semantics convention (folding
//!    `0 * (NULL-valued expr)` to `0`), documented and tested, not silent (F11).
//! 2. **Numeric-type policy** — [`div`] forces floating-point division by
//!    casting its numerator to `DOUBLE`, so `grad(x/y, y)` on integer columns
//!    does not silently truncate (integer `/` differs across engines). Literals
//!    are emitted with an explicit decimal point (F4).
//! 3. **Precedence-safe construction** — composite operands are wrapped in
//!    `Expr::Nested` exactly when the operator precedence requires it, because
//!    `sqlparser`'s `Display` for a binary op emits no precedence parentheses.
//!    Without this, a *constructed* `mul(add(a,b), c)` displays as `a + b * c`
//!    and reparses as the wrong expression — a wrong number in valid SQL (G1).

use sqlparser::ast::helpers::attached_token::AttachedToken;
use sqlparser::ast::{
    BinaryOperator, CaseWhen, CastKind, DataType, ExactNumberInfo, Expr, Function, FunctionArg,
    FunctionArgExpr, FunctionArgumentList, FunctionArguments, Ident, ObjectName, ObjectNamePart,
    UnaryOperator, Value,
};

use crate::error::{DiffError, Result};

// ---------------------------------------------------------------------------
// Literals and constant inspection
// ---------------------------------------------------------------------------

/// Format a *finite, non-negative* `f64` as the digits of a SQL numeric literal,
/// always with a decimal point (or exponent) so it reads as floating-point.
/// Negativity is represented structurally by [`num`] (as a unary minus), not in
/// the digits — so this never emits a leading `-`.
fn format_f64(v: f64) -> String {
    debug_assert!(
        v.is_finite() && v >= 0.0,
        "format_f64 expects a finite, non-negative value (got {v})"
    );
    let s = format!("{v}");
    if s.contains(['.', 'e', 'E']) {
        s
    } else {
        format!("{s}.0")
    }
}

/// A bare numeric-literal expression for the finite, non-negative value `v`.
fn raw_num(v: f64) -> Expr {
    Expr::Value(Value::Number(format_f64(v), false).with_empty_span())
}

/// A numeric literal expression for the finite value `v` (e.g. `1.0`, `2.0`,
/// `0.6931471805599453`).
///
/// A negative value is emitted as a *unary minus* applied to the magnitude
/// (`-1.0` ⇒ `UnaryOp{Minus, 1.0}`) — exactly the AST shape `sqlparser` produces
/// when it parses `-1.0`. This is what makes the §5 round-trip invariant
/// (`reparse(render(d)) == d` modulo `Nested`) hold for negative literals too;
/// emitting a `Value("-1.0")` would reparse to a `UnaryOp` and break it
/// (round-3 review #46).
///
/// `v` must be finite. For a *compile-time-known* finite constant (`0`, `1`,
/// `2`, `ln 2`, …) call `num` directly. For any value *computed from user input*
/// — which can overflow to `inf` or produce `NaN` — call [`finite_num`] instead,
/// which fails loud rather than emit an invalid `inf`/`NaN` literal (#33). This
/// is why `num` is not part of the public `build` surface: external callers get
/// the checked [`finite_num`], so a non-finite value can never silently become
/// `inf.0` in a release build.
pub(crate) fn num(v: f64) -> Expr {
    debug_assert!(v.is_finite(), "num expects a finite value (got {v})");
    if v < 0.0 {
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr: Box::new(raw_num(-v)),
        }
    } else {
        // `+0.0` and `-0.0` both land here (`-0.0 < 0.0` is false); normalize so
        // `-0.0` never renders as the literal `-0.0`.
        raw_num(if v == 0.0 { 0.0 } else { v })
    }
}

/// A numeric literal for a value that *might not be finite* — the checked
/// counterpart of [`num`]. Emits the literal if `v` is finite, else a typed
/// [`DiffError::NotImplemented`]: a non-finite value has no valid SQL literal
/// (`inf`/`NaN` are not numbers), so a derivative that would carry one must fail
/// loud, never emit invalid SQL (#33). Use this for every value derived from
/// user input or an arithmetic that can overflow (e.g. `ln(base)`, an
/// out-of-range exponent). This is the single seam through which computed
/// constants become literals.
pub fn finite_num(v: f64) -> Result<Expr> {
    if v.is_finite() {
        Ok(num(v))
    } else {
        Err(DiffError::NotImplemented(format!(
            "cannot emit a non-finite derivative constant ({v}); a non-finite \
             value has no valid SQL literal"
        )))
    }
}

/// The constant `0.0` — the derivative of anything independent of `wrt`.
pub fn zero() -> Expr {
    num(0.0)
}

/// The constant `1.0` — the derivative of `wrt` itself.
pub fn one() -> Expr {
    num(1.0)
}

/// The `f64` value of a numeric literal expression, if it is one.
///
/// Sees through a single `Expr::Nested` wrapper so folding still recognizes a
/// parenthesized literal.
pub fn as_const(e: &Expr) -> Option<f64> {
    match e {
        Expr::Value(v) => match &v.value {
            Value::Number(s, _) => s.parse::<f64>().ok(),
            _ => None,
        },
        Expr::Nested(inner) => as_const(inner),
        // `sqlparser` parses a negative literal `-2` as `UnaryOp{Minus,
        // Value("2")}`, not `Value("-2")` — so a negated constant must be seen
        // through here, or the `power` rule misclassifies a constant exponent
        // like `-2` as variable and wrongly rejects `power(x, -2)` (#46).
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => as_const(expr).map(|v| -v),
        Expr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => as_const(expr),
        _ => None,
    }
}

/// True if `e` is a numeric literal exactly equal to zero.
pub fn is_zero(e: &Expr) -> bool {
    matches!(as_const(e), Some(v) if v == 0.0)
}

/// True if `e` is a numeric literal exactly equal to one.
pub fn is_one(e: &Expr) -> bool {
    matches!(as_const(e), Some(v) if v == 1.0)
}

// ---------------------------------------------------------------------------
// Precedence-safe assembly (G1)
// ---------------------------------------------------------------------------

/// Binding-precedence of an expression's *top* operator, higher = binds tighter.
/// Self-delimiting forms (literals, identifiers, function calls, `CAST`,
/// already-`Nested`) are atoms and never need wrapping.
fn precedence(e: &Expr) -> u8 {
    match e {
        Expr::BinaryOp { op, .. } => match op {
            BinaryOperator::Plus | BinaryOperator::Minus => 10,
            BinaryOperator::Multiply | BinaryOperator::Divide | BinaryOperator::Modulo => 20,
            _ => 20,
        },
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            ..
        } => 30,
        _ => 100,
    }
}

/// Wrap `e` in `Expr::Nested` iff its precedence is below `threshold`
/// (`strict`) or at-or-below it (`!strict`, for the right operand of a
/// non-commutative operator, where equal precedence still needs parentheses:
/// `a - (b - c)` ≠ `a - b - c`).
fn wrap(e: Expr, threshold: u8, strict: bool) -> Expr {
    let needs = if strict {
        precedence(&e) < threshold
    } else {
        precedence(&e) <= threshold
    };
    if needs {
        Expr::Nested(Box::new(e))
    } else {
        e
    }
}

/// `left op right`, parenthesizing operands only where precedence demands it.
fn binary(left: Expr, op: BinaryOperator, right: Expr) -> Expr {
    let p = match op {
        BinaryOperator::Plus | BinaryOperator::Minus => 10,
        _ => 20,
    };
    let non_commutative = matches!(op, BinaryOperator::Minus | BinaryOperator::Divide);
    Expr::BinaryOp {
        left: Box::new(wrap(left, p, true)),
        op,
        right: Box::new(wrap(right, p, !non_commutative)),
    }
}

/// Wrap `e` in a `CAST(... AS DOUBLE)`. Self-delimiting, so it never needs
/// precedence parentheses as an operand.
pub fn cast_double(e: Expr) -> Expr {
    Expr::Cast {
        kind: CastKind::Cast,
        expr: Box::new(e),
        data_type: DataType::Double(ExactNumberInfo::None),
        array: false,
        format: None,
    }
}

// ---------------------------------------------------------------------------
// The folding builders
// ---------------------------------------------------------------------------

/// `a + b`, dropping a structurally-zero operand.
pub fn add(a: Expr, b: Expr) -> Expr {
    if is_zero(&a) {
        b
    } else if is_zero(&b) {
        a
    } else {
        binary(a, BinaryOperator::Plus, b)
    }
}

/// `a - b`, dropping a zero right operand and turning `0 - b` into `-b`.
pub fn sub(a: Expr, b: Expr) -> Expr {
    if is_zero(&b) {
        a
    } else if is_zero(&a) {
        neg(b)
    } else {
        binary(a, BinaryOperator::Minus, b)
    }
}

/// `a * b`, folding `0 * _ = 0` and `1 * b = b` (and the mirror cases).
pub fn mul(a: Expr, b: Expr) -> Expr {
    if is_zero(&a) || is_zero(&b) {
        zero()
    } else if is_one(&a) {
        b
    } else if is_one(&b) {
        a
    } else {
        binary(a, BinaryOperator::Multiply, b)
    }
}

/// `a / b`, folding `0 / _ = 0` and `a / 1 = a`.
///
/// When a real division is emitted, the numerator is cast to `DOUBLE` so the
/// division is floating-point on every engine — SQL integer division truncates
/// on some and not others, which would make `grad(x/y, y)` on a `BIGINT`
/// column silently wrong (F4). Casting one operand promotes the whole division;
/// casting the *numerator* (not the result) is essential — `CAST(a/b AS DOUBLE)`
/// would truncate before the cast.
pub fn div(a: Expr, b: Expr) -> Expr {
    if is_zero(&a) {
        zero()
    } else if is_one(&b) {
        a
    } else {
        binary(cast_double(a), BinaryOperator::Divide, b)
    }
}

/// `-a`, folding `-0 = 0` and `-(-e) = e`, and parenthesizing a binary operand
/// (`-(a + b)`, `-(a / b)`), since unary minus binds tighter than either.
pub fn neg(a: Expr) -> Expr {
    if is_zero(&a) {
        return zero();
    }
    match a {
        // Double negation cancels. This is not just simplification: without it,
        // `neg(neg(e))` renders as two adjacent minus tokens `--e`, which SQL
        // parses as a line comment — a silently-wrong result in valid-looking
        // SQL (e.g. d/dx(-cos(x)) = sin(x) would emit `--sin(x)`).
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => *expr,
        other => Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr: Box::new(wrap(other, 30, true)),
        },
    }
}

/// `e * e`.
pub fn square(e: Expr) -> Expr {
    mul(e.clone(), e)
}

/// The mathematical sign of `u` as a portable `CASE`, pinning `sign(0) = 0` on
/// every engine: `CASE WHEN u > 0 THEN 1.0 WHEN u < 0 THEN -1.0 ELSE 0.0 END`.
///
/// This is the derivative factor for `abs` (`d/du |u| = sign(u)`). It avoids the
/// engine-specific builtins — DuckDB has only `sign`, DataFusion only `signum`,
/// and the two disagree at `0` (`signum(0) = 1`) — so the emitted derivative is
/// both portable across the target engines and *actually* pins the documented
/// kink convention `abs'(0) = 0` (design.md §5, F12), which a bare `signum(u)`
/// call did not.
pub fn sign(u: Expr) -> Expr {
    let compare = |op: BinaryOperator| Expr::BinaryOp {
        left: Box::new(u.clone()),
        op,
        right: Box::new(zero()),
    };
    Expr::Case {
        case_token: AttachedToken::empty(),
        end_token: AttachedToken::empty(),
        operand: None,
        conditions: vec![
            CaseWhen {
                condition: compare(BinaryOperator::Gt),
                result: one(),
            },
            CaseWhen {
                condition: compare(BinaryOperator::Lt),
                result: num(-1.0),
            },
        ],
        else_result: Some(Box::new(zero())),
    }
}

// ---------------------------------------------------------------------------
// Function-call construction (for the outer factors of chain-rule terms)
// ---------------------------------------------------------------------------

/// Build an unqualified scalar function call `name(args...)`.
pub fn func(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Function(Function {
        name: ObjectName(vec![ObjectNamePart::Identifier(Ident::new(name))]),
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::List(FunctionArgumentList {
            duplicate_treatment: None,
            args: args
                .into_iter()
                .map(|e| FunctionArg::Unnamed(FunctionArgExpr::Expr(e)))
                .collect(),
            clauses: vec![],
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: vec![],
    })
}

/// `f(x)` — a unary call, the common case for chain-rule outer derivatives.
pub fn func1(name: &str, x: Expr) -> Expr {
    func(name, vec![x])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_additive_zero() {
        assert_eq!(add(one(), zero()).to_string(), "1.0");
        assert_eq!(add(zero(), one()).to_string(), "1.0");
    }

    #[test]
    fn folds_multiplicative_identity_and_zero() {
        assert_eq!(mul(one(), num(3.0)).to_string(), "3.0");
        assert_eq!(mul(num(3.0), one()).to_string(), "3.0");
        assert_eq!(mul(zero(), num(3.0)).to_string(), "0.0");
    }

    #[test]
    fn sub_zero_left_is_negation() {
        assert_eq!(
            sub(zero(), Expr::Identifier(Ident::new("b"))).to_string(),
            "-b"
        );
    }

    #[test]
    fn precedence_wrapping_is_semantic_not_cosmetic() {
        // (a+b)*c must keep its parentheses under Display (G1). Without the
        // Nested wrap this would render "a + b * c" and reparse wrongly.
        let a = Expr::Identifier(Ident::new("a"));
        let b = Expr::Identifier(Ident::new("b"));
        let c = Expr::Identifier(Ident::new("c"));
        let e = mul(add(a, b), c);
        assert_eq!(e.to_string(), "(a + b) * c");
    }

    #[test]
    fn non_commutative_right_operand_is_parenthesized() {
        let a = Expr::Identifier(Ident::new("a"));
        let b = Expr::Identifier(Ident::new("b"));
        let c = Expr::Identifier(Ident::new("c"));
        // a - (b + c) must keep parentheses; a - b + c would be wrong.
        assert_eq!(sub(a, add(b, c)).to_string(), "a - (b + c)");
    }

    #[test]
    fn div_casts_numerator_to_double() {
        let x = Expr::Identifier(Ident::new("x"));
        let y = Expr::Identifier(Ident::new("y"));
        // Forces float division; integer x/y would otherwise truncate (F4).
        assert_eq!(div(x, y).to_string(), "CAST(x AS DOUBLE) / y");
    }

    #[test]
    fn div_by_one_folds_without_cast() {
        let x = Expr::Identifier(Ident::new("x"));
        assert_eq!(div(x, one()).to_string(), "x");
    }

    #[test]
    fn num_emits_negatives_as_unary_minus() {
        // A negative literal must match sqlparser's parse shape (UnaryOp{Minus,
        // magnitude}) so derivatives round-trip; the rendered text is still
        // `-2.0` / `-0.5`.
        assert!(matches!(num(-2.0), Expr::UnaryOp { .. }));
        assert_eq!(num(-2.0).to_string(), "-2.0");
        assert_eq!(num(-0.5).to_string(), "-0.5");
        assert_eq!(num(0.0).to_string(), "0.0"); // incl. -0.0 normalization
        assert_eq!(num(-0.0).to_string(), "0.0");
    }

    #[test]
    fn finite_num_rejects_non_finite_values() {
        // The checked emission seam: finite values pass, inf/NaN fail loud
        // (never a silent `inf.0`/`NaN.0` token). This is the public `build`
        // surface, so external callers cannot emit invalid SQL in release.
        assert_eq!(finite_num(2.0).unwrap().to_string(), "2.0");
        assert!(finite_num(f64::INFINITY).is_err());
        assert!(finite_num(f64::NEG_INFINITY).is_err());
        assert!(finite_num(f64::NAN).is_err());
    }
}
