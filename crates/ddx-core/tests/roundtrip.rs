// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The semantic round-trip property test (design.md §5).
//!
//! A test that only checks the rewritten SQL *parses* sails right past the
//! precedence bug G1 found: `(a+b)*c` rendered without its parentheses becomes
//! `a + b * c`, which parses fine — just as the wrong tree. So the real
//! invariant is: rendering a constructed derivative and reparsing it must yield
//! the *same* expression, **modulo `Expr::Nested`** (normalize parentheses on
//! both sides before comparing).

use std::ops::ControlFlow;

use ddx_core::sqlparser::ast::{Expr, Ident, VisitMut, VisitorMut};
use ddx_core::sqlparser::dialect::GenericDialect;
use ddx_core::sqlparser::parser::Parser;
use ddx_core::{ColRef, Ddx};

/// Remove every `Expr::Nested` wrapper so two trees that differ only in
/// (redundant or precedence) parentheses compare equal.
struct StripNested;

impl VisitorMut for StripNested {
    type Break = ();
    fn post_visit_expr(&mut self, e: &mut Expr) -> ControlFlow<()> {
        // Post-order: children are already stripped. Collapse any Nested chain.
        while matches!(e, Expr::Nested(_)) {
            let owned = std::mem::replace(e, Expr::Identifier(Ident::new("")));
            if let Expr::Nested(inner) = owned {
                *e = *inner;
            }
        }
        ControlFlow::Continue(())
    }
}

fn strip(mut e: Expr) -> Expr {
    let _ = VisitMut::visit(&mut e, &mut StripNested);
    e
}

fn parse(text: &str) -> Expr {
    Parser::new(&GenericDialect {})
        .try_with_sql(text)
        .and_then(|mut p| p.parse_expr())
        .unwrap_or_else(|e| panic!("reparse of `{text}` failed: {e}"))
}

/// The load-bearing invariant (design.md §5): rendering the *constructed*
/// derivative AST `d` and reparsing it must yield the same AST, **modulo
/// `Nested`**. This compares against `d` itself (via [`Ddx::differentiate`],
/// which returns the AST) — not against another parse of the rendered text,
/// which would be vacuously equal and hide a constructed-vs-parsed mismatch
/// such as a negative literal emitted as `Value("-1.0")` (round-3 review #46).
fn assert_roundtrips(expr: &str, wrt: &str) {
    let ddx = Ddx::new();
    let d = ddx
        .differentiate(&parse(expr), &ColRef::bare(wrt))
        .unwrap_or_else(|e| panic!("differentiate d/d{wrt} ({expr}): {e}"));
    let rendered = d.to_string();
    let reparsed = parse(&rendered);
    assert_eq!(
        strip(reparsed),
        strip(d),
        "reparse(render(d)) != d modulo Nested for d/d{wrt} ({expr}); rendered = {rendered}"
    );
}

#[test]
fn precedence_sensitive_derivatives_round_trip() {
    // Each of these exercises a rule whose output nests a lower-precedence
    // operand inside a higher-precedence one — exactly where G1 bites.
    assert_roundtrips("(a + b) * c", "a"); // product rule over a sum
    assert_roundtrips("x / y", "x"); // quotient rule + DOUBLE cast
    assert_roundtrips("x / y", "y"); // quotient rule, negative numerator
    assert_roundtrips("sin(x) * x", "x"); // chain × product
    assert_roundtrips("power(x, 3)", "x"); // constant-exponent power
    assert_roundtrips("sin(x * y + x)", "x"); // chain over a compound argument
    assert_roundtrips("1 / (x + y)", "x"); // reciprocal of a sum
    assert_roundtrips("sqrt(x * x + y * y)", "x"); // nested composite
    assert_roundtrips("a * b * c * d", "a"); // n-factor product (swell shape)
    assert_roundtrips("exp(x) / (x - 1)", "x"); // quotient with composite denom
}

#[test]
fn negative_literal_derivatives_round_trip() {
    // The cases the original 10 avoided — every one emits a negative literal,
    // which must be a `UnaryOp{Minus, ..}` (matching sqlparser's parse), not a
    // `Value("-…")` that would reparse to a different tree (#46).
    assert_roundtrips("abs(x)", "x"); // CASE ... THEN -1.0 ...
    assert_roundtrips("power(x, -2)", "x"); // -2 * power(x, -3.0)
    assert_roundtrips("power(x, 0.5)", "x"); // emits exponent -0.5
    assert_roundtrips("x / y", "y"); // negative numerator, -(x)/(y*y)
    assert_roundtrips("cos(x) * x", "x"); // -sin(x) * x + cos(x)
}

#[test]
fn strip_nested_actually_normalizes() {
    // Sanity: the normalizer collapses parentheses, so an unparenthesized and a
    // parenthesized spelling of the same expression compare equal after strip.
    assert_eq!(strip(parse("a + b * c")), strip(parse("a + (b * c)")));
    // ...but genuinely different groupings stay different (the test has teeth).
    assert_ne!(strip(parse("(a + b) * c")), strip(parse("a + b * c")));
}
