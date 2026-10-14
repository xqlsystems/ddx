// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The public entry point: the [`Ddx`] object.
//!
//! The surface is an object rather than free functions so the user rule
//! registry and the dialect/identifier-folding policy have a home — no global
//! state, and room to add settings later without breaking the API.

use sqlparser::ast::Expr;

use crate::colref::{ColRef, IdentCasing};
use crate::engine::{differentiate, jvp, Rule, RuleRegistry};
use crate::error::Result;

/// The scalar differentiator.
///
/// Holds the (extensible) rule registry and the identifier-folding policy.
/// Construct one, optionally register custom rules, then drive it with
/// [`Ddx::differentiate`] or [`Ddx::jvp`].
///
/// # Example
///
/// ```
/// use ddx_core::sqlparser::dialect::GenericDialect;
/// use ddx_core::sqlparser::parser::Parser;
/// use ddx_core::{ColRef, Ddx};
///
/// let e = Parser::new(&GenericDialect {})
///     .try_with_sql("sin(x)")
///     .unwrap()
///     .parse_expr()
///     .unwrap();
/// let d = Ddx::new().differentiate(&e, &ColRef::bare("x")).unwrap();
/// assert_eq!(d.to_string(), "cos(x)");
/// ```
#[derive(Clone)]
pub struct Ddx {
    rules: RuleRegistry,
    casing: IdentCasing,
}

impl Default for Ddx {
    fn default() -> Self {
        Self::new()
    }
}

impl Ddx {
    /// A new engine with the built-in rule set and the generic
    /// (`FoldUnquoted`) identifier policy — the DataFusion/Postgres rule
    /// (unquoted identifiers fold to lowercase, quoted keep case).
    pub fn new() -> Self {
        Ddx {
            rules: RuleRegistry::new(),
            casing: IdentCasing::FoldUnquoted,
        }
    }

    /// A new engine with the built-in rules and an explicit identifier policy.
    pub fn with_casing(casing: IdentCasing) -> Self {
        Ddx {
            rules: RuleRegistry::new(),
            casing,
        }
    }

    /// The unary function names this engine can differentiate, sorted.
    ///
    /// Read from the rule registry, so it reflects what is implemented rather
    /// than a list kept in step by hand — and it includes anything registered
    /// through [`Ddx::register`].
    pub fn unary_rule_names(&self) -> Vec<String> {
        self.rules.unary_names()
    }

    /// The identifier-folding policy this engine compares columns under.
    pub fn casing(&self) -> IdentCasing {
        self.casing
    }

    /// Register (or override) a user differentiation rule for a unary function
    /// `name`: the rule supplies `f'(u)` and the engine applies the chain rule.
    pub fn register(&mut self, name: &str, rule: Rule) {
        self.rules.register(name, rule);
    }

    /// Differentiate an AST expression with respect to `wrt`.
    pub fn differentiate(&self, e: &Expr, wrt: &ColRef) -> Result<Expr> {
        differentiate(e, wrt, self.casing, &self.rules)
    }

    /// Forward-mode directional derivative: seed a tangent on each column in
    /// `seeds` and push it through `e`.
    ///
    /// (The seeds are a slice of pairs rather than a `HashMap<ColRef, Expr>`
    /// because `ColRef` equality is dialect-dependent — folding
    /// makes it a poor hash key — so an explicit, ordered seed list is clearer
    /// and preserves match order.)
    pub fn jvp(&self, e: &Expr, seeds: &[(ColRef, Expr)]) -> Result<Expr> {
        jvp(e, seeds, self.casing, &self.rules)
    }
}
