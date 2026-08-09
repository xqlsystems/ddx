// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The public entry point: the [`Ddx`] object.
//!
//! The surface is an object rather than free functions so the user rule
//! registry and the dialect/identifier-folding policy have a home — no global
//! state, no later API break (design.md §3.2, F9).

use sqlparser::ast::Expr;
use sqlparser::dialect::Dialect;
use sqlparser::parser::Parser;

use crate::colref::{ColRef, IdentCasing};
use crate::engine::{differentiate, jvp, Rule, RuleRegistry};
use crate::error::{DiffError, Result};
use crate::rewrite::{self, Explanation};

/// The ddx v1 differentiation engine.
///
/// Holds the (extensible) rule registry and the identifier-folding policy.
/// Construct one, optionally register custom rules, then drive it with
/// [`Ddx::rewrite_sql`] (the whole marker path) or the lower-level
/// [`Ddx::differentiate`] / [`Ddx::jvp`] (used by the DataFusion Path B bridge,
/// design.md §3.3).
///
/// # Example
///
/// ```
/// use ddx_core::Ddx;
/// use ddx_core::sqlparser::dialect::GenericDialect;
///
/// let ddx = Ddx::new();
/// let out = ddx
///     .rewrite_sql("SELECT grad(sin(x), x) AS d FROM t", &GenericDialect {})
///     .unwrap();
/// assert_eq!(out, "SELECT (cos(x)) AS d FROM t");
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

    /// The DataFusion-flavored engine (`FoldUnquoted`). Pair with
    /// `GenericDialect`/DataFusion's dialect when calling [`Ddx::rewrite_sql`].
    pub fn for_datafusion() -> Self {
        Self::with_casing(IdentCasing::FoldUnquoted)
    }

    /// The DuckDB-flavored engine (`FoldAll` — DuckDB folds quoted identifiers
    /// too). Pair with `DuckDbDialect` when calling [`Ddx::rewrite_sql`].
    pub fn for_duckdb() -> Self {
        Self::with_casing(IdentCasing::FoldAll)
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
    /// `name`: the rule supplies `f'(u)` and the engine applies the chain rule
    /// (design.md §3.2).
    pub fn register(&mut self, name: &str, rule: Rule) {
        self.rules.register(name, rule);
    }

    /// The whole marker path: rewrite every `grad`/`jvp` call in `sql` to
    /// derivative SQL and return the rewritten text. A statement with no marker
    /// is returned byte-identical (it is never even parsed, design.md §3.2).
    ///
    /// `dialect` is used to *parse* the marker-bearing statement; the
    /// identifier-folding policy used to *match* columns is this engine's
    /// (`casing`) — pair them (e.g. `Ddx::for_duckdb()` with `DuckDbDialect`).
    pub fn rewrite_sql(&self, sql: &str, dialect: &dyn Dialect) -> Result<String> {
        rewrite::rewrite_sql(sql, dialect, self.casing, &self.rules)
    }

    /// Preview what [`Ddx::rewrite_sql`] would do to `sql` — every `grad`/`jvp`
    /// marker and the derivative SQL it becomes, plus the fully rewritten
    /// statement — *without* running anything. The returned [`Explanation`] is
    /// inspectable field-by-field, and prints a readable summary via `Display`,
    /// so it doubles as a quick interactive "what will this do?" for a REPL or
    /// notebook.
    ///
    /// ```
    /// use ddx_core::Ddx;
    /// use ddx_core::sqlparser::dialect::GenericDialect;
    ///
    /// let ddx = Ddx::new();
    /// let ex = ddx
    ///     .explain("SELECT grad(sin(x), x) AS d FROM t", &GenericDialect {})
    ///     .unwrap();
    /// assert_eq!(ex.rewritten, "SELECT (cos(x)) AS d FROM t");
    /// assert_eq!(ex.steps.len(), 1);
    /// assert_eq!(ex.steps[0].marker, "grad(sin(x), x)");
    /// assert_eq!(ex.steps[0].derivative, "(cos(x))");
    /// println!("{ex}"); // human-readable, inspect-before-you-run summary
    /// ```
    pub fn explain(&self, sql: &str, dialect: &dyn Dialect) -> Result<Explanation> {
        rewrite::explain_sql(sql, dialect, self.casing, &self.rules)
    }

    /// Differentiate an AST expression with respect to `wrt`. The lower-level
    /// entry the DataFusion bridge drives (design.md §3.3, Path B).
    pub fn differentiate(&self, e: &Expr, wrt: &ColRef) -> Result<Expr> {
        differentiate(e, wrt, self.casing, &self.rules)
    }

    /// Forward-mode directional derivative: seed a tangent on each column in
    /// `seeds` and push it through `e` (design.md §3.6).
    ///
    /// (The design sketch names a `HashMap<ColRef, Expr>`; a slice of pairs is
    /// used instead because `ColRef` equality is dialect-dependent — folding
    /// makes it a poor hash key — so an explicit, ordered seed list is clearer
    /// and preserves match order.)
    pub fn jvp(&self, e: &Expr, seeds: &[(ColRef, Expr)]) -> Result<Expr> {
        jvp(e, seeds, self.casing, &self.rules)
    }

    /// The "calculus compiler" escape hatch: differentiate the scalar
    /// expression `expr` (SQL text) with respect to the column `wrt` (a bare
    /// column name), returning the derivative as SQL text — for embedding an
    /// update rule where a marker can't reach (design.md §3.6).
    pub fn differentiate_sql(
        &self,
        expr: &str,
        wrt: &str,
        dialect: &dyn Dialect,
    ) -> Result<String> {
        let parsed = parse_expr(expr, dialect)?;
        let wrt_expr = parse_expr(wrt, dialect)?;
        let wrt_col = ColRef::from_wrt_arg("differentiate_sql", &wrt_expr)?;
        let derivative = self.differentiate(&parsed, &wrt_col)?;
        Ok(derivative.to_string())
    }
}

/// Parse a single scalar expression from text under `dialect`.
fn parse_expr(text: &str, dialect: &dyn Dialect) -> Result<Expr> {
    Parser::new(dialect)
        .try_with_sql(text)
        .and_then(|mut p| p.parse_expr())
        .map_err(|e| DiffError::Parse(format!("failed to parse expression `{text}`: {e}")))
}
