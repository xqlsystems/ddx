// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Column identity read off the AST, compared with per-dialect identifier
//! folding rather than raw-string equality (design.md §3.2, F1).

use sqlparser::ast::{Expr, Ident};

use crate::error::{DiffError, Result};

/// How a dialect folds identifiers for case-insensitive comparison.
///
/// SQL unquoted identifiers are case-insensitive, so `grad(Temp*Temp, temp)`
/// must match — otherwise it silently differentiates to `0`. The exact rule is
/// per-dialect (F1):
///
/// * [`IdentCasing::FoldUnquoted`] — unquoted identifiers fold to lowercase;
///   quoted identifiers keep their case. (DataFusion, Postgres, the generic
///   dialect.)
/// * [`IdentCasing::FoldUnquotedUpper`] — the same rule with the opposite
///   target case: unquoted identifiers fold to *uppercase*. (Snowflake, Oracle.)
/// * [`IdentCasing::FoldAll`] — *all* identifiers fold, quoted included, so
///   case never distinguishes two columns. (DuckDB, Spark, MySQL.)
/// * [`IdentCasing::FoldNone`] — no identifier folds; `x` and `X` are simply
///   different columns. (ClickHouse.)
///
/// The two unquoted-only policies are not interchangeable, and the difference is
/// visible rather than cosmetic: `X` matches `"X"` under `FoldUnquotedUpper` and
/// `"x"` under `FoldUnquoted`. Applying the wrong one to Snowflake does not just
/// miss — it can match the *other* column and return a confidently wrong
/// nonzero derivative.
// More engines than these three families exist, and each one found is a new
// variant. Marking the enum non-exhaustive makes that an additive change for
// anyone matching on it downstream, instead of a breaking release per engine.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentCasing {
    /// Fold unquoted identifiers to lowercase (DataFusion / Postgres / generic).
    FoldUnquoted,
    /// Fold unquoted identifiers to uppercase (Snowflake / Oracle).
    FoldUnquotedUpper,
    /// Fold every identifier, quoted included (DuckDB / Spark / MySQL).
    FoldAll,
    /// Fold nothing; identifiers are compared as written (ClickHouse).
    FoldNone,
}

impl IdentCasing {
    /// The comparison key for a single identifier under this policy.
    ///
    /// Only equality of the returned keys is meaningful — which case they fold
    /// *to* is arbitrary, so long as an unquoted identifier and a quoted one
    /// land on the same key exactly when the engine would resolve them to the
    /// same column.
    pub fn fold(self, id: &Ident) -> String {
        match (self, id.quote_style) {
            // Case-insensitive throughout: quoting changes nothing.
            (IdentCasing::FoldAll, _) => id.value.to_ascii_lowercase(),
            // Case-sensitive throughout: nothing folds, quoted or not.
            (IdentCasing::FoldNone, _) => id.value.clone(),
            // Quoting pins the case exactly; an unquoted identifier folds to
            // whichever case the engine normalizes to, and that choice is what
            // decides which quoted identifiers it then collides with.
            (_, Some(_)) => id.value.clone(),
            (IdentCasing::FoldUnquoted, None) => id.value.to_ascii_lowercase(),
            (IdentCasing::FoldUnquotedUpper, None) => id.value.to_ascii_uppercase(),
        }
    }
}

/// A column reference: an optional qualifier and a name, taken straight off the
/// AST. Stores `sqlparser` [`Ident`]s (which keep quote-style) and compares
/// with dialect-aware folding, never raw-string equality.
#[derive(Debug, Clone)]
pub struct ColRef {
    /// The qualifier (`a` in `a.x`), if the reference was compound.
    pub qualifier: Option<Ident>,
    /// The column name (`x` in `a.x`, or in a bare `x`).
    pub name: Ident,
}

/// Whether a column occurrence *is* the differentiation variable, and if its
/// identity relative to `wrt` could be established syntactically at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Match {
    /// This occurrence is the `wrt` column — its tangent is the seed.
    Is,
    /// This occurrence is definitely a different column — tangent zero.
    Not,
    /// The occurrence's base name matches `wrt` but its qualification can't be
    /// pinned syntactically — a bare occurrence when `wrt` is qualified, or a
    /// qualified occurrence when `wrt` is bare. Hard error (F2).
    Ambiguous,
}

impl ColRef {
    /// Build a bare (unqualified) column reference by name.
    pub fn bare(name: impl Into<String>) -> Self {
        ColRef {
            qualifier: None,
            name: Ident::new(name.into()),
        }
    }

    /// Read a `ColRef` from a column-reference expression
    /// (`Identifier`/`CompoundIdentifier`, seeing through a `Nested` wrapper).
    /// Returns `None` for any expression that is not a column reference.
    pub fn from_expr(e: &Expr) -> Option<ColRef> {
        match e {
            Expr::Identifier(id) => Some(ColRef {
                qualifier: None,
                name: id.clone(),
            }),
            Expr::CompoundIdentifier(parts) => parts.last().map(|last| {
                let qualifier = if parts.len() >= 2 {
                    Some(parts[parts.len() - 2].clone())
                } else {
                    None
                };
                ColRef {
                    qualifier,
                    name: last.clone(),
                }
            }),
            Expr::Nested(inner) => ColRef::from_expr(inner),
            _ => None,
        }
    }

    /// Parse the `wrt` argument of a marker: it must be a bare column
    /// (`Identifier`/`CompoundIdentifier`), never an expression (F: the design
    /// rejects `grad(x*y, x+y)`).
    pub fn from_wrt_arg(func: &str, e: &Expr) -> Result<ColRef> {
        ColRef::from_expr(e).ok_or_else(|| {
            DiffError::InvalidMarker(format!(
                "{func}(): the differentiation variable must be a bare column, but got `{e}`. \
                 Differentiate with respect to a single column (e.g. `{func}(x * y, x)`), not an \
                 expression like `x + y`"
            ))
        })
    }

    /// Classify an occurrence `self` against the differentiation variable
    /// `wrt` under a folding policy — the whole of the ambiguity guard (F2).
    ///
    /// The guard fires (returns [`Match::Ambiguous`]) *only* on an uncertain
    /// occurrence of the `wrt` base name; a non-matching name is always
    /// [`Match::Not`], and a fully-qualified unambiguous match (e.g. `a.x`
    /// against `a.x`) is [`Match::Is`] with no error.
    pub fn classify(&self, wrt: &ColRef, casing: IdentCasing) -> Match {
        if casing.fold(&self.name) != casing.fold(&wrt.name) {
            // Different base name — unrelated column, no ambiguity possible.
            return Match::Not;
        }
        match (&self.qualifier, &wrt.qualifier) {
            // Both qualified: identity is fully determined by the qualifier.
            (Some(sq), Some(wq)) => {
                if casing.fold(sq) == casing.fold(wq) {
                    Match::Is
                } else {
                    Match::Not
                }
            }
            // Both bare, same name: this is the wrt.
            (None, None) => Match::Is,
            // A qualified occurrence when wrt is bare, or a bare occurrence
            // when wrt is qualified: cannot be pinned syntactically.
            (Some(_), None) | (None, Some(_)) => Match::Ambiguous,
        }
    }

    /// Render for error messages (e.g. `a.x` or `x`).
    pub fn display(&self) -> String {
        match &self.qualifier {
            Some(q) => format!("{q}.{}", self.name),
            None => self.name.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> Ident {
        Ident::new(s)
    }

    fn quoted(s: &str) -> Ident {
        Ident::with_quote('"', s)
    }

    #[test]
    fn unquoted_folds_case_in_every_dialect() {
        assert_eq!(
            IdentCasing::FoldUnquoted.fold(&id("Temp")),
            IdentCasing::FoldUnquoted.fold(&id("temp"))
        );
        assert_eq!(
            IdentCasing::FoldAll.fold(&id("Temp")),
            IdentCasing::FoldAll.fold(&id("temp"))
        );
        assert_eq!(
            IdentCasing::FoldUnquotedUpper.fold(&id("Temp")),
            IdentCasing::FoldUnquotedUpper.fold(&id("temp"))
        );
    }

    #[test]
    fn an_unquoted_identifier_matches_the_quoting_its_engine_normalizes_to() {
        // The whole reason FoldUnquotedUpper exists. Postgres resolves bare `X`
        // to "x"; Snowflake resolves it to "X". Using one engine's rule on the
        // other does not merely fail to match — it matches the *opposite*
        // column, which is a wrong nonzero derivative rather than a zero.
        assert_eq!(
            IdentCasing::FoldUnquoted.fold(&id("X")),
            IdentCasing::FoldUnquoted.fold(&quoted("x"))
        );
        assert_ne!(
            IdentCasing::FoldUnquoted.fold(&id("X")),
            IdentCasing::FoldUnquoted.fold(&quoted("X"))
        );

        assert_eq!(
            IdentCasing::FoldUnquotedUpper.fold(&id("X")),
            IdentCasing::FoldUnquotedUpper.fold(&quoted("X"))
        );
        assert_ne!(
            IdentCasing::FoldUnquotedUpper.fold(&id("X")),
            IdentCasing::FoldUnquotedUpper.fold(&quoted("x"))
        );
    }

    #[test]
    fn quoted_folding_is_per_dialect() {
        // DuckDB folds quoted; DataFusion/Postgres keep case.
        assert_eq!(
            IdentCasing::FoldAll.fold(&quoted("Temp")),
            IdentCasing::FoldAll.fold(&quoted("temp"))
        );
        assert_ne!(
            IdentCasing::FoldUnquoted.fold(&quoted("Temp")),
            IdentCasing::FoldUnquoted.fold(&quoted("temp"))
        );
    }

    #[test]
    fn bare_wrt_matches_bare_occurrence() {
        let x = ColRef::bare("x");
        assert_eq!(x.classify(&x, IdentCasing::FoldUnquoted), Match::Is);
        assert_eq!(
            ColRef::bare("y").classify(&x, IdentCasing::FoldUnquoted),
            Match::Not
        );
    }

    #[test]
    fn qualified_wrt_disambiguates_across_a_join() {
        // grad(a.x * b.x, a.x): a.x is the wrt, b.x is a different column.
        let ax = ColRef {
            qualifier: Some(id("a")),
            name: id("x"),
        };
        let bx = ColRef {
            qualifier: Some(id("b")),
            name: id("x"),
        };
        assert_eq!(ax.classify(&ax, IdentCasing::FoldUnquoted), Match::Is);
        assert_eq!(bx.classify(&ax, IdentCasing::FoldUnquoted), Match::Not);
    }

    #[test]
    fn bare_occurrence_with_qualified_wrt_is_ambiguous() {
        // grad(x * a.x, a.x): bare x might be a.x — demand qualification.
        let bare_x = ColRef::bare("x");
        let ax = ColRef {
            qualifier: Some(id("a")),
            name: id("x"),
        };
        assert_eq!(
            bare_x.classify(&ax, IdentCasing::FoldUnquoted),
            Match::Ambiguous
        );
    }

    #[test]
    fn qualified_occurrence_with_bare_wrt_is_ambiguous() {
        // grad(a.x * b.x, x): bare wrt x, qualified occurrences — ambiguous.
        let ax = ColRef {
            qualifier: Some(id("a")),
            name: id("x"),
        };
        let bare_x = ColRef::bare("x");
        assert_eq!(
            ax.classify(&bare_x, IdentCasing::FoldUnquoted),
            Match::Ambiguous
        );
    }
}
