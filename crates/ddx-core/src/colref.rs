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
/// * [`IdentCasing::FoldAll`] — *all* identifiers fold to lowercase, quoted
///   included. (DuckDB, which is fully case-insensitive.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentCasing {
    /// Fold unquoted identifiers only (DataFusion / Postgres / generic).
    FoldUnquoted,
    /// Fold every identifier, quoted included (DuckDB).
    FoldAll,
}

impl IdentCasing {
    /// The comparison key for a single identifier under this policy.
    pub fn fold(self, id: &Ident) -> String {
        match (id.quote_style, self) {
            // Unquoted: always case-folded.
            (None, _) => id.value.to_ascii_lowercase(),
            // Quoted: folded only for DuckDB.
            (Some(_), IdentCasing::FoldAll) => id.value.to_ascii_lowercase(),
            (Some(_), IdentCasing::FoldUnquoted) => id.value.clone(),
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
