// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The error type for differentiation and rewriting.

use std::fmt;

/// An error produced while differentiating or rewriting SQL.
///
/// Every failure mode of `ddx-core` is one of these. In keeping with design
/// principle 5 — *fail loud, never silently wrong* (design.md §2) — an
/// unsupported construct is always one of these typed errors, never an
/// approximate or silently-zero derivative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffError {
    /// A node or function has no differentiation rule. This is the "permanent
    /// or roadmap" bucket: `atan2` (no rule yet), general `u^v`, `CASE`,
    /// comparisons, string/temporal expressions (design.md §3.6).
    NotImplemented(String),

    /// The `wrt` (or a marker call) is malformed: `wrt` is not a bare column,
    /// wrong argument count, etc.
    InvalidMarker(String),

    /// An occurrence of the `wrt` base name could not be pinned syntactically —
    /// a bare occurrence when `wrt` is qualified, or a qualified occurrence
    /// when `wrt` is bare. Hard error demanding full qualification
    /// (design.md §3.2, F2).
    AmbiguousColumn(String),

    /// A marker argument references an identifier that is a *computed*
    /// select-list alias of a CTE/derived table in the same statement, used as
    /// a non-`wrt` term — differentiation would silently drop terms across the
    /// projection boundary (design.md §3.5, F3/G4).
    ProjectionBoundary(String),

    /// The input SQL did not parse under the given dialect. Only ever reported
    /// for a statement that *contains* a marker (the parse-free pre-gate means
    /// marker-free statements are never parsed, design.md §3.2, F5).
    Parse(String),

    /// An internal invariant was violated (e.g. an empty source span the API
    /// documents as possible, with no safe fallback). Should not occur in
    /// normal use.
    Internal(String),
}

impl fmt::Display for DiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffError::NotImplemented(m) => write!(f, "not implemented: {m}"),
            DiffError::InvalidMarker(m) => write!(f, "invalid marker call: {m}"),
            DiffError::AmbiguousColumn(m) => write!(f, "ambiguous column: {m}"),
            DiffError::ProjectionBoundary(m) => write!(f, "projection boundary: {m}"),
            DiffError::Parse(m) => write!(f, "parse error: {m}"),
            DiffError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for DiffError {}

/// The result type used throughout `ddx-core`.
pub type Result<T> = std::result::Result<T, DiffError>;
