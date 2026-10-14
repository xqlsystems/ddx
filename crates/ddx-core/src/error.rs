// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The error type for differentiation.

use std::fmt;

/// An error produced while differentiating an expression.
///
/// Every failure mode of `ddx-core` is one of these. ddx fails loud rather than
/// silently wrong: an unsupported construct is always one of these typed errors, never an
/// approximate or silently-zero derivative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffError {
    /// A node or function has no differentiation rule. This is the "permanent
    /// or roadmap" bucket: `atan2` (no rule yet), general `u^v`, `CASE`,
    /// comparisons, string/temporal expressions.
    NotImplemented(String),

    /// An occurrence of the `wrt` base name could not be pinned syntactically —
    /// a bare occurrence when `wrt` is qualified, or a qualified occurrence
    /// when `wrt` is bare. Hard error demanding full qualification, since a
    /// guess could differentiate with respect to the wrong column.
    AmbiguousColumn(String),

    /// An internal invariant was violated (e.g. an empty source span the API
    /// documents as possible, with no safe fallback). Should not occur in
    /// normal use.
    Internal(String),
}

impl fmt::Display for DiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffError::NotImplemented(m) => write!(f, "not implemented: {m}"),
            DiffError::AmbiguousColumn(m) => write!(f, "ambiguous column: {m}"),
            DiffError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for DiffError {}

/// The result type used throughout `ddx-core`.
pub type Result<T> = std::result::Result<T, DiffError>;
