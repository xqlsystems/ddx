// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The error type for plan analysis and differentiation.

use std::fmt;

/// An error from the analysis or the differentiation of a plan.
///
/// Design principle 5 in design.md §2 is "fail loud, never silently wrong", and
/// it holds one layer up as well. A relation or an expression that `ddx-ad` does
/// not understand becomes one of these errors. The backward walk never passes
/// over such a relation in silence.
///
/// The variants match the granularity of `ddx_core::DiffError`. A caller can
/// tell a missing rule from a malformed request, and both from a defect in ddx.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdError {
    /// The plan uses a relation or an expression that ddx-ad cannot
    /// differentiate yet.
    NotImplemented(String),

    /// The plan is malformed. For example, the plan has no root, a relation
    /// misses an input that it needs, or a function anchor has no declaration.
    InvalidPlan(String),

    /// The `wrt` list does not match the plan. For example, the plan never reads
    /// the named table, or the table has no column of that name.
    InvalidWrt(String),

    /// A ddx marker is malformed, or the marker sits where its meaning is
    /// ambiguous. For example, a contraction marker outside a `SUM`, or a marker
    /// with two arguments.
    InvalidMarker(String),

    /// An operation that needs a tag has none (design.md §2, principle 3).
    ///
    /// For example, a `SUM` over a gradient-carrying column that names neither a
    /// contraction nor a reduction. The user wrote no marker at all. For a
    /// marker that is present but wrong, see [`AdError::InvalidMarker`].
    Untagged(String),

    /// An internal invariant of ddx-ad is broken. Normal use does not produce
    /// this error. It is an error and not a panic. In a library that must be
    /// correct, a crash and a wrong answer are both worse than a typed
    /// failure.
    Internal(String),
}

impl fmt::Display for AdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdError::NotImplemented(m) => write!(f, "not implemented: {m}"),
            AdError::InvalidPlan(m) => write!(f, "invalid plan: {m}"),
            AdError::InvalidWrt(m) => write!(f, "invalid wrt: {m}"),
            AdError::InvalidMarker(m) => write!(f, "invalid marker: {m}"),
            AdError::Untagged(m) => write!(f, "untagged operation: {m}"),
            AdError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for AdError {}

/// The result type used throughout `ddx-ad`.
pub type Result<T> = std::result::Result<T, AdError>;
