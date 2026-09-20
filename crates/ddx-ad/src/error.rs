// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The error type for plan analysis and differentiation.

use std::fmt;

/// An error produced while analysing or differentiating a plan.
///
/// Design principle 5 — *fail loud, never silently wrong* (design.md §2) — holds
/// one layer up: a relation or expression `ddx-ad` doesn't understand is a typed
/// error, never something the backward walk quietly skips. The variants mirror
/// `ddx_core::DiffError`'s granularity, so a caller can tell a missing rule from
/// a malformed request from a bug in ddx.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdError {
    /// The plan uses a relation, expression or construct ddx-ad can't
    /// differentiate through yet.
    NotImplemented(String),

    /// The plan is malformed: no root, an input missing, a field reference past
    /// the end of its input, a function anchor that was never declared.
    InvalidPlan(String),

    /// The `wrt` list doesn't describe the plan: a table the plan never reads, a
    /// column the table doesn't have, or an output that depends on none of them.
    InvalidWrt(String),

    /// A ddx marker is malformed or sits where its meaning would be ambiguous —
    /// a contraction marker outside a `SUM`, a marker with two arguments.
    InvalidMarker(String),

    /// An operation that must be tagged isn't (design.md §2, principle 3): a
    /// `SUM` over a gradient-carrying column that says neither *contraction* nor
    /// *reduction*, or a `MAX` that says neither *route* nor *stop gradient*.
    /// The user wrote no marker at all, which is why this is not
    /// [`AdError::InvalidMarker`].
    Untagged(String),

    /// An internal invariant was violated. Should not occur in normal use; it is
    /// an error rather than a panic because a wrong answer and a crash are both
    /// worse than a typed failure in a correctness-critical library.
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
