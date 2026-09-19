// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The error type for plan analysis and differentiation.

use std::fmt;

/// An error produced while analysing or differentiating a plan.
///
/// Design principle 5 — *fail loud, never silently wrong* (design.md §2) — holds
/// one layer up: a relation or expression `ddx-ad` doesn't understand is a typed
/// error, never something the backward walk quietly skips.
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

    /// A ddx marker is missing, misplaced, or malformed — or an operation that
    /// must be tagged (design.md §2, principle 3) isn't.
    Marker(String),
}

impl fmt::Display for AdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdError::NotImplemented(m) => write!(f, "not implemented: {m}"),
            AdError::InvalidPlan(m) => write!(f, "invalid plan: {m}"),
            AdError::InvalidWrt(m) => write!(f, "invalid wrt: {m}"),
            AdError::Marker(m) => write!(f, "marker: {m}"),
        }
    }
}

impl std::error::Error for AdError {}

/// The result type used throughout `ddx-ad`.
pub type Result<T> = std::result::Result<T, AdError>;
