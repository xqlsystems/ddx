// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The error type for plan analysis and differentiation.

use std::fmt;

/// An error produced while analysing or differentiating a plan.
///
/// Design principle 5 — *fail loud, never silently wrong* (design.md §2) — holds
/// one layer up: a relation `ddx-ad` doesn't understand is a typed error, never
/// a relation silently skipped by the backward walk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdError {
    /// The plan uses a relation or construct with no transpose rule (or, at this
    /// stage, no place in the annotated-node index).
    NotImplemented(String),

    /// The plan is not the shape `vjp_query` requires: no root, several roots,
    /// or a relation missing an input it cannot do without.
    InvalidPlan(String),
}

impl fmt::Display for AdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdError::NotImplemented(m) => write!(f, "not implemented: {m}"),
            AdError::InvalidPlan(m) => write!(f, "invalid plan: {m}"),
        }
    }
}

impl std::error::Error for AdError {}

/// The result type used throughout `ddx-ad`.
pub type Result<T> = std::result::Result<T, AdError>;
