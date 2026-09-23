// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The error type.

use std::fmt;

use ddx_core::DiffError;

/// Why `ddx-ad` refused a plan.
///
/// Every refusal is one of these, never a gradient that is silently wrong or
/// silently zero (design.md §2, principle 5).
#[derive(Debug, Clone, PartialEq)]
pub enum AdError {
    /// The plan uses something `ddx-ad` has no transpose rule for yet, on a
    /// path the gradient flows along.
    NotImplemented(String),
    /// `grad` was asked for the gradient of something that is not a loss: one
    /// row and one column that depends on `wrt`.
    NotScalar(String),
    /// A `wrt` column names a table or column the plan does not read.
    UnknownWrt(String),
    /// The plan is not one ddx can read: malformed, or missing a field every
    /// producer fills in.
    InvalidPlan(String),
    /// The scalar engine refused an elementwise expression.
    Diff(DiffError),
    /// A bug in `ddx-ad`: an invariant one part relies on another to keep.
    Internal(String),
}

impl fmt::Display for AdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdError::NotImplemented(m) => write!(f, "not supported by ddx-ad yet: {m}"),
            AdError::NotScalar(m) => write!(f, "not a scalar loss: {m}"),
            AdError::UnknownWrt(m) => write!(f, "unknown wrt column: {m}"),
            AdError::InvalidPlan(m) => write!(f, "invalid Substrait plan: {m}"),
            AdError::Diff(e) => write!(f, "{e}"),
            AdError::Internal(m) => write!(f, "internal error in ddx-ad (a bug): {m}"),
        }
    }
}

impl std::error::Error for AdError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AdError::Diff(e) => Some(e),
            _ => None,
        }
    }
}

impl From<DiffError> for AdError {
    fn from(e: DiffError) -> Self {
        AdError::Diff(e)
    }
}

/// `Result` with [`AdError`].
pub type Result<T, E = AdError> = std::result::Result<T, E>;
