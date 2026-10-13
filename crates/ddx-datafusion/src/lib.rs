// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-datafusion` — the DataFusion adapter for [ddx](https://github.com/xqlsystems/ddx).
//!
//! Differentiate a whole query, as JAX differentiates a function: write the
//! forward pass and the loss as SQL, and get back the gradient of the loss
//! with respect to table columns, computed by DataFusion as a sequence of
//! plans. [`ad`] is the whole API; see its docs.
//!
//! ```sql
//! WITH loss AS (SELECT SUM(val * val) AS l FROM w)
//! SELECT w.i, w.val - 0.1 * g.val AS val
//! FROM w JOIN grad(loss, w.val) g ON w.i = g.i
//! ```
//!
//! DataFusion plans and optimizes the query and writes it as Substrait;
//! [`ddx_ad`] differentiates the plan and returns Substrait plans for
//! DataFusion to run. [`register_stop_gradient`] adds `ddx_stop_gradient`, the
//! one function a forward query may call that ddx gives a meaning to.

#![forbid(unsafe_code)]

pub mod ad;
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod engine;
#[cfg(not(feature = "internals"))]
mod engine;
mod stop_gradient;

pub use stop_gradient::{register_stop_gradient, stop_gradient_udf, STOP_GRADIENT};
