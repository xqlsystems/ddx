// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-ad` — query-level reverse-mode automatic differentiation (ddx v2).
//!
//! **Status: work in progress (M3).** This crate is the home for the v2 engine
//! described in [`docs/design.md`](../../../docs/design.md) §4: differentiating
//! whole queries (not scalar expressions) by applying one transpose rule per
//! relational primitive — contraction, elementwise, reduce, route,
//! stop-gradient — over `substrait::proto` plans tagged with
//! extension-function markers.
//!
//! What exists so far is the analysis the rules stand on:
//!
//! - [`Marker`] — the four marker functions a user writes in SQL
//!   (`ddx_contract_mark`, `ddx_reduce_mark`, `ddx_route_mark`,
//!   `ddx_stop_gradient`), recognized by name in the plan's function table.
//! - [`PlanIndex`] and [`Columns`] — the plan's nodes in backward order, and
//!   every output column traced back to where it came from, through the
//!   `Project`/`emit` layers real producers insert.
//! - [`Activity`] — which columns carry gradient, derived from the parameters
//!   the user names ([`Param`], e.g. `weight.val`) rather than from a naming
//!   convention.
//! - [`Analysis`] — all of the above for one plan, plus enforcement of *tag,
//!   don't infer*: a misplaced marker, or an untagged aggregate over a
//!   gradient-carrying column, is an error.
//!
//! The transpose rules and `vjp_query` land over M3/M4. The scalar core,
//! [`ddx-core`](../ddx_core/index.html), becomes the *elementwise leaf* of this
//! engine (design.md §4.3).
//!
//! Planned surface (design.md §4.4):
//!
//! ```ignore
//! pub fn vjp_query(plan: &Plan, wrt: &[Param]) -> Result<BackwardProgram, AdError>;
//! ```

#![forbid(unsafe_code)]

mod activity;
mod analysis;
mod columns;
mod error;
mod expr;
mod index;
mod markers;

#[cfg(test)]
mod test_plans;

/// Re-exported so a downstream crate can't accidentally link a different
/// `substrait` than the one `ddx-ad`'s API is written against (the same reason
/// `ddx-core` re-exports `sqlparser`, design.md §6).
pub use substrait;

pub use activity::{Activity, Param};
pub use analysis::Analysis;
pub use columns::{ColumnDef, Columns};
pub use error::{AdError, Result};
pub use index::{Node, NodeId, NodeKind, PlanIndex, RelRef};
pub use markers::{Functions, Marker};
