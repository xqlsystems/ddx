// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-ad` — query-level reverse-mode automatic differentiation (ddx v2).
//!
//! **Status: work in progress (M3).** This crate is the home for the v2 engine described in
//! [`docs/design.md`](../../../docs/design.md) §4: differentiating whole
//! queries (not scalar expressions) by applying one transpose rule per
//! relational primitive — contraction, elementwise, reduce, route,
//! stop-gradient — over `substrait::proto` plans tagged with
//! extension-function markers.
//!
//! So far this crate has only its seams: the [`substrait`] dependency, a typed
//! [`AdError`], and the [`PlanIndex`] the backward walk schedules itself with.
//! The markers, the transpose rules and `vjp_query` land over M3/M4. The scalar
//! core, [`ddx-core`](../ddx_core/index.html), becomes the *elementwise leaf* of
//! this engine (design.md §4.3).
//!
//! Planned surface (design.md §4.4):
//!
//! ```ignore
//! pub fn vjp_query(plan: &Plan, wrt: &[RelRef]) -> Result<BackwardProgram, AdError>;
//!
//! pub struct BackwardProgram {
//!     pub forward_steps: Vec<(Ident, Plan)>,
//!     pub backward_steps: Vec<(Ident, Plan)>,
//!     pub gradients: HashMap<RelRef, Ident>,
//! }
//! ```
//!
//! The four marker names it recognizes (`ddx_contract_mark`,
//! `ddx_reduce_mark`, `ddx_route_mark`, `ddx_stop_gradient`) are Substrait
//! extension-function markers, the same "tag, don't infer" mechanism `grad()`
//! uses in v1, one layer down in the plan.

#![forbid(unsafe_code)]

mod error;
mod index;

/// Re-exported so a downstream crate can't accidentally link a different
/// `substrait` than the one `ddx-ad`'s API is written against (the same reason
/// `ddx-core` re-exports `sqlparser`, design.md §6).
pub use substrait;

pub use error::{AdError, Result};
pub use index::{Node, NodeId, NodeKind, PlanIndex, RelRef};
