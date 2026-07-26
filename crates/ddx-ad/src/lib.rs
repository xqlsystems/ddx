// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-ad` — query-level reverse-mode automatic differentiation (ddx v2).
//!
//! **Status: scaffold.** This crate is the home for the v2 engine described in
//! [`docs/design.md`](../../../docs/design.md) §4: differentiating whole
//! queries (not scalar expressions) by applying one transpose rule per
//! relational primitive — contraction, elementwise, reduce, route,
//! stop-gradient — over `substrait::proto` plans tagged with
//! extension-function markers.
//!
//! Nothing here is implemented yet. M0 (the current milestone) delivers only
//! the scalar core, [`ddx-core`](../ddx_core/index.html), which becomes the
//! *elementwise leaf* of this engine (design.md §4.3). The public surface
//! sketched below is the M3/M4 target and exists here as a compile-checked
//! seam, not a working API.
//!
//! Planned surface (design.md §4.4):
//!
//! ```ignore
//! pub fn vjp_query(plan: &Plan, wrt: &[RelRef]) -> Result<BackwardProgram, DiffError>;
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

/// Milestone marker: v2 (`ddx-ad`) is not implemented in M0.
///
/// See [`docs/design.md`](../../../docs/design.md) §4 and milestones M3/M4.
pub const STATUS: &str = "scaffold: query-level reverse-mode AD lands in M3/M4";
