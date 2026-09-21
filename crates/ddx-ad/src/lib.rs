// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-ad`: query-level reverse-mode automatic differentiation, which is ddx
//! v2.
//!
//! Status: work in progress (M3). This crate holds the v2 engine that
//! [`docs/design.md`](../../../docs/design.md) §4 describes. The engine
//! differentiates whole queries rather than scalar expressions. It applies one
//! transpose rule for each relational primitive, and the primitives are
//! contraction, elementwise, reduce, route and stop-gradient. It operates on
//! `substrait::proto` plans that carry extension-function markers.
//!
//! What exists so far is the analysis that the rules stand on:
//!
//! - [`Marker`] holds the four marker functions that a user writes in SQL:
//!   `ddx_contract_mark`, `ddx_reduce_mark`, `ddx_route_mark` and
//!   `ddx_stop_gradient`. ddx recognizes each one by name in the function table
//!   of the plan. There are four markers and five rules, because elementwise
//!   needs no marker, and because `ddx_stop_gradient` is an operation on the
//!   gradient rather than a tag ([`Marker::is_tag`]).
//! - [`PlanIndex`] and [`Columns`] give the nodes of the plan in backward order,
//!   and trace every output column back to its origin. The trace passes through
//!   the `Project` and `emit` layers that a producer inserts. The output columns
//!   of a node ([`Col`]) and the positions that its expressions read
//!   ([`Field`]) have separate types, because they are separate numberings.
//! - [`Activity`] reports which columns carry gradient. It derives the answer
//!   from the columns that the user names ([`ColumnRef`], such as
//!   `weight.val`), and not from a naming convention. It answers that question
//!   only. Dim-ness is structural: ddx reads it from the join conditions and the
//!   grouping keys of the consuming node, and never infers it.
//! - [`Analysis`] holds all of the above for one plan. It also enforces "tag
//!   explicitly, never infer": a misplaced marker is an error, and so is an
//!   untagged aggregate over a column that carries gradient.
//!
//! The transpose rules and `vjp_query` arrive in M3 and M4. The scalar core,
//! [`ddx-core`](../ddx_core/index.html), becomes the elementwise leaf of this
//! engine (design.md §4.3).
//!
//! The planned surface (design.md §4.4):
//!
//! ```ignore
//! pub fn vjp_query(plan: &Plan, wrt: &[ColumnRef]) -> Result<BackwardProgram, AdError>;
//! ```

#![forbid(unsafe_code)]

mod activity;
mod analysis;
mod columns;
mod error;
mod expr;
mod index;
mod markers;
mod names;

#[cfg(test)]
mod test_plans;

/// The `substrait` crate that this crate was built against, re-exported. A
/// downstream crate that reaches for `substrait` through this path cannot link a
/// different version from the one that the API of `ddx-ad` uses. `ddx-core`
/// re-exports `sqlparser` for the same reason (design.md §6).
pub use substrait;

pub use activity::{Activity, ColumnRef};
pub use analysis::Analysis;
pub use columns::{Col, ColumnDef, Columns, Field};
pub use error::{AdError, Result};
pub use index::{Node, NodeId, NodeKind, PlanIndex, TableRef};
pub use markers::{Functions, Marker};
pub use names::AggKind;
