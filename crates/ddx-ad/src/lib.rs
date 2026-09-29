// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-ad`: query-level reverse-mode automatic differentiation (ddx v2).
//!
//! v1 ([`ddx_core`]) differentiates one scalar expression. v2 differentiates a
//! whole query: given a Substrait plan whose root is a loss, it emits the
//! queries that compute the loss's gradient with respect to chosen table
//! columns (design.md §4). It works by applying one transpose rule per
//! relational operator, and uses `ddx-core` for the elementwise one.
//!
//! The user marks the operations whose meaning ddx must not guess with four
//! identity functions: `ddx_contract_mark`, `ddx_reduce_mark`,
//! `ddx_route_mark` and `ddx_stop_gradient` (§4.3).
//!
//! This crate is being built across milestones M3 and M4. So far it pins the
//! plan type.
//!
//! # `substrait` version policy
//!
//! The public API takes and returns [`substrait::proto`] types, so the version
//! is pinned exactly and re-exported as [`crate::substrait`], like
//! `ddx_core::sqlparser`. An adapter should reach for plan types through this
//! re-export.

#![forbid(unsafe_code)]

/// The exact `substrait` this crate was built against, re-exported so an
/// adapter links the same version.
pub use substrait;

/// The exact `ddx-core` this crate was built against.
pub use ddx_core;
