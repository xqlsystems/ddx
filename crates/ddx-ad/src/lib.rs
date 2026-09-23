// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-ad`: query-level reverse-mode automatic differentiation (ddx v2).
//!
//! v1 ([`ddx_core`]) differentiates one scalar expression. v2 differentiates a
//! whole query, the way `jax.grad` differentiates a function: given a Substrait
//! plan whose result is a loss, it emits the queries that compute the loss's
//! gradient with respect to chosen table columns (design.md §4).
//!
//! A query is a composition of relational operators (map, select, join,
//! aggregate), and each has a transpose rule, as each JAX primitive does. The
//! map rule's local derivatives come from `ddx-core`. Nothing in the query has
//! to be labelled for this; the one function ddx claims is
//! `ddx_stop_gradient`, JAX's `lax.stop_gradient`.
//!
//! This crate is being built across milestones M3 and M4. So far it pins the
//! plan type, reads a plan's function table ([`Functions`]), and has the map
//! primitive's local derivatives ([`Elementwise`]). [`emit`] writes the plans
//! of a backward program, and [`expr`] builds the expressions inside them.
//!
//! # `substrait` version policy
//!
//! The public API takes and returns [`substrait::proto`] types, so the version
//! is pinned exactly and re-exported as [`crate::substrait`], like
//! `ddx_core::sqlparser`. An adapter should reach for plan types through this
//! re-export.

#![forbid(unsafe_code)]

mod elementwise;
pub mod emit;
mod error;
pub mod expr;
mod functions;

pub use elementwise::Elementwise;
pub use error::{AdError, Result};
pub use functions::{normalize, Extensions, Functions, STOP_GRADIENT};

/// The exact `substrait` this crate was built against, re-exported so an
/// adapter links the same version.
pub use substrait;

/// The exact `ddx-core` this crate was built against.
pub use ddx_core;
