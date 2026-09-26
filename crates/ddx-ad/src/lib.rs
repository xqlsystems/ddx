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
//! [`grad`] and [`vjp`] take a plan and the `wrt` columns and return a
//! [`BackwardProgram`]: plain Substrait plans for an engine to run in order,
//! the last of which hold the gradients, shaped like the tables they are
//! gradients of. The pieces are public for adapters: [`relation`] states what
//! dims and values are, [`forward`] reads the plan, [`Elementwise`] gives the
//! map primitive's local derivatives, and [`emit`] writes plans and binds their
//! reads. [`sql`] finds `grad(loss, table.column)` in a SQL statement, for
//! adapters that let users write `grad` in SQL.
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
pub mod forward;
mod functions;
mod program;
pub mod relation;
pub mod sql;
mod transpose;

pub use elementwise::Elementwise;
pub use error::{AdError, Result};
pub use forward::Forward;
pub use functions::{normalize, Extensions, Functions, STOP_GRADIENT};
pub use program::{
    grad, grad_with, vjp, vjp_with, BackwardProgram, Gradient, Step, COTANGENT, VALUE,
};
pub use relation::{ColumnRef, Table};
pub use sql::{GradCall, GradCalls, Loss};

/// The exact `substrait` this crate was built against, re-exported so an
/// adapter links the same version.
pub use substrait;

/// The exact `ddx-core` this crate was built against.
pub use ddx_core;
