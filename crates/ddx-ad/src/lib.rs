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
//! gradients of. An adapter runs the steps in order, late-binding each step's
//! reads of earlier steps with [`unbound_reads`] and [`bind_reads`], and runs
//! the program's [`checks`](BackwardProgram::checks) first. The table names
//! a program writes all start with a prefix unique to the program,
//! `__ddx_{id}_`; the `__ddx_` prefix is reserved.
//!
//! The modules that read and write plans (`forward`, `emit`, `expr`,
//! `relation`) are public only so this workspace's adapters and tests can
//! reach them; they are hidden from the docs and not part of the stable API.
//!
//! # `substrait` version policy
//!
//! The public API takes and returns [`substrait::proto`] types, so the version
//! is pinned exactly and re-exported as [`crate::substrait`], like
//! `ddx_core::sqlparser`. An adapter should reach for plan types through this
//! re-export.

#![forbid(unsafe_code)]

mod elementwise;
#[doc(hidden)]
pub mod emit;
mod error;
#[doc(hidden)]
pub mod expr;
#[doc(hidden)]
pub mod forward;
mod functions;
mod program;
#[doc(hidden)]
pub mod relation;
mod run;
mod transpose;

#[doc(hidden)]
pub use elementwise::Elementwise;
pub use emit::{bind_reads, unbound_reads};
pub use error::{AdError, Result};
#[doc(hidden)]
pub use forward::Forward;
pub use functions::STOP_GRADIENT;
#[doc(hidden)]
pub use functions::{normalize, Extensions, Functions};
pub use program::{
    decode_plan, grad, grad_with, vjp, vjp_with, BackwardProgram, Check, Gradient, Options, Step,
};
pub use relation::ColumnRef;
#[doc(hidden)]
pub use relation::Table;
pub use run::{run, Action, Backend, RunError, Runner};

/// The exact `substrait` this crate was built against, re-exported so an
/// adapter links the same version.
pub use substrait;

/// The exact `ddx-core` this crate was built against.
pub use ddx_core;
