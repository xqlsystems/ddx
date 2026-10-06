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
//! [`jvp`] is the forward-mode half: it rewrites the query so each relation
//! carries the tangent of its columns beside them, along a tangent of the
//! `wrt` columns the caller supplies, and returns a [`ForwardProgram`] that runs
//! by the same protocol.
//!
//! All three take a query's plan or a program of either kind
//! ([`Differentiable`]), so they compose as JAX's do: [`jvp`] of a `grad`
//! program is forward over reverse, a Hessian-vector product beside each
//! gradient; [`vjp`] of a `jvp` program is reverse over forward.
//!
//! The modules that read and write plans (`forward`, `emit`, `expr`,
//! `relation`) are public only with the `internals` feature, for this
//! workspace's tests: they are not part of the API, and a field renamed in
//! them is not a breaking change. They may become a supported plan builder
//! once a second adapter shows what it needs.
//!
//! # What you can extend, and what you cannot yet
//!
//! - **A scalar function's derivative.** Register a rule on a `ddx-core`
//!   engine ([`ddx_core::Ddx::register`]) and pass it with [`Options::ddx`],
//!   much as `jax.custom_jvp` does for one function. A unary function with a
//!   rule is differentiated wherever it appears in a projected expression.
//! - **What not to differentiate.** `ddx_stop_gradient(x)`, an identity the
//!   engine registers under that name, is a constant to ddx, like JAX's
//!   `lax.stop_gradient`.
//! - **The engine.** A program is plain Substrait plans plus the run protocol
//!   ([`Runner`], or [`Backend`] and [`run`] for a synchronous engine), so a
//!   new engine writes four primitives and nothing about AD.
//!
//! A relational operator of the host's own has no extension point: a
//! user-defined aggregate, or a Substrait `ExtensionSingleRel`,
//! `ExtensionMultiRel` or `ExtensionLeafRel`, on a path the gradient flows
//! along, is refused as [`AdError::NotImplemented`]. Off that path (constant
//! data, or a subtree no gradient reaches) it is copied as it is.
//!
//! [`sql`] finds `grad(loss, table.column)` and `jvp(f, table.column,
//! tangent)` in a SQL statement, for adapters that let users write them in
//! SQL.
//!
//! # `substrait` version policy
//!
//! The public API takes and returns [`substrait::proto`] types, so the version
//! is pinned exactly and re-exported as [`crate::substrait`], like
//! `ddx_core::sqlparser`. An adapter should reach for plan types through this
//! re-export.

#![forbid(unsafe_code)]

/// Modules public only with the `internals` feature (see the crate docs).
macro_rules! internal {
    ($(mod $m:ident;)*) => {$(
        #[cfg(feature = "internals")]
        #[doc(hidden)]
        pub mod $m;
        #[cfg(not(feature = "internals"))]
        mod $m;
    )*};
}

mod compose;
mod elementwise;
mod error;
mod functions;
mod fuse;
mod jvp;
mod program;
mod prune;
mod run;
pub mod sql;
mod tables;
mod tangent;
mod transpose;
internal! {
    mod emit;
    mod expr;
    mod forward;
    mod relation;
}

pub use compose::Differentiable;
pub use emit::{bind_reads, unbound_reads};
pub use error::{AdError, Result};
pub use functions::STOP_GRADIENT;
pub use jvp::{jvp, jvp_with};
pub use program::{
    decode_plan, grad, grad_with, vjp, vjp_with, BackwardProgram, Check, ForwardProgram, Options,
    Step,
};
pub use relation::ColumnRef;
pub use run::{run, run_verified, Action, Backend, Program, RunError, Runner, Verified};
pub use sql::{Calls, GradCall, Job, JvpCall, JvpJob, JvpTangent, Objective, Statements};
pub use tables::{InputTable, Of, OutputTable, Tangent};
#[cfg(feature = "internals")]
#[doc(hidden)]
pub use {
    elementwise::Elementwise,
    forward::Forward,
    functions::{normalize, Extensions, Functions},
    relation::Table,
};

/// The exact `substrait` this crate was built against, re-exported so an
/// adapter links the same version.
pub use substrait;

/// The exact `ddx-core` this crate was built against.
pub use ddx_core;
