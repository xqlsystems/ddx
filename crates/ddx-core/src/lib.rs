// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-core` — engine-neutral symbolic differentiation of SQL scalar
//! expressions. The v1 core of [`ddx`](https://github.com/xqlsystems/ddx):
//! write calculus directly in SQL and let the engine evaluate the derivative
//! per row, the relational equivalent of `jax.vmap(jax.grad(f))`.
//!
//! ```sql
//! SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g
//! ```
//!
//! `grad`/`jvp` are **markers**, not row functions: they carry a
//! differentiation request through parsing and are always rewritten away
//! *before* execution. [`Ddx::rewrite_sql`] is the whole path — find every
//! marker, differentiate what it wraps, splice the derivative back by source
//! span, return plain SQL.
//!
//! The engine differentiates [`sqlparser::ast::Expr`] directly — there is no
//! bespoke IR and no adapter layer; the AST *is* the IR (design.md §3.2). The
//! single load-bearing dependency is [`sqlparser`], which is **re-exported**
//! (see below) so downstream adapters cannot accidentally link a mismatched
//! version.
//!
//! # What v1 supports
//!
//! `+ - * /`; the unary chain rule for the trig / inverse-trig / exp / log /
//! hyperbolic set plus `abs`; `power` with a constant base or exponent;
//! higher-order via nesting; through-aggregate via linearity
//! (`AVG(grad(loss, theta))`). Anything else is a typed [`DiffError`], never a
//! silently-wrong number (design principle 5).
//!
//! Scalar `vjp` is deliberately **not** part of the surface: the name is
//! reserved for the query-level reverse-mode operation in
//! [`ddx-ad`](https://docs.rs/ddx-ad) (design.md §3.6, §4, decision Q7).
//!
//! # `sqlparser` version policy
//!
//! `ddx-core`'s public API takes and returns `sqlparser::ast::Expr`, so a
//! `sqlparser` bump is a breaking release of `ddx-core`. The version is pinned
//! exactly (see `Cargo.toml`) and re-exported here as [`ddx_core::sqlparser`]:
//! always reach for `sqlparser` types through this re-export so your build
//! links the same version the engine was compiled against (design.md §6, G2).

#![forbid(unsafe_code)]

/// The exact `sqlparser` this crate was built against, re-exported so downstream
/// code links a matching version (design.md §6, G2).
pub use sqlparser;

mod colref;
mod constructors;
mod ddx;
mod engine;
mod error;
mod rewrite;

/// Smart constructors for building derivative expressions — useful when writing
/// a custom [`Rule`], which returns `f'(u)` as an [`sqlparser::ast::Expr`].
pub mod build {
    pub use crate::constructors::{
        add, cast_double, div, finite_num, func, func1, mul, neg, one, sign, square, sub, zero,
    };
}

pub use colref::{ColRef, IdentCasing, Match};
pub use ddx::Ddx;
pub use engine::{Rule, RuleRegistry};
pub use error::{DiffError, Result};
pub use rewrite::{ExplainStep, Explanation};
