// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-core` — the scalar derivative rules of
//! [`ddx`](https://github.com/xqlsystems/ddx). [`ddx-ad`](https://docs.rs/ddx-ad)
//! differentiates a whole query over Substrait; for each expression in it, it
//! asks [`Ddx::differentiate`] for the partial derivatives.
//!
//! The engine differentiates [`sqlparser::ast::Expr`] directly. Its one
//! dependency is [`sqlparser`], which is **re-exported** (see below) so a
//! caller links the same version.
//!
//! # What it supports
//!
//! `+ - * /`; the unary chain rule for the trig / inverse-trig / exp / log /
//! hyperbolic set plus `abs`; `power` with a constant base or exponent. Anything
//! else is a typed [`DiffError`], never a silently-wrong number (design
//! principle 5).
//!
//! # `sqlparser` version policy
//!
//! `ddx-core`'s public API takes and returns `sqlparser::ast::Expr`, so a
//! `sqlparser` bump is a breaking release of `ddx-core`. The version is pinned
//! exactly (see `Cargo.toml`) and re-exported here as [`crate::sqlparser`]:
//! always reach for `sqlparser` types through this re-export so your build
//! links the same version the engine was compiled against. Two versions in one
//! build would be two unrelated Rust types, and an expression from one could
//! not be handed to the other.

#![forbid(unsafe_code)]
// The Google Rust style guide's rules that clippy checks
// (https://google.github.io/styleguide/rustguide.html).
#![warn(
    missing_docs,
    clippy::unwrap_used,
    clippy::needless_pass_by_value,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::wildcard_imports
)]

/// The exact `sqlparser` this crate was built against, re-exported so downstream
/// code links a matching version.
pub use sqlparser;

mod colref;
mod constructors;
mod ddx;
mod engine;
mod error;

// The shared simulation harness — a random expression generator, a reference
// interpreter, the numeric conditioning gates, and the failure reporter. Behind
// a feature so it never ships in a normal build, but public when enabled so
// every crate in the workspace fuzzes against the *same* generator: a
// cross-crate agreement test proves nothing if each side invented its own idea
// of "a random derivable expression".
#[cfg(feature = "test-utils")]
pub mod test_utils;

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
