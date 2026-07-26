// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx-datafusion` — the DataFusion adapter for ddx.
//!
//! **Status: scaffold.** M0 lands only the crate seam. The real adapter
//! arrives in M2 (design.md §3.3–§3.4, milestone M2) and exposes two paths,
//! both driving the *same* [`ddx_core::Ddx`] engine:
//!
//! * **Path A — [`ddx_sql`]:** the one-liner
//!   `ctx.sql(&ddx.rewrite_sql(sql, dialect)?)`. This works today on top of
//!   [`ddx_core`] and is the universal path (design.md §3.3 Path A); it is
//!   stubbed here rather than wired to a live `SessionContext` because M0 does
//!   not take a `datafusion` dependency.
//! * **Path B — marker UDFs + an `AnalyzerRule` bridge:** makes bare `grad()`
//!   work across the SQL and DataFrame APIs by unparsing the marker's argument
//!   with DataFusion's `expr_to_sql` (which emits exactly `ddx-core`'s
//!   `sqlparser::ast::Expr` input type), differentiating via `ddx-core`, and
//!   re-planning back to a DataFusion `Expr` (design.md §3.3 Path B). This is
//!   the reference proof that `ddx-core` can drive an *in-engine* rewrite.
//!
//! Path B carries two documented implementation constraints to honor in M2:
//! `add_analyzer_rule` runs after `TypeCoercion` (so the marker UDF must be
//! coercion-tolerant and there must be a `Cast` rule — there is), and the
//! re-plan seam is `SessionState::create_logical_expr`.

#![forbid(unsafe_code)]

/// Milestone marker: the DataFusion adapter is not implemented in M0.
pub const STATUS: &str = "scaffold: DataFusion Path A + Path B land in M2";

// Re-exported so downstream code and docs can name the engine this adapter
// will drive, even while the adapter itself is a stub.
pub use ddx_core;
