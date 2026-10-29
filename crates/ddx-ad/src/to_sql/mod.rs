// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A Substrait plan written as SQL text, for an engine that runs a program's
//! steps as SQL rather than through its Substrait consumer.
//!
//! DuckDB's consumer (`from_substrait`, in its community `substrait`
//! extension) reads a table as if on a connection of its own, so it cannot
//! see a dataset registered on the caller's connection, a temporary table or
//! the caller's open transaction, and it rejects every window function. The
//! same plan written as SQL runs on the caller's own connection, where
//! windows are ordinary SQL. [`to_sql`] writes it.
//!
//! # In two parts: what a plan means, then how a dialect writes it
//!
//! [`to_sql`] reads the plan into a typed SQL tree ([`ir`]), then prints the
//! tree in one dialect ([`print`](mod@print)).
//!
//! - **Reading is a fold over the plan, bottom up** (a catamorphism,
//!   [`crate::optics::cata`]): each relation's SQL is made from its inputs'
//!   finished SQL, by one case per kind of relation, and the fold keeps a
//!   stack rather than recursing, so a plan of any depth is read.
//! - **The order rows come in travels with them.** A sort sets it, and a
//!   projection, filter or fetch keeps it, as Substrait's do; but SQL
//!   orders only the rows of the query an `ORDER BY` is in, not a query
//!   reading it. So each relation's order is kept as keys among its columns
//!   (a key its projection drops is passed on as an extra column), and the
//!   query that needs it writes it again: one that picks rows with
//!   `LIMIT`, and the whole query.
//! - **A field reference is resolved in an environment** of the columns its
//!   relation reads, chained to the columns of each relation a subquery is
//!   inside: Substrait's `steps_out` counts links of that chain outward (a
//!   de Bruijn index), and a reference past either end is refused.
//! - **The tree says what the plan means**: a `divide` knows whether it is
//!   integer division, a literal its type, a window its frame, a function
//!   its canonical name. Reading needs no dialect, and printing makes every
//!   choice a dialect makes (`//` or `/`, `signum` or `sign`, `BLOB` or
//!   `BYTEA`), parenthesizing every compound expression so that no
//!   dialect's precedence comes into it.
//!
//! # How a plan is written
//!
//! - **Every relation is a `SELECT` whose columns are named by position**,
//!   `c0`, `c1`, …, since Substrait refers to a column by its position. It
//!   reads its input under an alias of its own (`t0`, `t1`, …), so a column
//!   is always `alias.cN` and a user's column names never clash with ddx's.
//! - **Each relation is a CTE** (`__ddx_q0`, `__ddx_q1`, …), so a deep plan
//!   is a long `WITH` clause rather than deeply nested subqueries, which an
//!   engine's SQL planner recurses through. A relation the plan shares
//!   (`ReferenceRel`) is one CTE read wherever it is shared. A relation
//!   inside a subquery expression is written in place: it may read the
//!   enclosing query's columns, which a CTE cannot see. The root relation
//!   is the query's own `SELECT`, so its `ORDER BY` and `LIMIT` order the
//!   result.
//! - **The plan's meaning is written out, not left to the dialect.** Every
//!   literal is written with its type, so `0.1` is a `DOUBLE` on an engine
//!   that would read it as a `DECIMAL`; a `divide` whose output is an integer
//!   is the dialect's integer division (DuckDB's `/` of two integers is a
//!   `DOUBLE`); and a window's frame is always written, since SQL's default
//!   frame for an ordered window ends at the current row, and Substrait's
//!   at the end of the partition.
//! - **A function is named by its normalized name**, and one ddx knows no
//!   name of is refused, naming it, rather than written as a guess.
//! - **A join that keeps one side's rows** (semi, anti, mark) is a correlated
//!   `EXISTS`, which every engine reads the same way.
//!
//! Anything else (an extension relation, a grouping set, an interval
//! literal) is refused with [`AdError::NotImplemented`]. A plan that does
//! not hold together (a reference past its relations, a shared relation
//! that reads itself) is refused with [`AdError::InvalidPlan`].
//!
//! # What a plan leaves to its engine
//!
//! Substrait does not say everything: how a cast of a float to an integer
//! rounds, whether integer overflow wraps, where NaN sorts. A plan means
//! what the engine that planned it does there, so write it in that engine's
//! dialect, as ddx does (DuckDB plans what DuckDB runs). The same plan run
//! on another engine can differ where the two do.
//!
//! The names `to_sql` makes up begin `__ddx_` (its CTEs are `__ddx_q0`, …),
//! which ddx reserves: a table of such a name is shadowed by a CTE.

mod ir;
mod print;
mod reader;

use substrait::proto::Plan;

use crate::error::Result;

/// The SQL dialect [`to_sql`] writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Dialect {
    /// DataFusion's SQL, which ddx's tests run every program in to check
    /// [`to_sql`] against DataFusion's Substrait consumer.
    DataFusion,
    /// DuckDB's SQL.
    DuckDb,
}

/// `plan` as one SQL query in `dialect`, its columns named as the plan's
/// root names them.
///
/// The plan's reads must be bound (each with its `base_schema`), as a
/// program's steps are when [`crate::run`](fn@crate::run) hands them to a [`crate::Backend`].
/// A table is read by the names the plan gives it, quoted, so the query
/// must run where those names resolve to the tables the plan was made from.
pub fn to_sql(plan: &Plan, dialect: Dialect) -> Result<String> {
    Ok(print::query(&reader::query(plan)?, dialect))
}
