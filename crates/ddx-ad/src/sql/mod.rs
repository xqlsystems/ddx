// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad`, `vjp` and `jvp` in SQL: a query's derivative as a column of
//! another query.
//!
//! Each is a column of a `SELECT`, with the rows a derivative has, as JAX's
//! results have the shapes of what they differentiate:
//!
//! | call | in a `SELECT` over | is, row by row |
//! |---|---|---|
//! | `grad(f, c)` | the table whose column `c` is | the derivative at that row's `c` of the number the CTE `f` computes (one row, one column) |
//! | `vjp(f, c, cotangent)` | the table whose column `c` is | the same for any `f`, pulled back along `cotangent`, a CTE or table shaped like `f`'s output (its keys, then its values) |
//! | `jvp(e, t.c, …, tangent)` | anything: `e` is an expression of the `SELECT` | the tangent of `e`, pushed forward along `tangent`: a CTE or table shaped like `t` (its dims, then a tangent under each column's name), or a number, the same at every row |
//!
//! A gradient has its table's rows, so `grad` and `vjp` sit in a `SELECT`
//! over the table (`c`; `t.c` for the table the `SELECT` calls `t` when it
//! reads more than one; or `s.t.c`, its whole name, when it has no alias,
//! spelled as `f` spells it: ddx matches tables by name, so it cannot know
//! that an engine's default schema makes `t` and `s.t` one table),
//! anywhere an expression can: its select list, `WHERE`, `GROUP BY`,
//! `HAVING`, a join's `ON`, or the `ORDER BY` of the query whose `SELECT` it
//! is. A call that is a column of its own, with no alias, is named
//! `{c}_grad`:
//!
//! ```sql
//! WITH f AS (SELECT SUM(val * val) AS l FROM w)
//! SELECT i, val, grad(f, val) AS df_dval FROM w        -- 2·val
//! ```
//!
//! A tangent has `e`'s rows, so `jvp` is a column of the `SELECT` that
//! computes `e`: the statement's own, or a CTE's. It is a column of its own
//! (computing with it is for a query over that `SELECT`), named by its alias,
//! or `{e}_tangent` when `e` is a column. With a tangent of 1, it is the
//! derivative of an expression row by row:
//!
//! ```sql
//! SELECT x, sin(x) AS y, jvp(sin(x), t.x, 1) AS dy FROM t   -- cos(x)
//! ```
//!
//! Several `jvp` columns of one `SELECT` push forward along the same columns
//! and tangent; along others, write a `SELECT` each. The `SELECT` becomes a
//! read of what its program computed, so its `ORDER BY` may name only its
//! output, by name or position. A row a cotangent or tangent table lacks
//! counts as 0.
//!
//! A gradient is what an optimizer steps along, so a step,
//! `params - lr * grad(f)(params)`, is one expression, whatever `f` scores:
//! a model's loss, a curve's misfit, or a physical system's energy, whose
//! gradient is the force on each body with its sign flipped:
//!
//! ```sql
//! -- Masses at heights mass(i, y), joined by springs spring(lo, hi, k, rest),
//! -- under their weights weight(i, w).
//! WITH springs AS (
//!        SELECT SUM(0.5 * s.k * power(b.y - a.y - s.rest, 2)) AS e
//!        FROM spring s JOIN mass a ON s.lo = a.i JOIN mass b ON s.hi = b.i),
//!      gravity AS (SELECT SUM(w.w * m.y) AS e FROM mass m JOIN weight w ON m.i = w.i),
//!      energy AS (SELECT springs.e + gravity.e AS e FROM springs CROSS JOIN gravity)
//! SELECT i, y - 0.01 * grad(energy, y) AS y FROM mass  -- one step towards equilibrium
//! ```
//!
//! Calls nest, as JAX's transformations do: `f`, `e`'s `SELECT`, or a
//! cotangent or tangent can read another call, and `jvp` of a gradient is a
//! Hessian-vector product:
//!
//! ```sql
//! WITH loss AS (SELECT SUM(val * val * val) AS l FROM w),
//!      g AS (SELECT i, grad(loss, val) AS val FROM w),
//!      v AS (SELECT i, 1.0 AS val FROM w)
//! SELECT i, jvp(val, w.val, v) AS hv FROM g
//! ```
//!
//! # How it runs
//!
//! No engine could run those calls: a function receives values, and `f` is a
//! query. So [`rewrite`] computes what they read first and rewrites each
//! statement to read it. It finds the calls of several statements at once
//! and plans one program per distinct derivative: a `grad` or `vjp` program
//! differentiates one query with respect to every column any call of it
//! names, so statements that update different tables from one objective pay
//! for one backward pass; a `jvp` program pushes one `SELECT` forward along
//! one tangent. A program comes after any whose result its query reads, and
//! differentiates through it; a query that reads, through its CTEs, a call
//! of itself is refused, since no order runs either first. Each is run on
//! the engine as [`crate::run`] runs one. An engine inlines a CTE at every
//! read, so a query whose CTEs read one another twice (a model's layers)
//! would reach ddx, and the engine's optimizer, at a size exponential in
//! its depth: each CTE the engine would copy is planned on its own instead,
//! reading those before it as empty placeholder tables, and the plans are
//! joined into one that reads each through a `ReferenceRel`, which ddx
//! differentiates once. A CTE that could be misread that way (one a nested
//! `WITH` defines again, one read inside an expression's subquery) is left
//! to the engine, and if planning the pieces fails, the query is planned
//! whole, as it always was. A view the objective reads, whose SQL the
//! engine gives ([`crate::Backend::view_sql`]), is written into it as CTEs
//! first, so a model that is a view has its CTEs planned once too. When a statement's `WHERE` keeps
//! some of a table's rows by comparing its columns with literals
//! (`layer = 1`), the program computes only those rows' gradients, if the
//! engine's plan of that restriction is one the program can take; otherwise
//! it computes every row, which is always right. Then each statement's tree
//! is rewritten: a table `grad` reads becomes its rows joined with their
//! gradients, and a `SELECT` with `jvp` columns becomes a read of its
//! program's value. It is written out as SQL again; a statement with no
//! call comes back as it was.
//!
//! A training loop runs the same statements on every batch. [`Programs`]
//! keeps each program from one call of [`rewrite`] to the next, but each
//! call still plans the statements' queries to find it. [`prepare`] goes
//! further, as `jax.jit` does: it computes the calls once, and
//! [`Prepared::run`] runs every plan of that run again, asking the engine
//! only to run them, and whether anything a plan depends on has changed (a
//! table's schema, a view, a setting, where names are searched); if so, it
//! runs nothing, for the statements to be prepared again.
//!
//! The names `grad`, `vjp` and `jvp` are ddx's: a function of that name in a
//! statement is one of these calls, and a misused one is refused with its
//! usage.

mod build;
mod cache;
mod find;
mod parse;
mod plan;
mod prepare;
mod rewrite;
mod rewritten;
mod share;
#[cfg(test)]
mod tests;
mod views;

pub use cache::Programs;
pub use prepare::{prepare, Prepared};
pub(crate) use rewrite::quote;
pub use rewritten::{rewrite, Rewritten};
pub use views::view_probe;
