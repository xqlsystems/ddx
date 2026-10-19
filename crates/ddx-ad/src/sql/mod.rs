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
//! the engine as [`crate::run`] runs one. When a statement's `WHERE` keeps
//! some of a table's rows by comparing its columns with literals
//! (`layer = 1`), the program computes only those rows' gradients, if the
//! engine's plan of that restriction is one the program can take; otherwise
//! it computes every row, which is always right. Then each statement's tree
//! is rewritten: a table `grad` reads becomes its rows joined with their
//! gradients, and a `SELECT` with `jvp` columns becomes a read of its
//! program's value. It is written out as SQL again; a statement with no
//! call comes back as it was.
//!
//! The names `grad`, `vjp` and `jvp` are ddx's: a function of that name in a
//! statement is one of these calls, and a misused one is refused with its
//! usage.

mod build;
mod cache;
mod find;
mod plan;
mod rewrite;
#[cfg(test)]
mod tests;

use std::future::Future;

use ddx_core::sqlparser::dialect::Dialect;

use crate::error::AdError;
use crate::program::{Options, Unbound};
use crate::run::{Backend, RunError};

pub use cache::Programs;
pub(crate) use rewrite::quote;

/// What [`rewrite`] returns: the statements to run, and the tables their
/// rewritten text reads, which the caller drops once it has planned (or run)
/// the statements ([`Rewritten::release`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Rewritten {
    /// Each statement, its calls replaced by reads of what their programs
    /// computed; a statement with no call as it was.
    pub statements: Vec<String>,
    /// The tables the programs left (values and gradients) and the
    /// cotangents and tangents registered for them.
    pub tables: Vec<String>,
}

impl Rewritten {
    /// Drop [`Rewritten::tables`] from `backend`, every one even after a
    /// failed drop; the first failure is reported.
    pub async fn release<B: Backend>(&self, backend: &mut B) -> Result<(), B::Error> {
        let mut first = Ok(());
        for t in &self.tables {
            if let (Err(e), Ok(())) = (backend.drop_table(t).await, &first) {
                first = Err(e);
            }
        }
        first
    }
}

/// Compute what the `grad`, `vjp` and `jvp` calls in `statements` read, on
/// `backend`, and rewrite each statement to read it (see the module docs).
/// `dialect` is how the statements are written.
///
/// A program kept in `programs` from an earlier call is run again instead
/// of built, and each program built is kept there (see [`Programs`]), so a
/// training loop that passes the same `programs` each step builds its
/// programs once.
///
/// Every program's tables but its results are dropped as it runs. What the
/// statements read stays until [`Rewritten::release`]; if anything fails,
/// nothing does.
///
/// Dropping the future before it completes (a timeout, a cancelled task)
/// skips that cleanup, as dropping any future skips the code after its
/// await: the tables written so far stay on the backend. Every one is named
/// under a namespace of ddx's, starting `__ddx_`, so they can be found and
/// dropped.
pub fn rewrite<'a, B: Backend>(
    backend: &'a mut B,
    programs: &'a Programs,
    statements: &[&str],
    dialect: &dyn Dialect,
) -> impl Future<Output = Result<Rewritten, RunError<B::Error>>> + Send + 'a {
    // Parsed and planned before the first await, so the future holds no
    // dialect (which need not be `Sync`) and stays `Send`.
    let planned = plan::Derivatives::plan(statements, dialect);
    async move {
        let mut planned = planned.map_err(RunError::Refused)?;
        let mut tables = Vec::new();
        let mut keys = Vec::new();
        let result = run_all(backend, &mut planned, programs, &mut keys, &mut tables).await;
        // Every program built or taken goes back, whatever happened: a run
        // that failed (a check that returned a row, an engine's error)
        // leaves its program as it was.
        for (key, program) in keys.into_iter().zip(planned.into_built()) {
            if let Some(key) = key {
                programs.keep(key, program);
            }
        }
        match result {
            Ok(statements) => Ok(Rewritten { statements, tables }),
            Err(e) => {
                let _ = (Rewritten {
                    statements: Vec::new(),
                    tables,
                })
                .release(backend)
                .await;
                Err(e)
            }
        }
    }
}

/// Build (or take from `programs`) and run every planned program in order,
/// recording in `keys` what each is kept under and in `tables` what each
/// leaves; then the rewritten statements.
async fn run_all<B: Backend>(
    backend: &mut B,
    planned: &mut plan::Derivatives,
    programs: &Programs,
    keys: &mut Vec<Option<cache::Key>>,
    tables: &mut Vec<String>,
) -> Result<Vec<String>, RunError<B::Error>> {
    while let Some(job) = planned.next().map_err(RunError::Refused)? {
        let key = cache::Key::of(backend, &job)
            .await
            .map_err(RunError::Engine)?;
        match key.as_ref().and_then(|k| programs.take(k)) {
            Some(program) => planned.adopt(&job, program).map_err(RunError::Refused)?,
            None => build(backend, planned, &job).await?,
        }
        keys.push(key);
        for (name, query) in planned.seeds(&job).map_err(RunError::Refused)? {
            let plan = backend.plan_sql(&query).await.map_err(RunError::Engine)?;
            // The engine's own plan of the query: every read already typed.
            let plan = Unbound::from_plan(plan)
                .bind(&mut |_| None)
                .map_err(RunError::Refused)?;
            tables.push(name.clone());
            backend
                .materialize(&name, &plan)
                .await
                .map_err(RunError::Engine)?;
        }
        let program = planned.built(job.index);
        crate::run::run(backend, program).await?;
        tables.extend(
            std::iter::once(program.value().step.clone())
                .chain(program.gradients().iter().map(|g| g.step.clone())),
        );
    }
    planned.rewrite().map_err(RunError::Refused)
}

/// Build `job`'s program from the engine's plans of its query and of its
/// restrictions.
async fn build<B: Backend>(
    backend: &mut B,
    planned: &mut plan::Derivatives,
    job: &plan::Job,
) -> Result<(), RunError<B::Error>> {
    // Only the gradient rows the statements read, where the engine can
    // plan the restriction and the program can take it; otherwise every
    // row, which is always right.
    let mut options = Options::new();
    for r in &job.pushdown {
        if let Ok(plan) = backend.plan_sql(&r.query).await {
            options = options.restrict(r.table.clone(), plan);
        }
    }
    let plan = backend
        .plan_sql(&job.query)
        .await
        .map_err(RunError::Engine)?;
    let built = planned.build(job, &plan, &options).map(|_| ());
    match built {
        // An engine's plan of the restriction need not have the shape a
        // program takes one in (its optimizer may fold the filter away,
        // or push it into the scan): compute every row instead.
        Err(AdError::InvalidOptions(_)) if !job.pushdown.is_empty() => {
            planned
                .build(job, &plan, &Options::new())
                .map_err(RunError::Refused)?;
        }
        other => other.map_err(RunError::Refused)?,
    }
    Ok(())
}
