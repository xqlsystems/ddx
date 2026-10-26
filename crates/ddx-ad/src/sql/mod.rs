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
mod share;
#[cfg(test)]
mod tests;
mod views;

use std::fmt;
use std::future::Future;

use ddx_core::sqlparser::dialect::Dialect;

use crate::error::AdError;
use crate::program::{Options, Unbound};
use crate::run::{Backend, Plans, RunError};

pub use cache::Programs;
pub use prepare::{prepare, Prepared};
use prepare::{results, Trace};
pub(crate) use rewrite::quote;
pub use views::view_probe;

/// What [`rewrite`] returns: the statements to run, and the tables their
/// rewritten text reads, which the caller drops once it has planned (or run)
/// the statements ([`Rewritten::release`]).
#[non_exhaustive]
pub struct Rewritten {
    /// Each statement, its calls replaced by reads of what their programs
    /// computed; a statement with no call as it was.
    pub statements: Vec<String>,
    /// The tables the programs left (values and gradients) and the
    /// cotangents and tangents registered for them.
    pub tables: Vec<String>,
    /// The programs that wrote them, to keep once they are dropped.
    kept: Vec<(cache::Key, Box<dyn plan::Built>)>,
}

impl fmt::Debug for Rewritten {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rewritten")
            .field("statements", &self.statements)
            .field("tables", &self.tables)
            .field("kept", &self.kept.len())
            .finish()
    }
}

impl Rewritten {
    /// Drop [`Rewritten::tables`] from `backend`, every one even after a
    /// failed drop, reporting the first failure; then keep in `programs`
    /// the programs that wrote them, for a later [`rewrite`] to run again.
    ///
    /// Not before: until the tables are dropped, a program kept would be
    /// free for another call to run, writing the same tables while this
    /// call's statements still read them.
    pub async fn release<B: Backend>(
        self,
        backend: &mut B,
        programs: &Programs,
    ) -> Result<(), B::Error> {
        let mut first = Ok(());
        for t in &self.tables {
            if let (Err(e), Ok(())) = (backend.drop_table(t).await, &first) {
                first = Err(e);
            }
        }
        for (key, program) in self.kept {
            programs.keep(key, program);
        }
        first
    }
}

/// Compute what the `grad`, `vjp` and `jvp` calls in `statements` read, on
/// `backend`, and rewrite each statement to read it (see the module docs).
/// `dialect` is how the statements are written.
///
/// A program kept in `programs` from an earlier call is run again instead
/// of built, and each program built is kept there once its tables are
/// released (see [`Programs`]), so a training loop that passes the same
/// `programs` each step builds its programs once.
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
        let result = run_all(
            backend,
            &mut planned,
            Some(programs),
            &mut keys,
            &mut tables,
            None,
        )
        .await;
        // Every program built or taken goes back once its tables are
        // dropped, whatever happened: a run that failed (a check that
        // returned a row, an engine's error) leaves its program as it was.
        let kept = keys
            .into_iter()
            .zip(planned.into_built())
            .filter_map(|(key, program)| Some((key?, program)))
            .collect();
        match result {
            Ok(statements) => Ok(Rewritten {
                statements,
                tables,
                kept,
            }),
            Err(e) => {
                let _ = (Rewritten {
                    statements: Vec::new(),
                    tables,
                    kept,
                })
                .release(backend, programs)
                .await;
                Err(e)
            }
        }
    }
}

/// Build (or take from `programs`) and run every planned program in order,
/// recording in `keys` what each is kept under, in `tables` what each
/// leaves, and in `traces`, if given, the plans each ran; then the
/// rewritten statements. With no `programs`, each is built, and kept under
/// no key.
async fn run_all<B: Backend>(
    backend: &mut B,
    planned: &mut plan::Derivatives,
    programs: Option<&Programs>,
    keys: &mut Vec<Option<cache::Key>>,
    tables: &mut Vec<String>,
    mut traces: Option<&mut Vec<Trace<B::Kept>>>,
) -> Result<Vec<String>, RunError<B::Error>> {
    while let Some(job) = planned.next().map_err(RunError::Refused)? {
        // Each CTE the engine would copy is planned once (see `share`); if
        // anything in that fails, the query is planned whole, as it always
        // was, which is always right.
        // The views the objective reads, written as its CTEs, so theirs
        // are planned once too.
        let tree = match &job.tree {
            Some(q) => Some(views::expand(backend, q).await.unwrap_or_else(|| q.clone())),
            None => None,
        };
        let split = tree
            .as_ref()
            .and_then(|q| share::split(q, &crate::forward::new_namespace()));
        let key = match split {
            Some(split) => match program_of_split(backend, planned, programs, &job, &split).await {
                Ok(key) => key,
                Err(_) if planned.built_count() == job.index => {
                    program_of_query(backend, planned, programs, &job).await?
                }
                Err(e) => return Err(e),
            },
            None => program_of_query(backend, planned, programs, &job).await?,
        };
        keys.push(key);
        let mut trace = Trace::default();
        for (name, query) in planned.seeds(&job).map_err(RunError::Refused)? {
            let plan = backend.plan_sql(&query).await.map_err(RunError::Engine)?;
            // The engine's own plan of the query: every read already typed.
            let plan = Unbound::from_plan(plan)
                .bind(&mut |_| None)
                .map_err(RunError::Refused)?;
            tables.push(name.clone());
            let kept = crate::run::keep(backend, &plan, traces.is_some()).await?;
            crate::run::materialize(backend, &name, &plan, kept.as_ref()).await?;
            trace.seeds.push((name, plan, kept));
        }
        let program = planned.built(job.index);
        let traced = traces.is_some().then_some(&mut trace.plans);
        crate::run::run_bound(backend, program, Plans::Bind(traced)).await?;
        tables.extend(results(program));
        if let Some(traces) = traces.as_deref_mut() {
            traces.push(trace);
        }
    }
    planned.rewrite().map_err(RunError::Refused)
}

/// `job`'s program, taken from `programs` or built from the engine's plan
/// of its query, and the key it is kept under.
async fn program_of_query<B: Backend>(
    backend: &mut B,
    planned: &mut plan::Derivatives,
    programs: Option<&Programs>,
    job: &plan::Job,
) -> Result<Option<cache::Key>, RunError<B::Error>> {
    let key = match programs {
        Some(_) => cache::Key::of(backend, job, std::slice::from_ref(&job.query), None)
            .await
            .map_err(RunError::Engine)?,
        None => None,
    };
    match key.as_ref().zip(programs).and_then(|(k, p)| p.take(k)) {
        Some(program) => planned.adopt(job, program).map_err(RunError::Refused)?,
        None => {
            let plan = backend
                .plan_sql(&job.query)
                .await
                .map_err(RunError::Engine)?;
            build(backend, planned, job, &plan).await?;
        }
    }
    Ok(key)
}

/// [`program_of_query`], with each CTE of `split` planned on its own: its
/// placeholder made from the engine's plan of it with `LIMIT 0`, then the
/// key taken from every piece's plan, then the pieces planned and stitched
/// into one plan. The placeholders are dropped whatever happens; the
/// program reads none.
async fn program_of_split<B: Backend>(
    backend: &mut B,
    planned: &mut plan::Derivatives,
    programs: Option<&Programs>,
    job: &plan::Job,
    split: &share::Split,
) -> Result<Option<cache::Key>, RunError<B::Error>> {
    let mut made: Vec<String> = Vec::new();
    let result = async {
        for piece in &split.ctes {
            let empty = backend
                .plan_sql(&share::Split::empty(piece))
                .await
                .map_err(RunError::Engine)?;
            let empty = Unbound::from_plan(empty)
                .bind(&mut |_| None)
                .map_err(RunError::Refused)?;
            made.push(piece.placeholder.clone());
            backend
                .materialize(&piece.placeholder, &empty)
                .await
                .map_err(RunError::Engine)?;
        }
        let queries: Vec<String> = split
            .ctes
            .iter()
            .map(|p| p.sql.clone())
            .chain(std::iter::once(split.root.clone()))
            .collect();
        let key = match programs {
            Some(_) => cache::Key::of(backend, job, &queries, Some(&split.namespace))
                .await
                .map_err(RunError::Engine)?,
            None => None,
        };
        if let Some(program) = key.as_ref().zip(programs).and_then(|(k, p)| p.take(k)) {
            planned.adopt(job, program).map_err(RunError::Refused)?;
            return Ok(key);
        }
        let mut pieces = Vec::with_capacity(split.ctes.len());
        for piece in &split.ctes {
            pieces.push(
                backend
                    .plan_sql(&piece.sql)
                    .await
                    .map_err(RunError::Engine)?,
            );
        }
        let root = backend
            .plan_sql(&split.root)
            .await
            .map_err(RunError::Engine)?;
        let plan = share::stitch(split, pieces, &root).map_err(RunError::Refused)?;
        build(backend, planned, job, &plan).await?;
        Ok(key)
    }
    .await;
    let mut dropped = Ok(());
    for name in made {
        if let (Err(e), Ok(())) = (backend.drop_table(&name).await, &dropped) {
            dropped = Err(RunError::Engine(e));
        }
    }
    let key = result?;
    dropped?;
    Ok(key)
}

/// Build `job`'s program from `plan`, the engine's plan of its query, and
/// the engine's plans of its restrictions.
async fn build<B: Backend>(
    backend: &mut B,
    planned: &mut plan::Derivatives,
    job: &plan::Job,
    plan: &substrait::proto::Plan,
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
    let built = planned.build(job, plan, &options).map(|_| ());
    match built {
        // An engine's plan of the restriction need not have the shape a
        // program takes one in (its optimizer may fold the filter away,
        // or push it into the scan): compute every row instead.
        Err(AdError::InvalidOptions(_)) if !job.pushdown.is_empty() => {
            planned
                .build(job, plan, &Options::new())
                .map_err(RunError::Refused)?;
        }
        other => other.map_err(RunError::Refused)?,
    }
    Ok(())
}
