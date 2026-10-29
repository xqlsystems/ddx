// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`rewrite`]: computing the calls in SQL statements, and the statements
//! that read what they computed (see the [module docs](super)).

use std::fmt;
use std::future::Future;

use ddx_core::sqlparser::dialect::Dialect;

use crate::error::AdError;
use crate::program::{Options, Unbound};
use crate::run::{Backend, Drops, Plans, RunError};

use super::cache::{self, Programs};
use super::prepare::{results, Trace};
use super::{plan, share, views};

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
    pub(super) kept: Vec<(cache::Key, Box<dyn plan::Built>)>,
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
pub(super) async fn run_all<B: Backend>(
    backend: &mut B,
    planned: &mut plan::Derivatives,
    programs: Option<&Programs>,
    keys: &mut Vec<Option<cache::Key>>,
    tables: &mut Vec<String>,
    mut traces: Option<&mut Vec<Trace<B::Compiled>>>,
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
        let mut seeds = Vec::new();
        for (name, query) in planned.seeds(&job).map_err(RunError::Refused)? {
            let plan = backend.plan_sql(&query).await.map_err(RunError::Engine)?;
            // The engine's own plan of the query: every read already typed.
            let plan = Unbound::from_plan(plan)
                .bind(&mut |_| None)
                .map_err(RunError::Refused)?;
            tables.push(name.clone());
            let seed = crate::run::trace(backend, plan, traces.is_some()).await?;
            crate::run::materialize(backend, &name, &seed).await?;
            seeds.push((name, seed));
        }
        let program = planned.built(job.index);
        let drops = Drops::of(program).map_err(RunError::Refused)?;
        let mut plans = Vec::new();
        let traced = traces.is_some().then_some(&mut plans);
        crate::run::run_bound(backend, program, Plans::Bind(traced), &drops).await?;
        tables.extend(results(program));
        if let Some(traces) = traces.as_deref_mut() {
            traces.push(Trace {
                seeds,
                plans,
                drops,
            });
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
        let plan = share::stitch(split, &pieces, &root).map_err(RunError::Refused)?;
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
