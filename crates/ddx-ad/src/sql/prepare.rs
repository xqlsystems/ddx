// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Statements prepared once and run again, as `jax.jit` traces a function
//! once and runs what it traced after that.

use std::fmt;
use std::future::Future;

use ddx_core::sqlparser::dialect::Dialect;

use super::plan::{self, Built};
use super::{run_all, Programs, Rewritten};
use crate::run::{Backend, Drops, Plans, RunError, Traced};

/// What a job's run did on the engine, beyond what its program says: the
/// plans it ran, each as [`Traced`], and when it dropped each table.
pub(super) struct Trace<C> {
    /// Each seed's table, and the engine's plan of its query.
    pub seeds: Vec<(String, Traced<C>)>,
    /// Its program's checks', then its steps'.
    pub plans: Vec<Traced<C>>,
    pub drops: Drops,
}

/// The tables `program` leaves for the statements to read: its value, then
/// its gradients.
pub(super) fn results(program: &dyn Built) -> impl Iterator<Item = String> + '_ {
    std::iter::once(program.value().step.clone())
        .chain(program.gradients().iter().map(|g| g.step.clone()))
}

/// Statements whose `grad`, `vjp` and `jvp` calls were computed once by
/// [`prepare`], to compute again on the same tables: a training step, run
/// once per batch.
///
/// [`Prepared::run`] runs what the first run ran, each plan as it was bound
/// then, or as the engine compiled it then ([`Backend::compile`]), on the tables'
/// rows as they are: between runs, a table may change its rows, or be
/// registered again under its name. It asks the engine for
/// no plan, key or schema, only for its [`Backend::session_key`], and runs
/// nothing once a table's schema, a view, a setting or where names are
/// searched has changed.
///
/// Every run writes the same tables, so two must not overlap: a run's
/// tables stay until its [`Rewritten::release`].
pub struct Prepared<C = ()> {
    /// Each job's program, in order, with what its run did.
    jobs: Vec<(Box<dyn Built>, Trace<C>)>,
    statements: Vec<String>,
    /// The engine's session, as [`prepare`] found it.
    session: Option<String>,
}

impl<C> fmt::Debug for Prepared<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Prepared")
            .field("statements", &self.statements)
            .field("programs", &self.jobs.len())
            .finish()
    }
}

/// What [`prepare`] returns.
type Preparing<B> =
    Result<(Prepared<<B as Backend>::Compiled>, Rewritten), RunError<<B as Backend>::Error>>;

/// [`crate::rewrite`], once: compute what the calls in `statements` read on
/// `backend`, and rewrite each statement to read it, keeping what ran to
/// run again ([`Prepared::run`]). It builds every program, keeping none in
/// a [`Programs`], so it asks for no [`Backend::plan_key`]. It has the
/// engine compile each plan it runs ([`Backend::compile`]), for later runs
/// to run what was compiled.
///
/// Returns the prepared statements, and this run's [`Rewritten`] (whose
/// [`Rewritten::release`] keeps no program, in whichever `programs` it is
/// given). If anything fails, it drops every table it wrote, and nothing
/// is prepared.
pub fn prepare<'a, B: Backend>(
    backend: &'a mut B,
    statements: &[&str],
    dialect: &dyn Dialect,
) -> impl Future<Output = Preparing<B>> + Send + 'a {
    // Parsed and planned before the first await, as in `rewrite`.
    let planned = plan::Derivatives::plan(statements, dialect);
    async move {
        let mut planned = planned.map_err(RunError::Refused)?;
        let session = backend.session_key().await.map_err(RunError::Engine)?;
        let (mut tables, mut keys, mut traces) = (Vec::new(), Vec::new(), Vec::new());
        let result = run_all(
            backend,
            &mut planned,
            None,
            &mut keys,
            &mut tables,
            Some(&mut traces),
        )
        .await;
        let statements = match result {
            Ok(statements) => statements,
            Err(e) => return Err(released(backend, tables, e).await),
        };
        let prepared = Prepared {
            jobs: planned.into_built().into_iter().zip(traces).collect(),
            statements: statements.clone(),
            session,
        };
        let rewritten = Rewritten {
            statements,
            tables,
            kept: Vec::new(),
        };
        Ok((prepared, rewritten))
    }
}

impl<C: Send + Sync> Prepared<C> {
    /// Run what [`prepare`] ran, on the same `backend`: each job's seeds,
    /// then its program's checks and steps, each plan as it was bound then
    /// (see the type's docs for what that asks of the tables). Returns the
    /// statements, rewritten as they were, and the tables they read, which
    /// the caller drops once it has planned or run them
    /// ([`Rewritten::release`], which keeps no program). If anything
    /// fails, every table this run wrote is dropped.
    ///
    /// Returns `None`, and runs nothing, if the engine's session is not as
    /// [`prepare`] found it, or it cannot say ([`Backend::session_key`]):
    /// the statements are then prepared again.
    // Not an `async fn`, so that its future is promised `Send`, as
    // `rewrite`'s is.
    #[allow(clippy::manual_async_fn)]
    pub fn run<'a, B: Backend<Compiled = C>>(
        &'a self,
        backend: &'a mut B,
    ) -> impl Future<Output = Result<Option<Rewritten>, RunError<B::Error>>> + Send + 'a {
        async move {
            let session = backend.session_key().await.map_err(RunError::Engine)?;
            if session.is_none() || session != self.session {
                return Ok(None);
            }
            let mut tables = Vec::new();
            for (program, trace) in &self.jobs {
                let result = async {
                    for (name, seed) in &trace.seeds {
                        tables.push(name.clone());
                        crate::run::materialize(backend, name, seed).await?;
                    }
                    crate::run::run_bound(
                        backend,
                        &**program,
                        Plans::Bound(&trace.plans),
                        &trace.drops,
                    )
                    .await
                }
                .await;
                if let Err(e) = result {
                    return Err(released(backend, tables, e).await);
                }
                tables.extend(results(&**program));
            }
            Ok(Some(Rewritten {
                statements: self.statements.clone(),
                tables,
                kept: Vec::new(),
            }))
        }
    }

    /// The statements, rewritten as every run gives them.
    #[must_use]
    pub fn statements(&self) -> &[String] {
        &self.statements
    }
}

/// `error`, once `tables` are dropped from `backend` (a failed drop is not
/// reported over it).
async fn released<B: Backend>(
    backend: &mut B,
    tables: Vec<String>,
    error: RunError<B::Error>,
) -> RunError<B::Error> {
    let written = Rewritten {
        statements: Vec::new(),
        tables,
        kept: Vec::new(),
    };
    let _ = written.release(backend, &Programs::new()).await;
    error
}
