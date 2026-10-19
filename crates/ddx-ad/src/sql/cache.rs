// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The programs [`super::rewrite`] keeps from one call to the next.

use std::fmt;
use std::sync::{Mutex, MutexGuard};

use super::plan::{Built, Job};
use crate::run::Backend;

/// How many programs a [`Programs`] keeps.
const CAPACITY: usize = 16;

/// Programs [`crate::rewrite`] built, kept to run again, as a training loop
/// that runs the same statements each step needs: the second call builds
/// nothing.
///
/// A program depends on the engine's plans of its query and of the
/// restrictions its statements put on a gradient's rows, not on the
/// tables' values, so each is kept under its query's text, its transform
/// and `wrt` columns, the restrictions' text, and the engine's
/// [`Backend::plan_key`] of each query. The engine's key changes when a
/// table the query reads is replaced by one of another schema, or a view
/// it reads is redefined, so the program is built again. An engine whose
/// key is `None` gets nothing kept.
///
/// A program is taken out while it runs and put back only when the caller
/// drops the tables it wrote ([`crate::Rewritten::release`]), so calls that
/// overlap never use one program at once (its runs write the same tables):
/// the second builds its own. At most 16 are kept, the least recently used
/// dropped first; dropping one costs only building it again.
///
/// The engine's optimizer may fold a stable function, such as `now()`,
/// into a literal when it plans. A kept program holds what it folded to
/// when it was built.
#[derive(Default)]
pub struct Programs {
    /// Least recently used first.
    kept: Mutex<Vec<(Key, Box<dyn Built>)>>,
}

impl Programs {
    /// None kept yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// The program kept under `key`, taken out.
    pub(super) fn take(&self, key: &Key) -> Option<Box<dyn Built>> {
        let mut kept = self.lock();
        let i = kept.iter().position(|(k, _)| k == key)?;
        Some(kept.remove(i).1)
    }

    /// Keep `program` under `key`, as the most recently used, in place of
    /// one kept under it.
    pub(super) fn keep(&self, key: Key, program: Box<dyn Built>) {
        let mut kept = self.lock();
        kept.retain(|(k, _)| *k != key);
        kept.push((key, program));
        if kept.len() > CAPACITY {
            kept.remove(0);
        }
    }

    fn lock(&self) -> MutexGuard<'_, Vec<(Key, Box<dyn Built>)>> {
        // A panic while the lock was held left the list whole: every change
        // to it is one call that does not panic.
        self.kept.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl fmt::Debug for Programs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Programs")
            .field("kept", &self.lock().len())
            .finish()
    }
}

/// What a job's program is built from.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Key {
    derivative: String,
    /// The job's query, then each restriction's.
    queries: Vec<String>,
    /// The engine's key of each of `queries`.
    plans: Vec<String>,
}

impl Key {
    /// `job`'s key, its query planned as `queries` (the query, or each CTE
    /// planned on its own and then the query), or `None` if the engine
    /// cannot give one for one of them or a restriction's. An engine that
    /// fails on a query fails here as it would have planning it; one that
    /// fails on a restriction's, which the build would have left out, keeps
    /// nothing.
    ///
    /// `fresh` is a namespace the queries name tables under that is new to
    /// this call (a split's placeholders): it is written as one name in the
    /// key, in the queries and in the engine's keys of them, so the next
    /// call, with a namespace of its own, finds the program.
    pub(super) async fn of<B: Backend>(
        backend: &mut B,
        job: &Job,
        queries: &[String],
        fresh: Option<&str>,
    ) -> Result<Option<Key>, B::Error> {
        let same = |s: &str| match fresh {
            Some(ns) => s.replace(ns, "__ddx_fresh_"),
            None => s.to_string(),
        };
        let mut plans = Vec::new();
        for q in queries {
            let Some(plan) = backend.plan_key(q).await? else {
                return Ok(None);
            };
            plans.push(same(&plan));
        }
        let mut queries: Vec<String> = queries.iter().map(|q| same(q)).collect();
        for r in &job.pushdown {
            let Ok(Some(plan)) = backend.plan_key(&r.query).await else {
                return Ok(None);
            };
            queries.push(r.query.clone());
            plans.push(plan);
        }
        Ok(Some(Key {
            derivative: job.derivative.clone(),
            queries,
            plans,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{Of, OutputTable};

    fn key(n: usize) -> Key {
        Key {
            derivative: "grad []".into(),
            queries: vec![format!("SELECT {n}")],
            plans: vec![String::new()],
        }
    }

    /// A program whose value is the step `name`.
    fn program(name: &str) -> Box<dyn Built> {
        Box::new(crate::ForwardProgram {
            inputs: Vec::new(),
            checks: Vec::new(),
            steps: Vec::new(),
            value: OutputTable {
                step: name.into(),
                of: Of::Output,
                columns: Vec::new(),
                tangents: Vec::new(),
            },
            gradients: Vec::new(),
        })
    }

    #[test]
    fn a_program_is_taken_out_while_it_runs() {
        let programs = Programs::new();
        programs.keep(key(0), program("a"));
        assert_eq!(programs.take(&key(0)).unwrap().value().step, "a");
        assert!(programs.take(&key(0)).is_none(), "taken twice");
        assert!(programs.take(&key(1)).is_none());
    }

    #[test]
    fn a_program_kept_under_a_key_replaces_the_one_there() {
        let programs = Programs::new();
        programs.keep(key(0), program("a"));
        programs.keep(key(0), program("b"));
        assert_eq!(programs.take(&key(0)).unwrap().value().step, "b");
        assert!(programs.take(&key(0)).is_none());
    }

    #[test]
    fn the_least_recently_used_program_goes_first() {
        let programs = Programs::new();
        for n in 0..CAPACITY {
            programs.keep(key(n), program(&n.to_string()));
        }
        // Program 0 runs again, so program 1 is now the least recently used.
        let p = programs.take(&key(0)).unwrap();
        programs.keep(key(0), p);
        programs.keep(key(CAPACITY), program("new"));
        assert!(
            programs.take(&key(1)).is_none(),
            "the least recently used stayed"
        );
        for n in (0..=CAPACITY).filter(|&n| n != 1) {
            assert!(programs.take(&key(n)).is_some(), "program {n} went");
        }
    }
}
