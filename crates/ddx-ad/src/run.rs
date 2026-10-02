// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Running a [`BackwardProgram`]: the protocol every engine follows.
//!
//! A program is data, and running it takes four things only an engine can
//! do: say whether a plan returns any row (a [check](BackwardProgram::checks)),
//! materialize a plan's rows as a named table, drop a table, and state a
//! table's schema (to bind a plan's reads of earlier steps; see
//! [`crate::bind_reads`]). The rest is the same everywhere and lives here: the
//! checks run first and a row from any refuses the program, then every step in
//! order, and the intermediate tables are dropped whether the steps succeed or
//! fail, leaving the value and the gradients.
//!
//! [`Runner`] is that protocol without I/O: it says what to do next and is
//! told how it went, so a synchronous engine, an asynchronous one and a
//! Python one drive the same state machine from their own loop. [`Backend`]
//! and [`run`] are the shortcut for a synchronous engine.
//!
//! ```
//! # use ddx_ad::{Action, BackwardProgram, Runner};
//! # fn engine_check(_: &ddx_ad::Check) -> Result<bool, String> { Ok(false) }
//! # fn engine_materialize(_: &ddx_ad::Step) -> Result<(), String> { Ok(()) }
//! # fn engine_drop(_: &str) -> Result<(), String> { Ok(()) }
//! fn run(program: &BackwardProgram) -> Result<(), ddx_ad::RunError<String>> {
//!     let mut runner = Runner::new(program);
//!     while let Some(action) = runner.next() {
//!         let result = match &action {
//!             Action::Check(i) => engine_check(&program.checks[*i]),
//!             Action::Materialize(i) => engine_materialize(program.step(*i)).map(|()| false),
//!             Action::Drop(name) => engine_drop(name).map(|()| false),
//!         };
//!         runner.report(result);
//!     }
//!     runner.finish()
//! }
//! ```

use std::fmt;

use substrait::proto::{NamedStruct, Plan};

use crate::emit::{bind_reads, unbound_reads};
use crate::error::AdError;
use crate::program::BackwardProgram;

/// What a [`Runner`] asks its engine to do next.
///
/// Not `non_exhaustive`: an engine has to do every action, so a new one is a
/// breaking change on purpose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Run [`BackwardProgram::checks`]`[i]` and report whether its plan
    /// returned any row (`Ok(true)`), which refuses the program.
    Check(usize),
    /// Run [`BackwardProgram::step`]`(i)` and materialize its rows as the
    /// table its name gives, replacing any table of that name; report
    /// `Ok(false)`.
    Materialize(usize),
    /// Drop the table of this name if it exists; report `Ok(false)`.
    Drop(String),
}

/// Why a program's run failed.
#[derive(Debug)]
pub enum RunError<E> {
    /// A check returned a row ([`AdError::InvalidWrt`]: a `wrt` table's rows
    /// are not what the program assumed), or a plan's reads could not be
    /// bound.
    Refused(AdError),
    /// The engine failed.
    Engine(E),
}

impl<E: fmt::Display> fmt::Display for RunError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Refused(e) => write!(f, "{e}"),
            RunError::Engine(e) => write!(f, "{e}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for RunError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RunError::Refused(e) => Some(e),
            RunError::Engine(e) => Some(e),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Check(usize),
    Step(usize),
    Drop(usize),
    Done,
}

/// The run protocol for one program, as a state machine (see the module
/// docs). It holds names and counts, not the program, so it can live apart
/// from it (across an FFI boundary, say).
#[derive(Debug)]
pub struct Runner<E> {
    messages: Vec<String>,
    steps: usize,
    intermediates: Vec<String>,
    phase: Phase,
    waiting: bool,
    failure: Option<RunError<E>>,
}

impl<E> Runner<E> {
    /// A run of `program`, not yet started.
    pub fn new(program: &BackwardProgram) -> Self {
        let mut runner = Runner {
            messages: program.checks.iter().map(|c| c.message.clone()).collect(),
            steps: program.steps().count(),
            intermediates: program
                .intermediate_steps()
                .map(|s| s.name.clone())
                .collect(),
            phase: Phase::Check(0),
            waiting: false,
            failure: None,
        };
        runner.settle();
        runner
    }

    /// The next action, or `None` once the run is over; then call
    /// [`Runner::finish`]. Each action must be [reported](Runner::report)
    /// before the next is asked for.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<Action> {
        assert!(
            !self.waiting,
            "report the last action before asking for the next"
        );
        let action = match self.phase {
            Phase::Check(i) => Action::Check(i),
            Phase::Step(i) => Action::Materialize(i),
            Phase::Drop(i) => Action::Drop(self.intermediates[i].clone()),
            Phase::Done => return None,
        };
        self.waiting = true;
        Some(action)
    }

    /// How the last action went: for a check, whether its plan returned a
    /// row; `Ok(false)` for any other action that succeeded.
    pub fn report(&mut self, result: Result<bool, E>) {
        assert!(self.waiting, "report follows an action");
        self.waiting = false;
        match (self.phase, result) {
            (Phase::Check(i), Ok(false)) => self.phase = Phase::Check(i + 1),
            (Phase::Check(i), Ok(true)) => {
                // Nothing has been written yet, so there is nothing to drop.
                self.failure = Some(RunError::Refused(AdError::InvalidWrt(
                    self.messages[i].clone(),
                )));
                self.phase = Phase::Done;
            }
            (Phase::Check(_), Err(e)) => {
                self.failure = Some(RunError::Engine(e));
                self.phase = Phase::Done;
            }
            (Phase::Step(i), Ok(_)) => self.phase = Phase::Step(i + 1),
            (Phase::Step(_), Err(e)) => {
                self.failure = Some(RunError::Engine(e));
                self.phase = Phase::Drop(0);
            }
            // Every intermediate is dropped even after a failed drop; the
            // first failure is the one reported.
            (Phase::Drop(i), result) => {
                if let (Err(e), None) = (result, &self.failure) {
                    self.failure = Some(RunError::Engine(e));
                }
                self.phase = Phase::Drop(i + 1);
            }
            (Phase::Done, _) => unreachable!("no action is pending once the run is done"),
        }
        self.settle();
    }

    /// Abandon the last action with a refusal of ddx's own (a plan's reads
    /// that could not be bound, say), as if the engine had failed it.
    pub fn refuse(&mut self, e: AdError) {
        assert!(self.waiting, "refuse follows an action");
        self.waiting = false;
        if self.failure.is_none() {
            self.failure = Some(RunError::Refused(e));
        }
        self.phase = match self.phase {
            Phase::Check(_) | Phase::Done => Phase::Done,
            Phase::Step(_) => Phase::Drop(0),
            Phase::Drop(i) => Phase::Drop(i + 1),
        };
        self.settle();
    }

    /// The run's result, once [`Runner::next`] has returned `None`.
    pub fn finish(self) -> Result<(), RunError<E>> {
        assert!(self.phase == Phase::Done, "finish follows the last action");
        self.failure.map_or(Ok(()), Err)
    }

    /// Move past phases with nothing left in them.
    fn settle(&mut self) {
        loop {
            self.phase = match self.phase {
                Phase::Check(i) if i >= self.messages.len() => Phase::Step(0),
                Phase::Step(i) if i >= self.steps => Phase::Drop(0),
                Phase::Drop(i) if i >= self.intermediates.len() => Phase::Done,
                _ => return,
            };
        }
    }
}

/// A synchronous engine, for [`run`].
pub trait Backend {
    /// The engine's error.
    type Error;

    /// The schema of the table `name` as the engine's own Substrait producer
    /// states it: the `base_schema` of the read in its plan of
    /// `SELECT * FROM name`. ddx binds a plan's reads of earlier steps with
    /// it, so names and types must be the ones the engine's consumer expects.
    fn table_schema(&mut self, name: &str) -> Result<NamedStruct, Self::Error>;

    /// Run `plan` and say whether it returned any row.
    fn returns_rows(&mut self, plan: &Plan) -> Result<bool, Self::Error>;

    /// Run `plan` and store its rows as the table `name`, replacing one.
    fn materialize(&mut self, name: &str, plan: &Plan) -> Result<(), Self::Error>;

    /// Drop the table `name` if it exists.
    fn drop_table(&mut self, name: &str) -> Result<(), Self::Error>;
}

/// Run `program` on `backend`: [`Runner`]'s protocol, binding each plan's
/// reads with [`Backend::table_schema`].
pub fn run<B: Backend>(
    backend: &mut B,
    program: &BackwardProgram,
) -> Result<(), RunError<B::Error>> {
    let mut runner = Runner::new(program);
    while let Some(action) = runner.next() {
        let plan = match &action {
            Action::Check(i) => Some(&program.checks[*i].plan),
            Action::Materialize(i) => Some(&program.step(*i).plan),
            Action::Drop(_) => None,
        };
        let bound = match plan.map(|p| bind(backend, p)).transpose() {
            Ok(bound) => bound,
            Err(Bind::Refused(e)) => {
                runner.refuse(e);
                continue;
            }
            Err(Bind::Engine(e)) => {
                runner.report(Err(e));
                continue;
            }
        };
        let result = match (&action, bound) {
            (Action::Check(_), Some(plan)) => backend.returns_rows(&plan),
            (Action::Materialize(i), Some(plan)) => backend
                .materialize(&program.step(*i).name, &plan)
                .map(|()| false),
            (Action::Drop(name), _) => backend.drop_table(name).map(|()| false),
            _ => unreachable!("checks and steps have plans"),
        };
        runner.report(result);
    }
    runner.finish()
}

enum Bind<E> {
    Refused(AdError),
    Engine(E),
}

fn bind<B: Backend>(backend: &mut B, plan: &Plan) -> Result<Plan, Bind<B::Error>> {
    let mut plan = plan.clone();
    let mut schemas = std::collections::HashMap::new();
    for name in unbound_reads(&plan) {
        let schema = backend.table_schema(&name).map_err(Bind::Engine)?;
        schemas.insert(name, schema);
    }
    bind_reads(&mut plan, &mut |name| schemas.get(name).cloned()).map_err(Bind::Refused)?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The actions a runner takes when every action gets `outcome(action)`.
    fn trace(
        checks: usize,
        steps: &[&str],
        intermediates: &[&str],
        outcome: &dyn Fn(&Action) -> Result<bool, &'static str>,
    ) -> (Vec<Action>, Result<(), String>) {
        let mut runner: Runner<&'static str> = Runner {
            messages: (0..checks).map(|i| format!("check {i}")).collect(),
            steps: steps.len(),
            intermediates: intermediates.iter().map(|s| s.to_string()).collect(),
            phase: Phase::Check(0),
            waiting: false,
            failure: None,
        };
        runner.settle();
        let mut seen = Vec::new();
        while let Some(a) = runner.next() {
            runner.report(outcome(&a));
            seen.push(a);
        }
        (seen, runner.finish().map_err(|e| e.to_string()))
    }

    #[test]
    fn checks_then_steps_then_the_intermediates_are_dropped() {
        let (seen, result) = trace(2, &["s0", "s1", "v"], &["s0", "s1"], &|_| Ok(false));
        assert_eq!(
            seen,
            vec![
                Action::Check(0),
                Action::Check(1),
                Action::Materialize(0),
                Action::Materialize(1),
                Action::Materialize(2),
                Action::Drop("s0".into()),
                Action::Drop("s1".into()),
            ]
        );
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn a_check_that_returns_a_row_refuses_before_anything_is_written() {
        let (seen, result) = trace(2, &["s0"], &["s0"], &|a| Ok(*a == Action::Check(1)));
        assert_eq!(seen, vec![Action::Check(0), Action::Check(1)]);
        assert_eq!(result, Err("invalid wrt column: check 1".into()));
    }

    #[test]
    fn a_failed_step_still_drops_every_intermediate_and_reports_the_step() {
        let (seen, result) = trace(0, &["s0", "s1", "v"], &["s0", "s1"], &|a| match a {
            Action::Materialize(1) => Err("disk full"),
            Action::Drop(n) if n == "s0" => Err("busy"),
            _ => Ok(false),
        });
        assert_eq!(
            seen,
            vec![
                Action::Materialize(0),
                Action::Materialize(1),
                Action::Drop("s0".into()),
                Action::Drop("s1".into()),
            ]
        );
        assert_eq!(result, Err("disk full".into()));
    }
}
