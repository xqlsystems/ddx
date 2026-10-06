// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Running a [`Program`] (a [`BackwardProgram`] or a [`ForwardProgram`]): the
//! protocol every engine follows.
//!
//! A program is data, and running it takes four things only an engine can
//! do: say whether a plan returns any row (a [check](BackwardProgram::checks)),
//! materialize a plan's rows as a named table, drop a table, and state a
//! table's schema (to bind a plan's reads of earlier steps; see
//! [`crate::bind_reads`]). The rest is the same everywhere and lives here: the
//! checks run first and a row from any refuses the program, then every step in
//! order. A run that succeeds drops the intermediate tables and leaves the
//! value and the gradients. A run that fails, at a check or a step, leaves
//! none of the program's tables, so a loop that reruns one program (under a
//! fixed [namespace](crate::Options::namespace), say) never finds this run's
//! value beside an earlier run's gradient. Two programs that share a namespace
//! write the same tables, so they must not run on one engine.
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
//!         match &action {
//!             Action::Check(i) => runner.checked(engine_check(&program.checks[*i])),
//!             Action::Materialize(i) => runner.done(engine_materialize(program.step(*i))),
//!             Action::Drop(name) => runner.done(engine_drop(name)),
//!         }
//!     }
//!     runner.finish()
//! }
//! ```

use std::collections::HashMap;
use std::fmt;

use substrait::proto::{NamedStruct, Plan};

use crate::emit::{bind_reads, unbound_reads};
use crate::error::AdError;
use crate::program::{BackwardProgram, Check, ForwardProgram, Step};

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::program::BackwardProgram {}
    impl Sealed for crate::program::ForwardProgram {}
}

/// What the run protocol needs of a program: its checks, its steps in order,
/// and which steps' tables a caller reads once it has run (the others are
/// dropped). A [`BackwardProgram`] and a [`ForwardProgram`] are programs.
///
/// Sealed: an engine runs programs, and ddx makes them; a new kind of
/// program is a change to this crate, not an extension point.
pub trait Program: sealed::Sealed {
    /// Plans that must return no rows, run before the steps.
    fn checks(&self) -> &[Check];
    /// The number of steps.
    fn step_count(&self) -> usize;
    /// Step `i`, in the order they run.
    fn step(&self, i: usize) -> &Step;
    /// Whether the table of the step named `step` stays after a successful
    /// run, for a caller to read.
    fn is_result(&self, step: &str) -> bool;
}

impl Program for BackwardProgram {
    fn checks(&self) -> &[Check] {
        &self.checks
    }
    fn step_count(&self) -> usize {
        self.forward_steps.len() + self.backward_steps.len()
    }
    fn step(&self, i: usize) -> &Step {
        BackwardProgram::step(self, i)
    }
    fn is_result(&self, step: &str) -> bool {
        self.value.step == step || self.gradients.iter().any(|g| g.step == step)
    }
}

impl Program for ForwardProgram {
    fn checks(&self) -> &[Check] {
        &self.checks
    }
    fn step_count(&self) -> usize {
        self.steps.len()
    }
    fn step(&self, i: usize) -> &Step {
        &self.steps[i]
    }
    fn is_result(&self, step: &str) -> bool {
        self.value.step == step || self.gradients.iter().any(|g| g.step == step)
    }
}

/// What a [`Runner`] asks its engine to do next.
///
/// Not `non_exhaustive`: an engine has to do every action, so a new one is a
/// breaking change on purpose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Run [`BackwardProgram::checks`]`[i]` and say whether its plan returned
    /// any row, with [`Runner::checked`]; a row refuses the program.
    Check(usize),
    /// Run [`BackwardProgram::step`]`(i)` and materialize its rows as the
    /// table its name gives, replacing any table of that name; then
    /// [`Runner::done`].
    Materialize(usize),
    /// Drop the table of this name if it exists; then [`Runner::done`].
    Drop(String),
}

/// Why a program's run failed.
#[derive(Debug)]
#[non_exhaustive]
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
    /// The checks to run: each one's index in the program, and its message.
    checks: Vec<(usize, String)>,
    steps: Vec<String>,
    intermediates: Vec<String>,
    /// What the drop phase drops: the intermediates after a success, every
    /// step's table after a failure.
    dropping: Vec<String>,
    phase: Phase,
    waiting: bool,
    failure: Option<RunError<E>>,
}

impl<E> Runner<E> {
    /// A run of `program`, not yet started.
    pub fn new<P: Program + ?Sized>(program: &P) -> Self {
        Runner::verified(program, &Verified::new())
    }

    /// A run of `program` that skips each check whose fact `verified` already
    /// holds (see [`Verified`]). [`Action::Check`] still numbers checks as
    /// the program does.
    pub fn verified<P: Program + ?Sized>(program: &P, verified: &Verified) -> Self {
        let steps: Vec<&Step> = (0..program.step_count()).map(|i| program.step(i)).collect();
        let mut runner = Runner {
            checks: program
                .checks()
                .iter()
                .enumerate()
                .filter(|(_, c)| !verified.knows(c))
                .map(|(i, c)| (i, c.message.clone()))
                .collect(),
            steps: steps.iter().map(|s| s.name.clone()).collect(),
            intermediates: steps
                .iter()
                .filter(|s| !program.is_result(&s.name))
                .map(|s| s.name.clone())
                .collect(),
            dropping: Vec::new(),
            phase: Phase::Check(0),
            waiting: false,
            failure: None,
        };
        runner.settle();
        runner
    }

    /// The next action, or `None` once the run is over; then call
    /// [`Runner::finish`]. Each action must be answered, with
    /// [`Runner::checked`] for a check and [`Runner::done`] otherwise, before
    /// the next is asked for.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<Action> {
        assert!(
            !self.waiting,
            "answer the last action before asking for the next"
        );
        let action = match self.phase {
            Phase::Check(k) => Action::Check(self.checks[k].0),
            Phase::Step(i) => Action::Materialize(i),
            Phase::Drop(i) => Action::Drop(self.dropping[i].clone()),
            Phase::Done => return None,
        };
        self.waiting = true;
        Some(action)
    }

    /// How the pending [`Action::Check`] went: whether its plan returned a
    /// row.
    pub fn checked(&mut self, result: Result<bool, E>) {
        let Phase::Check(i) = self.answer("checked", true) else {
            unreachable!()
        };
        match result {
            Ok(false) => self.phase = Phase::Check(i + 1),
            Ok(true) => self.fail(RunError::Refused(AdError::InvalidWrt(
                self.checks[i].1.clone(),
            ))),
            Err(e) => self.fail(RunError::Engine(e)),
        }
        self.settle();
    }

    /// How the pending [`Action::Materialize`] or [`Action::Drop`] went.
    pub fn done(&mut self, result: Result<(), E>) {
        match (self.answer("done", false), result) {
            (Phase::Step(i), Ok(())) => self.phase = Phase::Step(i + 1),
            (Phase::Step(_), Err(e)) => self.fail(RunError::Engine(e)),
            // Every table is dropped even after a failed drop; the first
            // failure is the one reported.
            (Phase::Drop(i), result) => {
                if let (Err(e), None) = (result, &self.failure) {
                    self.failure = Some(RunError::Engine(e));
                }
                self.phase = Phase::Drop(i + 1);
            }
            _ => unreachable!(),
        }
        self.settle();
    }

    /// Abandon the pending action with a refusal of ddx's own (a plan's
    /// reads that could not be bound, say), as if the engine had failed it.
    pub fn refuse(&mut self, e: AdError) {
        assert!(self.waiting, "refuse answers an action");
        self.waiting = false;
        match self.phase {
            Phase::Check(_) | Phase::Step(_) => self.fail(RunError::Refused(e)),
            Phase::Drop(i) => {
                if self.failure.is_none() {
                    self.failure = Some(RunError::Refused(e));
                }
                self.phase = Phase::Drop(i + 1);
            }
            Phase::Done => unreachable!("no action is pending once the run is done"),
        }
        self.settle();
    }

    /// The run's result, once [`Runner::next`] has returned `None`.
    pub fn finish(self) -> Result<(), RunError<E>> {
        assert!(self.phase == Phase::Done, "finish follows the last action");
        self.failure.map_or(Ok(()), Err)
    }

    /// Take the answer to the pending action, which must be a check exactly
    /// when `check`.
    fn answer(&mut self, how: &str, check: bool) -> Phase {
        assert!(self.waiting, "{how} answers an action");
        assert_eq!(
            matches!(self.phase, Phase::Check(_)),
            check,
            "a check is answered with `checked`, any other action with `done`"
        );
        self.waiting = false;
        self.phase
    }

    /// A check or a step failed: drop every table of the program.
    fn fail(&mut self, failure: RunError<E>) {
        self.failure = Some(failure);
        self.dropping = self.steps.clone();
        self.phase = Phase::Drop(0);
    }

    /// Move past phases with nothing left in them.
    fn settle(&mut self) {
        loop {
            self.phase = match self.phase {
                Phase::Check(i) if i >= self.checks.len() => Phase::Step(0),
                Phase::Step(i) if i >= self.steps.len() => {
                    self.dropping = self.intermediates.clone();
                    Phase::Drop(0)
                }
                Phase::Drop(i) if i >= self.dropping.len() => Phase::Done,
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
    /// [`run`] asks once per table, and again after the table is rewritten.
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
pub fn run<B: Backend, P: Program + ?Sized>(
    backend: &mut B,
    program: &P,
) -> Result<(), RunError<B::Error>> {
    run_verified(backend, program, &mut Verified::new())
}

/// [`run`], skipping each check whose fact `verified` holds, and recording
/// in it every check's fact once the run has succeeded (see [`Verified`]).
pub fn run_verified<B: Backend, P: Program + ?Sized>(
    backend: &mut B,
    program: &P,
    verified: &mut Verified,
) -> Result<(), RunError<B::Error>> {
    let mut runner = Runner::verified(program, verified);
    let mut schemas: HashMap<String, NamedStruct> = HashMap::new();
    while let Some(action) = runner.next() {
        let plan = match &action {
            Action::Check(i) => &program.checks()[*i].plan,
            Action::Materialize(i) => &program.step(*i).plan,
            Action::Drop(name) => {
                schemas.remove(name);
                runner.done(backend.drop_table(name));
                continue;
            }
        };
        let plan = match bind(backend, plan, &mut schemas) {
            Ok(plan) => plan,
            Err(Bind::Refused(e)) => {
                runner.refuse(e);
                continue;
            }
            Err(Bind::Engine(e)) => {
                match action {
                    Action::Check(_) => runner.checked(Err(e)),
                    _ => runner.done(Err(e)),
                }
                continue;
            }
        };
        match action {
            Action::Check(_) => runner.checked(backend.returns_rows(&plan)),
            Action::Materialize(i) => {
                let name = &program.step(i).name;
                // A table rewritten has the schema of this run's plan.
                schemas.remove(name);
                runner.done(backend.materialize(name, &plan));
            }
            Action::Drop(_) => unreachable!("handled above"),
        }
    }
    runner.finish()?;
    verified.record_all(program);
    Ok(())
}

/// Facts a program's checks have proved, kept by a caller across runs: that
/// in a table, the key columns a check names identify its rows. A run given
/// them ([`Runner::verified`], [`run_verified`]) skips each check whose fact
/// is already known, and a run that succeeds records every check's fact.
///
/// A check costs a grouped scan of its table, every run. When many runs read
/// the same tables, most of those scans prove again what the first proved:
/// conjugate gradient runs one `H·v` program per iteration against the same
/// parameters, and a training loop one `grad` program per step whose update
/// joins on the parameters' dims and so keeps them.
///
/// The fact is about the table's rows, which ddx cannot watch: it holds only
/// until the table's keys change, and keeping it past that is the caller's
/// promise. [`Verified::forget`] a table whenever its keys may have changed:
/// rewritten from another query, rows inserted, a new tangent or cotangent
/// written under the same name. A fact kept for a table whose keys now
/// repeat lets a program run that its check would have refused, and give
/// rows that share keys their summed gradient.
#[derive(Debug, Clone, Default)]
pub struct Verified {
    facts: std::collections::BTreeSet<(Vec<String>, Vec<String>)>,
}

impl Verified {
    /// No facts.
    pub fn new() -> Self {
        Verified::default()
    }

    /// Does it hold the fact `check` proves?
    pub fn knows(&self, check: &Check) -> bool {
        self.facts
            .contains(&(check.table.clone(), check.keys.clone()))
    }

    /// Hold the fact `check` proves, as a run that passed it would.
    pub fn record(&mut self, check: &Check) {
        self.facts.insert((check.table.clone(), check.keys.clone()));
    }

    /// Hold the facts every check of `program` proves.
    pub fn record_all<P: Program + ?Sized>(&mut self, program: &P) {
        for c in program.checks() {
            self.record(c);
        }
    }

    /// Drop every fact about the table `table` (named as a
    /// [`crate::ColumnRef`] names one: a bare name matches a table whose last
    /// part it is), because its keys may have changed.
    pub fn forget(&mut self, table: &str) {
        self.facts
            .retain(|(names, _)| !crate::relation::table_matches(table, names));
    }

    /// Drop every fact.
    pub fn clear(&mut self) {
        self.facts.clear();
    }
}

enum Bind<E> {
    Refused(AdError),
    Engine(E),
}

fn bind<B: Backend>(
    backend: &mut B,
    plan: &Plan,
    schemas: &mut HashMap<String, NamedStruct>,
) -> Result<Plan, Bind<B::Error>> {
    let mut plan = plan.clone();
    for name in unbound_reads(&plan) {
        if let std::collections::hash_map::Entry::Vacant(slot) = schemas.entry(name) {
            let schema = backend.table_schema(slot.key()).map_err(Bind::Engine)?;
            slot.insert(schema);
        }
    }
    bind_reads(&mut plan, &mut |name| schemas.get(name).cloned()).map_err(Bind::Refused)?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The actions a runner takes when every action gets `outcome(action)`
    /// (for a check, whether it returned a row).
    fn trace(
        checks: usize,
        steps: &[&str],
        intermediates: &[&str],
        outcome: &dyn Fn(&Action) -> Result<bool, &'static str>,
    ) -> (Vec<Action>, Result<(), String>) {
        let mut runner: Runner<&'static str> = Runner {
            checks: (0..checks).map(|i| (i, format!("check {i}"))).collect(),
            steps: steps.iter().map(|s| s.to_string()).collect(),
            intermediates: intermediates.iter().map(|s| s.to_string()).collect(),
            dropping: Vec::new(),
            phase: Phase::Check(0),
            waiting: false,
            failure: None,
        };
        runner.settle();
        let mut seen = Vec::new();
        while let Some(a) = runner.next() {
            match a {
                Action::Check(_) => runner.checked(outcome(&a)),
                _ => runner.done(outcome(&a).map(|_| ())),
            }
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
    fn a_check_that_returns_a_row_refuses_and_leaves_none_of_the_programs_tables() {
        // An earlier run's value and gradients may still be there; a failed
        // run leaves none, so they never sit beside a different run's.
        let (seen, result) = trace(2, &["s0", "v"], &["s0"], &|a| Ok(*a == Action::Check(1)));
        assert_eq!(
            seen,
            vec![
                Action::Check(0),
                Action::Check(1),
                Action::Drop("s0".into()),
                Action::Drop("v".into()),
            ]
        );
        assert_eq!(result, Err("invalid wrt column: check 1".into()));
    }

    #[test]
    fn a_failed_step_drops_every_table_and_reports_the_step() {
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
                Action::Drop("v".into()),
            ]
        );
        assert_eq!(result, Err("disk full".into()));
    }

    #[test]
    #[should_panic(expected = "a check is answered with `checked`")]
    fn a_step_answered_as_a_check_is_a_bug_in_the_engine_loop() {
        let mut runner: Runner<()> = Runner {
            checks: Vec::new(),
            steps: vec!["v".into()],
            intermediates: Vec::new(),
            dropping: Vec::new(),
            phase: Phase::Check(0),
            waiting: false,
            failure: None,
        };
        runner.settle();
        assert_eq!(runner.next(), Some(Action::Materialize(0)));
        runner.checked(Ok(true));
    }
}
