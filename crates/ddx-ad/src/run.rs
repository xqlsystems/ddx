// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Running a [`Program`] (a [`BackwardProgram`] or a [`ForwardProgram`]) on
//! an engine.
//!
//! A program is data, and running it takes four things only an engine can
//! do, a [`Backend`]: say whether a plan returns any row (a
//! [check](BackwardProgram::checks)), materialize a plan's rows as a named
//! table, drop a table, and state a table's schema, to bind a plan's reads
//! of earlier steps ([`Unbound::bind`]). The rest is the same everywhere and
//! is [`run`]'s: the checks run first and a row from any refuses the program,
//! then every step in order. A run that succeeds drops the intermediate
//! tables and leaves the value and the gradients. A run that fails, at a
//! check or a step, leaves none of the program's tables, so a loop that
//! reruns one program (under a fixed [namespace](crate::Options::namespace),
//! say) never finds this run's value beside an earlier run's gradient. Two
//! programs that share a namespace write the same tables, so they must not
//! run on one engine.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;

use substrait::proto::NamedStruct;

use crate::error::AdError;
use crate::program::{BackwardProgram, Bound, Check, ForwardProgram, Step, Unbound};
use crate::tables::{InputTable, OutputTable};

pub(crate) mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::program::BackwardProgram {}
    impl Sealed for crate::program::ForwardProgram {}
    #[cfg(test)]
    impl Sealed for super::tests::Fake {}
}

/// What the run protocol needs of a program: the tables a caller registers
/// before it runs, its checks, its steps in order, and the tables it leaves.
/// A [`BackwardProgram`] and a [`ForwardProgram`] are programs.
///
/// Sealed: an engine runs programs, and ddx makes them; a new kind of
/// program is a change to this crate, not an extension point.
pub trait Program: sealed::Sealed + Sync {
    /// The tables to register before it runs: a `vjp`'s cotangent, a
    /// `jvp`'s tangents.
    fn inputs(&self) -> &[InputTable];
    /// Plans that must return no rows, run before the steps.
    fn checks(&self) -> &[Check];
    /// The number of steps.
    fn step_count(&self) -> usize;
    /// Step `i`, in the order they run.
    fn step(&self, i: usize) -> &Step;
    /// The query's output, with its tangents for a `jvp` program.
    fn value(&self) -> &OutputTable;
    /// One per `wrt` table: its gradient, with its tangents for `jvp` of a
    /// program that has gradients.
    fn gradients(&self) -> &[OutputTable];
    /// Whether the table of the step named `step` stays after a successful
    /// run, for a caller to read: the value's or a gradient's.
    fn is_result(&self, step: &str) -> bool {
        self.value().step == step || self.gradients().iter().any(|g| g.step == step)
    }
}

impl Program for BackwardProgram {
    fn inputs(&self) -> &[InputTable] {
        &self.inputs
    }
    fn checks(&self) -> &[Check] {
        &self.checks
    }
    fn step_count(&self) -> usize {
        self.forward_steps.len() + self.backward_steps.len()
    }
    fn step(&self, i: usize) -> &Step {
        BackwardProgram::step(self, i)
    }
    fn value(&self) -> &OutputTable {
        &self.value
    }
    fn gradients(&self) -> &[OutputTable] {
        &self.gradients
    }
}

impl Program for ForwardProgram {
    fn inputs(&self) -> &[InputTable] {
        &self.inputs
    }
    fn checks(&self) -> &[Check] {
        &self.checks
    }
    fn step_count(&self) -> usize {
        self.steps.len()
    }
    fn step(&self, i: usize) -> &Step {
        &self.steps[i]
    }
    fn value(&self) -> &OutputTable {
        &self.value
    }
    fn gradients(&self) -> &[OutputTable] {
        &self.gradients
    }
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

/// An engine, as [`run`] needs it: the four things only an engine can do.
///
/// Each method returns a future, so an asynchronous engine implements it
/// directly and a synchronous one returns a future that is already done.
/// The futures are `Send`, so a run can be spawned on a multi-threaded
/// runtime.
pub trait Backend: Send {
    /// The engine's error.
    type Error: Send;

    /// The schema of the table `name` as the engine's own Substrait producer
    /// states it: the `base_schema` of the read in its plan of
    /// `SELECT * FROM name`. ddx binds a plan's reads of earlier steps with
    /// it, so names and types must be the ones the engine's consumer expects.
    /// [`run`] asks once per table, and again after the table is rewritten.
    fn table_schema(
        &mut self,
        name: &str,
    ) -> impl Future<Output = Result<NamedStruct, Self::Error>> + Send;

    /// Run each of `plans`, a program's checks, and say whether each returned
    /// any row, in order. An engine may run them at once: none reads
    /// another, or any table a step writes.
    fn returns_rows(
        &mut self,
        plans: &[Bound],
    ) -> impl Future<Output = Result<Vec<bool>, Self::Error>> + Send;

    /// Run `plan` and store its rows as the table `name`, replacing one.
    fn materialize(
        &mut self,
        name: &str,
        plan: &Bound,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Drop the table `name` if it exists.
    fn drop_table(&mut self, name: &str) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// The engine's Substrait plan of the SQL query `sql`, planned and
    /// optimized as it would run it. Only [`crate::rewrite`] asks for one,
    /// to differentiate a query a SQL statement names; an engine that only
    /// runs programs may refuse.
    fn plan_sql(
        &mut self,
        sql: &str,
    ) -> impl Future<Output = Result<substrait::proto::Plan, Self::Error>> + Send;

    /// Text that is the same at two calls only if [`Backend::plan_sql`] of
    /// `sql` is: what the engine's plan depends on (the tables `sql` reads,
    /// with their columns, types and nullability, and the definitions of
    /// the views it reads), found without optimizing. [`crate::rewrite`]
    /// keys the programs it keeps ([`crate::Programs`]) by it, so a key
    /// that leaves out something the plan depends on would rerun a stale
    /// program. `None`, the default, means the engine cannot say, and
    /// nothing is kept.
    fn plan_key(
        &mut self,
        sql: &str,
    ) -> impl Future<Output = Result<Option<String>, Self::Error>> + Send {
        let _ = sql;
        async { Ok(None) }
    }
}

/// Run `program` on `backend`, as the module docs describe: its checks, then
/// its steps, each plan bound to the tables the engine holds; then the
/// intermediate tables are dropped, or, if anything failed, every table the
/// program writes.
pub async fn run<B: Backend, P: Program + ?Sized>(
    backend: &mut B,
    program: &P,
) -> Result<(), RunError<B::Error>> {
    let mut schemas: HashMap<String, NamedStruct> = HashMap::new();
    let steps: Vec<&Step> = (0..program.step_count()).map(|i| program.step(i)).collect();
    let result = async {
        let mut checks = Vec::with_capacity(program.checks().len());
        for check in program.checks() {
            checks.push(bind(backend, &check.plan, &mut schemas).await?);
        }
        let rows = if checks.is_empty() {
            Vec::new()
        } else {
            backend
                .returns_rows(&checks)
                .await
                .map_err(RunError::Engine)?
        };
        if let Some(i) = rows.iter().position(|&r| r) {
            return Err(RunError::Refused(AdError::InvalidWrt(
                program.checks()[i].message.clone(),
            )));
        }
        for step in &steps {
            let plan = bind(backend, &step.plan, &mut schemas).await?;
            // A table rewritten has the schema of this run's plan.
            schemas.remove(&step.name);
            backend
                .materialize(&step.name, &plan)
                .await
                .map_err(RunError::Engine)?;
        }
        Ok(())
    }
    .await;
    // On success the intermediates go; on failure every step's table, so an
    // earlier run's value never sits beside this run's gradient. Every table
    // is dropped even after a failed drop; the first failure is reported.
    let failed = result.is_err();
    let dropping = steps
        .iter()
        .filter(|s| failed || !program.is_result(&s.name))
        .map(|s| s.name.as_str());
    let mut first = result.err();
    for name in dropping {
        if let (Err(e), None) = (backend.drop_table(name).await, &first) {
            first = Some(RunError::Engine(e));
        }
    }
    first.map_or(Ok(()), Err)
}

/// `plan` bound to the engine's tables, asking it for each schema not yet
/// known.
async fn bind<B: Backend>(
    backend: &mut B,
    plan: &Unbound,
    schemas: &mut HashMap<String, NamedStruct>,
) -> Result<Bound, RunError<B::Error>> {
    for name in plan.reads() {
        if let std::collections::hash_map::Entry::Vacant(slot) = schemas.entry(name) {
            let schema = backend
                .table_schema(slot.key())
                .await
                .map_err(RunError::Engine)?;
            slot.insert(schema);
        }
    }
    plan.bind(&mut |name| schemas.get(name).cloned())
        .map_err(RunError::Refused)
}

#[cfg(test)]
mod tests {
    use std::future::ready;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    use super::*;
    use crate::emit::{plan, read_step};

    /// Run a future whose every await is already done, as each here is.
    fn now<F: Future>(f: F) -> F::Output {
        match pin!(f).poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(out) => out,
            Poll::Pending => panic!("a test backend's future is never pending"),
        }
    }

    /// A backend that logs what it is asked, and fails where `fail` says.
    struct Log {
        seen: Vec<String>,
        rows: Vec<bool>,
        fail: &'static str,
    }

    impl Log {
        fn outcome(&mut self, what: String) -> Result<(), String> {
            let failed = what == self.fail;
            self.seen.push(what);
            if failed {
                Err(format!("{} failed", self.fail))
            } else {
                Ok(())
            }
        }
    }

    impl Backend for Log {
        type Error = String;
        fn table_schema(
            &mut self,
            name: &str,
        ) -> impl Future<Output = Result<NamedStruct, String>> + Send {
            ready(self.outcome(format!("schema {name}")).map(|_| NamedStruct {
                names: vec!["x".into()],
                r#struct: Some(substrait::proto::r#type::Struct {
                    types: vec![crate::expr::fp64()],
                    ..Default::default()
                }),
            }))
        }
        fn returns_rows(
            &mut self,
            plans: &[Bound],
        ) -> impl Future<Output = Result<Vec<bool>, String>> + Send {
            ready(
                self.outcome(format!("check {}", plans.len()))
                    .map(|_| self.rows.clone()),
            )
        }
        fn materialize(
            &mut self,
            name: &str,
            _: &Bound,
        ) -> impl Future<Output = Result<(), String>> + Send {
            ready(self.outcome(format!("write {name}")))
        }
        fn drop_table(&mut self, name: &str) -> impl Future<Output = Result<(), String>> + Send {
            ready(self.outcome(format!("drop {name}")))
        }
        fn plan_sql(
            &mut self,
            sql: &str,
        ) -> impl Future<Output = Result<substrait::proto::Plan, String>> + Send {
            ready(Err(format!("no SQL here: {sql}")))
        }
    }

    /// A program of steps, each reading the one before, whose value is the
    /// last, and checks.
    pub(super) struct Fake {
        checks: Vec<Check>,
        steps: Vec<Step>,
        value: OutputTable,
    }

    impl Program for Fake {
        fn inputs(&self) -> &[InputTable] {
            &[]
        }
        fn checks(&self) -> &[Check] {
            &self.checks
        }
        fn step_count(&self) -> usize {
            self.steps.len()
        }
        fn step(&self, i: usize) -> &Step {
            &self.steps[i]
        }
        fn value(&self) -> &OutputTable {
            &self.value
        }
        fn gradients(&self) -> &[OutputTable] {
            &[]
        }
    }

    fn program(checks: usize, steps: &[&str]) -> Fake {
        let ext = crate::functions::Extensions::new(&Default::default());
        let read = |name: &str| {
            Unbound(plan(
                read_step(name, vec!["x".into()]),
                vec!["x".into()],
                &ext,
            ))
        };
        Fake {
            checks: (0..checks)
                .map(|i| Check {
                    plan: read("t"),
                    message: format!("check {i}"),
                })
                .collect(),
            steps: steps
                .iter()
                .enumerate()
                .map(|(i, s)| Step {
                    name: s.to_string(),
                    plan: read(if i == 0 { "t" } else { steps[i - 1] }),
                })
                .collect(),
            value: OutputTable {
                step: steps.last().unwrap().to_string(),
                of: crate::tables::Of::Output,
                columns: vec!["x".into()],
                tangents: Vec::new(),
            },
        }
    }

    fn trace(p: &Fake, rows: &[bool], fail: &'static str) -> (Vec<String>, Result<(), String>) {
        let mut b = Log {
            seen: Vec::new(),
            rows: rows.to_vec(),
            fail,
        };
        let result = now(run(&mut b, p)).map_err(|e| e.to_string());
        (b.seen, result)
    }

    #[test]
    fn checks_then_steps_then_the_intermediates_are_dropped() {
        let (seen, result) = trace(&program(2, &["s0", "s1", "v"]), &[false, false], "");
        assert_eq!(
            seen,
            [
                "schema t",
                "check 2",
                "write s0",
                "schema s0",
                "write s1",
                "schema s1",
                "write v",
                "drop s0",
                "drop s1"
            ]
        );
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn a_check_that_returns_a_row_refuses_and_leaves_none_of_the_programs_tables() {
        // An earlier run's value and gradients may still be there; a failed
        // run leaves none, so they never sit beside a different run's.
        let (seen, result) = trace(&program(2, &["s0", "v"]), &[false, true], "");
        assert_eq!(seen, ["schema t", "check 2", "drop s0", "drop v"]);
        assert_eq!(result, Err("invalid wrt column: check 1".into()));
    }

    #[test]
    fn a_failed_step_drops_every_table_and_reports_the_step() {
        let (seen, result) = trace(&program(0, &["s0", "s1", "v"]), &[], "write s1");
        assert_eq!(
            seen,
            [
                "schema t",
                "write s0",
                "schema s0",
                "write s1",
                "drop s0",
                "drop s1",
                "drop v"
            ]
        );
        assert_eq!(result, Err("write s1 failed".into()));
    }

    #[test]
    fn a_failed_drop_still_drops_the_rest_and_is_reported() {
        let (seen, result) = trace(&program(0, &["s0", "s1", "v"]), &[], "drop s0");
        assert_eq!(seen.last().unwrap(), "drop s1");
        assert_eq!(result, Err("drop s0 failed".into()));
    }
}
