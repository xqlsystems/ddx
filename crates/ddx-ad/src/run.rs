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
//! then every step in order. Each intermediate table is dropped once the
//! last step that reads it has run, so a run holds only the tables later
//! steps still read; one that succeeds leaves the value and the gradients.
//! A run that fails, at a
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

    /// A plan the engine compiled to run again ([`Backend::compile`]): its
    /// physical plan, say. `()` for an engine that compiles none.
    type Compiled: Send + Sync;

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
        plans: &[&Bound],
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

    /// The SQL that defines `name`, as a query writes it, if it names a view
    /// whose SQL the engine keeps: a `CREATE VIEW` statement, or its query.
    /// [`crate::rewrite`] writes such a view into the objective as CTEs, so
    /// a model that is a view has the CTEs it reads more than once planned
    /// once each. `None`, the default, leaves every view to the engine.
    /// The SQL must still mean what the engine runs for the view: an engine
    /// that keeps the plan a view was made with, as DataFusion does, runs
    /// the old definition of a view it reads after that view is redefined,
    /// and its backend returns `None` for it ([`crate::view_probe`] helps
    /// tell).
    fn view_sql(
        &mut self,
        name: &str,
    ) -> impl Future<Output = Result<Option<String>, Self::Error>> + Send {
        let _ = name;
        async { Ok(None) }
    }

    /// Text that is the same at two calls only if the engine would plan
    /// any query as it did: what [`Backend::plan_key`] depends on but the
    /// query. That is every table's name and schema, every view's name and
    /// what the engine runs for it, the settings that change what a query
    /// means, and where names are searched.
    /// [`crate::Prepared::run`] compares it with the one [`crate::prepare`]
    /// read, and runs nothing once it differs, for the statements to be
    /// prepared again. `None`, the default, means the engine cannot say,
    /// and nothing prepared is run again.
    fn session_key(&mut self) -> impl Future<Output = Result<Option<String>, Self::Error>> + Send {
        async { Ok(None) }
    }

    /// `plan`, compiled to run again, each time on the rows its tables hold
    /// then: an engine's physical plan, say. [`crate::prepare`] compiles
    /// each plan it runs, and [`crate::Prepared::run`] runs what was
    /// compiled ([`Backend::materialize_compiled`],
    /// [`Backend::returns_rows_compiled`]) rather than having the engine
    /// plan `plan` again, as `jax.jit` compiles what it traced once.
    ///
    /// Optional: the default compiles nothing, returning `None`, and every
    /// run gives the engine `plan` itself ([`Backend::materialize`],
    /// [`Backend::returns_rows`]).
    fn compile(
        &mut self,
        _plan: &Bound,
    ) -> impl Future<Output = Result<Option<Self::Compiled>, Self::Error>> + Send {
        async { Ok(None) }
    }

    /// [`Backend::materialize`] of `plan`, by running `compiled`, which
    /// [`Backend::compile`] made of it. The default runs `plan`.
    fn materialize_compiled(
        &mut self,
        name: &str,
        plan: &Bound,
        _compiled: &Self::Compiled,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.materialize(name, plan)
    }

    /// [`Backend::returns_rows`] of each plan, by running what
    /// [`Backend::compile`] made of it. The default runs the plans.
    fn returns_rows_compiled(
        &mut self,
        plans: &[(&Bound, &Self::Compiled)],
    ) -> impl Future<Output = Result<Vec<bool>, Self::Error>> + Send {
        let plans: Vec<&Bound> = plans.iter().map(|(plan, _)| *plan).collect();
        async move { self.returns_rows(&plans).await }
    }
}

/// Run `program` on `backend`, as the module docs describe: its checks, then
/// its steps, each plan bound to the tables the engine holds, and each
/// intermediate table dropped after its last reader; if anything failed,
/// every table the program writes is dropped.
pub async fn run<B: Backend, P: Program + ?Sized>(
    backend: &mut B,
    program: &P,
) -> Result<(), RunError<B::Error>> {
    run_bound(backend, program, Plans::Bind(None)).await
}

/// A plan as a run ran it, to run again: bound to the engine's tables, and
/// what the engine compiled of it, if anything ([`Backend::compile`]).
pub(crate) struct Traced<C> {
    pub plan: Bound,
    pub compiled: Option<C>,
}

/// Where [`run_bound`] takes each plan from.
pub(crate) enum Plans<'a, C> {
    /// Bound as it runs, asking the engine for each schema; if given,
    /// compiled by the engine and traced here, the checks' then the
    /// steps'.
    Bind(Option<&'a mut Vec<Traced<C>>>),
    /// As an earlier run of the program, on the same tables, traced them.
    Bound(&'a [Traced<C>]),
}

/// [`run`], with each plan from `plans`.
pub(crate) async fn run_bound<B: Backend, P: Program + ?Sized>(
    backend: &mut B,
    program: &P,
    plans: Plans<'_, B::Compiled>,
) -> Result<(), RunError<B::Error>> {
    let steps: Vec<&Step> = (0..program.step_count()).map(|i| program.step(i)).collect();
    let mut live = Live::new(program, &steps);
    let result = match plans {
        Plans::Bind(traced) => bind_and_run(backend, program, &steps, &mut live, traced).await,
        Plans::Bound(traced) => replay(backend, program, &steps, &mut live, traced).await,
    };
    // On success every intermediate is gone already; on failure every
    // step's table goes, so an earlier run's value never sits beside this
    // run's gradient. Every table is dropped even after a failed drop; the
    // first failure is reported.
    let mut first = result.err();
    let dropping: Vec<usize> = (0..steps.len())
        .filter(|&i| !live.dropped[i] && (first.is_some() || !program.is_result(&steps[i].name)))
        .collect();
    for i in dropping {
        if let (Err(e), None) = (backend.drop_table(&steps[i].name).await, &first) {
            first = Some(RunError::Engine(e));
        }
    }
    first.map_or(Ok(()), Err)
}

/// Which intermediate tables a run drops after each step: those no later
/// step reads. A table no step reads is dropped as soon as it is written.
struct Live {
    /// By step, the steps whose tables are dropped once it has run.
    dead_after: Vec<Vec<usize>>,
    /// By step, whether its table has been dropped.
    dropped: Vec<bool>,
}

impl Live {
    fn new<P: Program + ?Sized>(program: &P, steps: &[&Step]) -> Live {
        let index: HashMap<&str, usize> = steps
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name.as_str(), i))
            .collect();
        // Each step's last reader, or itself: by every table a plan names,
        // a read already typed included.
        let mut last: Vec<usize> = (0..steps.len()).collect();
        for (j, step) in steps.iter().enumerate() {
            for name in crate::emit::tables_read(&step.plan.0) {
                if let Some(&i) = index.get(name.as_str()) {
                    last[i] = last[i].max(j);
                }
            }
        }
        let mut dead_after = vec![Vec::new(); steps.len()];
        for (i, &j) in last.iter().enumerate() {
            if !program.is_result(&steps[i].name) {
                dead_after[j].push(i);
            }
        }
        Live {
            dead_after,
            dropped: vec![false; steps.len()],
        }
    }

    /// Drop the tables no step after step `j` reads.
    async fn ran<B: Backend>(
        &mut self,
        backend: &mut B,
        steps: &[&Step],
        j: usize,
    ) -> Result<(), RunError<B::Error>> {
        for &i in &self.dead_after[j] {
            backend
                .drop_table(&steps[i].name)
                .await
                .map_err(RunError::Engine)?;
            self.dropped[i] = true;
        }
        Ok(())
    }
}

/// `program`'s checks, then its steps, each plan bound as it runs, and, if
/// `traced` is given, compiled by the engine and traced there.
async fn bind_and_run<B: Backend, P: Program + ?Sized>(
    backend: &mut B,
    program: &P,
    steps: &[&Step],
    live: &mut Live,
    mut traced: Option<&mut Vec<Traced<B::Compiled>>>,
) -> Result<(), RunError<B::Error>> {
    let compiling = traced.is_some();
    let mut schemas: HashMap<String, NamedStruct> = HashMap::new();
    let mut checks = Vec::new();
    for check in program.checks() {
        let plan = bind(backend, &check.plan, &mut schemas).await?;
        checks.push(trace(backend, plan, compiling).await?);
    }
    refuse_on_rows(backend, program, &checks).await?;
    if let Some(traced) = traced.as_deref_mut() {
        traced.append(&mut checks);
    }
    for (j, step) in steps.iter().enumerate() {
        let plan = bind(backend, &step.plan, &mut schemas).await?;
        let ran = trace(backend, plan, compiling).await?;
        // A table rewritten has the schema of this run's plan.
        schemas.remove(&step.name);
        materialize(backend, &step.name, &ran).await?;
        if let Some(traced) = traced.as_deref_mut() {
            traced.push(ran);
        }
        live.ran(backend, steps, j).await?;
    }
    Ok(())
}

/// `program`'s checks, then its steps, each plan as `traced` holds it: read
/// where it was traced, not copied.
async fn replay<B: Backend, P: Program + ?Sized>(
    backend: &mut B,
    program: &P,
    steps: &[&Step],
    live: &mut Live,
    traced: &[Traced<B::Compiled>],
) -> Result<(), RunError<B::Error>> {
    let n = program.checks().len();
    if traced.len() != n + steps.len() {
        return Err(RunError::Refused(AdError::Internal(format!(
            "a program of {} plans replayed from {}",
            n + steps.len(),
            traced.len()
        ))));
    }
    let (checks, plans) = traced.split_at(n);
    refuse_on_rows(backend, program, checks).await?;
    for (j, (step, ran)) in steps.iter().zip(plans).enumerate() {
        materialize(backend, &step.name, ran).await?;
        live.ran(backend, steps, j).await?;
    }
    Ok(())
}

/// `plan`, traced to run again: compiled by `backend` if `compiling`.
pub(crate) async fn trace<B: Backend>(
    backend: &mut B,
    plan: Bound,
    compiling: bool,
) -> Result<Traced<B::Compiled>, RunError<B::Error>> {
    let compiled = if compiling {
        backend.compile(&plan).await.map_err(RunError::Engine)?
    } else {
        None
    };
    Ok(Traced { plan, compiled })
}

/// Run `ran`'s plan, or what was compiled of it, storing its rows as
/// `name`.
pub(crate) async fn materialize<B: Backend>(
    backend: &mut B,
    name: &str,
    ran: &Traced<B::Compiled>,
) -> Result<(), RunError<B::Error>> {
    match &ran.compiled {
        Some(compiled) => {
            backend
                .materialize_compiled(name, &ran.plan, compiled)
                .await
        }
        None => backend.materialize(name, &ran.plan).await,
    }
    .map_err(RunError::Engine)
}

/// Refuse `program` if any of its `checks` (each compiled, if all were)
/// returns a row.
async fn refuse_on_rows<B: Backend, P: Program + ?Sized>(
    backend: &mut B,
    program: &P,
    checks: &[Traced<B::Compiled>],
) -> Result<(), RunError<B::Error>> {
    if checks.is_empty() {
        return Ok(());
    }
    let compiled: Option<Vec<(&Bound, &B::Compiled)>> = checks
        .iter()
        .map(|c| Some((&c.plan, c.compiled.as_ref()?)))
        .collect();
    let rows = match compiled {
        Some(compiled) => backend.returns_rows_compiled(&compiled).await,
        None => {
            let plans: Vec<&Bound> = checks.iter().map(|c| &c.plan).collect();
            backend.returns_rows(&plans).await
        }
    }
    .map_err(RunError::Engine)?;
    if let Some(i) = rows.iter().position(|&r| r) {
        return Err(RunError::Refused(AdError::InvalidWrt(
            program.checks()[i].message.clone(),
        )));
    }
    Ok(())
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
        type Compiled = ();
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
            plans: &[&Bound],
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
        let reads: Vec<&str> = (0..steps.len())
            .map(|i| if i == 0 { "t" } else { steps[i - 1] })
            .collect();
        program_reading(checks, steps, &reads)
    }

    /// A program whose step `i` is named `steps[i]` and reads `reads[i]`.
    fn program_reading(checks: usize, steps: &[&str], reads: &[&str]) -> Fake {
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
                    plan: read(reads[i]),
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
    fn checks_then_steps_each_intermediate_dropped_after_its_last_reader() {
        let (seen, result) = trace(&program(2, &["s0", "s1", "v"]), &[false, false], "");
        assert_eq!(
            seen,
            [
                "schema t",
                "check 2",
                "write s0",
                "schema s0",
                "write s1",
                "drop s0",
                "schema s1",
                "write v",
                "drop s1"
            ]
        );
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn a_table_read_later_is_kept_until_then_and_one_never_read_goes_at_once() {
        // s1 reads s0, nothing reads s2, and v reads s0 again.
        let p = program_reading(0, &["s0", "s1", "s2", "v"], &["t", "s0", "t", "s0"]);
        let (seen, result) = trace(&p, &[], "");
        assert_eq!(
            seen,
            [
                "schema t",
                "write s0",
                "schema s0",
                "write s1",
                "drop s1",
                "write s2",
                "drop s2",
                "write v",
                "drop s0"
            ]
        );
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn a_read_already_typed_keeps_its_table_until_it_runs() {
        // v reads s0 with its schema stated, so binding asks for none; s0
        // must still stay until v has run.
        let mut p = program_reading(0, &["s0", "s1", "v"], &["t", "t", "s0"]);
        let typed = p.steps[2]
            .plan
            .bind(&mut |_| {
                Some(NamedStruct {
                    names: vec!["x".into()],
                    r#struct: Some(substrait::proto::r#type::Struct {
                        types: vec![crate::expr::fp64()],
                        ..Default::default()
                    }),
                })
            })
            .unwrap();
        p.steps[2].plan = Unbound(typed.into_plan());
        let (seen, result) = trace(&p, &[], "");
        assert_eq!(
            seen,
            ["schema t", "write s0", "write s1", "drop s1", "write v", "drop s0"]
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
    fn a_failed_step_drops_every_table_left_and_reports_the_step() {
        let (seen, result) = trace(&program(0, &["s0", "s1", "s2", "v"]), &[], "write s2");
        assert_eq!(
            seen,
            [
                "schema t",
                "write s0",
                "schema s0",
                "write s1",
                "drop s0",
                "schema s1",
                "write s2",
                "drop s1",
                "drop s2",
                "drop v"
            ]
        );
        assert_eq!(result, Err("write s2 failed".into()));
    }

    #[test]
    fn a_failed_drop_fails_the_run_drops_the_rest_and_is_reported() {
        let (seen, result) = trace(&program(0, &["s0", "s1", "v"]), &[], "drop s0");
        assert_eq!(
            seen,
            [
                "schema t",
                "write s0",
                "schema s0",
                "write s1",
                "drop s0",
                "drop s0",
                "drop s1",
                "drop v"
            ]
        );
        assert_eq!(result, Err("drop s0 failed".into()));
    }
}
