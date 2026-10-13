// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad`, `vjp` and `jvp` in SQL: see [`Statements`].

use std::ops::ControlFlow;

use ddx_core::sqlparser::ast::{
    Expr, FunctionArg, FunctionArgExpr, ObjectNamePart, Query, Statement, TableFactor, Visit,
    Visitor,
};
use ddx_core::sqlparser::dialect::Dialect;
use ddx_core::sqlparser::parser::Parser;
use ddx_core::sqlparser::tokenizer::{Location, Token, TokenWithSpan, Tokenizer};
use substrait::proto::Plan;

use crate::compose::{sealed::Sealed, After, Differentiable, Subject};
use crate::error::{AdError, Result};
use crate::program::{BackwardProgram, Check, ForwardProgram, Options, Step};
use crate::relation::{table_matches, ColumnRef};
use crate::tables::{InputTable, OutputTable};

/// Which derivative a call takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// `grad(f, t.c, …)`: [`crate::grad`].
    Grad,
    /// `vjp(f, t.c, …, cotangent)`: [`crate::vjp`].
    Vjp,
    /// `jvp(f, t.c, …, tangent)`: [`crate::jvp`].
    Jvp,
}

impl Kind {
    /// The kind a call's name names, case-insensitively.
    fn of(name: &str) -> Option<Kind> {
        [Kind::Grad, Kind::Vjp, Kind::Jvp]
            .into_iter()
            .find(|k| k.name().eq_ignore_ascii_case(name))
    }

    /// The call's name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Kind::Grad => "grad",
            Kind::Vjp => "vjp",
            Kind::Jvp => "jvp",
        }
    }

    /// What its last argument is, if it takes one after the columns.
    fn seed(self) -> Option<&'static str> {
        match self {
            Kind::Grad => None,
            Kind::Vjp => Some("cotangent"),
            Kind::Jvp => Some("tangent"),
        }
    }

    fn usage(self) -> String {
        match self {
            Kind::Grad => "write grad(f, table.column, …): the name of a CTE computing one \
                           number (a loss, a likelihood, an energy), then the columns of one \
                           table to differentiate with respect to"
                .into(),
            Kind::Vjp => "write vjp(f, table.column, …, cotangent): the name of a CTE, the \
                          columns of one table, then the name of a CTE or table holding the \
                          cotangent of f's output (its keys, then its values)"
                .into(),
            Kind::Jvp => "write jvp(f, table.column, …, tangent): the name of a CTE, the \
                          columns of one table, then the name of a CTE or table holding their \
                          tangent (the table's dims, then a tangent under each column's name)"
                .into(),
        }
    }
}

/// One `grad`, `vjp` or `jvp` call.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Call {
    /// Which.
    pub kind: Kind,
    /// A query computing the CTE `f`: the statement's CTEs up to and
    /// including it, then `SELECT * FROM` it.
    pub query: String,
    /// The table, as written.
    pub table: String,
    /// The columns, as written.
    pub columns: Vec<String>,
    /// For `vjp` and `jvp`, a query computing the cotangent or tangent: a
    /// CTE's, as for `f`, or `SELECT * FROM` a table.
    pub seed: Option<String>,
    /// For `grad` and `vjp`, the rows of the gradient the statement reads at
    /// most, when its `WHERE` says: the conjuncts that compare one of the
    /// call's dims with a literal, written over the table's column names
    /// (`layer = 0 AND i < 10`). Such a comparison rejects NULL, so computing
    /// only those rows leaves the statement's result as it was, whatever the
    /// joins.
    pub filter: Option<String>,
    /// The byte range of the call in the statement: the name through the
    /// closing parenthesis, not an alias after it.
    span: (usize, usize),
}

/// The calls in a statement, in source order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Calls {
    /// The calls.
    pub calls: Vec<Call>,
    sql: String,
}

/// One program to build: what [`Statements::plan`] found a call needs.
#[derive(Debug, Clone)]
struct Spec {
    kind: Kind,
    /// The query to differentiate, as the statement wrote it.
    query: String,
    /// The columns, each once, compared case-insensitively: of every table
    /// a `grad` or `vjp` call on this query asks about, of one table for
    /// `jvp`.
    wrt: Vec<ColumnRef>,
    /// For `vjp` and `jvp`, a query computing the cotangent or tangent,
    /// which the program's input table holds.
    seed: Option<String>,
    restrict: Vec<(String, String)>,
    /// `query` and `seed`, with the calls they hold themselves.
    query_text: Text,
    seed_text: Option<Text>,
}

/// The next program to build, from [`Statements::next`]: its query to plan,
/// and the rows of the gradient to restrict it to.
#[derive(Debug, Clone, PartialEq)]
pub struct Job {
    index: usize,
    query: String,
    restrict: Vec<(String, String)>,
}

/// SQL that may hold calls (a statement, or a job's query or cotangent or
/// tangent), with the job each call needs.
#[derive(Debug, Clone)]
struct Text {
    sql: String,
    calls: Option<Calls>,
    /// One per call, in order.
    jobs: Vec<usize>,
}

/// The program a [`Job`] builds: a `grad` or `vjp` job's
/// [`BackwardProgram`], a `jvp` job's [`ForwardProgram`]. It runs as either
/// does ([`crate::Program`]), and [`Statements::rewrite`] reads it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum JobProgram {
    /// Of a `grad` or `vjp` job.
    Backward(BackwardProgram),
    /// Of a `jvp` job.
    Forward(ForwardProgram),
}

impl JobProgram {
    /// The tables it reads that no step writes: [`Statements::seeds`] says what to
    /// register as each.
    pub fn inputs(&self) -> &[InputTable] {
        match self {
            JobProgram::Backward(p) => &p.inputs,
            JobProgram::Forward(p) => &p.inputs,
        }
    }
}

impl Sealed for JobProgram {
    fn subject(&self) -> Subject<'_> {
        match self {
            JobProgram::Backward(p) => p.subject(),
            JobProgram::Forward(p) => p.subject(),
        }
    }
}
impl Differentiable for JobProgram {}

impl crate::run::sealed::Sealed for JobProgram {}
impl crate::Program for JobProgram {
    fn checks(&self) -> &[Check] {
        match self {
            JobProgram::Backward(p) => crate::Program::checks(p),
            JobProgram::Forward(p) => crate::Program::checks(p),
        }
    }
    fn step_count(&self) -> usize {
        match self {
            JobProgram::Backward(p) => p.step_count(),
            JobProgram::Forward(p) => p.step_count(),
        }
    }
    fn step(&self, i: usize) -> &Step {
        match self {
            JobProgram::Backward(p) => crate::Program::step(p, i),
            JobProgram::Forward(p) => crate::Program::step(p, i),
        }
    }
    fn is_result(&self, step: &str) -> bool {
        match self {
            JobProgram::Backward(p) => p.is_result(step),
            JobProgram::Forward(p) => p.is_result(step),
        }
    }
}

impl Job {
    /// The query to differentiate, for the engine to plan: as the statement
    /// wrote it, but with each call it holds itself replaced by a read of
    /// what that call's program, built and run before it, computed.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// For a table every call of which reads only some rows of its
    /// gradient, as a `WHERE` on the call's dims says: the table, as
    /// written, and a predicate over its columns that holds on every row
    /// some call reads. Only those rows need computing; an adapter passes
    /// each it can plan as [`Options::restrict`].
    pub fn restrict(&self) -> &[(String, String)] {
        &self.restrict
    }
}

impl Spec {
    /// Does `call` need this job?
    fn takes(&self, call: &Call) -> bool {
        self.kind == call.kind
            && self.query == call.query
            && self.seed == call.seed
            && (call.kind != Kind::Jvp || pushes_along(&self.wrt, call))
    }

    fn build<D: Differentiable + ?Sized>(&self, of: &D, options: &Options) -> Result<JobProgram> {
        Ok(match self.kind {
            Kind::Grad => JobProgram::Backward(crate::grad_with(of, &self.wrt, options)?),
            Kind::Vjp => JobProgram::Backward(crate::vjp_with(of, &self.wrt, options)?),
            Kind::Jvp => JobProgram::Forward(crate::jvp_with(of, &self.wrt, options)?),
        })
    }
}

impl Text {
    /// `sql`, with its calls, each matched to the one of `specs` it needs.
    fn of(sql: &str, dialect: &dyn Dialect, specs: &[Spec]) -> Result<Text> {
        let calls = Calls::find(sql, dialect)?;
        let jobs = calls
            .iter()
            .flat_map(|c| &c.calls)
            .map(|call| {
                specs.iter().position(|s| s.takes(call)).ok_or_else(|| {
                    AdError::Internal(format!(
                        "a {} call inside another's query that the statement lacks",
                        call.kind.name()
                    ))
                })
            })
            .collect::<Result<_>>()?;
        Ok(Text {
            sql: sql.to_string(),
            calls,
            jobs,
        })
    }

    /// The jobs whose programs it reads, each once.
    fn needs(&self) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        for &j in &self.jobs {
            if !out.contains(&j) {
                out.push(j);
            }
        }
        out
    }

    /// The text with each call replaced by a read of what its job's program
    /// computed. A `grad` or `vjp` call reads its table's dims, then the
    /// columns it named, from the gradient its job's program wrote. A
    /// column is one of the table's values when some `wrt` entry of the job
    /// names it; every other is a dim. A `jvp` call reads its program's
    /// value: `f`'s columns, then each tangent as `{column}_tangent`.
    fn rewrite(&self, specs: &[Spec], programs: &[JobProgram]) -> Result<String> {
        let Some(calls) = &self.calls else {
            return Ok(self.sql.clone());
        };
        let mut relations = Vec::with_capacity(calls.calls.len());
        for (call, &j) in calls.calls.iter().zip(&self.jobs) {
            let program = programs
                .get(j)
                .ok_or_else(|| AdError::Internal(format!("job {j} is read before it is built")))?;
            let Subject::Program(parts) = program.subject() else {
                return Err(AdError::Internal(format!("job {j}'s program is a query")));
            };
            relations.push(match call.kind {
                Kind::Jvp => value_and_tangents(parts.value),
                _ => gradient(call, &specs[j], parts.gradients)?,
            });
        }
        let mut relations = relations.into_iter();
        Ok(calls.rewrite(&mut |_| relations.next().expect("one per call")))
    }
}

/// `grad`, `vjp` and `jvp` in SQL: a query's derivative as a relation in
/// another query.
///
/// Each is a call in a `FROM` clause, on a CTE `f` of the statement and the
/// columns of one table, as JAX's are on a function and its arguments:
///
/// | call | is | a relation shaped like |
/// |---|---|---|
/// | `grad(f, t.c, …)` | the gradient of the number `f` computes (its **objective**: one row, one column) | `t`: its dims, then each column's gradient under its name |
/// | `vjp(f, t.c, …, cotangent)` | `f`'s output pulled back along `cotangent`, a relation shaped like `f`'s output (its keys, then its values) | `t`, as for `grad` |
/// | `jvp(f, t.c, …, tangent)` | `f`'s output pushed forward along `tangent`, a relation shaped like `t` (its dims, then a tangent under each column's name) | `f`'s output: its columns, then each one's tangent as `{column}_tangent` |
///
/// The cotangent or tangent is a CTE of the statement or a table; a row it
/// lacks counts as 0. Several tables are several calls, as several columns of
/// one table are one: a gradient is shaped like one table, and so is a
/// tangent. A `jvp` relation names a tangent `{column}_tangent_{n}`, for the
/// first free `n`, if `f` already has a column of that name.
///
/// The objective of `grad` is any query that returns one number: an ML
/// loss, a log-likelihood, a portfolio's risk, or a physical system's energy,
/// whose gradient is the force on each body with its sign flipped:
///
/// ```sql
/// -- Masses at heights mass(i, y), joined by springs spring(lo, hi, k, rest),
/// -- under their weights weight(i, w).
/// WITH springs AS (
///        SELECT SUM(0.5 * s.k * power(b.y - a.y - s.rest, 2)) AS e
///        FROM spring s JOIN mass a ON s.lo = a.i JOIN mass b ON s.hi = b.i),
///      gravity AS (SELECT SUM(w.w * m.y) AS e FROM mass m JOIN weight w ON m.i = w.i),
///      energy AS (SELECT springs.e + gravity.e AS e FROM springs CROSS JOIN gravity)
/// SELECT m.i, m.y - 0.01 * g.y AS y            -- one step towards equilibrium
/// FROM mass m JOIN grad(energy, mass.y) g ON m.i = g.i
/// ```
///
/// An SGD step is the same join: `params - lr * grad(loss)(params)`, JAX's
/// shape of an update. And a directional derivative is a `jvp`:
///
/// ```sql
/// WITH loss AS (SELECT SUM(val * val) AS l FROM w),
///      v AS (SELECT i, 1.0 AS val FROM w)
/// SELECT l, l_tangent FROM jvp(loss, w.val, v)
/// ```
///
/// No engine could run those calls: a table function receives values, and
/// `f` is a query. So, like v1's `grad` (design.md §3.3, Path A), each is
/// rewritten before the engine sees the statement. [`Statements::plan`] finds
/// the calls in several statements at once and the jobs they need, one per
/// program: a job differentiates one query, with respect to every column any
/// call of its kind asks about (one table's, for `jvp`), so statements that
/// update different tables from one objective pay for one backward pass.
///
/// Calls nest, as JAX's transformations do: `f`, or the cotangent or
/// tangent, can read another call, and `jvp` of a query over a `grad` is a
/// Hessian-vector product:
///
/// ```sql
/// WITH loss AS (SELECT SUM(val * val * val) AS l FROM w),
///      g AS (SELECT * FROM grad(loss, w.val)),
///      v AS (SELECT i, 1.0 AS val FROM w)
/// SELECT i, val_tangent AS hv FROM jvp(g, w.val, v)
/// ```
///
/// The adapter drives it, as it does a program with [`crate::Runner`]: it
/// asks for the [next](Statements::next) job, plans its query, has it
/// [built](Statements::build), registers its [seeds](Statements::seeds) and
/// runs the program, until no job is left. A job's query reads what the jobs
/// before it computed, and its program differentiates through theirs. Then
/// [`Statements::rewrite`] splices a relation over what each call needs in
/// its place, by source span, leaving the rest of each statement
/// byte-identical. Only planning and running needs the engine.
///
/// ```ignore
/// let mut planned = Statements::plan(&[sql], &dialect)?;
/// while let Some(job) = planned.next()? {
///     let plan = engine.plan(job.query());
///     let program = planned.build(&job, &plan, &Options::new())?;
///     for (name, query) in planned.seeds(&job)? { engine.register(name, query) }
///     engine.run(program);
/// }
/// let rewritten = planned.rewrite()?;
/// ```
///
/// This is the one part of `ddx-ad` about SQL text rather than Substrait, so
/// its API names `sqlparser` (a [`Dialect`]), through the `ddx-core` this
/// crate re-exports; the rest of the crate needs only `substrait`.
///
/// The scalar `grad(expr, column)` and `jvp(expr, column, dx)` of v1 are
/// expressions, in a select list; these are relations, in a `FROM` clause.
/// The rewriter tells them apart by where they appear.
#[derive(Debug, Clone)]
pub struct Statements {
    statements: Vec<Text>,
    specs: Vec<Spec>,
    /// The programs built so far, one per spec, in order.
    programs: Vec<JobProgram>,
}

/// Does a `jvp` job along `wrt` push forward along exactly `call`'s columns?
/// Each call gets its own tangents, so two calls share a job only when they
/// name the same columns: one along both would add their tangents.
fn pushes_along(wrt: &[ColumnRef], call: &Call) -> bool {
    let named = |w: &ColumnRef| {
        w.table.eq_ignore_ascii_case(&call.table)
            && call
                .columns
                .iter()
                .any(|c| c.eq_ignore_ascii_case(&w.column))
    };
    wrt.iter().all(named)
        && call
            .columns
            .iter()
            .all(|c| wrt.iter().any(|w| w.column.eq_ignore_ascii_case(c)))
}

impl Statements {
    /// Find every statement's calls, and the jobs they need: for `grad`
    /// and for `vjp`, one per query (and cotangent), with respect to every
    /// column asked about; for `jvp`, one per distinct call. A call inside
    /// another's query or cotangent or tangent (in a CTE it reads) comes
    /// first in the statement, so its job comes first.
    pub fn plan(statements: &[&str], dialect: &dyn Dialect) -> Result<Statements> {
        let mut specs: Vec<Spec> = Vec::new();
        let mut texts = Vec::with_capacity(statements.len());
        for sql in statements {
            let calls = Calls::find(sql, dialect)?;
            let mut jobs = Vec::new();
            for call in calls.iter().flat_map(|c| &c.calls) {
                let j = match specs.iter().position(|s| s.takes(call)) {
                    Some(j) => j,
                    None => {
                        let query_text = Text::of(&call.query, dialect, &specs)?;
                        let seed_text = match &call.seed {
                            Some(seed) => Some(Text::of(seed, dialect, &specs)?),
                            None => None,
                        };
                        specs.push(Spec {
                            kind: call.kind,
                            query: call.query.clone(),
                            wrt: Vec::new(),
                            seed: call.seed.clone(),
                            restrict: Vec::new(),
                            query_text,
                            seed_text,
                        });
                        specs.len() - 1
                    }
                };
                for column in &call.columns {
                    let w = ColumnRef::new(call.table.clone(), column.clone());
                    let same = |x: &ColumnRef| {
                        x.table.eq_ignore_ascii_case(&w.table)
                            && x.column.eq_ignore_ascii_case(&w.column)
                    };
                    if !specs[j].wrt.iter().any(same) {
                        specs[j].wrt.push(w);
                    }
                }
                jobs.push(j);
            }
            texts.push(Text {
                sql: sql.to_string(),
                calls,
                jobs,
            });
        }
        // A table's rows can be restricted only if every call reading its
        // gradient in the job says which rows it reads.
        for (j, spec) in specs.iter_mut().enumerate() {
            if spec.kind == Kind::Jvp {
                continue;
            }
            let mut by_table: Vec<(String, Vec<Option<String>>)> = Vec::new();
            for text in &texts {
                let Some(calls) = &text.calls else { continue };
                for (call, &k) in calls.calls.iter().zip(&text.jobs) {
                    if k != j {
                        continue;
                    }
                    match by_table
                        .iter_mut()
                        .find(|(t, _)| t.eq_ignore_ascii_case(&call.table))
                    {
                        Some((_, filters)) => filters.push(call.filter.clone()),
                        None => by_table.push((call.table.clone(), vec![call.filter.clone()])),
                    }
                }
            }
            for (table, filters) in by_table {
                let Some(filters) = filters.into_iter().collect::<Option<Vec<_>>>() else {
                    continue;
                };
                let mut distinct: Vec<String> = Vec::new();
                for f in filters {
                    if !distinct.contains(&f) {
                        distinct.push(f);
                    }
                }
                let predicate = if distinct.len() == 1 {
                    distinct.pop().expect("one")
                } else {
                    distinct
                        .iter()
                        .map(|f| format!("({f})"))
                        .collect::<Vec<_>>()
                        .join(" OR ")
                };
                spec.restrict.push((table, predicate));
            }
        }
        Ok(Statements {
            statements: texts,
            specs,
            programs: Vec::new(),
        })
    }

    /// The next job to build, once every job before it is built and has
    /// run, or `None` when none is left. Until it is built, this is the
    /// same job again.
    pub fn next(&self) -> Result<Option<Job>> {
        let index = self.programs.len();
        let Some(spec) = self.specs.get(index) else {
            return Ok(None);
        };
        Ok(Some(Job {
            index,
            query: spec.query_text.rewrite(&self.specs, &self.programs)?,
            restrict: spec.restrict.clone(),
        }))
    }

    /// `job`'s program, from `plan`, the engine's plan of [`Job::query`],
    /// with `options`. If the query reads what other calls computed, the
    /// program differentiates through their programs too.
    pub fn build(&mut self, job: &Job, plan: &Plan, options: &Options) -> Result<&JobProgram> {
        if job.index != self.programs.len() {
            return Err(AdError::Internal(format!(
                "job {} built out of turn, after {}",
                job.index,
                self.programs.len()
            )));
        }
        let spec = &self.specs[job.index];
        let needs = spec.query_text.needs();
        let program = if needs.is_empty() {
            spec.build(plan, options)?
        } else {
            let inner: Vec<&dyn Differentiable> = needs
                .iter()
                .map(|&j| &self.programs[j] as &dyn Differentiable)
                .collect();
            spec.build(&After::new(&inner, plan)?, options)?
        };
        self.programs.push(program);
        Ok(self.programs.last().expect("just pushed"))
    }

    /// What to register before `job`'s program, [built](Statements::build),
    /// runs: each of its input tables' names, and a query for its rows, the
    /// cotangent or tangent with the columns the program asks for, by name.
    /// An input it carries from a program it reads was registered for that
    /// one.
    pub fn seeds(&self, job: &Job) -> Result<Vec<(String, String)>> {
        let spec = &self.specs[job.index];
        let program = self.programs.get(job.index).ok_or_else(|| {
            AdError::Internal(format!("job {}'s seeds before it is built", job.index))
        })?;
        let carried: Vec<&str> = spec
            .query_text
            .needs()
            .into_iter()
            .flat_map(|j| self.programs[j].inputs())
            .map(|i| i.name.as_str())
            .collect();
        let mut out = Vec::new();
        for input in program.inputs() {
            if carried.contains(&input.name.as_str()) {
                continue;
            }
            let seed = match &spec.seed_text {
                Some(text) => text.rewrite(&self.specs, &self.programs)?,
                None => {
                    return Err(AdError::Internal(format!(
                        "a {} job has no table to register as `{}`",
                        spec.kind.name(),
                        input.name
                    )))
                }
            };
            let columns: Vec<String> = input.columns.iter().map(|c| quote(c)).collect();
            let query = format!("SELECT {} FROM ({seed}) AS __ddx_seed", columns.join(", "));
            out.push((input.name.clone(), query));
        }
        Ok(out)
    }

    /// The programs built so far, in order: what an adapter releases once
    /// the statements are planned.
    pub fn programs(&self) -> &[JobProgram] {
        &self.programs
    }

    /// Each statement with its calls replaced by reads of what their
    /// programs computed, once every job is built and has run. A statement
    /// with no call comes back as it was.
    pub fn rewrite(&self) -> Result<Vec<String>> {
        if self.programs.len() != self.specs.len() {
            return Err(AdError::Internal(format!(
                "{} programs built for {} jobs",
                self.programs.len(),
                self.specs.len()
            )));
        }
        self.statements
            .iter()
            .map(|t| t.rewrite(&self.specs, &self.programs))
            .collect()
    }
}

/// SQL for a `grad` or `vjp` call's relation: its table's dims, then the
/// columns it named, from the gradient of that table in `gradients`.
fn gradient(call: &Call, job: &Spec, gradients: &[OutputTable]) -> Result<String> {
    let found = gradients.iter().find_map(|g| {
        let table = g.of.table()?;
        table_matches(&call.table, table).then_some((g, table))
    });
    let Some((g, table)) = found else {
        return Err(AdError::Internal(format!(
            "no gradient was computed for `{}`",
            call.table
        )));
    };
    let is_value = |c: &String| {
        job.wrt
            .iter()
            .any(|w| table_matches(&w.table, table) && w.column.eq_ignore_ascii_case(c))
    };
    let picked: Vec<String> = g
        .columns
        .iter()
        .filter(|c| !is_value(c))
        .cloned()
        .chain(call.columns.iter().filter_map(|c| {
            g.columns
                .iter()
                .find(|v| is_value(v) && v.eq_ignore_ascii_case(c))
                .cloned()
        }))
        .map(|c| quote(&c))
        .collect();
    Ok(format!(
        "(SELECT {} FROM {})",
        picked.join(", "),
        quote(&g.step)
    ))
}

/// SQL for a `jvp` call's relation: `value`'s own columns, then each
/// tangent, named `{column}_tangent` (see [`Statements`]).
fn value_and_tangents(value: &OutputTable) -> String {
    let own = &value.columns[..value.columns.len() - value.tangents.len()];
    let mut taken: Vec<String> = own.iter().map(|c| c.to_ascii_lowercase()).collect();
    let mut picked: Vec<String> = own.iter().map(|c| quote(c)).collect();
    for t in &value.tangents {
        let base = format!("{}_tangent", t.column);
        let name = std::iter::once(base.clone())
            .chain((2..).map(|n| format!("{base}_{n}")))
            .find(|n| !taken.contains(&n.to_ascii_lowercase()))
            .expect("a free name");
        taken.push(name.to_ascii_lowercase());
        picked.push(format!("{} AS {}", quote(&t.tangent), quote(&name)));
    }
    format!("(SELECT {} FROM {})", picked.join(", "), quote(&value.step))
}

/// `name` as a quoted SQL identifier.
fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

impl Calls {
    /// Find the calls in `sql`, or `None` if it has none.
    ///
    /// A statement without the text `grad(`, `vjp(` or `jvp(` is not parsed
    /// at all, and one `sqlparser` cannot parse is passed over: it may be
    /// engine syntax `sqlparser` lacks, and if it does hold a call, the
    /// engine refuses it loudly as an unknown table function.
    pub fn find(sql: &str, dialect: &dyn Dialect) -> Result<Option<Calls>> {
        // Tokens, not text: a comment or a string can hold `(`, `)` or
        // `grad(`, and a comment can sit between `grad` and its `(`.
        let tokens = Tokenizer::new(dialect, sql).tokenize_with_location().ok();
        if !mentions_call(sql, tokens.as_deref()) {
            return Ok(None);
        }
        let Ok(statements) = Parser::parse_sql(dialect, sql) else {
            return Ok(None);
        };
        let [Statement::Query(query)] = statements.as_slice() else {
            return find_none(&statements);
        };

        let mut finder = Finder::default();
        let _ = query.visit(&mut finder);
        let mut filters = Filters::default();
        let _ = query.visit(&mut filters);
        if finder.found.is_empty() {
            return Ok(None);
        }

        let mut calls = Vec::new();
        for (kind, at, args) in finder.found {
            let (name, table, columns, seed) = parse_args(kind, &args)?;
            let query_of =
                |name: &str| -> Result<Option<String>> { cte_query(query, name, kind.name()) };
            let f = query_of(&name)?.ok_or_else(|| {
                AdError::InvalidPlan(format!(
                    "{}'s first argument must name a CTE in the statement's WITH clause, and \
                     `{name}` is not one",
                    kind.name()
                ))
            })?;
            let seed = match seed {
                Some(s) => {
                    Some(query_of(&s.value)?.unwrap_or_else(|| format!("SELECT * FROM {s}")))
                }
                None => None,
            };
            let filter = match kind {
                Kind::Jvp => None,
                _ => filters.of(at, &columns),
            };
            calls.push(Call {
                kind,
                query: f,
                table,
                columns,
                seed,
                filter,
                span: call_span(sql, tokens.as_deref(), at)?,
            });
        }
        calls.sort_by_key(|c| c.span.0);
        if calls.windows(2).any(|w| w[1].span.0 < w[0].span.1) {
            return Err(AdError::Internal(
                "two calls whose spans overlap in the statement".into(),
            ));
        }
        Ok(Some(Calls {
            calls,
            sql: sql.to_string(),
        }))
    }

    /// The statement with each call replaced by `relation(call)`: SQL for a
    /// relation holding what it computes, such as a subquery over the table
    /// an engine materialized. An alias written after a call stays. Each is
    /// asked for in source order.
    pub fn rewrite(&self, relation: &mut dyn FnMut(&Call) -> String) -> String {
        let mut out = String::with_capacity(self.sql.len());
        let mut at = 0;
        for call in &self.calls {
            out.push_str(&self.sql[at..call.span.0]);
            out.push_str(&relation(call));
            at = call.span.1;
        }
        out.push_str(&self.sql[at..]);
        out
    }
}

/// Does `sql` call `grad`, `vjp` or `jvp`: the word, then `(`, with any
/// whitespace or comments between, in any case? Without tokens (the
/// dialect's tokenizer failed), any mention of one lets the parser decide.
fn mentions_call(sql: &str, tokens: Option<&[TokenWithSpan]>) -> bool {
    let lower = sql.to_ascii_lowercase();
    if !["grad", "vjp", "jvp"].iter().any(|n| lower.contains(n)) {
        return false;
    }
    let Some(tokens) = tokens else {
        return true;
    };
    let significant: Vec<&Token> = tokens
        .iter()
        .map(|t| &t.token)
        .filter(|t| !matches!(t, Token::Whitespace(_)))
        .collect();
    significant.windows(2).any(|w| {
        matches!(w[0], Token::Word(word) if Kind::of(&word.value).is_some())
            && *w[1] == Token::LParen
    })
}

fn find_none(statements: &[Statement]) -> Result<Option<Calls>> {
    // Anything that is not a single query cannot hold a relation-valued
    // call ddx knows how to place; make sure there is none before saying so.
    let mut finder = Finder::default();
    for st in statements {
        let _ = st.visit(&mut finder);
    }
    match finder.found.first() {
        None => Ok(None),
        Some((kind, _, _)) => Err(AdError::NotImplemented(format!(
            "{}(f, …) in a statement that is not a single query",
            kind.name()
        ))),
    }
}

/// Collects `grad(…)`, `vjp(…)` and `jvp(…)` table-function calls:
/// unqualified, case-folded.
#[derive(Default)]
struct Finder {
    found: Vec<(Kind, Location, Vec<FunctionArg>)>,
}

impl Visitor for Finder {
    type Break = ();

    fn pre_visit_table_factor(&mut self, factor: &TableFactor) -> ControlFlow<()> {
        if let TableFactor::Table {
            name,
            args: Some(args),
            ..
        } = factor
        {
            if let [ObjectNamePart::Identifier(id)] = name.0.as_slice() {
                if let Some(kind) = Kind::of(&id.value) {
                    self.found.push((kind, id.span.start, args.args.clone()));
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// Each `grad` and `vjp` call's alias and the `WHERE` of the `SELECT` whose
/// `FROM` holds it, by the location of the call's name.
#[derive(Default)]
struct Filters {
    found: Vec<(Location, String, Option<Expr>)>,
}

impl Filters {
    /// The conjuncts of the call at `at`'s `WHERE` that compare one of its
    /// dims with a literal, over bare column names; `None` if there are
    /// none. `values` are the columns the call names, which in its result
    /// hold gradients, not the table's values, so they are never pushed.
    fn of(&self, at: Location, values: &[String]) -> Option<String> {
        let (_, alias, selection) = self.found.iter().find(|(l, _, _)| *l == at)?;
        let mut kept = Vec::new();
        for c in conjuncts(selection.as_ref()?) {
            if let Some(text) = pushable(c, alias, values) {
                kept.push(text);
            }
        }
        (!kept.is_empty()).then(|| kept.join(" AND "))
    }
}

impl Visitor for Filters {
    type Break = ();

    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<()> {
        let mut stack = vec![query.body.as_ref()];
        while let Some(body) = stack.pop() {
            match body {
                ddx_core::sqlparser::ast::SetExpr::Select(select) => {
                    for twj in &select.from {
                        let factors = std::iter::once(&twj.relation)
                            .chain(twj.joins.iter().map(|j| &j.relation));
                        for factor in factors {
                            if let TableFactor::Table {
                                name,
                                args: Some(_),
                                alias: Some(alias),
                                ..
                            } = factor
                            {
                                if let [ObjectNamePart::Identifier(id)] = name.0.as_slice() {
                                    let kind = Kind::of(&id.value);
                                    if matches!(kind, Some(Kind::Grad | Kind::Vjp)) {
                                        self.found.push((
                                            id.span.start,
                                            alias.name.value.clone(),
                                            select.selection.clone(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                ddx_core::sqlparser::ast::SetExpr::SetOperation { left, right, .. } => {
                    stack.push(left);
                    stack.push(right);
                }
                _ => {}
            }
        }
        ControlFlow::Continue(())
    }
}

/// The `AND`-ed parts of `e`.
fn conjuncts(e: &Expr) -> Vec<&Expr> {
    use ddx_core::sqlparser::ast::BinaryOperator;
    match e {
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            let mut out = conjuncts(left);
            out.extend(conjuncts(right));
            out
        }
        Expr::Nested(inner) => conjuncts(inner),
        other => vec![other],
    }
}

/// `c` over bare column names, if it compares a dim of the call aliased
/// `alias` with a literal: `=`, `<>`, `<`, `<=`, `>`, `>=`, `IN (…)` or
/// `BETWEEN`, none negated. Each rejects NULL.
fn pushable(c: &Expr, alias: &str, values: &[String]) -> Option<String> {
    use ddx_core::sqlparser::ast::BinaryOperator;
    let dim = |e: &Expr| -> Option<Expr> {
        let Expr::CompoundIdentifier(parts) = e else {
            return None;
        };
        let [table, column] = parts.as_slice() else {
            return None;
        };
        let is_value = values.iter().any(|v| v.eq_ignore_ascii_case(&column.value));
        (table.value.eq_ignore_ascii_case(alias) && !is_value)
            .then(|| Expr::Identifier(column.clone()))
    };
    let literal = |e: &Expr| match e {
        Expr::Value(_) => true,
        Expr::UnaryOp { expr, .. } => matches!(expr.as_ref(), Expr::Value(_)),
        _ => false,
    };
    match c {
        Expr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Lt
                    | BinaryOperator::LtEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq
            ) =>
        {
            if let (Some(d), true) = (dim(left), literal(right)) {
                return Some(
                    Expr::BinaryOp {
                        left: Box::new(d),
                        op: op.clone(),
                        right: right.clone(),
                    }
                    .to_string(),
                );
            }
            if let (true, Some(d)) = (literal(left), dim(right)) {
                return Some(
                    Expr::BinaryOp {
                        left: left.clone(),
                        op: op.clone(),
                        right: Box::new(d),
                    }
                    .to_string(),
                );
            }
            None
        }
        Expr::InList {
            expr,
            list,
            negated: false,
        } if list.iter().all(literal) => dim(expr).map(|d| {
            Expr::InList {
                expr: Box::new(d),
                list: list.clone(),
                negated: false,
            }
            .to_string()
        }),
        Expr::Between {
            expr,
            negated: false,
            low,
            high,
        } if literal(low) && literal(high) => dim(expr).map(|d| {
            Expr::Between {
                expr: Box::new(d),
                negated: false,
                low: low.clone(),
                high: high.clone(),
            }
            .to_string()
        }),
        _ => None,
    }
}

/// A call's arguments: `f`'s name, the table, its columns, and the
/// cotangent's or tangent's name for `vjp` and `jvp`.
fn parse_args(
    kind: Kind,
    args: &[FunctionArg],
) -> Result<(
    String,
    String,
    Vec<String>,
    Option<ddx_core::sqlparser::ast::Ident>,
)> {
    let usage = kind.usage();
    let exprs: Vec<&Expr> = args
        .iter()
        .map(|a| match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Ok(e),
            _ => Err(AdError::InvalidPlan(usage.clone())),
        })
        .collect::<Result<_>>()?;
    let [Expr::Identifier(f), rest @ ..] = exprs.as_slice() else {
        return Err(AdError::InvalidPlan(usage));
    };
    let (columns, seed) = match (kind.seed(), rest) {
        (None, columns) => (columns, None),
        (Some(_), [columns @ .., Expr::Identifier(seed)]) => (columns, Some(seed.clone())),
        (Some(what), _) => {
            return Err(AdError::InvalidPlan(format!(
                "{}'s last argument is the name of its {what}; {usage}",
                kind.name()
            )))
        }
    };
    if columns.is_empty() {
        return Err(AdError::InvalidPlan(usage));
    }
    let mut table: Option<String> = None;
    let mut names = Vec::with_capacity(columns.len());
    for e in columns {
        let Expr::CompoundIdentifier(parts) = e else {
            return Err(AdError::InvalidPlan(format!(
                "`{e}` is not a table column; {usage}"
            )));
        };
        let Some((column, prefix)) = parts.split_last().filter(|(_, p)| !p.is_empty()) else {
            return Err(AdError::InvalidPlan(format!(
                "`{e}` is not a table column; {usage}"
            )));
        };
        let t = prefix
            .iter()
            .map(|p| p.value.clone())
            .collect::<Vec<_>>()
            .join(".");
        match &table {
            Some(first) if !first.eq_ignore_ascii_case(&t) => {
                return Err(AdError::NotImplemented(format!(
                    "{}({}, …) takes columns of one table, so its {} is shaped like that \
                     table; call it once per table",
                    kind.name(),
                    f.value,
                    match kind {
                        Kind::Jvp => "tangent",
                        _ => "gradient",
                    }
                )))
            }
            Some(_) => {}
            None => table = Some(t),
        }
        names.push(column.value.clone());
    }
    Ok((
        f.value.clone(),
        table.expect("one column at least"),
        names,
        seed,
    ))
}

/// A query computing the CTE `name` of `query`, for a call of `call`: the
/// statement's CTEs up to and including it, then `SELECT * FROM` it; `None`
/// if no CTE has that name.
fn cte_query(query: &Query, name: &str, call: &str) -> Result<Option<String>> {
    let Some(with) = query.with.as_ref() else {
        return Ok(None);
    };
    let Some(idx) = with
        .cte_tables
        .iter()
        .position(|c| c.alias.name.value.eq_ignore_ascii_case(name))
    else {
        return Ok(None);
    };
    if with.recursive {
        return Err(AdError::NotImplemented(format!(
            "{call} of a CTE defined in a WITH RECURSIVE clause"
        )));
    }
    let ctes: Vec<String> = with.cte_tables[..=idx]
        .iter()
        .map(|c| c.to_string())
        .collect();
    let quoted = &with.cte_tables[idx].alias.name;
    Ok(Some(format!(
        "WITH {} SELECT * FROM {quoted}",
        ctes.join(", ")
    )))
}

/// The byte range of the call starting at `start`: through the matching
/// closing parenthesis, counted in tokens, so a parenthesis in a comment or
/// a string does not count.
fn call_span(
    sql: &str,
    tokens: Option<&[TokenWithSpan]>,
    start: Location,
) -> Result<(usize, usize)> {
    let begin = byte_offset(sql, start)
        .ok_or_else(|| AdError::Internal(format!("no byte offset for {start:?}")))?;
    let tokens = tokens
        .ok_or_else(|| AdError::Internal("a statement that parses but does not tokenize".into()))?;
    let from = tokens
        .iter()
        .position(|t| t.span.start == start)
        .ok_or_else(|| AdError::Internal(format!("no token at {start:?}")))?;
    let mut depth = 0;
    for t in &tokens[from..] {
        match t.token {
            Token::LParen => depth += 1,
            Token::RParen => {
                depth -= 1;
                if depth == 0 {
                    let end = byte_offset(sql, t.span.end).ok_or_else(|| {
                        AdError::Internal(format!("no byte offset for {:?}", t.span.end))
                    })?;
                    return Ok((begin, end));
                }
            }
            _ => {}
        }
    }
    Err(AdError::Internal("an unclosed call".into()))
}

/// The byte offset of a 1-based line/column (in characters) position.
fn byte_offset(sql: &str, at: Location) -> Option<usize> {
    let (mut line, mut column) = (1u64, 1u64);
    for (i, ch) in sql.char_indices() {
        if line == at.line && column == at.column {
            return Some(i);
        }
        if ch == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    (line == at.line && column == at.column).then_some(sql.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddx_core::sqlparser::dialect::GenericDialect;

    const SQL: &str = "WITH d AS (SELECT * FROM x), \
                       loss AS (SELECT SUM(w.val * d.v) AS l FROM w JOIN d ON w.i = d.i) \
                       SELECT w.i, w.val - 0.1 * g.val AS val \
                       FROM w JOIN GRAD(loss, w.val) g ON w.i = g.i";

    fn filter_of(sql: &str) -> Option<String> {
        Calls::find(sql, &GenericDialect {}).unwrap().unwrap().calls[0]
            .filter
            .clone()
    }

    #[test]
    fn one_table_named_in_two_cases_is_one_table() {
        let found = Calls::find(
            "WITH loss AS (SELECT SUM(a * b) AS l FROM w) SELECT * FROM grad(loss, w.a, W.b)",
            &GenericDialect {},
        )
        .unwrap()
        .unwrap();
        assert_eq!(found.calls[0].columns, vec!["a", "b"]);
    }

    #[test]
    fn a_comparison_of_a_gradient_dim_with_a_literal_is_pushed_and_nothing_else() {
        let with = |w: &str| {
            format!(
                "WITH loss AS (SELECT SUM(val) AS l FROM w) \
                 SELECT g.i FROM w JOIN grad(loss, w.val) g ON w.i = g.i WHERE {w}"
            )
        };
        assert_eq!(filter_of(&with("g.layer = 0")), Some("layer = 0".into()));
        assert_eq!(
            filter_of(&with(
                "g.layer IN (0, 2) AND g.i BETWEEN 1 AND 4 AND 3 > g.k"
            )),
            Some("layer IN (0, 2) AND i BETWEEN 1 AND 4 AND 3 > k".into())
        );
        // g.val is the gradient, not the table's value; w is another table;
        // an OR, a negation, or a comparison with a column is not pushed.
        for w in [
            "g.val > 0",
            "w.layer = 0",
            "g.layer = 0 OR g.i = 1",
            "g.layer NOT IN (0)",
            "g.layer = w.layer",
        ] {
            assert_eq!(filter_of(&with(w)), None, "{w}");
        }
        assert_eq!(
            filter_of(&with("g.layer = 0 AND g.val > 0")),
            Some("layer = 0".into())
        );
    }

    #[test]
    fn a_job_restricts_a_table_only_when_every_call_on_it_does() {
        let stmt = |w: &str| {
            format!(
                "WITH loss AS (SELECT SUM(val) AS l FROM w) \
                 SELECT g.i FROM grad(loss, w.val) g{w}"
            )
        };
        let plan = |ws: &[&str]| {
            let stmts: Vec<String> = ws.iter().map(|w| stmt(w)).collect();
            let refs: Vec<&str> = stmts.iter().map(String::as_str).collect();
            Statements::plan(&refs, &GenericDialect {}).unwrap().specs[0]
                .restrict
                .clone()
        };
        assert_eq!(
            plan(&[" WHERE g.layer = 0"]),
            vec![("w".to_string(), "layer = 0".to_string())]
        );
        assert_eq!(
            plan(&[" WHERE g.layer = 0", " WHERE g.layer = 1"]),
            vec![("w".to_string(), "(layer = 0) OR (layer = 1)".to_string())]
        );
        assert_eq!(plan(&[" WHERE g.layer = 0", ""]), vec![]);
    }

    #[test]
    fn a_call_is_found_with_its_objective_query() {
        let found = Calls::find(SQL, &GenericDialect {}).unwrap().unwrap();
        let call = &found.calls[0];
        assert_eq!((call.kind, call.seed.as_deref()), (Kind::Grad, None));
        assert!(
            call.query
                .starts_with("WITH d AS (SELECT * FROM x), loss AS ("),
            "{}",
            call.query
        );
        assert!(call.query.ends_with("SELECT * FROM loss"), "{}", call.query);
        assert_eq!(found.calls[0].table, "w");
        assert_eq!(found.calls[0].columns, vec!["val"]);
    }

    #[test]
    fn the_call_is_spliced_out_and_its_alias_kept() {
        let found = Calls::find(SQL, &GenericDialect {}).unwrap().unwrap();
        let out = found.rewrite(&mut |_| "(SELECT i, val FROM g_w)".into());
        assert!(
            out.ends_with("FROM w JOIN (SELECT i, val FROM g_w) g ON w.i = g.i"),
            "{out}"
        );
        assert!(
            out.starts_with("WITH d AS (SELECT * FROM x), loss AS"),
            "{out}"
        );
    }

    #[test]
    fn two_calls_on_one_objective_share_it() {
        let sql = "WITH loss AS (SELECT SUM(w.val * b.val) AS l FROM w JOIN b ON w.o = b.o) \
                   SELECT * FROM grad(loss, w.val) gw, grad(loss, b.val) gb";
        let found = Calls::find(sql, &GenericDialect {}).unwrap().unwrap();
        let planned = Statements::plan(&[sql], &GenericDialect {}).unwrap();
        assert_eq!(planned.specs.len(), 1);
        assert_eq!(
            planned.specs[0].wrt,
            vec![ColumnRef::new("w", "val"), ColumnRef::new("b", "val")]
        );
        let out = found.rewrite(&mut |c| format!("g_{}", c.table));
        assert!(out.ends_with("SELECT * FROM g_w gw, g_b gb"), "{out}");
    }

    #[test]
    fn case_variants_of_a_column_are_one_wrt_column() {
        let sql = "WITH loss AS (SELECT SUM(val) AS l FROM w) \
                   SELECT * FROM grad(loss, w.val) a, grad(loss, W.VAL) b";
        let planned = Statements::plan(&[sql], &GenericDialect {}).unwrap();
        assert_eq!(planned.specs[0].wrt, vec![ColumnRef::new("w", "val")]);
    }

    #[test]
    fn a_statement_without_grad_is_not_parsed() {
        assert_eq!(
            Calls::find("not even SQL", &GenericDialect {}).unwrap(),
            None
        );
        assert_eq!(
            Calls::find("SELECT gradient FROM t", &GenericDialect {}).unwrap(),
            None
        );
        assert_eq!(
            Calls::find("SELECT grad ( FROM", &GenericDialect {}).unwrap(),
            None
        );
        let scalar = "SELECT grad(x * x, x) FROM t";
        assert_eq!(Calls::find(scalar, &GenericDialect {}).unwrap(), None);
    }

    #[test]
    fn misuse_is_refused() {
        let bad = [
            "SELECT * FROM grad(nope, w.val)",
            "WITH loss AS (SELECT 1 AS l) SELECT * FROM grad(loss)",
            "WITH loss AS (SELECT 1 AS l) SELECT * FROM grad(loss, val)",
            "WITH loss AS (SELECT 1 AS l) SELECT * FROM grad(loss, w.val, b.val)",
        ];
        for sql in bad {
            assert!(Calls::find(sql, &GenericDialect {}).is_err(), "{sql}");
        }
    }

    const CALLS: &str = "WITH h AS (SELECT x.n, SUM(x.v * w.val) + MAX(b.val) AS y \
                         FROM x JOIN w ON x.i = w.i JOIN b ON w.o = b.o GROUP BY x.n), \
                         dw AS (SELECT i, o, 0.5 AS val FROM w) \
                         SELECT * FROM JVP(h, w.val, dw) j, jvp(h, b.val, db) k, \
                         vjp(h, w.val, c) g WHERE g.o = 1";

    #[test]
    fn vjp_and_jvp_take_one_table_s_columns_then_their_seed() {
        let found = Calls::find(CALLS, &GenericDialect {}).unwrap().unwrap();
        let kinds: Vec<Kind> = found.calls.iter().map(|c| c.kind).collect();
        assert_eq!(kinds, vec![Kind::Jvp, Kind::Jvp, Kind::Vjp]);
        let [j, k, g] = &found.calls[..] else {
            unreachable!()
        };
        assert!(j.query.ends_with("SELECT * FROM h"), "{}", j.query);
        // dw is a CTE of the statement, db and c are tables.
        let dw = j.seed.as_deref().unwrap();
        assert!(
            dw.starts_with("WITH h AS (") && dw.ends_with("SELECT * FROM dw"),
            "{dw}"
        );
        assert_eq!(k.seed.as_deref(), Some("SELECT * FROM db"));
        assert_eq!(g.seed.as_deref(), Some("SELECT * FROM c"));
        // A vjp's rows are a gradient's, so its WHERE is pushed; a jvp's
        // are f's.
        assert_eq!(
            (j.filter.as_deref(), g.filter.as_deref()),
            (None, Some("o = 1"))
        );
        let out = found.rewrite(&mut |c| format!("({})", c.kind.name()));
        assert!(
            out.ends_with("SELECT * FROM (jvp) j, (jvp) k, (vjp) g WHERE g.o = 1"),
            "{out}"
        );
    }

    #[test]
    fn a_job_takes_its_kind_s_calls_and_jvps_of_two_tables_are_two_jobs() {
        let planned = Statements::plan(&[CALLS, CALLS], &GenericDialect {}).unwrap();
        let jobs = &planned.specs;
        // jvp along w, jvp along b, vjp: twice over, the same three jobs.
        assert_eq!(jobs.len(), 3);
        assert_eq!(jobs[0].wrt, vec![ColumnRef::new("w", "val")]);
        assert_eq!(jobs[1].wrt, vec![ColumnRef::new("b", "val")]);
        assert_eq!(jobs[2].kind, Kind::Vjp);
        assert_eq!(
            jobs[2].restrict,
            vec![("w".to_string(), "o = 1".to_string())]
        );
        assert!(jobs[0].restrict.is_empty());
        // Two vjps of one query along one cotangent share a backward pass.
        let two = "WITH h AS (SELECT w.o, w.val * b.val AS y FROM w JOIN b ON w.o = b.o) \
                   SELECT * FROM vjp(h, w.val, c) g, vjp(h, b.val, c) gb, vjp(h, b.val, d) gd";
        let planned = Statements::plan(&[two], &GenericDialect {}).unwrap();
        assert_eq!(planned.specs.len(), 2);
        assert_eq!(planned.specs[0].wrt.len(), 2);
    }

    /// A program that takes `inputs`, as if `planned` had built it for
    /// its next job.
    fn built(planned: &mut Statements, inputs: Vec<InputTable>) -> Job {
        let job = planned.next().unwrap().unwrap();
        planned
            .programs
            .push(JobProgram::Forward(crate::ForwardProgram {
                inputs,
                checks: Vec::new(),
                steps: Vec::new(),
                value: OutputTable {
                    step: "__ddx_1_value".into(),
                    of: crate::tables::Of::Output,
                    columns: Vec::new(),
                    tangents: Vec::new(),
                },
                gradients: Vec::new(),
            }));
        job
    }

    #[test]
    fn a_job_s_seed_has_the_columns_its_program_asks_for() {
        let mut planned = Statements::plan(&[CALLS], &GenericDialect {}).unwrap();
        let input = InputTable {
            name: "__ddx_1_tangent_0_w".into(),
            of: crate::tables::Of::Table(vec!["w".into()]),
            columns: vec!["i".into(), "o".into(), "val".into()],
            keys: 2,
        };
        built(&mut planned, vec![]);
        let job = built(&mut planned, vec![input.clone()]);
        assert_eq!(
            planned.seeds(&job).unwrap(),
            vec![(
                input.name.clone(),
                "SELECT \"i\", \"o\", \"val\" FROM (SELECT * FROM db) AS __ddx_seed".into()
            )]
        );
        // A grad job writes its cotangent, 1, into its plan: it has none.
        let grad = "WITH l AS (SELECT SUM(val) AS l FROM w) SELECT * FROM grad(l, w.val)";
        let mut planned = Statements::plan(&[grad], &GenericDialect {}).unwrap();
        let job = built(&mut planned, vec![input]);
        assert!(planned.seeds(&job).is_err());
    }

    #[test]
    fn a_call_inside_another_s_query_is_a_job_before_it_and_its_query_reads_it() {
        let sql = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), \
                        g AS (SELECT * FROM grad(loss, w.val)), \
                        v AS (SELECT i, 1.0 AS val FROM w) \
                   SELECT * FROM jvp(g, w.val, v) j";
        let mut planned = Statements::plan(&[sql], &GenericDialect {}).unwrap();
        assert_eq!(planned.specs.len(), 2);
        assert_eq!(planned.specs[0].kind, Kind::Grad);
        assert_eq!(planned.specs[1].query_text.needs(), vec![0]);
        let first = planned.next().unwrap().unwrap();
        assert!(!first.query().contains("grad("), "{}", first.query());
        // The outer job's query cannot be written until the inner one is
        // built: it reads its program's table.
        planned
            .programs
            .push(JobProgram::Backward(crate::BackwardProgram {
                inputs: Vec::new(),
                checks: Vec::new(),
                forward_steps: Vec::new(),
                backward_steps: Vec::new(),
                value: OutputTable {
                    step: "__ddx_1_value".into(),
                    of: crate::tables::Of::Output,
                    columns: vec!["l".into()],
                    tangents: Vec::new(),
                },
                gradients: vec![OutputTable {
                    step: "__ddx_1_grad_0_w".into(),
                    of: crate::tables::Of::Table(vec!["w".into()]),
                    columns: vec!["i".into(), "val".into()],
                    tangents: Vec::new(),
                }],
            }));
        let second = planned.next().unwrap().unwrap();
        assert!(
            second
                .query()
                .contains("(SELECT \"i\", \"val\" FROM \"__ddx_1_grad_0_w\")"),
            "{}",
            second.query()
        );
        assert!(!second.query().contains("grad("), "{}", second.query());
    }

    #[test]
    fn a_malformed_vjp_or_jvp_call_is_refused_with_its_usage() {
        let f = "WITH f AS (SELECT SUM(a * b) AS l FROM w JOIN v ON w.i = v.i) SELECT * FROM ";
        for call in [
            "jvp(f, w.a)",      // no tangent
            "vjp(f, w.a)",      // no cotangent
            "jvp(f, t)",        // a tangent with no columns
            "jvp(f, w.a, 1.0)", // not a name
            "vjp(g, w.a, t)",   // not a CTE
            "grad(f, w.a, t)",  // grad takes no seed
        ] {
            let e = Calls::find(&format!("{f}{call}"), &GenericDialect {}).unwrap_err();
            assert!(matches!(e, AdError::InvalidPlan(_)), "{call}: {e}");
        }
        for call in ["jvp(f, w.a, v.b, t)", "vjp(f, w.a, v.b, t)"] {
            let e = Calls::find(&format!("{f}{call}"), &GenericDialect {}).unwrap_err();
            assert!(
                matches!(&e, AdError::NotImplemented(m) if m.contains("once per table")),
                "{call}: {e}"
            );
        }
    }

    #[test]
    fn a_tangent_column_is_named_after_its_column_and_never_twice() {
        let value = OutputTable {
            step: "__ddx_1_jvp".into(),
            of: crate::tables::Of::Output,
            columns: vec![
                "l".into(),
                "l_tangent".into(),
                "__ddx_tangent_0".into(),
                "__ddx_tangent_1".into(),
            ],
            tangents: vec![
                crate::tables::Tangent {
                    column: "l".into(),
                    tangent: "__ddx_tangent_0".into(),
                },
                crate::tables::Tangent {
                    column: "l_tangent".into(),
                    tangent: "__ddx_tangent_1".into(),
                },
            ],
        };
        assert_eq!(
            value_and_tangents(&value),
            "(SELECT \"l\", \"l_tangent\", \"__ddx_tangent_0\" AS \"l_tangent_2\", \
             \"__ddx_tangent_1\" AS \"l_tangent_tangent\" FROM \"__ddx_1_jvp\")"
        );
    }
}
