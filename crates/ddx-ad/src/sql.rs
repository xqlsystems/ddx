// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad` in SQL: a query's gradient as a relation in another query.
//!
//! `grad(f, table.column, …)` in a `FROM` clause is the gradient of the
//! number the CTE `f` computes (its **objective**: one row, one column), with
//! respect to the named columns of one table. It is a relation shaped like
//! that table: the table's dims and, under the columns' own names, their
//! gradients. The objective is any query that returns one number: an ML
//! loss, a log-likelihood, a portfolio's risk, or a physical system's energy,
//! whose gradient is the force on each body with its sign flipped:
//!
//! ```sql
//! -- Masses at heights mass(i, y), joined by springs spring(lo, hi, k, rest),
//! -- under their weights weight(i, w).
//! WITH springs AS (
//!        SELECT SUM(0.5 * s.k * power(b.y - a.y - s.rest, 2)) AS e
//!        FROM spring s JOIN mass a ON s.lo = a.i JOIN mass b ON s.hi = b.i),
//!      gravity AS (SELECT SUM(w.w * m.y) AS e FROM mass m JOIN weight w ON m.i = w.i),
//!      energy AS (SELECT springs.e + gravity.e AS e FROM springs CROSS JOIN gravity)
//! SELECT m.i, m.y - 0.01 * g.y AS y            -- one step towards equilibrium
//! FROM mass m JOIN grad(energy, mass.y) g ON m.i = g.i
//! ```
//!
//! An SGD step is the same join: `params - lr * grad(loss)(params)`, JAX's
//! shape of an update. The objective must be one number because `grad` is
//! the gradient of a scalar function; for a query with any output, the
//! vector-Jacobian product is [`crate::vjp`].
//!
//! No engine could run that call: a table function receives values, and
//! `f` is a query. So, like v1's `grad` (design.md §3.3, Path A), it is
//! rewritten before the engine sees the statement. [`Calls::find`] finds
//! the calls and the objectives they need; the engine adapter runs each
//! objective's [`crate::grad`] program; [`Calls::rewrite`] splices a relation
//! holding each gradient in place of each call, by source span, leaving the
//! rest of the statement byte-identical.
//!
//! An adapter runs several statements at once with [`Statements`]: it plans
//! one job per distinct objective, differentiated with respect to every
//! column any statement asks about, so statements that update different
//! tables from one objective pay for one backward pass. The adapter runs each
//! job's program and hands the programs back to [`Statements::rewrite`]. Only
//! those two calls need the engine.
//!
//! This is the one part of `ddx-ad` about SQL text rather than Substrait, so
//! its API names `sqlparser` (a [`Dialect`]), through the `ddx-core` this
//! crate re-exports; the rest of the crate needs only `substrait`.
//!
//! The scalar `grad(expr, column)` of v1 is an expression, in a select list;
//! this one is a relation, in a `FROM` clause. The rewriter tells them apart by
//! where they appear.
//!
//! # `jvp` in SQL
//!
//! `jvp(f, table.column, …, tangent, …)` in a `FROM` clause is `f`'s output
//! with, beside it, its tangent along the given tangents, as `jax.jvp`
//! returns both ([`crate::jvp`]). `f` is any CTE, not only one number. Each
//! `wrt` table's columns are followed by the relation holding its tangent,
//! a CTE of the statement or a table, shaped like the table: its dims, then
//! a tangent under each column's name. A row the tangent lacks has tangent 0.
//!
//! ```sql
//! WITH h AS (SELECT x.n, tanh(SUM(x.v * w.val) + b.val) AS y
//!            FROM x JOIN w ON x.i = w.i JOIN b ON w.o = b.o GROUP BY x.n, w.o, b.val),
//!      dw AS (SELECT i, o, 0.01 AS val FROM w),
//!      db AS (SELECT o, 1.0 AS val FROM b)
//! SELECT n, y, y_tangent FROM jvp(h, w.val, dw, b.val, db)
//! ```
//!
//! The relation has `f`'s columns, then the tangent of each that has one,
//! named `{column}_tangent` (`{column}_tangent_{n}` for the first free `n`
//! if `f` already has that name). [`Statements::jvp_jobs`] are the programs
//! to run, one per distinct call, and [`Statements::rewrite`] reads their
//! values.

use std::collections::BTreeMap;
use std::ops::ControlFlow;

use ddx_core::sqlparser::ast::{
    Expr, FunctionArg, FunctionArgExpr, ObjectNamePart, Query, Statement, TableFactor, Visit,
    Visitor,
};
use ddx_core::sqlparser::dialect::Dialect;
use ddx_core::sqlparser::parser::Parser;
use ddx_core::sqlparser::tokenizer::{Location, Token, TokenWithSpan, Tokenizer};

use crate::error::{AdError, Result};
use crate::program::{BackwardProgram, ForwardProgram};
use crate::relation::{table_matches, ColumnRef};
use crate::tables::OutputTable;

/// The CTE some `grad` call differentiates: a query computing one number
/// (a loss, a likelihood, an energy, …).
#[derive(Debug, Clone, PartialEq)]
pub struct Objective {
    /// The CTE's name.
    pub name: String,
    /// A query computing just the objective: the statement's CTEs up to and
    /// including this one, then `SELECT * FROM` it.
    pub query: String,
    /// Every column any call takes this objective's gradient with respect
    /// to, each once, compared case-insensitively.
    pub wrt: Vec<ColumnRef>,
}

/// One `grad(f, table.column, …)` call.
#[derive(Debug, Clone, PartialEq)]
pub struct GradCall {
    /// Index into [`Calls::objectives`].
    pub objective: usize,
    /// The table, as written.
    pub table: String,
    /// The columns, as written.
    pub columns: Vec<String>,
    /// The rows of the gradient the statement reads at most, when its
    /// `WHERE` says: the conjuncts that compare one of the call's dims with
    /// a literal, written over the table's column names (`layer = 0 AND i <
    /// 10`). Such a comparison rejects NULL, so computing only those rows
    /// leaves the statement's result as it was, whatever the joins.
    pub filter: Option<String>,
    /// The byte range of `grad(…)` in the statement: the name through the
    /// closing parenthesis, not an alias after it.
    span: (usize, usize),
}

/// One `jvp(f, table.column, …, tangent, …)` call (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct JvpCall {
    /// What to run for it.
    pub job: JvpJob,
    /// The byte range of `jvp(…)` in the statement.
    span: (usize, usize),
}

/// One `jvp` to run for [`Statements`]: its program's query, `wrt` columns
/// and tangents.
#[derive(Debug, Clone, PartialEq)]
pub struct JvpJob {
    /// A query computing `f`: the statement's CTEs up to and including it,
    /// then `SELECT * FROM` it.
    pub query: String,
    /// The `wrt` columns, every table's, as written.
    pub wrt: Vec<ColumnRef>,
    /// Each `wrt` table's tangent, in the order written.
    pub tangents: Vec<JvpTangent>,
}

impl JvpJob {
    /// The tangent of the table whose name parts are `table` (an
    /// [`InputTable`](crate::InputTable)'s [`Of::Table`](crate::Of::Table)),
    /// matched as a [`ColumnRef`] names a table.
    pub fn tangent_of(&self, table: &[String]) -> Option<&JvpTangent> {
        self.tangents
            .iter()
            .find(|t| table_matches(&t.table, table))
    }
}

/// A `wrt` table's tangent, in a `jvp` call.
#[derive(Debug, Clone, PartialEq)]
pub struct JvpTangent {
    /// The table, as written.
    pub table: String,
    /// A query computing its tangent: a CTE of the statement (its CTEs up to
    /// and including it, then `SELECT * FROM` it), or `SELECT * FROM` a
    /// table.
    pub query: String,
}

/// The `grad` and `jvp` calls in a statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Calls {
    /// The objectives the `grad` calls differentiate, each once.
    pub objectives: Vec<Objective>,
    /// The `grad` calls, in source order.
    pub calls: Vec<GradCall>,
    /// The `jvp` calls, in source order.
    pub jvps: Vec<JvpCall>,
    sql: String,
}

/// One objective to differentiate for [`Statements`]: a query, and every
/// column any statement takes its gradient with respect to.
#[derive(Debug, Clone, PartialEq)]
pub struct Job {
    /// The objective's query.
    pub query: String,
    /// The columns, each once, compared case-insensitively.
    pub wrt: Vec<ColumnRef>,
    /// For a table every call of which reads only some rows of its gradient
    /// ([`GradCall::filter`]): the table, as written, and a predicate over
    /// its columns that holds on every row some call reads. Only those rows
    /// need computing ([`crate::Options::restrict`]).
    pub restrict: Vec<(String, String)>,
}

/// Several statements' `grad` calls, planned together (see the module docs).
#[derive(Debug, Clone)]
pub struct Statements {
    statements: Vec<String>,
    found: Vec<Option<Calls>>,
    jobs: Vec<Job>,
    /// (statement, objective within it) → job.
    job_of: BTreeMap<(usize, usize), usize>,
    jvp_jobs: Vec<JvpJob>,
    /// (statement, `jvp` call within it) → its job.
    jvp_of: BTreeMap<(usize, usize), usize>,
}

impl Statements {
    /// Find every statement's calls, and the jobs they need: one per
    /// distinct objective.
    pub fn plan(statements: &[&str], dialect: &dyn Dialect) -> Result<Statements> {
        let mut found = Vec::with_capacity(statements.len());
        for sql in statements {
            found.push(Calls::find(sql, dialect)?);
        }
        let mut jobs: Vec<Job> = Vec::new();
        let mut job_of = BTreeMap::new();
        for (s, calls) in found.iter().enumerate() {
            let Some(calls) = calls else { continue };
            for (l, objective) in calls.objectives.iter().enumerate() {
                let j = match jobs.iter().position(|j| j.query == objective.query) {
                    Some(j) => j,
                    None => {
                        jobs.push(Job {
                            query: objective.query.clone(),
                            wrt: Vec::new(),
                            restrict: Vec::new(),
                        });
                        jobs.len() - 1
                    }
                };
                for w in &objective.wrt {
                    let same = |x: &ColumnRef| {
                        x.table.eq_ignore_ascii_case(&w.table)
                            && x.column.eq_ignore_ascii_case(&w.column)
                    };
                    if !jobs[j].wrt.iter().any(same) {
                        jobs[j].wrt.push(w.clone());
                    }
                }
                job_of.insert((s, l), j);
            }
        }
        // A table's rows can be restricted only if every call reading its
        // gradient in the job says which rows it reads.
        for (j, job) in jobs.iter_mut().enumerate() {
            let mut by_table: Vec<(String, Vec<Option<String>>)> = Vec::new();
            for (s, calls) in found.iter().enumerate() {
                let Some(calls) = calls else { continue };
                for call in &calls.calls {
                    if job_of.get(&(s, call.objective)) != Some(&j) {
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
                job.restrict.push((table, predicate));
            }
        }
        // One program per distinct jvp call.
        let mut jvp_jobs: Vec<JvpJob> = Vec::new();
        let mut jvp_of = BTreeMap::new();
        for (s, calls) in found.iter().enumerate() {
            let Some(calls) = calls else { continue };
            for (c, call) in calls.jvps.iter().enumerate() {
                let j = match jvp_jobs.iter().position(|j| *j == call.job) {
                    Some(j) => j,
                    None => {
                        jvp_jobs.push(call.job.clone());
                        jvp_jobs.len() - 1
                    }
                };
                jvp_of.insert((s, c), j);
            }
        }
        Ok(Statements {
            statements: statements.iter().map(|s| s.to_string()).collect(),
            found,
            jobs,
            job_of,
            jvp_jobs,
            jvp_of,
        })
    }

    /// The objectives the `grad` calls differentiate, in the order
    /// [`Statements::rewrite`] expects their programs.
    pub fn jobs(&self) -> &[Job] {
        &self.jobs
    }

    /// The `jvp`s to run, in the order [`Statements::rewrite`] expects their
    /// programs: [`crate::jvp`] of the query, with each tangent's query
    /// registered as the program's input table for its table.
    pub fn jvp_jobs(&self) -> &[JvpJob] {
        &self.jvp_jobs
    }

    /// Each statement with its calls replaced by reads of what the programs
    /// computed: `programs`, one per [job](Statements::jobs) in order, and
    /// `jvps`, one per [`jvp` job](Statements::jvp_jobs), each already run
    /// so its tables exist. A statement with no call comes back as it was.
    ///
    /// A `grad` call reads its table's dims, then the columns it named, from
    /// the table its job's program wrote. A column is one of the table's
    /// values when some `wrt` entry of the job names it; every other is a
    /// dim. A `jvp` call reads its program's value: `f`'s columns, then each
    /// tangent as `{column}_tangent`.
    pub fn rewrite(
        &self,
        programs: &[&BackwardProgram],
        jvps: &[&ForwardProgram],
    ) -> Result<Vec<String>> {
        if programs.len() != self.jobs.len() || jvps.len() != self.jvp_jobs.len() {
            return Err(AdError::Internal(format!(
                "{} grad and {} jvp programs for {} and {} jobs",
                programs.len(),
                jvps.len(),
                self.jobs.len(),
                self.jvp_jobs.len()
            )));
        }
        let mut out = Vec::with_capacity(self.statements.len());
        for (s, sql) in self.statements.iter().enumerate() {
            let Some(calls) = &self.found[s] else {
                out.push(sql.clone());
                continue;
            };
            let mut failure = None;
            let mut jvp_index = 0;
            let rewritten = calls.rewrite(
                &mut |call| {
                    let j = self.job_of[&(s, call.objective)];
                    let found = programs[j].gradients.iter().find_map(|g| {
                        let table = g.of.table()?;
                        table_matches(&call.table, table).then_some((g, table))
                    });
                    let Some((g, table)) = found else {
                        failure = Some(AdError::Internal(format!(
                            "no gradient was computed for `{}`",
                            call.table
                        )));
                        return String::new();
                    };
                    let is_value = |c: &String| {
                        self.jobs[j].wrt.iter().any(|w| {
                            table_matches(&w.table, table) && w.column.eq_ignore_ascii_case(c)
                        })
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
                    format!("(SELECT {} FROM {})", picked.join(", "), quote(&g.step))
                },
                &mut |_| {
                    let program = jvps[self.jvp_of[&(s, jvp_index)]];
                    jvp_index += 1;
                    value_and_tangents(&program.value)
                },
            );
            if let Some(e) = failure {
                return Err(e);
            }
            out.push(rewritten);
        }
        Ok(out)
    }
}

/// SQL for a `jvp` call's relation: `value`'s own columns, then each
/// tangent, named `{column}_tangent` (see the module docs).
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
    /// Find the `grad` and `jvp` calls in `sql`, or `None` if it has none.
    ///
    /// A statement without the text `grad(` or `jvp(` is not parsed at all,
    /// and one `sqlparser` cannot parse is passed over: it may be engine
    /// syntax `sqlparser` lacks, and if it does hold a `grad(f, …)`, the
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
        if finder.found.is_empty() && finder.jvps.is_empty() {
            return Ok(None);
        }

        let mut objectives: Vec<Objective> = Vec::new();
        let mut by_name: BTreeMap<String, usize> = BTreeMap::new();
        let mut calls = Vec::new();
        for (factor, args) in finder.found {
            let (name, wrt) = parse_args(&args)?;
            let objective = match by_name.get(&name.to_ascii_lowercase()) {
                Some(&i) => i,
                None => {
                    objectives.push(Objective {
                        query: objective_query(query, &name)?,
                        name: name.clone(),
                        wrt: Vec::new(),
                    });
                    by_name.insert(name.to_ascii_lowercase(), objectives.len() - 1);
                    objectives.len() - 1
                }
            };
            let table = wrt[0].table.clone();
            if wrt.iter().any(|w| !w.table.eq_ignore_ascii_case(&table)) {
                return Err(AdError::NotImplemented(format!(
                    "grad({name}, …) takes columns of one table, so it can return a \
                     relation shaped like that table; call it once per table"
                )));
            }
            for w in &wrt {
                // `w.VAL` and `w.val` are one column: identifiers are
                // compared case-insensitively, as `ddx_ad::grad` does.
                let same = |x: &ColumnRef| {
                    x.table.eq_ignore_ascii_case(&w.table)
                        && x.column.eq_ignore_ascii_case(&w.column)
                };
                if !objectives[objective].wrt.iter().any(same) {
                    objectives[objective].wrt.push(w.clone());
                }
            }
            let columns: Vec<String> = wrt.into_iter().map(|w| w.column).collect();
            let filter = filters.of(factor, &columns);
            calls.push(GradCall {
                objective,
                table,
                filter,
                columns,
                span: call_span(sql, tokens.as_deref(), factor)?,
            });
        }
        calls.sort_by_key(|c| c.span.0);
        let mut jvps = Vec::new();
        for (at, args) in finder.jvps {
            jvps.push(JvpCall {
                job: jvp_job(query, &args)?,
                span: call_span(sql, tokens.as_deref(), at)?,
            });
        }
        jvps.sort_by_key(|c| c.span.0);
        let mut spans: Vec<(usize, usize)> = calls
            .iter()
            .map(|c| c.span)
            .chain(jvps.iter().map(|c| c.span))
            .collect();
        spans.sort();
        if spans.windows(2).any(|w| w[1].0 < w[0].1) {
            return Err(AdError::Internal(
                "two grad(…) or jvp(…) calls whose spans overlap in the statement".into(),
            ));
        }
        Ok(Some(Calls {
            objectives,
            calls,
            jvps,
            sql: sql.to_string(),
        }))
    }

    /// The statement with each `grad` call replaced by `grad(call)` and each
    /// `jvp` call by `jvp(call)`: SQL for a relation holding what the call
    /// computes, such as a subquery over the table an engine materialized.
    /// An alias written after a call stays. Each is asked for in source
    /// order.
    pub fn rewrite(
        &self,
        grad: &mut dyn FnMut(&GradCall) -> String,
        jvp: &mut dyn FnMut(&JvpCall) -> String,
    ) -> String {
        enum Of<'a> {
            Grad(&'a GradCall),
            Jvp(&'a JvpCall),
        }
        let mut all: Vec<((usize, usize), Of)> = self
            .calls
            .iter()
            .map(|c| (c.span, Of::Grad(c)))
            .chain(self.jvps.iter().map(|c| (c.span, Of::Jvp(c))))
            .collect();
        all.sort_by_key(|(span, _)| *span);
        let mut out = String::with_capacity(self.sql.len());
        let mut at = 0;
        for (span, call) in all {
            out.push_str(&self.sql[at..span.0]);
            out.push_str(&match call {
                Of::Grad(c) => grad(c),
                Of::Jvp(c) => jvp(c),
            });
            at = span.1;
        }
        out.push_str(&self.sql[at..]);
        out
    }
}

/// Does `sql` call `grad` or `jvp`: the word, then `(`, with any whitespace
/// or comments between, in any case? Without tokens (the dialect's tokenizer
/// failed), any mention of either lets the parser decide.
fn mentions_call(sql: &str, tokens: Option<&[TokenWithSpan]>) -> bool {
    let lower = sql.to_ascii_lowercase();
    if !lower.contains("grad") && !lower.contains("jvp") {
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
        matches!(w[0], Token::Word(word)
            if word.value.eq_ignore_ascii_case("grad") || word.value.eq_ignore_ascii_case("jvp"))
            && *w[1] == Token::LParen
    })
}

fn find_none(statements: &[Statement]) -> Result<Option<Calls>> {
    // Anything that is not a single query cannot hold a relation-valued
    // `grad` ddx knows how to place; make sure there is none before saying so.
    let mut finder = Finder::default();
    for st in statements {
        let _ = st.visit(&mut finder);
    }
    if finder.found.is_empty() && finder.jvps.is_empty() {
        Ok(None)
    } else {
        Err(AdError::NotImplemented(
            "grad(f, …) or jvp(f, …) in a statement that is not a single query".into(),
        ))
    }
}

/// Collects `grad(…)` and `jvp(…)` table-function calls: unqualified,
/// case-folded.
#[derive(Default)]
struct Finder {
    found: Vec<(Location, Vec<FunctionArg>)>,
    jvps: Vec<(Location, Vec<FunctionArg>)>,
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
                if id.value.eq_ignore_ascii_case("grad") {
                    self.found.push((id.span.start, args.args.clone()));
                } else if id.value.eq_ignore_ascii_case("jvp") {
                    self.jvps.push((id.span.start, args.args.clone()));
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// Each `grad` call's alias and the `WHERE` of the `SELECT` whose `FROM`
/// holds it, by the location of the call's name.
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
                                    if id.value.eq_ignore_ascii_case("grad") {
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

/// `grad(f, t.c, …)`'s objective name and columns.
fn parse_args(args: &[FunctionArg]) -> Result<(String, Vec<ColumnRef>)> {
    let usage = "write grad(f, table.column, …): the name of a CTE computing one number \
                 (a loss, a likelihood, an energy), then the columns to differentiate with \
                 respect to";
    let exprs: Vec<&Expr> = args
        .iter()
        .map(|a| match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Ok(e),
            _ => Err(AdError::InvalidPlan(usage.into())),
        })
        .collect::<Result<_>>()?;
    let [objective, wrt @ ..] = exprs.as_slice() else {
        return Err(AdError::InvalidPlan(usage.into()));
    };
    let Expr::Identifier(objective) = objective else {
        return Err(AdError::InvalidPlan(usage.into()));
    };
    if wrt.is_empty() {
        return Err(AdError::InvalidPlan(usage.into()));
    }
    let wrt = wrt
        .iter()
        .map(|e| match e {
            Expr::CompoundIdentifier(parts) if parts.len() >= 2 => {
                let (column, table) = parts.split_last().expect("two or more parts");
                Ok(ColumnRef::new(
                    table
                        .iter()
                        .map(|p| p.value.clone())
                        .collect::<Vec<_>>()
                        .join("."),
                    column.value.clone(),
                ))
            }
            _ => Err(AdError::InvalidPlan(format!(
                "`{e}` is not a table column; {usage}"
            ))),
        })
        .collect::<Result<_>>()?;
    Ok((objective.value.clone(), wrt))
}

/// A query computing just the objective CTE `name`: the statement's CTEs up
/// to and including it, then `SELECT * FROM` it.
fn objective_query(query: &Query, name: &str) -> Result<String> {
    cte_query(query, name, "grad")?.ok_or_else(|| no_cte("grad", name))
}

/// A query computing the CTE `name` of `query`, for the call `call`: the
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

fn no_cte(call: &str, name: &str) -> AdError {
    AdError::InvalidPlan(format!(
        "{call}'s first argument must name a CTE in the statement's WITH clause, and \
         `{name}` is not one"
    ))
}

/// `jvp(f, t.c, …, tangent, …)`'s job: `f`'s query, then each table's
/// columns, each group followed by the relation holding that table's
/// tangent (a CTE of the statement, or a table).
fn jvp_job(query: &Query, args: &[FunctionArg]) -> Result<JvpJob> {
    let usage = "write jvp(f, table.column, …, tangent, …): the name of a CTE, then each \
                 wrt table's columns followed by the name of a CTE or table holding its \
                 tangent (its dims, then a tangent under each column's name)";
    let exprs: Vec<&Expr> = args
        .iter()
        .map(|a| match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Ok(e),
            _ => Err(AdError::InvalidPlan(usage.into())),
        })
        .collect::<Result<_>>()?;
    let [Expr::Identifier(f), rest @ ..] = exprs.as_slice() else {
        return Err(AdError::InvalidPlan(usage.into()));
    };
    let query_of = |name: &str| -> Result<String> {
        cte_query(query, name, "jvp")?.ok_or_else(|| no_cte("jvp", name))
    };
    let mut job = JvpJob {
        query: query_of(&f.value)?,
        wrt: Vec::new(),
        tangents: Vec::new(),
    };
    let mut group: Vec<ColumnRef> = Vec::new();
    for e in rest {
        match e {
            Expr::CompoundIdentifier(parts) if parts.len() >= 2 => {
                let (column, table) = parts.split_last().expect("two or more parts");
                let table = table
                    .iter()
                    .map(|p| p.value.clone())
                    .collect::<Vec<_>>()
                    .join(".");
                if group
                    .first()
                    .is_some_and(|w| !w.table.eq_ignore_ascii_case(&table))
                {
                    return Err(AdError::InvalidPlan(format!(
                        "jvp takes each table's columns, then its tangent: `{table}.{}` \
                         follows `{}`'s columns with no tangent between; {usage}",
                        column.value, group[0].table
                    )));
                }
                group.push(ColumnRef::new(table, column.value.clone()));
            }
            Expr::Identifier(tangent) => {
                let Some(first) = group.first() else {
                    return Err(AdError::InvalidPlan(format!(
                        "`{tangent}` names a tangent with no columns before it; {usage}"
                    )));
                };
                let table = first.table.clone();
                if job
                    .tangents
                    .iter()
                    .any(|t| t.table.eq_ignore_ascii_case(&table))
                {
                    return Err(AdError::InvalidPlan(format!(
                        "jvp names table `{table}` twice; list all its columns before its \
                         tangent"
                    )));
                }
                let query = match cte_query(query, &tangent.value, "jvp")? {
                    Some(q) => q,
                    None => format!("SELECT * FROM {tangent}"),
                };
                job.wrt.append(&mut group);
                job.tangents.push(JvpTangent { table, query });
            }
            other => {
                return Err(AdError::InvalidPlan(format!(
                    "`{other}` is neither a table column nor a tangent's name; {usage}"
                )))
            }
        }
    }
    if !group.is_empty() || job.tangents.is_empty() {
        return Err(AdError::InvalidPlan(format!(
            "jvp's last columns have no tangent after them; {usage}"
        )));
    }
    Ok(job)
}

/// The byte range of `grad(…)` starting at `start`: through the matching
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
    Err(AdError::Internal("an unclosed grad(".into()))
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
        assert_eq!(found.objectives[0].wrt.len(), 2);
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
            Statements::plan(&refs, &GenericDialect {}).unwrap().jobs()[0]
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
        assert_eq!(found.objectives.len(), 1);
        let objective = &found.objectives[0];
        assert_eq!(objective.name, "loss");
        assert_eq!(objective.wrt, vec![ColumnRef::new("w", "val")]);
        assert!(
            objective
                .query
                .starts_with("WITH d AS (SELECT * FROM x), loss AS ("),
            "{}",
            objective.query
        );
        assert!(
            objective.query.ends_with("SELECT * FROM loss"),
            "{}",
            objective.query
        );
        assert_eq!(found.calls[0].table, "w");
        assert_eq!(found.calls[0].columns, vec!["val"]);
    }

    #[test]
    fn the_call_is_spliced_out_and_its_alias_kept() {
        let found = Calls::find(SQL, &GenericDialect {}).unwrap().unwrap();
        let out = found.rewrite(
            &mut |_| "(SELECT i, val FROM g_w)".into(),
            &mut |_| unreachable!(),
        );
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
        assert_eq!(found.objectives.len(), 1);
        assert_eq!(
            found.objectives[0].wrt,
            vec![ColumnRef::new("w", "val"), ColumnRef::new("b", "val")]
        );
        let out = found.rewrite(&mut |c| format!("g_{}", c.table), &mut |_| unreachable!());
        assert!(out.ends_with("SELECT * FROM g_w gw, g_b gb"), "{out}");
    }

    #[test]
    fn case_variants_of_a_column_are_one_wrt_column() {
        let sql = "WITH loss AS (SELECT SUM(val) AS l FROM w) \
                   SELECT * FROM grad(loss, w.val) a, grad(loss, W.VAL) b";
        let found = Calls::find(sql, &GenericDialect {}).unwrap().unwrap();
        assert_eq!(found.objectives[0].wrt, vec![ColumnRef::new("w", "val")]);
        assert_eq!(found.calls.len(), 2);
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

    const JVP: &str = "WITH h AS (SELECT x.n, SUM(x.v * w.val) + MAX(b.val) AS y \
                       FROM x JOIN w ON x.i = w.i JOIN b ON w.o = b.o GROUP BY x.n), \
                       dw AS (SELECT i, o, 0.5 AS val FROM w) \
                       SELECT n, y_tangent FROM JVP(h, w.val, dw, b.val, db) j";

    #[test]
    fn a_jvp_call_takes_each_table_s_columns_then_its_tangent() {
        let found = Calls::find(JVP, &GenericDialect {}).unwrap().unwrap();
        assert!(found.calls.is_empty());
        let job = &found.jvps[0].job;
        assert!(job.query.ends_with("SELECT * FROM h"), "{}", job.query);
        assert_eq!(
            job.wrt,
            vec![ColumnRef::new("w", "val"), ColumnRef::new("b", "val")]
        );
        // dw is a CTE of the statement, db is a table.
        assert_eq!(job.tangents.len(), 2);
        assert_eq!(job.tangents[0].table, "w");
        assert!(
            job.tangents[0].query.starts_with("WITH h AS (")
                && job.tangents[0].query.ends_with("SELECT * FROM dw"),
            "{}",
            job.tangents[0].query
        );
        assert_eq!(job.tangents[1].query, "SELECT * FROM db");
        assert!(job.tangent_of(&["B".to_string()]).is_some());
        assert!(job.tangent_of(&["x".to_string()]).is_none());
        let out = found.rewrite(&mut |_| unreachable!(), &mut |_| "(J)".into());
        assert!(out.ends_with("SELECT n, y_tangent FROM (J) j"), "{out}");
    }

    #[test]
    fn several_columns_of_one_table_share_its_tangent() {
        let sql = "WITH f AS (SELECT SUM(a * b) AS l FROM w) SELECT * FROM jvp(f, w.a, W.b, t)";
        let job = &Calls::find(sql, &GenericDialect {}).unwrap().unwrap().jvps[0].job;
        assert_eq!(job.wrt.len(), 2);
        assert_eq!(job.tangents.len(), 1);
    }

    #[test]
    fn a_malformed_jvp_call_is_refused_with_its_usage() {
        let f = "WITH f AS (SELECT SUM(a * b) AS l FROM w JOIN v ON w.i = v.i) SELECT * FROM ";
        for call in [
            "jvp(f, w.a)",            // no tangent
            "jvp(f, t)",              // a tangent with no columns
            "jvp(f, w.a, v.b, t)",    // two tables before one tangent
            "jvp(f, w.a, t, w.b, u)", // one table twice
            "jvp(f, w.a, 1.0)",       // not a name
            "jvp(g, w.a, t)",         // not a CTE
        ] {
            let e = Calls::find(&format!("{f}{call}"), &GenericDialect {}).unwrap_err();
            assert!(matches!(e, AdError::InvalidPlan(_)), "{call}: {e}");
        }
    }

    #[test]
    fn grad_and_jvp_calls_in_one_statement_are_spliced_in_source_order() {
        let sql = "WITH loss AS (SELECT SUM(val * val) AS l FROM w) \
                   SELECT * FROM jvp(loss, w.val, t) j, grad(loss, w.val) g, jvp(loss, w.val, u) k";
        let found = Calls::find(sql, &GenericDialect {}).unwrap().unwrap();
        assert_eq!((found.calls.len(), found.jvps.len()), (1, 2));
        let mut n = 0;
        let out = found.rewrite(&mut |_| "G".into(), &mut |_| {
            n += 1;
            format!("J{n}")
        });
        assert!(out.ends_with("SELECT * FROM J1 j, G g, J2 k"), "{out}");
    }

    #[test]
    fn identical_jvp_calls_share_a_job_and_different_ones_do_not() {
        let a =
            "WITH loss AS (SELECT SUM(val * val) AS l FROM w) SELECT * FROM jvp(loss, w.val, t)";
        let b =
            "WITH loss AS (SELECT SUM(val * val) AS l FROM w) SELECT * FROM jvp(loss, w.val, u)";
        let planned = Statements::plan(&[a, a, b], &GenericDialect {}).unwrap();
        assert_eq!(planned.jvp_jobs().len(), 2);
        assert!(planned.jobs().is_empty());
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
