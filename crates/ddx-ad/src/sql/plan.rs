// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Planning the programs a batch of statements' calls need: one per distinct
//! derivative, each after the derivatives its query or seed reads.

use ddx_core::sqlparser::ast::Query;
use ddx_core::sqlparser::dialect::Dialect;
use substrait::proto::Plan;

use super::find::{same_column, same_columns, Call, Calls, Derivative, Site, Tangent, Transform};
use super::rewrite::{quote, Answer};
use crate::compose::{After, Differentiable, Subject};
use crate::error::{AdError, Result};
use crate::program::Options;
use crate::relation::{table_matches, ColumnRef};
use crate::tables::OutputTable;
use crate::Program;

/// A built program of either kind, as planning holds it: run as a
/// [`Program`], and composed into a later one as a [`Differentiable`].
pub(super) trait Built: Program + Differentiable + Send {
    /// It, as a [`Differentiable`], made from the concrete type. Not
    /// `self as &dyn Differentiable`, an upcast of one trait object to
    /// another (stable since Rust 1.86): built with Rust 1.88, the oldest
    /// ddx supports, a program composed from the upcast object read invalid
    /// memory, and newer compilers did not.
    fn differentiable(&self) -> &dyn Differentiable;
}
impl<T: Program + Differentiable + Send> Built for T {
    fn differentiable(&self) -> &dyn Differentiable {
        self
    }
}

/// One program to build: a derivative, merged across every call that needs
/// it, with what its query and seed read.
struct Planned {
    /// For `grad` and `vjp`, with respect to every column any call of it
    /// names, each once, case aside; for `jvp`, one call's columns.
    derivative: Derivative,
    /// Rows of a `wrt` table's gradient the calls read, where they all say
    /// (see [`Call::pushdown`]).
    pushdown: Vec<Restriction>,
    /// The query, with the calls it holds itself.
    query: Text,
    /// The cotangent or tangent query, with the calls it holds itself.
    seed: Option<Text>,
}

/// SQL that may hold calls (a statement, or a derivative's query or seed),
/// with the derivative each call needs.
struct Text {
    /// As a statement was written, or a query ddx built, written out: what
    /// runs when it holds no call.
    sql: String,
    calls: Option<Calls>,
    /// Which of the planned derivatives each call needs, in order.
    needs: Vec<usize>,
}

/// The next program to build ([`Derivatives::next`]): its query, with the
/// calls it holds rewritten to read the programs before it, and the rows of
/// a gradient to push down.
pub(super) struct Job {
    pub index: usize,
    /// The transformation and its `wrt` columns, in order: with the query
    /// and the restrictions' queries, what the program is built from
    /// besides the engine's plans.
    pub derivative: String,
    pub query: String,
    pub pushdown: Vec<Restriction>,
}

/// The rows of a `wrt` table whose gradient a program computes, at most
/// ([`crate::Options::restrict`]).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Restriction {
    /// The table, its name parts joined by `.`, as its `wrt` columns name it.
    pub table: String,
    /// `SELECT * FROM` the table, as the statement writes it, `WHERE` the
    /// rows: for the engine to plan.
    pub query: String,
}

/// Every derivative a batch of statements needs, in the order to build
/// them, and the programs built so far.
pub(super) struct Derivatives {
    statements: Vec<Text>,
    planned: Vec<Planned>,
    built: Vec<Box<dyn Built>>,
}

impl Derivatives {
    /// Find each statement's calls and plan the derivatives they need: for
    /// `grad` and `vjp`, one per query (and cotangent), with respect to
    /// every column asked about; for `jvp`, one per `SELECT`, columns and
    /// tangent. A call that another's query or seed reads is planned first.
    pub fn plan(statements: &[&str], dialect: &dyn Dialect) -> Result<Derivatives> {
        let mut planned = Vec::new();
        let mut texts = Vec::with_capacity(statements.len());
        for sql in statements {
            texts.push(Text::statement(sql, dialect, &mut planned)?);
        }
        for (j, p) in planned.iter_mut().enumerate() {
            p.pushdown = pushdown(&texts, j, p.derivative.wrt());
        }
        Ok(Derivatives {
            statements: texts,
            planned,
            built: Vec::new(),
        })
    }

    /// The next program to build, once every one before it is built and has
    /// run; `None` when none is left.
    pub fn next(&self) -> Result<Option<Job>> {
        let index = self.built.len();
        let Some(p) = self.planned.get(index) else {
            return Ok(None);
        };
        Ok(Some(Job {
            index,
            derivative: format!(
                "{} {:?}",
                p.derivative.transform().name(),
                p.derivative.wrt()
            ),
            query: p.query.rewrite(&self.planned, &self.built)?,
            pushdown: p.pushdown.clone(),
        }))
    }

    /// `job`'s program, from `plan`, the engine's plan of its query. If the
    /// query reads what earlier programs computed, it differentiates
    /// through them too.
    pub fn build(&mut self, job: &Job, plan: &Plan, options: &Options) -> Result<&dyn Built> {
        if job.index != self.built.len() {
            return Err(AdError::Internal(format!(
                "job {} built out of turn, after {}",
                job.index,
                self.built.len()
            )));
        }
        let p = &self.planned[job.index];
        let needs = p.query.needs();
        let inner: Vec<&dyn Differentiable> = needs
            .iter()
            .map(|&j| self.built[j].differentiable())
            .collect();
        let program = if inner.is_empty() {
            build(&p.derivative, plan, options)?
        } else {
            build(&p.derivative, &After::new(&inner, plan)?, options)?
        };
        self.built.push(program);
        Ok(&*self.built[job.index])
    }

    /// `program`, built for `job` before (by [`Derivatives::build`], from
    /// the same plans), as `job`'s program.
    pub fn adopt(&mut self, job: &Job, program: Box<dyn Built>) -> Result<()> {
        if job.index != self.built.len() {
            return Err(AdError::Internal(format!(
                "job {} adopted out of turn, after {}",
                job.index,
                self.built.len()
            )));
        }
        self.built.push(program);
        Ok(())
    }

    /// The programs built so far, in the order of their jobs.
    pub fn into_built(self) -> Vec<Box<dyn Built>> {
        self.built
    }

    /// What to register before `job`'s program, built, runs: each of its
    /// input tables' names, and a query for its rows, the cotangent or
    /// tangent with the columns the program asks for, by name. An input it
    /// carries from a program it reads was registered for that one.
    pub fn seeds(&self, job: &Job) -> Result<Vec<(String, String)>> {
        let p = &self.planned[job.index];
        let program = self.built.get(job.index).ok_or_else(|| {
            AdError::Internal(format!("job {}'s seeds before it is built", job.index))
        })?;
        let carried: Vec<&str> = p
            .query
            .needs()
            .into_iter()
            .flat_map(|j| self.built[j].inputs())
            .map(|i| i.name.as_str())
            .collect();
        let seed = match &p.seed {
            Some(text) => Some(text.rewrite(&self.planned, &self.built)?),
            None => None,
        };
        let mut out = Vec::new();
        for input in program.inputs() {
            if carried.contains(&input.name.as_str()) {
                continue;
            }
            let columns: Vec<String> = input.columns.iter().map(|c| quote(c)).collect();
            let query = match (&p.derivative, &seed) {
                (_, Some(seed)) => {
                    format!("SELECT {} FROM ({seed}) AS __ddx_seed", columns.join(", "))
                }
                // The same number at every row of the table: the table's own
                // dims, and the number under each wrt column's name.
                (
                    Derivative::Jvp {
                        wrt,
                        tangent: Tangent::Literal { value, table },
                        ..
                    },
                    None,
                ) => {
                    let picked: Vec<String> = input
                        .columns
                        .iter()
                        .map(|c| {
                            if wrt.iter().any(|w| w.column.eq_ignore_ascii_case(c)) {
                                format!("CAST({value:?} AS DOUBLE) AS {}", quote(c))
                            } else {
                                quote(c)
                            }
                        })
                        .collect();
                    format!("SELECT {} FROM {table}", picked.join(", "))
                }
                _ => {
                    return Err(AdError::Internal(format!(
                        "a {} job has no table to register as `{}`",
                        p.derivative.transform().name(),
                        input.name
                    )))
                }
            };
            out.push((input.name.clone(), query));
        }
        Ok(out)
    }

    /// Each statement with its calls replaced by reads of what their
    /// programs computed, once every program is built and has run. A
    /// statement with no call comes back as it was.
    pub fn rewrite(&self) -> Result<Vec<String>> {
        if self.built.len() != self.planned.len() {
            return Err(AdError::Internal(format!(
                "{} programs built for {} jobs",
                self.built.len(),
                self.planned.len()
            )));
        }
        self.statements
            .iter()
            .map(|t| t.rewrite(&self.planned, &self.built))
            .collect()
    }

    /// Program `i`, once built.
    pub fn built(&self, i: usize) -> &dyn Built {
        &*self.built[i]
    }

    #[cfg(test)]
    pub fn derivatives(&self) -> Vec<&Derivative> {
        self.planned.iter().map(|p| &p.derivative).collect()
    }

    /// The derivatives derivative `j`'s query reads.
    #[cfg(test)]
    pub fn query_needs(&self, j: usize) -> Vec<usize> {
        self.planned[j].query.needs()
    }

    /// The derivatives statement `i` reads.
    #[cfg(test)]
    pub fn statement_needs(&self, i: usize) -> Vec<usize> {
        self.statements[i].needs()
    }

    /// `program`, as if built for the next job.
    #[cfg(test)]
    pub fn push_built(&mut self, program: Box<dyn Built>) {
        self.built.push(program);
    }
}

/// The program `derivative` takes of `of`.
fn build<D: Differentiable + ?Sized>(
    derivative: &Derivative,
    of: &D,
    options: &Options,
) -> Result<Box<dyn Built>> {
    Ok(match derivative {
        Derivative::Grad { wrt, .. } => Box::new(crate::grad_with(of, wrt, options)?),
        Derivative::Vjp { wrt, .. } => Box::new(crate::vjp_with(of, wrt, options)?),
        Derivative::Jvp { wrt, .. } => Box::new(crate::jvp_with(of, wrt, options)?),
    })
}

impl Planned {
    /// The one of `planned` that `call` needs, its columns added to its
    /// `wrt`; planned now if none is, after the derivatives its query and
    /// seed read, so a derivative comes after every one whose program it
    /// reads.
    ///
    /// `open` holds the queries being planned, outermost first: a query or
    /// seed that reads, through its CTEs, a call of itself is refused.
    fn of(call: &Call, planned: &mut Vec<Planned>, open: &mut Vec<Query>) -> Result<usize> {
        let j = match planned.iter().position(|p| p.takes(&call.derivative)) {
            Some(j) => j,
            None => {
                let query = Text::query(call.derivative.of(), planned, open)?;
                let seed = match call.derivative.seed_query() {
                    Some(seed) => Some(Text::query(seed, planned, open)?),
                    None => None,
                };
                let mut derivative = call.derivative.clone();
                derivative.wrt_mut().clear();
                planned.push(Planned {
                    derivative,
                    pushdown: Vec::new(),
                    query,
                    seed,
                });
                planned.len() - 1
            }
        };
        let wrt = planned[j].derivative.wrt_mut();
        for w in call.derivative.wrt() {
            if !wrt.iter().any(|x| same_column(x, w)) {
                wrt.push(w.clone());
            }
        }
        Ok(j)
    }

    /// Does a call of `derivative` need this program? The same
    /// transformation of the same query along the same seed; for `jvp`,
    /// along exactly the same columns too, since each call gets its own
    /// tangents and one program along both would add them.
    fn takes(&self, derivative: &Derivative) -> bool {
        match (&self.derivative, derivative) {
            (Derivative::Grad { of: a, .. }, Derivative::Grad { of: b, .. }) => a == b,
            (
                Derivative::Vjp {
                    of: a,
                    cotangent: x,
                    ..
                },
                Derivative::Vjp {
                    of: b,
                    cotangent: y,
                    ..
                },
            ) => a == b && x == y,
            (
                Derivative::Jvp {
                    of: a,
                    wrt: u,
                    tangent: x,
                },
                Derivative::Jvp {
                    of: b,
                    wrt: v,
                    tangent: y,
                },
            ) => a == b && x == y && same_columns(u, v),
            _ => false,
        }
    }
}

/// The rows of each table's gradient that derivative `j`'s calls read,
/// where every call on that table says which (see [`Call::pushdown`]).
///
/// A call's conjuncts compare columns of its table other than its own. One
/// that compares a column `j` differentiates is dropped, since its program
/// has no such dim: what is left still holds every row the call reads.
fn pushdown(texts: &[Text], j: usize, wrt: &[ColumnRef]) -> Vec<Restriction> {
    // Per table: its name as `wrt` has it and as the statement writes it,
    // then each call's predicate, `None` for every row.
    let mut by_table: Vec<(String, String, Vec<Option<String>>)> = Vec::new();
    for text in texts {
        let Some(calls) = &text.calls else { continue };
        for (call, &k) in calls.calls.iter().zip(&text.needs) {
            let Site::Column { read, .. } = call.site else {
                continue;
            };
            if k != j {
                continue;
            }
            let table = &call.derivative.wrt()[0].table;
            let kept: Vec<&str> = call
                .pushdown
                .iter()
                .filter(|c| {
                    !wrt.iter().any(|w| {
                        w.table.eq_ignore_ascii_case(table)
                            && w.column.eq_ignore_ascii_case(&c.column)
                    })
                })
                .map(|c| c.sql.as_str())
                .collect();
            let predicate = (!kept.is_empty()).then(|| kept.join(" AND "));
            match by_table
                .iter_mut()
                .find(|(t, _, _)| t.eq_ignore_ascii_case(table))
            {
                Some((_, _, filters)) => filters.push(predicate),
                None => by_table.push((
                    table.clone(),
                    calls.reads[read].name.to_string(),
                    vec![predicate],
                )),
            }
        }
    }
    let mut out = Vec::new();
    for (table, written, filters) in by_table {
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
        out.push(Restriction {
            table,
            query: format!("SELECT * FROM {written} WHERE {predicate}"),
        });
    }
    out
}

impl Text {
    /// A statement, as written in `dialect`. One with no call is not
    /// parsed, or not understood ([`Calls::find`]), and comes back as it
    /// was.
    fn statement(sql: &str, dialect: &dyn Dialect, planned: &mut Vec<Planned>) -> Result<Text> {
        let calls = Calls::find(sql, dialect)?;
        Text::of(sql.to_string(), calls, planned, &mut Vec::new())
    }

    /// A derivative's query or seed, which ddx built from a statement's
    /// nodes, so the calls in it are the statement's.
    fn query(query: &Query, planned: &mut Vec<Planned>, open: &mut Vec<Query>) -> Result<Text> {
        if open.contains(query) {
            return Err(AdError::InvalidPlan(
                "a call whose function, cotangent or tangent reads that same call, through \
                 the CTEs it reads: a derivative cannot be part of what it differentiates"
                    .into(),
            ));
        }
        let calls = Calls::in_query(query)?;
        Text::of(query.to_string(), calls, planned, open)
    }

    /// `sql`, holding `calls`, each planned.
    fn of(
        sql: String,
        calls: Option<Calls>,
        planned: &mut Vec<Planned>,
        open: &mut Vec<Query>,
    ) -> Result<Text> {
        let mut needs = Vec::new();
        if let Some(calls) = &calls {
            open.push(calls.query.clone());
            for call in &calls.calls {
                needs.push(Planned::of(call, planned, open)?);
            }
            open.pop();
        }
        Ok(Text { sql, calls, needs })
    }

    /// The derivatives whose programs it reads, each once.
    fn needs(&self) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        for &j in &self.needs {
            if !out.contains(&j) {
                out.push(j);
            }
        }
        out
    }

    /// The text with each call replaced by a read of what its program
    /// computed ([`Calls::rewrite`]).
    fn rewrite(&self, planned: &[Planned], built: &[Box<dyn Built>]) -> Result<String> {
        let Some(calls) = &self.calls else {
            return Ok(self.sql.clone());
        };
        calls.rewrite(&|i| {
            let j = self.needs[i];
            let program = built
                .get(j)
                .ok_or_else(|| AdError::Internal(format!("job {j} is read before it is built")))?;
            let Subject::Program(parts) = program.subject() else {
                return Err(AdError::Internal(format!("job {j}'s program is a query")));
            };
            match calls.calls[i].derivative.transform() {
                Transform::Jvp => Ok(Answer::Value(parts.value)),
                _ => gradient(&calls.calls[i], &planned[j].derivative, parts.gradients),
            }
        })
    }
}

/// What a `grad` or `vjp` call reads: the gradient of its table in
/// `gradients`, which holds the table's dims, then a gradient under each
/// column the program is taken with respect to.
fn gradient<'a>(call: &Call, of: &Derivative, gradients: &'a [OutputTable]) -> Result<Answer<'a>> {
    let ColumnRef { table, column } = &call.derivative.wrt()[0];
    let found = gradients.iter().find_map(|g| {
        let names = g.of.table()?;
        table_matches(table, names).then_some((g, names))
    });
    let Some((g, names)) = found else {
        return Err(AdError::Internal(format!(
            "no gradient was computed for `{table}`"
        )));
    };
    let is_value = |c: &str| {
        of.wrt()
            .iter()
            .any(|w| table_matches(&w.table, names) && w.column.eq_ignore_ascii_case(c))
    };
    let found = g
        .columns
        .iter()
        .find(|c| is_value(c) && c.eq_ignore_ascii_case(column))
        .ok_or_else(|| {
            AdError::Internal(format!("no gradient was computed for `{table}.{column}`"))
        })?;
    Ok(Answer::Gradient {
        step: &g.step,
        dims: g
            .columns
            .iter()
            .filter(|c| !is_value(c))
            .map(String::as_str)
            .collect(),
        column: found,
    })
}
