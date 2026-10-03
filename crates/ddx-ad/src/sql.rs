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
//! rewritten before the engine sees the statement. [`GradCalls::find`] finds
//! the calls and the objectives they need; the engine adapter runs each
//! objective's [`crate::grad`] program; [`GradCalls::rewrite`] splices a relation
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
use crate::program::BackwardProgram;
use crate::relation::{table_matches, ColumnRef};

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
    /// Index into [`GradCalls::objectives`].
    pub objective: usize,
    /// The table, as written.
    pub table: String,
    /// The columns, as written.
    pub columns: Vec<String>,
    /// The byte range of `grad(…)` in the statement: the name through the
    /// closing parenthesis, not an alias after it.
    span: (usize, usize),
}

/// The `grad` calls in a statement.
#[derive(Debug, Clone, PartialEq)]
pub struct GradCalls {
    /// The objectives the calls differentiate, each once.
    pub objectives: Vec<Objective>,
    /// The calls, in source order.
    pub calls: Vec<GradCall>,
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
}

/// Several statements' `grad` calls, planned together (see the module docs).
#[derive(Debug, Clone)]
pub struct Statements {
    statements: Vec<String>,
    found: Vec<Option<GradCalls>>,
    jobs: Vec<Job>,
    /// (statement, objective within it) → job.
    job_of: BTreeMap<(usize, usize), usize>,
}

impl Statements {
    /// Find every statement's calls, and the jobs they need: one per
    /// distinct objective.
    pub fn plan(statements: &[&str], dialect: &dyn Dialect) -> Result<Statements> {
        let mut found = Vec::with_capacity(statements.len());
        for sql in statements {
            found.push(GradCalls::find(sql, dialect)?);
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
        Ok(Statements {
            statements: statements.iter().map(|s| s.to_string()).collect(),
            found,
            jobs,
            job_of,
        })
    }

    /// The objectives to differentiate, in the order [`Statements::rewrite`]
    /// expects their programs.
    pub fn jobs(&self) -> &[Job] {
        &self.jobs
    }

    /// Each statement with its calls replaced by reads of the gradients
    /// `programs` computed, one program per [job](Statements::jobs) in order,
    /// each already run so its gradient tables exist. A statement with no
    /// call comes back as it was.
    ///
    /// A call reads its table's dims, then the columns it named, from the
    /// table its job's program wrote. A column is one of the table's values
    /// when some `wrt` entry of the job names it; every other is a dim.
    pub fn rewrite(&self, programs: &[&BackwardProgram]) -> Result<Vec<String>> {
        if programs.len() != self.jobs.len() {
            return Err(AdError::Internal(format!(
                "{} programs for {} jobs",
                programs.len(),
                self.jobs.len()
            )));
        }
        let mut out = Vec::with_capacity(self.statements.len());
        for (s, sql) in self.statements.iter().enumerate() {
            let Some(calls) = &self.found[s] else {
                out.push(sql.clone());
                continue;
            };
            let mut failure = None;
            let rewritten = calls.rewrite(&mut |call| {
                let j = self.job_of[&(s, call.objective)];
                let found = programs[j]
                    .gradients
                    .iter()
                    .find(|g| table_matches(&call.table, &g.table));
                let Some(g) = found else {
                    failure = Some(AdError::Internal(format!(
                        "no gradient was computed for `{}`",
                        call.table
                    )));
                    return String::new();
                };
                let is_value = |c: &String| {
                    self.jobs[j].wrt.iter().any(|w| {
                        table_matches(&w.table, &g.table) && w.column.eq_ignore_ascii_case(c)
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
            });
            if let Some(e) = failure {
                return Err(e);
            }
            out.push(rewritten);
        }
        Ok(out)
    }
}

/// `name` as a quoted SQL identifier.
fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

impl GradCalls {
    /// Find the `grad` calls in `sql`, or `None` if it has none.
    ///
    /// A statement without the text `grad(` is not parsed at all, and one
    /// `sqlparser` cannot parse is passed over: it may be engine syntax
    /// `sqlparser` lacks, and if it does hold a `grad(f, …)`, the engine
    /// refuses it loudly as an unknown table function.
    pub fn find(sql: &str, dialect: &dyn Dialect) -> Result<Option<GradCalls>> {
        // Tokens, not text: a comment or a string can hold `(`, `)` or
        // `grad(`, and a comment can sit between `grad` and its `(`.
        let tokens = Tokenizer::new(dialect, sql).tokenize_with_location().ok();
        if !mentions_grad_call(sql, tokens.as_deref()) {
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
        if finder.found.is_empty() {
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
            if wrt.iter().any(|w| w.table != table) {
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
            calls.push(GradCall {
                objective,
                table,
                columns: wrt.into_iter().map(|w| w.column).collect(),
                span: call_span(sql, tokens.as_deref(), factor)?,
            });
        }
        calls.sort_by_key(|c| c.span.0);
        if calls.windows(2).any(|w| w[1].span.0 < w[0].span.1) {
            return Err(AdError::Internal(
                "two grad(…) calls whose spans overlap in the statement".into(),
            ));
        }
        Ok(Some(GradCalls {
            objectives,
            calls,
            sql: sql.to_string(),
        }))
    }

    /// The statement with each call replaced by `relation(call)`: SQL for a
    /// relation holding that call's gradient, such as a subquery over the
    /// table an engine materialized. An alias written after the call stays.
    pub fn rewrite(&self, relation: &mut dyn FnMut(&GradCall) -> String) -> String {
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

/// Does `sql` call `grad`: the word, then `(`, with any whitespace or
/// comments between, in any case? Without tokens (the dialect's tokenizer
/// failed), any mention of `grad` lets the parser decide.
fn mentions_grad_call(sql: &str, tokens: Option<&[TokenWithSpan]>) -> bool {
    if !sql.to_ascii_lowercase().contains("grad") {
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
        matches!(w[0], Token::Word(word) if word.value.eq_ignore_ascii_case("grad"))
            && *w[1] == Token::LParen
    })
}

fn find_none(statements: &[Statement]) -> Result<Option<GradCalls>> {
    // Anything that is not a single query cannot hold a relation-valued
    // `grad` ddx knows how to place; make sure there is none before saying so.
    let mut finder = Finder::default();
    for st in statements {
        let _ = st.visit(&mut finder);
    }
    if finder.found.is_empty() {
        Ok(None)
    } else {
        Err(AdError::NotImplemented(
            "grad(f, …) in a statement that is not a single query".into(),
        ))
    }
}

/// Collects `grad(…)` table-function calls: unqualified, case-folded.
#[derive(Default)]
struct Finder {
    found: Vec<(Location, Vec<FunctionArg>)>,
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
                }
            }
        }
        ControlFlow::Continue(())
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
    let with = query.with.as_ref().ok_or_else(|| no_cte(name))?;
    if with.recursive {
        return Err(AdError::NotImplemented(
            "grad of an objective defined in a WITH RECURSIVE clause".into(),
        ));
    }
    let idx = with
        .cte_tables
        .iter()
        .position(|c| c.alias.name.value.eq_ignore_ascii_case(name))
        .ok_or_else(|| no_cte(name))?;
    let ctes: Vec<String> = with.cte_tables[..=idx]
        .iter()
        .map(|c| c.to_string())
        .collect();
    let quoted = &with.cte_tables[idx].alias.name;
    Ok(format!("WITH {} SELECT * FROM {quoted}", ctes.join(", ")))
}

fn no_cte(name: &str) -> AdError {
    AdError::InvalidPlan(format!(
        "grad's first argument must name a CTE in the statement's WITH clause, and \
         `{name}` is not one"
    ))
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

    #[test]
    fn a_call_is_found_with_its_objective_query() {
        let found = GradCalls::find(SQL, &GenericDialect {}).unwrap().unwrap();
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
        let found = GradCalls::find(SQL, &GenericDialect {}).unwrap().unwrap();
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
        let found = GradCalls::find(sql, &GenericDialect {}).unwrap().unwrap();
        assert_eq!(found.objectives.len(), 1);
        assert_eq!(
            found.objectives[0].wrt,
            vec![ColumnRef::new("w", "val"), ColumnRef::new("b", "val")]
        );
        let out = found.rewrite(&mut |c| format!("g_{}", c.table));
        assert!(out.ends_with("SELECT * FROM g_w gw, g_b gb"), "{out}");
    }

    #[test]
    fn case_variants_of_a_column_are_one_wrt_column() {
        let sql = "WITH loss AS (SELECT SUM(val) AS l FROM w) \
                   SELECT * FROM grad(loss, w.val) a, grad(loss, W.VAL) b";
        let found = GradCalls::find(sql, &GenericDialect {}).unwrap().unwrap();
        assert_eq!(found.objectives[0].wrt, vec![ColumnRef::new("w", "val")]);
        assert_eq!(found.calls.len(), 2);
    }

    #[test]
    fn a_statement_without_grad_is_not_parsed() {
        assert_eq!(
            GradCalls::find("not even SQL", &GenericDialect {}).unwrap(),
            None
        );
        assert_eq!(
            GradCalls::find("SELECT gradient FROM t", &GenericDialect {}).unwrap(),
            None
        );
        assert_eq!(
            GradCalls::find("SELECT grad ( FROM", &GenericDialect {}).unwrap(),
            None
        );
        let scalar = "SELECT grad(x * x, x) FROM t";
        assert_eq!(GradCalls::find(scalar, &GenericDialect {}).unwrap(), None);
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
            assert!(GradCalls::find(sql, &GenericDialect {}).is_err(), "{sql}");
        }
    }
}
