// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad` in SQL: a query's gradient as a relation in another query.
//!
//! ```sql
//! WITH loss AS (SELECT SUM(power(x.v * w.val - y.v, 2)) AS l
//!               FROM x JOIN w ON x.i = w.i JOIN y ON x.s = y.s)
//! SELECT w.i, w.val - 0.1 * g.val AS val
//! FROM w JOIN grad(loss, w.val) g ON w.i = g.i
//! ```
//!
//! `grad(loss, table.column, …)` in a `FROM` clause is the gradient of the
//! loss the CTE `loss` computes, with respect to the named columns of one
//! table. It is a relation shaped like that table: the table's dims and, under
//! the columns' own names, their gradients. That is `params - lr *
//! grad(loss)(params)`, JAX's shape of an update, written as a join.
//!
//! No engine could run that call: a table function receives values, and
//! `loss` is a query. So, like v1's `grad` (design.md §3.3, Path A), it is
//! rewritten before the engine sees the statement. [`GradCalls::find`] finds
//! the calls and the loss queries they need; the engine adapter runs each
//! loss's [`crate::grad`] program; [`GradCalls::rewrite`] splices a relation
//! holding each gradient in place of each call, by source span, leaving the
//! rest of the statement byte-identical.
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
use ddx_core::sqlparser::tokenizer::Location;

use crate::error::{AdError, Result};
use crate::relation::ColumnRef;

/// A loss CTE some `grad` call differentiates.
#[derive(Debug, Clone, PartialEq)]
pub struct Loss {
    /// The CTE's name.
    pub name: String,
    /// A query computing just the loss: the statement's CTEs up to and
    /// including this one, then `SELECT * FROM` it.
    pub query: String,
    /// Every column any call takes this loss's gradient with respect to, each
    /// once, compared case-insensitively.
    pub wrt: Vec<ColumnRef>,
}

/// One `grad(loss, table.column, …)` call.
#[derive(Debug, Clone, PartialEq)]
pub struct GradCall {
    /// Index into [`GradCalls::losses`].
    pub loss: usize,
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
    /// The losses the calls differentiate, each once.
    pub losses: Vec<Loss>,
    /// The calls, in source order.
    pub calls: Vec<GradCall>,
    sql: String,
}

impl GradCalls {
    /// Find the `grad` calls in `sql`, or `None` if it has none.
    ///
    /// A statement without the text `grad(` is not parsed at all, and one
    /// `sqlparser` cannot parse is passed over: it may be engine syntax
    /// `sqlparser` lacks, and if it does hold a `grad(loss, …)`, the engine
    /// refuses it loudly as an unknown table function.
    pub fn find(sql: &str, dialect: &dyn Dialect) -> Result<Option<GradCalls>> {
        if !mentions_grad_call(sql) {
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

        let mut losses: Vec<Loss> = Vec::new();
        let mut by_name: BTreeMap<String, usize> = BTreeMap::new();
        let mut calls = Vec::new();
        for (factor, args) in finder.found {
            let (loss_name, wrt) = parse_args(&args)?;
            let loss = match by_name.get(&loss_name.to_ascii_lowercase()) {
                Some(&i) => i,
                None => {
                    losses.push(Loss {
                        query: loss_query(query, &loss_name)?,
                        name: loss_name.clone(),
                        wrt: Vec::new(),
                    });
                    by_name.insert(loss_name.to_ascii_lowercase(), losses.len() - 1);
                    losses.len() - 1
                }
            };
            let table = wrt[0].table.clone();
            if wrt.iter().any(|w| w.table != table) {
                return Err(AdError::NotImplemented(format!(
                    "grad({loss_name}, …) takes columns of one table, so it can return a \
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
                if !losses[loss].wrt.iter().any(same) {
                    losses[loss].wrt.push(w.clone());
                }
            }
            calls.push(GradCall {
                loss,
                table,
                columns: wrt.into_iter().map(|w| w.column).collect(),
                span: call_span(sql, factor)?,
            });
        }
        calls.sort_by_key(|c| c.span.0);
        Ok(Some(GradCalls {
            losses,
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

/// Does `sql` contain `grad`, then optional whitespace, then `(`, in any case?
fn mentions_grad_call(sql: &str) -> bool {
    let lower = sql.to_ascii_lowercase();
    lower
        .match_indices("grad")
        .any(|(i, _)| lower[i + 4..].trim_start().starts_with('('))
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
            "grad(loss, …) in a statement that is not a single query".into(),
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

/// `grad(loss, t.c, …)`'s loss name and columns.
fn parse_args(args: &[FunctionArg]) -> Result<(String, Vec<ColumnRef>)> {
    let usage = "write grad(loss, table.column, …): the name of a CTE computing the loss, \
                 then the columns to differentiate with respect to";
    let exprs: Vec<&Expr> = args
        .iter()
        .map(|a| match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Ok(e),
            _ => Err(AdError::InvalidPlan(usage.into())),
        })
        .collect::<Result<_>>()?;
    let [loss, wrt @ ..] = exprs.as_slice() else {
        return Err(AdError::InvalidPlan(usage.into()));
    };
    let Expr::Identifier(loss) = loss else {
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
    Ok((loss.value.clone(), wrt))
}

/// A query computing just the loss CTE `name`: the statement's CTEs up to and
/// including it, then `SELECT * FROM` it.
fn loss_query(query: &Query, name: &str) -> Result<String> {
    let with = query.with.as_ref().ok_or_else(|| no_cte(name))?;
    if with.recursive {
        return Err(AdError::NotImplemented(
            "grad of a loss defined in a WITH RECURSIVE clause".into(),
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
/// closing parenthesis, skipping quoted text.
fn call_span(sql: &str, start: Location) -> Result<(usize, usize)> {
    let begin = byte_offset(sql, start)
        .ok_or_else(|| AdError::Internal(format!("no byte offset for {start:?}")))?;
    let mut depth = 0;
    let mut quote: Option<char> = None;
    for (i, ch) in sql[begin..].char_indices() {
        match (quote, ch) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '\'' | '"') => quote = Some(ch),
            (None, '(') => depth += 1,
            (None, ')') => {
                depth -= 1;
                if depth == 0 {
                    return Ok((begin, begin + i + 1));
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
    fn a_call_is_found_with_its_loss_query() {
        let found = GradCalls::find(SQL, &GenericDialect {}).unwrap().unwrap();
        assert_eq!(found.losses.len(), 1);
        let loss = &found.losses[0];
        assert_eq!(loss.name, "loss");
        assert_eq!(loss.wrt, vec![ColumnRef::new("w", "val")]);
        assert!(
            loss.query
                .starts_with("WITH d AS (SELECT * FROM x), loss AS ("),
            "{}",
            loss.query
        );
        assert!(loss.query.ends_with("SELECT * FROM loss"), "{}", loss.query);
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
    fn two_calls_on_one_loss_share_it() {
        let sql = "WITH loss AS (SELECT SUM(w.val * b.val) AS l FROM w JOIN b ON w.o = b.o) \
                   SELECT * FROM grad(loss, w.val) gw, grad(loss, b.val) gb";
        let found = GradCalls::find(sql, &GenericDialect {}).unwrap().unwrap();
        assert_eq!(found.losses.len(), 1);
        assert_eq!(
            found.losses[0].wrt,
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
        assert_eq!(found.losses[0].wrt, vec![ColumnRef::new("w", "val")]);
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
