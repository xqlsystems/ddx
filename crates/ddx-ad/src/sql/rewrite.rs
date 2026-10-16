// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Rewriting a statement to read what its calls' programs computed, on its
//! parsed tree.

use std::collections::HashMap;
use std::ops::ControlFlow;

use ddx_core::sqlparser::ast::{
    ExcludeSelectItem, Expr, Ident, ObjectName, ObjectNamePart, Query, Select, SelectItem, SetExpr,
    Spanned, TableFactor, VisitMut, VisitorMut,
};
use ddx_core::sqlparser::dialect::GenericDialect;
use ddx_core::sqlparser::parser::Parser;
use ddx_core::sqlparser::tokenizer::Location;

use super::find::{select_at, Calls, Site};
use crate::error::{AdError, Result};
use crate::tables::OutputTable;

/// What a call's program computed, for [`Calls::rewrite`] to read.
#[derive(Debug, Clone)]
pub(crate) enum Answer<'a> {
    /// A `grad` or `vjp` call's: the table holding the gradient of its
    /// table, that table's dims, and the column holding the gradient of the
    /// call's column.
    Gradient {
        step: &'a str,
        dims: Vec<&'a str>,
        column: &'a str,
    },
    /// A `jvp` call's: its program's value, `f`'s columns then their
    /// tangents.
    Value(&'a OutputTable),
}

impl Calls {
    /// The statement with each call replaced by a read of what its program
    /// computed, `answer(i)` for the `i`th call:
    ///
    /// - each table `grad` and `vjp` calls are columns of becomes a
    ///   relation over its rows, under its alias, that also holds each
    ///   call's gradient, joined on the table's dims; each call reads its
    ///   own, and a `*` over the table excludes them;
    /// - each `SELECT` `jvp` calls are columns of becomes a read of their
    ///   program's value: its columns, with each call's tangent in the
    ///   call's place, under its name.
    ///
    /// A call inside a `jvp`'s `SELECT` is its program's to read. The
    /// statement is rewritten as a tree, then written out as SQL.
    pub fn rewrite<'a>(&self, answer: &dyn Fn(usize) -> Result<Answer<'a>>) -> Result<String> {
        let mut rewriter = Rewriter::default();
        // Each `grad` or `vjp` call's gradient column. Calls on one read
        // that read the same gradient share one, so that a call written
        // twice (in a SELECT and its GROUP BY) is the same expression twice.
        let mut hiddens: Vec<Option<String>> = vec![None; self.calls.len()];
        for (r, read) in self.reads.iter().enumerate() {
            let mut columns = Vec::new();
            let mut joins: Vec<(&str, Vec<&str>)> = Vec::new();
            let mut named: Vec<(&str, &str, String)> = Vec::new();
            for (i, call) in self.calls.iter().enumerate() {
                let Site::Column {
                    at,
                    read: of,
                    ref name,
                } = call.site
                else {
                    continue;
                };
                if of != r {
                    continue;
                }
                let Answer::Gradient { step, dims, column } = answer(i)? else {
                    return Err(AdError::Internal(format!(
                        "a {} call answered with a value",
                        call.derivative.transform().name()
                    )));
                };
                let g = match joins.iter().position(|(s, _)| *s == step) {
                    Some(g) => g,
                    None => {
                        joins.push((step, dims));
                        joins.len() - 1
                    }
                };
                let h = match named.iter().find(|(s, c, _)| *s == step && *c == column) {
                    Some((_, _, h)) => h.clone(),
                    None => {
                        let h = hidden(i);
                        columns.push(format!("__ddx_g{g}.{} AS {}", quote(column), quote(&h)));
                        named.push((step, column, h.clone()));
                        h
                    }
                };
                rewriter.calls.insert(
                    at,
                    Expr::CompoundIdentifier(vec![
                        read.alias.clone(),
                        Ident::with_quote('"', h.clone()),
                    ]),
                );
                hiddens[i] = Some(h);
                if let Some(name) = name {
                    rewriter.names.insert(at, Ident::with_quote('"', name));
                }
            }
            let mut from = format!("{TABLE} AS __ddx_t");
            for (g, (step, dims)) in joins.iter().enumerate() {
                let on: Vec<String> = dims
                    .iter()
                    .map(|d| {
                        format!(
                            "(__ddx_t.{d} IS NOT DISTINCT FROM __ddx_g{g}.{d})",
                            d = quote(d)
                        )
                    })
                    .collect();
                let on = if on.is_empty() {
                    "TRUE".to_string()
                } else {
                    on.join(" AND ")
                };
                from.push_str(&format!(" LEFT JOIN {} AS __ddx_g{g} ON {on}", quote(step)));
            }
            let sql = format!(
                "SELECT * FROM (SELECT __ddx_t.*, {} FROM {from}) AS __ddx_alias",
                columns.join(", ")
            );
            let mut factor = first_table(generated(&sql, self.lines)?)?;
            let _ = VisitMut::visit(&mut factor, &mut Named(&read.name));
            if let TableFactor::Derived { alias: Some(a), .. } = &mut factor {
                a.name = read.alias.clone();
            }
            rewriter.reads.insert(read.at, factor);
        }
        for (end, covered) in &self.stars {
            let mut names: Vec<&str> = Vec::new();
            for (call, h) in self.calls.iter().zip(&hiddens) {
                if let (Site::Column { read, .. }, Some(h)) = (&call.site, h) {
                    if covered.contains(read) && !names.contains(&h.as_str()) {
                        names.push(h);
                    }
                }
            }
            let excluded = names
                .into_iter()
                .map(|h| ObjectName::from(vec![Ident::with_quote('"', h)]))
                .collect();
            rewriter
                .stars
                .insert(*end, ExcludeSelectItem::Multiple(excluded));
        }
        for (s, &at) in self.selects.iter().enumerate() {
            let mine: Vec<(usize, &str, &str)> = self
                .calls
                .iter()
                .enumerate()
                .filter_map(|(i, c)| match &c.site {
                    Site::Tangent {
                        select,
                        hidden,
                        name,
                    } if *select == s => Some((i, hidden.as_str(), name.as_str())),
                    _ => None,
                })
                .collect();
            let Answer::Value(value) = answer(mine[0].0)? else {
                return Err(AdError::Internal(
                    "a jvp call answered with a gradient".into(),
                ));
            };
            let own = &value.columns[..value.columns.len() - value.tangents.len()];
            let picked: Vec<String> = own
                .iter()
                .map(|c| match mine.iter().find(|(_, h, _)| h == c) {
                    Some((_, _, name)) => {
                        let tangent = value
                            .tangents
                            .iter()
                            .find(|t| &t.column == c)
                            .map_or("CAST(0 AS DOUBLE)".to_string(), |t| quote(&t.tangent));
                        format!("{tangent} AS {}", quote(name))
                    }
                    None => quote(c),
                })
                .collect();
            let sql = format!("SELECT {} FROM {}", picked.join(", "), quote(&value.step));
            let SetExpr::Select(select) = *generated(&sql, self.lines)?.body else {
                return Err(AdError::Internal(format!("`{sql}` is not a SELECT")));
            };
            rewriter.selects.insert(at, *select);
        }
        let mut query = self.query.clone();
        let _ = VisitMut::visit(&mut query, &mut rewriter);
        Ok(query.to_string())
    }
}

/// The name a `grad` or `vjp` call's gradient column has in the relation
/// that replaces its table.
fn hidden(i: usize) -> String {
    format!("__ddx_d{i}")
}

/// The placeholder for a table's name in the relation that replaces it.
const TABLE: &str = "__ddx_table";

/// `sql`, which ddx wrote, as a query placed after the statement's `lines`
/// lines, so that none of its nodes starts where one of the statement's
/// does. Every name in it is ddx's or quoted; a table's own name goes in
/// afterwards ([`Named`]), so the user's dialect never meets this parser.
fn generated(sql: &str, lines: usize) -> Result<Query> {
    let text = format!("{}{sql}", "\n".repeat(lines));
    Parser::new(&GenericDialect {})
        .try_with_sql(&text)
        .and_then(|mut p| p.parse_query())
        .map(|q| *q)
        .map_err(|e| AdError::Internal(format!("ddx wrote SQL it cannot parse, `{sql}`: {e}")))
}

/// The first relation in `query`'s `FROM`.
fn first_table(query: Query) -> Result<TableFactor> {
    let SetExpr::Select(select) = *query.body else {
        return Err(AdError::Internal(
            "a query of ddx's that is not a SELECT".into(),
        ));
    };
    select
        .from
        .into_iter()
        .next()
        .map(|t| t.relation)
        .ok_or_else(|| AdError::Internal("a query of ddx's with no FROM".into()))
}

/// Puts a table's name where [`TABLE`] stands.
struct Named<'a>(&'a ObjectName);

impl VisitorMut for Named<'_> {
    type Break = ();

    fn pre_visit_relation(&mut self, name: &mut ObjectName) -> ControlFlow<()> {
        if matches!(name.0.as_slice(), [ObjectNamePart::Identifier(id)] if id.value == TABLE) {
            *name = self.0.clone();
        }
        ControlFlow::Continue(())
    }
}

/// Replaces nodes of a statement, each named by where it starts
/// ([`Site`]): `selects` before their contents are visited, so whatever a
/// replaced `SELECT` held goes with it.
#[derive(Default)]
struct Rewriter {
    /// A `jvp` `SELECT`'s replacement.
    selects: HashMap<Location, Select>,
    /// What a `*` excludes.
    stars: HashMap<Location, ExcludeSelectItem>,
    /// A table's replacement, by where its name starts.
    reads: HashMap<Location, TableFactor>,
    /// A `grad` or `vjp` call's replacement, by where its name starts.
    calls: HashMap<Location, Expr>,
    /// The name of a `grad` or `vjp` call that is a column of its own with
    /// no alias, by where its name starts.
    names: HashMap<Location, Ident>,
}

impl VisitorMut for Rewriter {
    type Break = ();

    fn pre_visit_select(&mut self, select: &mut Select) -> ControlFlow<()> {
        if let Some(new) = select_at(select)
            .ok()
            .and_then(|at| self.selects.remove(&at))
        {
            *select = new;
            return ControlFlow::Continue(());
        }
        for item in &mut select.projection {
            if let SelectItem::UnnamedExpr(expr) = item {
                let alias = match &*expr {
                    Expr::Function(f) => self.names.remove(&f.name.span().start),
                    _ => None,
                };
                if let Some(alias) = alias {
                    // Its contents are visited after this, so the call
                    // under the alias is still replaced.
                    *item = SelectItem::ExprWithAlias {
                        expr: expr.clone(),
                        alias,
                    };
                }
                continue;
            }
            let (SelectItem::Wildcard(options) | SelectItem::QualifiedWildcard(_, options)) = item
            else {
                continue;
            };
            if let Some(excluded) = self.stars.remove(&options.wildcard_token.0.span.start) {
                options.opt_exclude = Some(excluded);
            }
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<()> {
        if let TableFactor::Table { name, .. } = factor {
            if let Some(new) = self.reads.remove(&name.span().start) {
                *factor = new;
            }
        }
        ControlFlow::Continue(())
    }

    fn post_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<()> {
        if let Expr::Function(f) = expr {
            if let Some(new) = self.calls.remove(&f.name.span().start) {
                *expr = new;
            }
        }
        ControlFlow::Continue(())
    }
}

/// `name` as a quoted SQL identifier.
pub(crate) fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}
