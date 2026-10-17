// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Rewriting a statement to read what its calls' programs computed, on its
//! parsed tree.

use std::collections::HashMap;
use std::ops::ControlFlow;

use ddx_core::sqlparser::ast::{
    BinaryOperator, CastKind, DataType, ExactNumberInfo, ExcludeSelectItem, Expr, Ident, Join,
    JoinConstraint, JoinOperator, ObjectName, Query, Select, SelectItem,
    SelectItemQualifiedWildcardKind, SetExpr, Spanned, TableAlias, TableFactor, TableWithJoins,
    Value, VisitMut, VisitorMut, WildcardAdditionalOptions,
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
    /// statement is rewritten as a tree, every replacement built as nodes,
    /// then written out as SQL.
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
                        columns.push(SelectItem::ExprWithAlias {
                            expr: field(&joined(g), column),
                            alias: quoted(&h),
                        });
                        named.push((step, column, h.clone()));
                        h
                    }
                };
                rewriter.calls.insert(
                    at,
                    Expr::CompoundIdentifier(vec![read.alias.clone(), quoted(&h)]),
                );
                hiddens[i] = Some(h);
                if let Some(name) = name {
                    rewriter.names.insert(at, quoted(name));
                }
            }
            let joins = joins
                .into_iter()
                .enumerate()
                .map(|(g, (step, dims))| gradient_join(g, step, &dims))
                .collect();
            rewriter.reads.insert(
                read.at,
                Gradients {
                    alias: read.alias.clone(),
                    columns,
                    joins,
                },
            );
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
                .map(|h| ObjectName::from(vec![quoted(h)]))
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
            let projection = own
                .iter()
                .map(|c| match mine.iter().find(|(_, h, _)| h == c) {
                    Some((_, _, name)) => SelectItem::ExprWithAlias {
                        expr: value
                            .tangents
                            .iter()
                            .find(|t| &t.column == c)
                            .map_or_else(zero, |t| Expr::Identifier(quoted(&t.tangent))),
                        alias: quoted(name),
                    },
                    None => SelectItem::UnnamedExpr(Expr::Identifier(quoted(c))),
                })
                .collect();
            let from = TableWithJoins {
                relation: table(ObjectName::from(vec![quoted(&value.step)]), None),
                joins: Vec::new(),
            };
            let SetExpr::Select(select) = *select_from(projection, from).body else {
                unreachable!("select_from writes a SELECT");
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

/// What replaces a table `grad` or `vjp` calls are columns of: its rows,
/// under `alias`, with `columns` (each a gradient, from one of `joins`).
struct Gradients {
    alias: Ident,
    columns: Vec<SelectItem>,
    joins: Vec<Join>,
}

impl Gradients {
    /// `(SELECT __ddx_t.*, <columns> FROM <table> AS __ddx_t <joins>) AS
    /// <alias>`, over the statement's own `table`.
    fn over(self, mut table: TableFactor) -> TableFactor {
        if let TableFactor::Table { alias, .. } = &mut table {
            *alias = named(Ident::new(ROWS));
        }
        let mut projection = vec![SelectItem::QualifiedWildcard(
            SelectItemQualifiedWildcardKind::ObjectName(ObjectName::from(vec![Ident::new(ROWS)])),
            WildcardAdditionalOptions::default(),
        )];
        projection.extend(self.columns);
        let from = TableWithJoins {
            relation: table,
            joins: self.joins,
        };
        TableFactor::Derived {
            lateral: false,
            subquery: Box::new(select_from(projection, from)),
            alias: named(self.alias),
            sample: None,
        }
    }
}

/// The alias of a table's own rows in the relation that replaces it.
const ROWS: &str = "__ddx_t";

/// The alias of the `g`th gradient table joined to a table's rows.
fn joined(g: usize) -> String {
    format!("__ddx_g{g}")
}

/// `LEFT JOIN <step> AS __ddx_g<g> ON` each of `dims` matching, NULLs
/// included (`TRUE` for a table with no dims).
fn gradient_join(g: usize, step: &str, dims: &[&str]) -> Join {
    let on = dims
        .iter()
        .map(|d| {
            Expr::Nested(Box::new(Expr::IsNotDistinctFrom(
                Box::new(field(ROWS, d)),
                Box::new(field(&joined(g), d)),
            )))
        })
        .reduce(|a, b| Expr::BinaryOp {
            left: Box::new(a),
            op: BinaryOperator::And,
            right: Box::new(b),
        })
        .unwrap_or_else(|| Expr::Value(Value::Boolean(true).with_empty_span()));
    Join {
        relation: table(
            ObjectName::from(vec![quoted(step)]),
            named(Ident::new(joined(g))),
        ),
        global: false,
        join_operator: JoinOperator::Left(JoinConstraint::On(on)),
    }
}

/// `<table>."<column>"`.
fn field(table: &str, column: &str) -> Expr {
    Expr::CompoundIdentifier(vec![Ident::new(table), quoted(column)])
}

/// `CAST(0 AS DOUBLE)`: the tangent of a column that does not move.
fn zero() -> Expr {
    Expr::Cast {
        kind: CastKind::Cast,
        expr: Box::new(Expr::Value(
            Value::Number("0".into(), false).with_empty_span(),
        )),
        data_type: DataType::Double(ExactNumberInfo::None),
        array: false,
        format: None,
    }
}

/// `AS <name>`.
fn named(name: Ident) -> Option<TableAlias> {
    Some(TableAlias {
        explicit: true,
        name,
        columns: Vec::new(),
        at: None,
    })
}

/// The table `name`, read as it is.
fn table(name: ObjectName, alias: Option<TableAlias>) -> TableFactor {
    TableFactor::Table {
        name,
        alias,
        args: None,
        with_hints: Vec::new(),
        version: None,
        with_ordinality: false,
        partitions: Vec::new(),
        json_path: None,
        sample: None,
        index_hints: Vec::new(),
    }
}

/// `SELECT <projection> FROM <from>`. sqlparser's `Query` and `Select`
/// have dozens of fields and no `Default`, so it starts from a constant
/// query, which holds no name or value of the statement's, and sets those
/// two.
fn select_from(projection: Vec<SelectItem>, from: TableWithJoins) -> Query {
    let mut query = Parser::new(&GenericDialect {})
        .try_with_sql("SELECT 1")
        .and_then(|mut p| p.parse_query())
        .expect("a constant query parses");
    let SetExpr::Select(select) = query.body.as_mut() else {
        unreachable!("SELECT 1 is a SELECT");
    };
    select.projection = projection;
    select.from = vec![from];
    *query
}

/// `name` as a quoted identifier.
fn quoted(name: &str) -> Ident {
    Ident::with_quote('"', name)
}

/// Replaces nodes of a statement, each named by where it starts
/// ([`Site`]). A table or a `SELECT` is replaced after its contents are
/// visited, so a replacement is never visited itself: its nodes, which ddx
/// built, are never taken for the statement's, and whatever a replaced
/// `SELECT` held goes with it.
#[derive(Default)]
struct Rewriter {
    /// A `jvp` `SELECT`'s replacement.
    selects: HashMap<Location, Select>,
    /// What a `*` excludes.
    stars: HashMap<Location, ExcludeSelectItem>,
    /// What replaces a table, by where its name starts.
    reads: HashMap<Location, Gradients>,
    /// A `grad` or `vjp` call's replacement, by where its name starts.
    calls: HashMap<Location, Expr>,
    /// The name of a `grad` or `vjp` call that is a column of its own with
    /// no alias, by where its name starts.
    names: HashMap<Location, Ident>,
}

impl VisitorMut for Rewriter {
    type Break = ();

    fn pre_visit_select(&mut self, select: &mut Select) -> ControlFlow<()> {
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

    fn post_visit_select(&mut self, select: &mut Select) -> ControlFlow<()> {
        if let Some(new) = select_at(select)
            .ok()
            .and_then(|at| self.selects.remove(&at))
        {
            *select = new;
        }
        ControlFlow::Continue(())
    }

    fn post_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<()> {
        let TableFactor::Table { name, .. } = factor else {
            return ControlFlow::Continue(());
        };
        if let Some(gradients) = self.reads.remove(&name.span().start) {
            *factor = gradients.over(factor.clone());
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
