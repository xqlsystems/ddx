// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Finding `grad`, `vjp` and `jvp` calls in a statement: what each call
//! differentiates, and where in the statement it sits.

use std::collections::HashMap;
use std::ops::ControlFlow;

use ddx_core::sqlparser::ast::{
    Cte, Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, Ident, ObjectName,
    ObjectNamePart, OrderBy, OrderByKind, Query, Select, SelectItem,
    SelectItemQualifiedWildcardKind, SetExpr, Spanned, Statement, TableFactor, Visit, Visitor,
    With,
};
use ddx_core::sqlparser::dialect::Dialect;
use ddx_core::sqlparser::parser::Parser;
use ddx_core::sqlparser::tokenizer::{Location, Span};

use super::build;
use crate::error::{AdError, Result};
use crate::relation::ColumnRef;

/// Which derivative a call takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transform {
    /// `grad(f, t.c, …)`: [`crate::grad`].
    Grad,
    /// `vjp(f, t.c, …, cotangent)`: [`crate::vjp`].
    Vjp,
    /// `jvp(f, t.c, …, tangent)`: [`crate::jvp`].
    Jvp,
}

impl Transform {
    /// The kind a call's name names, case-insensitively.
    fn of(name: &str) -> Option<Transform> {
        [Transform::Grad, Transform::Vjp, Transform::Jvp]
            .into_iter()
            .find(|k| k.name().eq_ignore_ascii_case(name))
    }

    /// The call's name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Transform::Grad => "grad",
            Transform::Vjp => "vjp",
            Transform::Jvp => "jvp",
        }
    }

    /// What its last argument is, if it takes one after the columns.
    fn seed(self) -> Option<&'static str> {
        match self {
            Transform::Grad => None,
            Transform::Vjp => Some("cotangent"),
            Transform::Jvp => Some("tangent"),
        }
    }

    fn usage(self) -> String {
        match self {
            Transform::Grad => {
                "write grad(f, column) as a column of a SELECT over a table: the name \
                           of a CTE computing one number (a loss, a likelihood, an energy), \
                           then a column of a table in the SELECT's FROM (table.column if it \
                           reads more than one); it is that column's gradient, row by row"
                    .into()
            }
            Transform::Vjp => {
                "write vjp(f, column, cotangent) as a column of a SELECT over a table: \
                          the name of a CTE, a column of a table in the SELECT's FROM, then \
                          the name of a CTE or table holding the cotangent of f's output (its \
                          keys, then its values)"
                    .into()
            }
            Transform::Jvp => "write jvp(e, table.column, …, tangent) AS name as a column of the \
                          statement's SELECT or a CTE's: an expression of that SELECT, the \
                          columns of one table it reads, then their tangent: the name of a CTE \
                          or table holding it (the table's dims, then a tangent under each \
                          column's name), or a number, the same at every row; it is e's \
                          tangent, row by row"
                .into(),
        }
    }
}

/// What a call reads besides its function: a `jvp`'s tangent.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Tangent {
    /// A query computing it, a CTE's or `SELECT * FROM` a table: shaped like
    /// the `wrt` table, its dims then a tangent under each column's name.
    Query(Box<Query>),
    /// The same number at every row: `jvp(sin(x), t.x, 1)` is the derivative
    /// of `sin(x)` with respect to `x`, row by row. `table` is the `wrt`
    /// table as the statement writes it, quotes included, so that SQL ddx
    /// writes over its rows names the table the statement means.
    Literal { value: f64, table: String },
}

/// The derivative a call asks for: a transformation applied to a function
/// (a query), with respect to columns of one table, with its cotangent or
/// tangent. Planning merges calls that need the same program, so there a
/// `grad` or `vjp` derivative may be with respect to several tables.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Derivative {
    /// `grad(f, c)`: [`crate::grad`] of `of`, the CTE `f`'s query.
    Grad { of: Box<Query>, wrt: Vec<ColumnRef> },
    /// `vjp(f, c, cotangent)`: [`crate::vjp`] of `of`, pulled back along a
    /// query computing the cotangent, a CTE's or `SELECT * FROM` a table.
    Vjp {
        of: Box<Query>,
        wrt: Vec<ColumnRef>,
        cotangent: Box<Query>,
    },
    /// `jvp(e, t.c, …, tangent)`: [`crate::jvp`] of `of`, the `SELECT` the
    /// call is a column of, after the CTEs it reads, with each `jvp(e, …)`
    /// in it written `e` under a name of ddx's ([`Site::Tangent`]).
    Jvp {
        of: Box<Query>,
        wrt: Vec<ColumnRef>,
        tangent: Tangent,
    },
}

impl Derivative {
    /// Which transformation it applies.
    pub(crate) fn transform(&self) -> Transform {
        match self {
            Derivative::Grad { .. } => Transform::Grad,
            Derivative::Vjp { .. } => Transform::Vjp,
            Derivative::Jvp { .. } => Transform::Jvp,
        }
    }

    /// The query it differentiates.
    pub(crate) fn of(&self) -> &Query {
        match self {
            Derivative::Grad { of, .. }
            | Derivative::Vjp { of, .. }
            | Derivative::Jvp { of, .. } => of,
        }
    }

    /// The columns it is taken with respect to.
    pub(crate) fn wrt(&self) -> &[ColumnRef] {
        match self {
            Derivative::Grad { wrt, .. }
            | Derivative::Vjp { wrt, .. }
            | Derivative::Jvp { wrt, .. } => wrt,
        }
    }

    pub(crate) fn wrt_mut(&mut self) -> &mut Vec<ColumnRef> {
        match self {
            Derivative::Grad { wrt, .. }
            | Derivative::Vjp { wrt, .. }
            | Derivative::Jvp { wrt, .. } => wrt,
        }
    }

    /// A query computing its cotangent or tangent, if one does.
    pub(crate) fn seed_query(&self) -> Option<&Query> {
        match self {
            Derivative::Vjp { cotangent, .. } => Some(cotangent),
            Derivative::Jvp {
                tangent: Tangent::Query(q),
                ..
            } => Some(q),
            _ => None,
        }
    }
}

/// One `grad`, `vjp` or `jvp` call: a column of a `SELECT`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Call {
    /// What it differentiates.
    pub derivative: Derivative,
    /// For `grad` and `vjp`, the rows of the table the `SELECT` reads at
    /// most, when its `WHERE` says: the conjuncts that compare a column of
    /// the table other than the call's own with a literal (`layer = 0`, `i <
    /// 10`); empty if none does, which is every row. Such a comparison
    /// rejects NULL, so computing only those rows' gradients leaves the
    /// statement's result as it was, whatever the joins. Planning pushes
    /// the conjuncts that compare one of the table's dims into the program.
    pub pushdown: Vec<Conjunct>,
    pub(super) site: Site,
}

/// One conjunct of a `WHERE` that [`Call::pushdown`] can carry.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Conjunct {
    /// The column of the table it compares.
    pub column: String,
    /// The comparison, over bare column names (`layer IN (0, 2)`).
    pub sql: String,
}

/// Where a call sits, for [`Calls::rewrite`].
///
/// A node of the statement is named by where its first token starts: no
/// other node starts there, and the name survives cloning the statement.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Site {
    /// A `grad` or `vjp` call: where its name starts, which of [`Calls`]'s
    /// `reads` is the table whose rows it is a column of, and, when the call
    /// is a column of its own with no alias, the column's name:
    /// `{column}_grad`, as a `jvp`'s is `{e}_tangent`.
    Column {
        at: Location,
        read: usize,
        name: Option<String>,
    },
    /// A `jvp` call: which of [`Calls`]'s `selects` it is a column of, the
    /// name `f` gives its expression there, and the column's name.
    Tangent {
        select: usize,
        hidden: String,
        name: String,
        /// Set when its expression repeats a column before it in the
        /// `SELECT`, which `f` then writes once: DuckDB's Substrait producer
        /// drops a column that repeats another, so the plan would not say
        /// which is which.
        reuse: Option<Reuse>,
    },
}

/// A `jvp` call whose expression `f` writes once, as another column: that
/// column's position among `f`'s columns, and where the call's own column
/// goes among them (before the column at `at`, or last).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Reuse {
    pub(super) column: usize,
    pub(super) at: usize,
}

/// A table in a `FROM` clause whose rows `grad` or `vjp` calls are columns
/// of.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Read {
    /// Where its name starts.
    pub(super) at: Location,
    /// Its name, as written.
    pub(super) name: ObjectName,
    /// What the `SELECT` calls it: its alias, or its name's last part.
    pub(super) alias: Ident,
    /// Where the `SELECT` whose `FROM` it is in starts.
    pub(super) select: Location,
}

/// The calls in a statement, in the order the statement writes them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Calls {
    /// The calls.
    pub calls: Vec<Call>,
    pub(super) reads: Vec<Read>,
    /// Where each `SELECT` a `jvp` call is a column of starts.
    pub(super) selects: Vec<Location>,
    /// Each `*` of a `SELECT` whose calls read a table: where it starts, and
    /// the reads it covers, whose gradient columns it must not return.
    pub(super) stars: Vec<(Location, Vec<usize>)>,
    /// The statement.
    pub(super) query: Query,
}

impl Calls {
    /// Find the calls in `sql`, or `None` if it has none.
    ///
    /// A statement that does not mention `grad`, `vjp` or `jvp` is not
    /// parsed at all, and one `sqlparser` cannot parse is passed over: it may
    /// be engine syntax `sqlparser` lacks, and if it does hold a call, the
    /// engine refuses it loudly as an unknown function.
    pub fn find(sql: &str, dialect: &dyn Dialect) -> Result<Option<Calls>> {
        let lower = sql.to_ascii_lowercase();
        if !["grad", "vjp", "jvp"].iter().any(|n| lower.contains(n)) {
            return Ok(None);
        }
        let Ok(statements) = Parser::parse_sql(dialect, sql) else {
            return Ok(None);
        };
        let [Statement::Query(query)] = statements.as_slice() else {
            return find_none(&statements);
        };
        Calls::in_query(query)
    }

    /// Find the calls in `query`, or `None` if it has none: a statement's,
    /// or a query ddx built from one's nodes (a derivative's query or
    /// seed), whose calls are the nodes the statement placed there.
    pub fn in_query(query: &Query) -> Result<Option<Calls>> {
        let mut finder = Finder::default();
        let _ = query.visit(&mut finder);
        if let Some(kind) = finder.relations.first() {
            return Err(AdError::InvalidPlan(format!(
                "{}(…) is a column of a SELECT, not a relation in its FROM; {}",
                kind.name(),
                kind.usage()
            )));
        }
        if finder.found.is_empty() {
            return Ok(None);
        }
        let mut found = Found {
            query,
            ctes: query.with.as_ref().map_or(&[][..], |w| &w.cte_tables[..]),
            calls: Vec::new(),
            reads: Vec::new(),
            selects: Vec::new(),
        };
        for (kind, function, select) in &finder.found {
            let select = select.as_ref().ok_or_else(|| {
                AdError::InvalidPlan(format!(
                    "{}(…) outside a SELECT's columns; {}",
                    kind.name(),
                    kind.usage()
                ))
            })?;
            let call = found.call(*kind, function, select)?;
            found.calls.push(call);
        }
        found.write_jvp_queries(&finder.found)?;
        let stars = stars(&finder.found, &found.reads)?;
        Ok(Some(Calls {
            calls: found.calls,
            reads: found.reads,
            selects: found.selects,
            stars,
            query: query.clone(),
        }))
    }
}

/// What [`Calls::find`] has found in a statement so far.
struct Found<'q> {
    query: &'q Query,
    ctes: &'q [Cte],
    calls: Vec<Call>,
    reads: Vec<Read>,
    selects: Vec<Location>,
}

impl Found<'_> {
    /// The call `function`, of `kind`, a column of `select`.
    fn call(&mut self, kind: Transform, function: &Function, select: &Select) -> Result<Call> {
        let args = call_args(kind, function)?;
        let seeded = usize::from(kind.seed().is_some());
        let columns = &args[1..args.len() - seeded];
        match kind {
            Transform::Grad => self.column_call(kind, function, select, args[0], columns, None),
            Transform::Vjp => {
                let cotangent = Box::new(self.relation(kind, args[args.len() - 1])?);
                self.column_call(kind, function, select, args[0], columns, Some(cotangent))
            }
            Transform::Jvp => {
                let last = args[args.len() - 1];
                let tangent = match literal(last) {
                    Some(value) => Tangent::Literal {
                        value,
                        table: String::new(),
                    },
                    None => Tangent::Query(Box::new(self.relation(kind, last)?)),
                };
                self.tangent_call(function, select, columns, tangent)
            }
        }
    }

    /// A query computing `arg`, a call's last argument, the name of a CTE
    /// or a table: its cotangent or tangent.
    fn relation(&self, kind: Transform, arg: &Expr) -> Result<Query> {
        let what = kind.seed().unwrap_or("relation");
        match arg {
            Expr::Identifier(s) => Ok(cte_query(self.query, &s.value, kind.name())?
                .unwrap_or_else(|| {
                    build::query(None, build::select_star(vec![build::unplaced(s)].into()))
                })),
            _ => Err(AdError::InvalidPlan(format!(
                "{}'s last argument is the name of its {what}; {}",
                kind.name(),
                kind.usage()
            ))),
        }
    }

    /// A `grad` or `vjp` call of the CTE `f`, a column of `select`, the
    /// gradient of `columns`' one column.
    fn column_call(
        &mut self,
        kind: Transform,
        function: &Function,
        select: &Select,
        f: &Expr,
        columns: &[&Expr],
        cotangent: Option<Box<Query>>,
    ) -> Result<Call> {
        let [column] = columns else {
            return Err(AdError::InvalidPlan(format!(
                "{}(…) is the gradient of one column, so it takes one; {}",
                kind.name(),
                kind.usage()
            )));
        };
        let Expr::Identifier(f) = f else {
            return Err(AdError::InvalidPlan(format!(
                "{}'s first argument is the name of a CTE, and `{f}` is not a name; {}",
                kind.name(),
                kind.usage()
            )));
        };
        let query = cte_query(self.query, &f.value, kind.name())?.ok_or_else(|| {
            AdError::InvalidPlan(format!(
                "{}'s first argument must name a CTE in the statement's WITH clause, and \
                 `{f}` is not one; {}",
                kind.name(),
                kind.usage()
            ))
        })?;
        let (factor, name) = read_of(kind, select, column, self.ctes)?;
        let read = self.read(factor, select)?;
        let called = self.reads[read].alias.value.clone();
        let wrt = vec![ColumnRef::new(
            table_name(&self.reads[read].name),
            name.clone(),
        )];
        Ok(Call {
            derivative: match cotangent {
                None => Derivative::Grad {
                    of: Box::new(query),
                    wrt,
                },
                Some(cotangent) => Derivative::Vjp {
                    of: Box::new(query),
                    wrt,
                    cotangent,
                },
            },
            pushdown: filter_of(select, &called, one_table(select), &name),
            site: Site::Column {
                at: start_of(function.name.span())?,
                read,
                name: unnamed_column(select, function).then(|| format!("{name}_grad")),
            },
        })
    }

    /// Which of `reads` is `factor`, a table in `select`'s `FROM`, added
    /// the first time.
    fn read(&mut self, factor: &TableFactor, select: &Select) -> Result<usize> {
        let TableFactor::Table { name, alias, .. } = factor else {
            return Err(AdError::Internal("a read of something not a table".into()));
        };
        let at = start_of(name.span())?;
        if let Some(r) = self.reads.iter().position(|r| r.at == at) {
            return Ok(r);
        }
        let alias = match alias {
            Some(a) => a.name.clone(),
            None => name
                .0
                .last()
                .and_then(|p| p.as_ident())
                .cloned()
                .ok_or_else(|| AdError::Internal(format!("a table named `{name}`")))?,
        };
        self.reads.push(Read {
            at,
            name: name.clone(),
            alias,
            select: select_at(select)?,
        });
        Ok(self.reads.len() - 1)
    }

    /// A `jvp` call along `columns`, of one table, a column of `select`.
    /// Its query is written once every call is found.
    fn tangent_call(
        &mut self,
        function: &Function,
        select: &Select,
        columns: &[&Expr],
        mut tangent: Tangent,
    ) -> Result<Call> {
        let (table, written, names) = one_tables_columns(Transform::Jvp, columns)?;
        if let Tangent::Literal { table, .. } = &mut tangent {
            *table = written;
        }
        let at = select_at(select)?;
        let index = match self.selects.iter().position(|&s| s == at) {
            Some(i) => i,
            None => {
                self.selects.push(at);
                self.selects.len() - 1
            }
        };
        let earlier = self
            .calls
            .iter()
            .filter(|c| matches!(c.site, Site::Tangent { select, .. } if select == index))
            .count();
        Ok(Call {
            derivative: Derivative::Jvp {
                // Written once every call is found (`write_jvp_queries`).
                of: Box::new(build::query(None, select.clone())),
                wrt: names
                    .into_iter()
                    .map(|c| ColumnRef::new(table.clone(), c))
                    .collect(),
                tangent,
            },
            pushdown: Vec::new(),
            site: Site::Tangent {
                select: index,
                hidden: format!("__ddx_jvp_{earlier}"),
                name: tangent_name(select, function)?,
                reuse: None,
            },
        })
    }

    /// Write each `jvp` call's query: its `SELECT`, after the CTEs it reads,
    /// with each `jvp(e, …)` in it written `e` under the call's hidden name.
    /// One `SELECT`'s calls must push forward along the same columns and
    /// tangent, since their program has one tangent.
    fn write_jvp_queries(&mut self, found: &[(Transform, Function, Option<Select>)]) -> Result<()> {
        let recursive = self.query.with.as_ref().is_some_and(|w| w.recursive);
        for (index, &at) in self.selects.iter().enumerate() {
            let select = found
                .iter()
                .filter_map(|(_, _, s)| s.as_ref())
                .find(|s| select_at(s).ok() == Some(at))
                .expect("each SELECT has a call");
            let visible = visible_ctes(self.query, at).ok_or_else(|| {
                AdError::NotImplemented(
                    "jvp(…) in a SELECT other than the statement's own or one of its CTEs'; \
                     move it into a CTE"
                        .into(),
                )
            })?;
            if recursive {
                return Err(AdError::NotImplemented(
                    "jvp(…) in a statement with a WITH RECURSIVE clause".into(),
                ));
            }
            if let Some(order_by) = owner(self.query, at).and_then(|q| q.order_by.as_ref()) {
                orders_by_output(order_by, select)?;
            }
            let mut f = select.clone();
            let mut k = 0;
            for item in &mut f.projection {
                if let Some(e) = jvp_expr(item) {
                    *item = SelectItem::ExprWithAlias {
                        expr: e,
                        alias: Ident::with_quote('"', format!("__ddx_jvp_{k}")),
                    };
                    k += 1;
                }
            }
            let reuses = write_repeats_once(&mut f);
            let f = Box::new(build::query(
                with_of(self.ctes, visible, &reads_of(select)),
                f,
            ));
            let mut along: Option<Derivative> = None;
            for call in self.calls.iter_mut() {
                if !matches!(call.site, Site::Tangent { select, .. } if select == index) {
                    continue;
                }
                if let Derivative::Jvp { of, .. } = &mut call.derivative {
                    *of = f.clone();
                }
                if let Site::Tangent { hidden, reuse, .. } = &mut call.site {
                    *reuse = reuses.get(hidden.as_str()).copied();
                }
                match &along {
                    None => along = Some(call.derivative.clone()),
                    Some(first) => {
                        if !same_columns(first.wrt(), call.derivative.wrt())
                            || tangent_of(first) != tangent_of(&call.derivative)
                        {
                            return Err(AdError::NotImplemented(
                                "jvp(…) columns of one SELECT along different columns or \
                                 tangents; push forward along each in a SELECT of its own"
                                    .into(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// The number `arg` is, if it is a numeric literal (`1`, `-0.5`): a
/// tangent the same at every row.
fn literal(arg: &Expr) -> Option<f64> {
    match arg {
        Expr::Value(v) => match &v.value {
            ddx_core::sqlparser::ast::Value::Number(n, _) => n.parse().ok(),
            _ => None,
        },
        Expr::UnaryOp {
            op: ddx_core::sqlparser::ast::UnaryOperator::Minus,
            expr,
        } => literal(expr).map(|v| -v),
        _ => None,
    }
}

/// A `jvp` derivative's tangent.
fn tangent_of(d: &Derivative) -> Option<&Tangent> {
    match d {
        Derivative::Jvp { tangent, .. } => Some(tangent),
        _ => None,
    }
}

/// Do `a` and `b` name the same columns, in order, case aside?
pub(super) fn same_columns(a: &[ColumnRef], b: &[ColumnRef]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_column(x, y))
}

/// Do `a` and `b` name the same column of the same table, case aside?
pub(super) fn same_column(a: &ColumnRef, b: &ColumnRef) -> bool {
    a.table.eq_ignore_ascii_case(&b.table) && a.column.eq_ignore_ascii_case(&b.column)
}

/// The first argument of `item`, if it is a `jvp` call, maybe under an
/// alias.
/// `f`'s `jvp` columns (`__ddx_jvp_{k}`) that repeat a column before them,
/// taken out of `f`, by their names: DuckDB's Substrait producer drops a
/// column that repeats another (`SELECT x AS a, x AS b`), so `f` writes each
/// expression once. Not with a wildcard, whose columns are not its items.
fn write_repeats_once(f: &mut Select) -> HashMap<String, Reuse> {
    let mut reuses = HashMap::new();
    if f.projection.iter().any(|i| {
        matches!(
            i,
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(..)
        )
    }) {
        return reuses;
    }
    let expr = |item: &SelectItem| match item {
        SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => Some(e.clone()),
        _ => None,
    };
    let mut kept: Vec<SelectItem> = Vec::new();
    for item in std::mem::take(&mut f.projection) {
        if let SelectItem::ExprWithAlias { expr: e, alias } = &item {
            if alias.value.starts_with("__ddx_jvp_") {
                if let Some(column) = kept.iter().position(|k| expr(k).as_ref() == Some(e)) {
                    let at = kept.len();
                    reuses.insert(alias.value.clone(), Reuse { column, at });
                    continue;
                }
            }
        }
        kept.push(item);
    }
    f.projection = kept;
    reuses
}

fn jvp_expr(item: &SelectItem) -> Option<Expr> {
    let (SelectItem::UnnamedExpr(Expr::Function(f))
    | SelectItem::ExprWithAlias {
        expr: Expr::Function(f),
        ..
    }) = item
    else {
        return None;
    };
    let [ObjectNamePart::Identifier(id)] = f.name.0.as_slice() else {
        return None;
    };
    if Transform::of(&id.value) != Some(Transform::Jvp) {
        return None;
    }
    call_args(Transform::Jvp, f)
        .ok()?
        .first()
        .map(|e| (*e).clone())
}

/// Each `*` of a `SELECT` holding `found` that covers a table one of
/// `reads` is: where it starts, and the reads it covers.
fn stars(
    found: &[(Transform, Function, Option<Select>)],
    reads: &[Read],
) -> Result<Vec<(Location, Vec<usize>)>> {
    let mut stars = Vec::new();
    for select in found.iter().filter_map(|(_, _, s)| s.as_ref()) {
        let at = select_at(select)?;
        let mine: Vec<usize> = (0..reads.len())
            .filter(|&r| reads[r].select == at)
            .collect();
        if mine.is_empty() {
            continue;
        }
        for item in &select.projection {
            let (options, covered) = match item {
                SelectItem::Wildcard(o) => (o, mine.clone()),
                SelectItem::QualifiedWildcard(
                    SelectItemQualifiedWildcardKind::ObjectName(n),
                    o,
                ) => {
                    let last = n.0.last().and_then(|p| p.as_ident());
                    let covered = mine
                        .iter()
                        .copied()
                        .filter(|&r| {
                            last.is_some_and(|l| {
                                l.value.eq_ignore_ascii_case(&reads[r].alias.value)
                            })
                        })
                        .collect();
                    (o, covered)
                }
                _ => continue,
            };
            if covered.is_empty() {
                continue;
            }
            if options.opt_ilike.is_some()
                || options.opt_exclude.is_some()
                || options.opt_except.is_some()
                || options.opt_replace.is_some()
                || options.opt_rename.is_some()
                || options.opt_alias.is_some()
            {
                return Err(AdError::NotImplemented(
                    "a * with options in a SELECT whose grad or vjp columns read a table it \
                     covers"
                        .into(),
                ));
            }
            let star = start_of(options.wildcard_token.0.span)?;
            if !stars.iter().any(|(s, _)| *s == star) {
                stars.push((star, covered));
            }
        }
    }
    Ok(stars)
}

/// Where a node with `span` starts: its name in [`Site`]. A node the parser
/// did not place has no name, and nothing of the statement is one.
fn start_of(span: Span) -> Result<Location> {
    if span == Span::empty() {
        return Err(AdError::Internal(
            "a node of the statement with no place".into(),
        ));
    }
    Ok(span.start)
}

/// Where `select` starts: its `SELECT` keyword.
pub(super) fn select_at(select: &Select) -> Result<Location> {
    start_of(select.select_token.0.span)
}

/// Is any statement in `statements` holding a call? Anything that is not a
/// single query cannot hold one ddx knows how to place: make sure there is
/// none before saying so.
fn find_none(statements: &[Statement]) -> Result<Option<Calls>> {
    let mut finder = Finder::default();
    for st in statements {
        let _ = st.visit(&mut finder);
    }
    let kind = finder
        .found
        .first()
        .map(|(k, _, _)| *k)
        .or(finder.relations.first().copied());
    match kind {
        None => Ok(None),
        Some(kind) => Err(AdError::NotImplemented(format!(
            "{}(…) in a statement that is not a single query",
            kind.name()
        ))),
    }
}

/// Collects `grad(…)`, `vjp(…)` and `jvp(…)` calls (unqualified,
/// case-folded), each with the `SELECT` it is in, and any written as a
/// relation in a `FROM` clause.
#[derive(Default)]
struct Finder {
    found: Vec<(Transform, Function, Option<Select>)>,
    relations: Vec<Transform>,
    /// The `SELECT`s around the node being visited, innermost last; `None`
    /// for a query, so that a call in a query's own clauses outside its
    /// body (`LIMIT`, or a set operation's `ORDER BY`) is in no `SELECT`,
    /// never in the one around the query.
    open: Vec<Option<Select>>,
    /// The calls in a query's `ORDER BY`, by where their names start, each
    /// with the query's body: a `SELECT` orders its own rows, so a call
    /// there is a column of them as one in its select list is.
    ordering: HashMap<Location, Select>,
}

impl Visitor for Finder {
    type Break = ();

    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<()> {
        if let (Some(order_by), SetExpr::Select(select)) = (&query.order_by, &*query.body) {
            if let OrderByKind::Expressions(exprs) = &order_by.kind {
                for e in exprs {
                    let _ = e.expr.visit(&mut OurCalls {
                        found: |f: &Function| {
                            self.ordering
                                .insert(f.name.span().start, (**select).clone());
                        },
                        depth: 0,
                    });
                }
            }
        }
        self.open.push(None);
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _: &Query) -> ControlFlow<()> {
        self.open.pop();
        ControlFlow::Continue(())
    }

    fn pre_visit_select(&mut self, select: &Select) -> ControlFlow<()> {
        self.open.push(Some(select.clone()));
        ControlFlow::Continue(())
    }

    fn post_visit_select(&mut self, _: &Select) -> ControlFlow<()> {
        self.open.pop();
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if let Expr::Function(f) = expr {
            if let [ObjectNamePart::Identifier(id)] = f.name.0.as_slice() {
                if let Some(kind) = Transform::of(&id.value) {
                    let select = match self.ordering.remove(&f.name.span().start) {
                        Some(select) => Some(select),
                        None => self.open.last().cloned().flatten(),
                    };
                    self.found.push((kind, f.clone(), select));
                }
            }
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_table_factor(&mut self, factor: &TableFactor) -> ControlFlow<()> {
        if let TableFactor::Table {
            name,
            args: Some(_),
            ..
        } = factor
        {
            if let [ObjectNamePart::Identifier(id)] = name.0.as_slice() {
                if let Some(kind) = Transform::of(&id.value) {
                    self.relations.push(kind);
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// Calls `found` on each function call in what it visits whose name is one
/// of ours, outside any query nested in it: a call in a subquery is a
/// column of that subquery's rows.
struct OurCalls<F> {
    found: F,
    /// How many queries deep the node being visited is.
    depth: usize,
}

impl<F: FnMut(&Function)> Visitor for OurCalls<F> {
    type Break = ();

    fn pre_visit_query(&mut self, _: &Query) -> ControlFlow<()> {
        self.depth += 1;
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _: &Query) -> ControlFlow<()> {
        self.depth -= 1;
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if let (Expr::Function(f), 0) = (expr, self.depth) {
            if let [ObjectNamePart::Identifier(id)] = f.name.0.as_slice() {
                if Transform::of(&id.value).is_some() {
                    (self.found)(f);
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// A call's arguments, each a plain expression.
fn call_args(kind: Transform, function: &Function) -> Result<Vec<&Expr>> {
    let FunctionArguments::List(list) = &function.args else {
        return Err(AdError::InvalidPlan(kind.usage()));
    };
    if list.duplicate_treatment.is_some() || !list.clauses.is_empty() {
        return Err(AdError::InvalidPlan(kind.usage()));
    }
    let args: Vec<&Expr> = list
        .args
        .iter()
        .map(|a| match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Ok(e),
            _ => Err(AdError::InvalidPlan(kind.usage())),
        })
        .collect::<Result<_>>()?;
    let least = if kind.seed().is_some() { 3 } else { 2 };
    if args.len() < least {
        return Err(AdError::InvalidPlan(kind.usage()));
    }
    Ok(args)
}

/// The tables of `select`'s `FROM` clause, outermost joins included.
fn tables(select: &Select) -> impl Iterator<Item = &TableFactor> {
    select
        .from
        .iter()
        .flat_map(|twj| std::iter::once(&twj.relation).chain(twj.joins.iter().map(|j| &j.relation)))
}

/// Does `select` read one table, so a column may name it bare?
fn one_table(select: &Select) -> bool {
    tables(select).count() == 1
}

/// The table in `select`'s `FROM` whose column `column` (`c`, or `t.c` for
/// the table the `SELECT` calls `t`) a `grad` or `vjp` call names, and the
/// column's name.
fn read_of<'s>(
    kind: Transform,
    select: &'s Select,
    column: &Expr,
    ctes: &[Cte],
) -> Result<(&'s TableFactor, String)> {
    let usage = kind.usage();
    let (qualifier, name) = match column {
        Expr::Identifier(c) => (None, c.value.clone()),
        Expr::CompoundIdentifier(parts) => match parts.split_last() {
            Some((c, t)) if !t.is_empty() => (Some(t), c.value.clone()),
            _ => {
                return Err(AdError::InvalidPlan(format!(
                    "`{column}` is not a column of a table in the SELECT's FROM; {usage}"
                )))
            }
        },
        _ => {
            return Err(AdError::InvalidPlan(format!(
                "`{column}` is not a column; {usage}"
            )))
        }
    };
    let called = |f: &TableFactor| -> Option<String> {
        let TableFactor::Table { name, alias, .. } = f else {
            return None;
        };
        Some(match alias {
            Some(a) => a.name.value.clone(),
            None => name.0.last()?.as_ident()?.value.clone(),
        })
    };
    // A table is named by what the SELECT calls it or, when it has no
    // alias, by its whole name (`public.w.val` over `FROM public.w`).
    let whole = |f: &TableFactor, parts: &[Ident]| {
        matches!(f, TableFactor::Table { name, alias: None, .. }
        if name.0.len() == parts.len()
            && name.0.iter().zip(parts).all(|(n, p)| {
                n.as_ident().is_some_and(|n| n.value.eq_ignore_ascii_case(&p.value))
            }))
    };
    let found: Vec<&TableFactor> = match qualifier {
        Some([t]) => tables(select)
            .filter(|f| called(f).is_some_and(|n| n.eq_ignore_ascii_case(&t.value)))
            .collect(),
        Some(parts) => tables(select).filter(|f| whole(f, parts)).collect(),
        None if one_table(select) => tables(select).collect(),
        None => {
            return Err(AdError::InvalidPlan(format!(
                "`{column}` is ambiguous in a SELECT over more than one table: write \
                 table.{name}; {usage}"
            )))
        }
    };
    let [factor] = found.as_slice() else {
        return Err(AdError::InvalidPlan(format!(
            "`{column}` is not a column of a table in the SELECT's FROM; {usage}"
        )));
    };
    let TableFactor::Table {
        name: table,
        alias,
        args: None,
        ..
    } = factor
    else {
        return Err(AdError::InvalidPlan(format!(
            "{}(…) is a column of a table's rows, and `{column}` is not a table's column; \
             {usage}",
            kind.name()
        )));
    };
    if alias.as_ref().is_some_and(|a| !a.columns.is_empty()) {
        return Err(AdError::NotImplemented(format!(
            "{}(…) over a table whose alias renames its columns",
            kind.name()
        )));
    }
    if let [ObjectNamePart::Identifier(t)] = table.0.as_slice() {
        if ctes
            .iter()
            .any(|c| c.alias.name.value.eq_ignore_ascii_case(&t.value))
        {
            return Err(AdError::InvalidPlan(format!(
                "{}(…) is a column of a table's rows, and `{t}` is a CTE: select from the \
                 table it reads",
                kind.name()
            )));
        }
    }
    Ok((factor, name))
}

/// The name a table is written with, its parts joined by `.`, unquoted.
fn table_name(name: &ObjectName) -> String {
    name.0
        .iter()
        .map(|p| p.as_ident().map_or(p.to_string(), |i| i.value.clone()))
        .collect::<Vec<_>>()
        .join(".")
}

/// A `jvp` call's columns, all of one table: the table, its parts joined by
/// `.` unquoted and as the call writes it, and the columns' names.
fn one_tables_columns(kind: Transform, columns: &[&Expr]) -> Result<(String, String, Vec<String>)> {
    let usage = kind.usage();
    let mut table: Option<(String, String)> = None;
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
            Some((first, _)) if !first.eq_ignore_ascii_case(&t) => {
                return Err(AdError::NotImplemented(format!(
                    "{}(…) takes columns of one table, so its tangent is shaped like that \
                     table; call it once per table",
                    kind.name(),
                )))
            }
            Some(_) => {}
            None => {
                let written = ObjectName::from(prefix.to_vec()).to_string();
                table = Some((t, written));
            }
        }
        names.push(column.value.clone());
    }
    let (table, written) = table.expect("one column at least");
    Ok((table, written, names))
}

/// The select item a `jvp` call is: the call, maybe under an alias.
fn select_item<'s>(select: &'s Select, function: &Function) -> Result<&'s SelectItem> {
    select
        .projection
        .iter()
        .find(|item| {
            let e = match item {
                SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => e,
                _ => return false,
            };
            matches!(e, Expr::Function(f) if f.name.span() == function.name.span())
        })
        .ok_or_else(|| {
            AdError::InvalidPlan(format!(
                "jvp(…) is a column of its own, the tangent of a column of its SELECT; \
                 compute with it in a query over that SELECT; {}",
                Transform::Jvp.usage()
            ))
        })
}

/// Is `function` a column of `select` of its own, with no alias?
fn unnamed_column(select: &Select, function: &Function) -> bool {
    select.projection.iter().any(|item| {
        matches!(item, SelectItem::UnnamedExpr(Expr::Function(f))
            if f.name.span() == function.name.span())
    })
}

/// The name of a `jvp` call's column: its alias, or `{e}_tangent` when
/// its expression is a column `e`.
fn tangent_name(select: &Select, function: &Function) -> Result<String> {
    match select_item(select, function)? {
        SelectItem::ExprWithAlias { alias, .. } => Ok(alias.value.clone()),
        _ => match call_args(Transform::Jvp, function)?[0] {
            Expr::Identifier(c) => Ok(format!("{}_tangent", c.value)),
            Expr::CompoundIdentifier(parts) => Ok(format!(
                "{}_tangent",
                parts.last().map(|p| p.value.clone()).unwrap_or_default()
            )),
            e => Err(AdError::InvalidPlan(format!(
                "name the tangent of `{e}`: jvp({e}, …) AS name"
            ))),
        },
    }
}
/// How many of the statement's CTEs the `SELECT` starting `at` can read:
/// all of them if it is the statement's own, those before it if it is a
/// CTE's; `None` if it is neither.
fn visible_ctes(query: &Query, at: Location) -> Option<usize> {
    let starts =
        |body: &SetExpr| matches!(body, SetExpr::Select(s) if select_at(s).ok() == Some(at));
    let ctes = query.with.as_ref().map_or(&[][..], |w| &w.cte_tables[..]);
    if starts(&query.body) {
        return Some(ctes.len());
    }
    ctes.iter()
        .position(|c| c.query.with.is_none() && starts(&c.query.body))
}

/// The query whose body is the `SELECT` starting `at`: the statement, or
/// one of its CTEs ([`visible_ctes`]).
fn owner(query: &Query, at: Location) -> Option<&Query> {
    let starts =
        |body: &SetExpr| matches!(body, SetExpr::Select(s) if select_at(s).ok() == Some(at));
    if starts(&query.body) {
        return Some(query);
    }
    let ctes = query.with.as_ref().map_or(&[][..], |w| &w.cte_tables[..]);
    ctes.iter()
        .map(|c| &*c.query)
        .find(|q| q.with.is_none() && starts(&q.body))
}

/// Refuse an `ORDER BY` of a `SELECT` with `jvp` columns that names
/// anything but the `SELECT`'s output, by name or position. The `SELECT`
/// becomes a read of its program's value, which holds only that output:
/// the tables it read are gone.
fn orders_by_output(order_by: &OrderBy, select: &Select) -> Result<()> {
    let mut names = Vec::new();
    let mut star = false;
    for item in &select.projection {
        match item {
            SelectItem::ExprWithAlias { alias, .. } => names.push(alias.value.clone()),
            SelectItem::UnnamedExpr(Expr::Identifier(c)) => names.push(c.value.clone()),
            SelectItem::UnnamedExpr(Expr::CompoundIdentifier(parts)) => {
                names.extend(parts.last().map(|p| p.value.clone()));
            }
            SelectItem::UnnamedExpr(Expr::Function(f)) => {
                if let Ok(name) = tangent_name(select, f) {
                    names.push(name);
                }
            }
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(..) => star = true,
            _ => {}
        }
    }
    let refuse = |what: &dyn std::fmt::Display| {
        AdError::NotImplemented(format!(
            "ORDER BY {what} in a SELECT with jvp(…) columns: order by a column of the \
             SELECT's output, by name or position, or in a query over it"
        ))
    };
    let OrderByKind::Expressions(exprs) = &order_by.kind else {
        return Err(refuse(&"ALL"));
    };
    for e in exprs {
        let ok = match &e.expr {
            Expr::Identifier(c) => star || names.iter().any(|n| n.eq_ignore_ascii_case(&c.value)),
            other => literal(other).is_some(),
        };
        if !ok {
            return Err(refuse(&e.expr));
        }
    }
    Ok(())
}

/// The conjuncts of `select`'s `WHERE` that compare a column of the table
/// it calls `alias` (or that it reads alone, if `bare`) with a literal,
/// over bare column names. `value` is the column the call names, never
/// pushed.
fn filter_of(select: &Select, alias: &str, bare: bool, value: &str) -> Vec<Conjunct> {
    let Some(selection) = select.selection.as_ref() else {
        return Vec::new();
    };
    conjuncts(selection)
        .into_iter()
        .filter_map(|c| pushable(c, alias, bare, value))
        .collect()
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

/// `c` over bare column names, if it compares a column of the table called
/// `alias` (or read alone, if `bare`), other than `value`, with a literal:
/// `=`, `<>`, `<`, `<=`, `>`, `>=`, `IN (…)` or `BETWEEN`, none negated.
/// Each rejects NULL.
fn pushable(c: &Expr, alias: &str, bare: bool, value: &str) -> Option<Conjunct> {
    let sql = compared(c, alias, bare, value)?;
    let column = match &sql {
        Expr::BinaryOp { left, right, .. } => match (left.as_ref(), right.as_ref()) {
            (Expr::Identifier(c), _) | (_, Expr::Identifier(c)) => c,
            _ => return None,
        },
        Expr::InList { expr, .. } | Expr::Between { expr, .. } => match expr.as_ref() {
            Expr::Identifier(c) => c,
            _ => return None,
        },
        _ => return None,
    };
    Some(Conjunct {
        column: column.value.clone(),
        sql: sql.to_string(),
    })
}

/// [`pushable`]'s comparison, over the bare column name.
fn compared(c: &Expr, alias: &str, bare: bool, value: &str) -> Option<Expr> {
    use ddx_core::sqlparser::ast::BinaryOperator;
    let dim = |e: &Expr| -> Option<Expr> {
        let column = match e {
            Expr::CompoundIdentifier(parts) => match parts.as_slice() {
                [table, column] if table.value.eq_ignore_ascii_case(alias) => column,
                _ => return None,
            },
            Expr::Identifier(column) if bare => column,
            _ => return None,
        };
        (!column.value.eq_ignore_ascii_case(value)).then(|| Expr::Identifier(column.clone()))
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
                return Some(Expr::BinaryOp {
                    left: Box::new(d),
                    op: op.clone(),
                    right: right.clone(),
                });
            }
            if let (true, Some(d)) = (literal(left), dim(right)) {
                return Some(Expr::BinaryOp {
                    left: left.clone(),
                    op: op.clone(),
                    right: Box::new(d),
                });
            }
            None
        }
        Expr::InList {
            expr,
            list,
            negated: false,
        } if list.iter().all(literal) => dim(expr).map(|d| Expr::InList {
            expr: Box::new(d),
            list: list.clone(),
            negated: false,
        }),
        Expr::Between {
            expr,
            negated: false,
            low,
            high,
        } if literal(low) && literal(high) => dim(expr).map(|d| Expr::Between {
            expr: Box::new(d),
            negated: false,
            low: low.clone(),
            high: high.clone(),
        }),
        _ => None,
    }
}

/// A query computing the CTE `name` of `query`, for a call of `call`: the
/// CTEs it reads, then it, then `SELECT * FROM` it; `None` if no CTE has
/// that name.
fn cte_query(query: &Query, name: &str, call: &str) -> Result<Option<Query>> {
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
    let cte = build::unplaced(&with.cte_tables[idx].alias.name);
    Ok(Some(build::query(
        with_of(&with.cte_tables, idx + 1, &[name.to_string()]),
        build::select_star(vec![cte].into()),
    )))
}

/// The names of the relations `node` reads, nested queries included, and
/// every bare identifier in it, which may name a CTE a call takes (`f`, a
/// cotangent or a tangent).
fn reads_of<V: Visit>(node: &V) -> Vec<String> {
    struct Names(Vec<String>);
    impl Visitor for Names {
        type Break = ();
        fn pre_visit_relation(&mut self, name: &ObjectName) -> ControlFlow<()> {
            if let [ObjectNamePart::Identifier(id)] = name.0.as_slice() {
                self.0.push(id.value.clone());
            }
            ControlFlow::Continue(())
        }
        fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
            if let Expr::Identifier(id) = expr {
                self.0.push(id.value.clone());
            }
            ControlFlow::Continue(())
        }
    }
    let mut names = Names(Vec::new());
    let _ = node.visit(&mut names);
    names.0
}

/// A `WITH` clause of the first `upto` of `ctes` that the relations named
/// `roots` read, directly or through each other, in their order; `None` if
/// none. A CTE nothing reads is left out: one holding a call would make the
/// query read that call's program.
fn with_of(ctes: &[Cte], upto: usize, roots: &[String]) -> Option<With> {
    let mut wanted: Vec<String> = roots.to_vec();
    let mut kept = vec![false; upto];
    for k in (0..upto).rev() {
        let name = &ctes[k].alias.name.value;
        if wanted.iter().any(|w| w.eq_ignore_ascii_case(name)) {
            kept[k] = true;
            wanted.extend(reads_of(&*ctes[k].query));
        }
    }
    build::with(
        (0..upto)
            .filter(|&k| kept[k])
            .map(|k| ctes[k].clone())
            .collect(),
    )
}
