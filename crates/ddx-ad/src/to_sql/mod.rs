// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A Substrait plan written as SQL text, for an engine that runs a program's
//! steps as SQL rather than through its Substrait consumer.
//!
//! DuckDB's consumer (`from_substrait`, in its community `substrait`
//! extension) reads a table as if on a connection of its own, so it cannot
//! see a dataset registered on the caller's connection, a temporary table or
//! the caller's open transaction, and it rejects every window function. The
//! same plan written as SQL runs on the caller's own connection, where
//! windows are ordinary SQL. [`to_sql`] writes it.
//!
//! # In two parts: what a plan means, then how a dialect writes it
//!
//! [`to_sql`] reads the plan into a typed SQL tree ([`ir`]), then writes the
//! tree in one dialect ([`lower`]) with [polyglot-sql]'s generators, so the
//! dialects' syntax, quoting and escaping are polyglot's to keep, not ddx's.
//!
//! [polyglot-sql]: https://github.com/tobilg/polyglot
//!
//! - **Reading is a fold over the plan, bottom up** (a catamorphism,
//!   [`crate::optics::cata`]): each relation's SQL is made from its inputs'
//!   finished SQL, by one case per kind of relation, and the fold keeps a
//!   stack rather than recursing, so a plan of any depth is read.
//! - **The order rows come in travels with them.** A sort sets it, and a
//!   projection, filter or fetch keeps it, as Substrait's do; but SQL
//!   orders only the rows of the query an `ORDER BY` is in, not a query
//!   reading it. So each relation's order is kept as keys among its columns
//!   (a key its projection drops is passed on as an extra column), and the
//!   query that needs it writes it again: one that picks rows with
//!   `LIMIT`, and the whole query.
//! - **A field reference is resolved in an environment** of the columns its
//!   relation reads, chained to the columns of each relation a subquery is
//!   inside: Substrait's `steps_out` counts links of that chain outward (a
//!   de Bruijn index), and a reference past either end is refused.
//! - **The tree says what the plan means**: a `divide` knows whether it is
//!   integer division, a literal its type, a window its frame, a function
//!   its canonical name. Reading needs no dialect; writing makes every
//!   choice a dialect makes (`//` or `/`, `signum` or `sign`, `BLOB` or
//!   `BYTEA`), parenthesizing every compound expression so that no
//!   dialect's precedence comes into it.
//!
//! # How a plan is written
//!
//! - **Every relation is a `SELECT` whose columns are named by position**,
//!   `c0`, `c1`, …, since Substrait refers to a column by its position. It
//!   reads its input under an alias of its own (`t0`, `t1`, …), so a column
//!   is always `alias.cN` and a user's column names never clash with ddx's.
//! - **Each relation is a CTE** (`__ddx_q0`, `__ddx_q1`, …), so a deep plan
//!   is a long `WITH` clause rather than deeply nested subqueries, which an
//!   engine's SQL planner recurses through. A relation the plan shares
//!   (`ReferenceRel`) is one CTE read wherever it is shared. A relation
//!   inside a subquery expression is written in place: it may read the
//!   enclosing query's columns, which a CTE cannot see. The root relation
//!   is the query's own `SELECT`, so its `ORDER BY` and `LIMIT` order the
//!   result.
//! - **The plan's meaning is written out, not left to the dialect.** Every
//!   literal is written with its type, so `0.1` is a `DOUBLE` on an engine
//!   that would read it as a `DECIMAL`; a `divide` whose output is an integer
//!   is the dialect's integer division (DuckDB's `/` of two integers is a
//!   `DOUBLE`); and a window's frame is always written, since SQL's default
//!   frame for an ordered window ends at the current row, and Substrait's
//!   at the end of the partition.
//! - **A function is named by its normalized name**, and one ddx knows no
//!   name of is refused, naming it, rather than written as a guess.
//! - **A join that keeps one side's rows** (semi, anti, mark) is a correlated
//!   `EXISTS`, which every engine reads the same way.
//!
//! Anything else (an extension relation, a grouping set, an interval
//! literal) is refused with [`AdError::NotImplemented`]. A plan that does
//! not hold together (a reference past its relations, a shared relation
//! that reads itself) is refused with [`AdError::InvalidPlan`].
//!
//! # What a plan leaves to its engine
//!
//! Substrait does not say everything: how a cast of a float to an integer
//! rounds, whether integer overflow wraps, where NaN sorts. A plan means
//! what the engine that planned it does there, so write it in that engine's
//! dialect, as ddx does (DuckDB plans what DuckDB runs). The same plan run
//! on another engine can differ where the two do.
//!
//! The names `to_sql` makes up begin `__ddx_` (its CTEs are `__ddx_q0`, …),
//! which ddx reserves: a table of such a name is shadowed by a CTE.

mod ir;
mod lower;

use std::collections::{HashMap, HashSet};

use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::expression::field_reference::{ReferenceType, RootType};
use substrait::proto::expression::literal::LiteralType;
use substrait::proto::expression::reference_segment::ReferenceType as Segment;
use substrait::proto::expression::subquery::set_comparison::{ComparisonOp, ReductionOp};
use substrait::proto::expression::subquery::set_predicate::PredicateOp;
use substrait::proto::expression::subquery::SubqueryType;
use substrait::proto::expression::window_function::bound::Kind as BoundKind;
use substrait::proto::expression::window_function::{Bound as SubstraitBound, BoundsType};
use substrait::proto::expression::{cast, Literal as SubstraitLiteral, RexType, Subquery};
use substrait::proto::fetch_rel::{CountMode, OffsetMode};
use substrait::proto::function_argument::ArgType;
use substrait::proto::join_rel::JoinType;
use substrait::proto::plan_rel::RelType as PlanRelType;
use substrait::proto::r#type::Kind;
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::EmitKind;
use substrait::proto::set_rel::SetOp as SubstraitSetOp;
use substrait::proto::sort_field::{SortDirection, SortKind};
use substrait::proto::{
    AggregateRel, Expression, FetchRel, FunctionArgument, Plan, ReadRel, Rel, RelCommon, SortField,
    Type as SubstraitType,
};

use crate::error::{AdError, Result};
use crate::functions::{Functions, STOP_GRADIENT};
use crate::optics::cata;
use ir::{
    Alias, Bound, Call, Column, Expr, Frame, From, IsTest, JoinKind, Literal, Op, Order, Query,
    Select, SetOp, Type, Units,
};

/// The SQL dialect [`to_sql`] writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Dialect {
    /// DataFusion's SQL, which ddx's tests run every program in to check
    /// [`to_sql`] against DataFusion's Substrait consumer.
    DataFusion,
    /// DuckDB's SQL.
    DuckDb,
}

/// `plan` as one SQL query in `dialect`, its columns named as the plan's
/// root names them.
///
/// With the `to-sql` feature, which brings in polyglot-sql's generators.
///
/// The plan's reads must be bound (each with its `base_schema`), as a
/// program's steps are when [`crate::run`](fn@crate::run) hands them to a [`crate::Backend`].
/// A table is read by the names the plan gives it, quoted, so the query
/// must run where those names resolve to the tables the plan was made from.
pub fn to_sql(plan: &Plan, dialect: Dialect) -> Result<String> {
    let functions = Functions::from_plan(plan)?;
    let root = plan
        .relations
        .iter()
        .rev()
        .find_map(|r| match &r.rel_type {
            Some(PlanRelType::Root(root)) => Some(root),
            _ => None,
        })
        .ok_or_else(|| AdError::InvalidPlan("the plan has no root relation".into()))?;
    let input = root
        .input
        .as_ref()
        .ok_or_else(|| AdError::InvalidPlan("the root has no input".into()))?;
    let mut reader = Reader {
        plan,
        functions: &functions,
        aliases: 0,
        ctes: Vec::new(),
        shared: HashMap::new(),
        reading: HashSet::new(),
    };
    let source = reader.relation(input, None, Placement::Cte)?;
    let mut body = reader.select_of(source);
    if body.columns.len() != root.names.len() {
        return Err(AdError::InvalidPlan(format!(
            "the root names {} columns, but its relation has {}",
            root.names.len(),
            body.columns.len()
        )));
    }
    // A query of no columns, such as a program's check (which asks only
    // whether any row comes back), gets one placeholder column,
    // `__ddx_unit`, since SQL has no query of no columns: its rows are the
    // plan's, and nothing reads the column.
    body.names = Some(root.names.clone());
    let query = Query {
        ctes: reader.ctes,
        body,
    };
    lower::query(&query, dialect)
}

/// Where a relation's reader puts the `SELECT` it reads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Placement {
    /// In a CTE of its own.
    Cte,
    /// In place, as a subquery: inside a subquery expression, whose
    /// relations may read the enclosing query's columns.
    InPlace,
}

/// A relation read so far, with columns `c0`, `c1`, …: the first `width`
/// its own, and any after them only keys of the order its rows come in.
struct Source {
    origin: Origin,
    width: usize,
    /// The order its rows come in, if a sort below it set one and every
    /// relation since kept it (Substrait's projections, filters and fetches
    /// do; nothing else does). SQL orders only the rows of the query an
    /// `ORDER BY` is in, so the query that needs the order (one that picks
    /// rows with `LIMIT`, or the whole query's) writes it again from these
    /// keys, which a projection that dropped them still passes on.
    order: Vec<Sorted>,
}

/// Where a [`Source`]'s rows are: a `SELECT` its reader places, or a CTE
/// already written (a shared relation's).
enum Origin {
    Select(Box<Select>),
    Cte(usize),
}

/// A key of the order a relation's rows come in: one of its columns.
#[derive(Clone)]
struct Sorted {
    column: usize,
    descending: bool,
    nulls_first: bool,
}

/// A [`Source`] placed under an alias: its `FROM` item, its own columns,
/// and the order its rows come in, read through the alias.
struct Placed {
    from: From,
    columns: Vec<Expr>,
    order: Vec<Order>,
}

/// The columns an expression may read: its relation's input columns, and,
/// inside a subquery, those of each enclosing relation, which a field
/// reference reaches `steps_out` links outward.
struct Env<'e> {
    columns: Vec<Expr>,
    outer: Option<&'e Env<'e>>,
}

impl Env<'_> {
    fn field(&self, steps_out: usize, field: i32) -> Result<Expr> {
        let mut env = self;
        for _ in 0..steps_out {
            env = env.outer.ok_or_else(|| {
                AdError::InvalidPlan(format!(
                    "a field reference steps out {steps_out} levels, past the outermost"
                ))
            })?;
        }
        usize::try_from(field)
            .ok()
            .and_then(|i| env.columns.get(i))
            .cloned()
            .ok_or_else(|| {
                AdError::InvalidPlan(format!(
                    "field {field} is past its input's {} columns",
                    env.columns.len()
                ))
            })
    }
}

/// Reads a plan into a [`Query`]: the CTEs written so far, and the aliases
/// handed out.
struct Reader<'p> {
    plan: &'p Plan,
    functions: &'p Functions,
    aliases: usize,
    ctes: Vec<Select>,
    /// The CTE of each shared relation read so far (its index, width and
    /// order), by its position in the plan's relations.
    shared: HashMap<i32, (usize, usize, Vec<Sorted>)>,
    /// The shared relations being read, to refuse one that reads itself.
    reading: HashSet<i32>,
}

impl Reader<'_> {
    fn alias(&mut self) -> Alias {
        self.aliases += 1;
        Alias(self.aliases - 1)
    }

    /// `rel`, folded bottom up.
    fn relation(&mut self, rel: &Rel, outer: Option<&Env>, placement: Placement) -> Result<Source> {
        cata(rel, inputs, |rel, read| {
            self.node(rel, read, outer, placement)
        })
    }

    /// `select` as a new CTE: its index. Its `ORDER BY` goes unless it
    /// picks rows: a CTE's order is not its reader's.
    fn name(&mut self, mut select: Box<Select>) -> usize {
        if select.limit.is_none() && select.offset.is_none() {
            select.order_by.clear();
        }
        self.ctes.push(*select);
        self.ctes.len() - 1
    }

    /// `source` placed and read under a new alias.
    fn place(&mut self, source: Source, placement: Placement) -> Placed {
        let alias = self.alias();
        let from = match source.origin {
            Origin::Cte(index) => From::Cte { index, alias },
            Origin::Select(select) => match placement {
                Placement::Cte => From::Cte {
                    index: self.name(select),
                    alias,
                },
                Placement::InPlace => {
                    let mut query = select;
                    if query.limit.is_none() && query.offset.is_none() {
                        query.order_by.clear();
                    }
                    From::Subquery { query, alias }
                }
            },
        };
        let column = |i| Expr::Column {
            table: alias,
            column: Column::Position(i),
        };
        Placed {
            from,
            columns: (0..source.width).map(column).collect(),
            order: source
                .order
                .iter()
                .map(|s| Order {
                    expr: column(s.column),
                    descending: s.descending,
                    nulls_first: s.nulls_first,
                })
                .collect(),
        }
    }

    /// `source` as a `SELECT` of its own columns, in its order: the whole
    /// query's, or a subquery's.
    fn select_of(&mut self, source: Source) -> Select {
        match source.origin {
            Origin::Select(mut select) => {
                select.columns.truncate(source.width);
                *select
            }
            Origin::Cte(_) => {
                let Placed {
                    from,
                    columns,
                    order,
                } = self.place(source, Placement::Cte);
                Select {
                    columns,
                    from,
                    order_by: order,
                    ..Select::default()
                }
            }
        }
    }

    /// One relation, from the sources [`inputs`] named for it, in order.
    fn node(
        &mut self,
        rel: &Rel,
        read: Vec<Source>,
        outer: Option<&Env>,
        placement: Placement,
    ) -> Result<Source> {
        let kind = rel
            .rel_type
            .as_ref()
            .ok_or_else(|| AdError::InvalidPlan("a relation has no type".into()))?;
        let mut read = read.into_iter();
        let mut next = |reader: &mut Self| {
            let source = read.next().expect("`inputs` names each input a node reads");
            reader.place(source, placement)
        };
        let none = Vec::new;
        let env = |columns: &[Expr]| Env {
            columns: columns.to_vec(),
            outer,
        };
        // Each relation's SELECT, its emit, and the order its rows come in.
        let (select, common, order) = match kind {
            RelType::Read(r) => (self.read(r, outer)?, r.common.as_ref(), none()),
            RelType::Project(p) => {
                let Placed {
                    from,
                    columns,
                    order,
                } = next(self);
                let computed = self.exprs(&p.expressions, &env(&columns))?;
                let select = Select {
                    columns: columns.into_iter().chain(computed).collect(),
                    from,
                    ..Select::default()
                };
                (select, p.common.as_ref(), order)
            }
            RelType::Filter(f) => {
                let Placed {
                    from,
                    columns,
                    order,
                } = next(self);
                let condition = required(f.condition.as_deref(), "a filter's condition")?;
                let condition = self.expr(condition, &env(&columns))?;
                let select = Select {
                    columns,
                    from,
                    filter: vec![condition],
                    ..Select::default()
                };
                (select, f.common.as_ref(), order)
            }
            RelType::Join(j) => {
                let left = next(self);
                let right = next(self);
                let kind = JoinType::try_from(j.r#type).unwrap_or(JoinType::Unspecified);
                let select = self.join(
                    left,
                    right,
                    j.expression.as_deref(),
                    j.post_join_filter.as_deref(),
                    kind,
                    outer,
                )?;
                (select, j.common.as_ref(), none())
            }
            RelType::Cross(c) => {
                let Placed {
                    from: left,
                    mut columns,
                    ..
                } = next(self);
                let Placed {
                    from: right,
                    columns: right_columns,
                    ..
                } = next(self);
                columns.extend(right_columns);
                let from = From::Join {
                    left: Box::new(left),
                    right: Box::new(right),
                    kind: JoinKind::Cross,
                    on: None,
                };
                let select = Select {
                    columns,
                    from,
                    ..Select::default()
                };
                (select, c.common.as_ref(), none())
            }
            RelType::Aggregate(a) => {
                let input = next(self);
                (self.aggregate(a, input, outer)?, a.common.as_ref(), none())
            }
            RelType::Sort(s) => {
                let Placed { from, columns, .. } = next(self);
                let order = self.orders(&s.sorts, &env(&columns))?;
                let select = Select {
                    columns,
                    from,
                    ..Select::default()
                };
                (select, s.common.as_ref(), order)
            }
            RelType::Fetch(f) => {
                // It picks rows in the order its input's come in, which the
                // ORDER BY that `finish` writes from `order` keeps.
                let Placed {
                    from,
                    columns,
                    order,
                } = next(self);
                let (offset, limit) = fetch(f)?;
                let select = Select {
                    columns,
                    from,
                    offset,
                    limit,
                    ..Select::default()
                };
                (select, f.common.as_ref(), order)
            }
            RelType::Set(s) => {
                let op = match SubstraitSetOp::try_from(s.op).unwrap_or(SubstraitSetOp::Unspecified)
                {
                    SubstraitSetOp::UnionAll => SetOp::UnionAll,
                    SubstraitSetOp::UnionDistinct => SetOp::Union,
                    SubstraitSetOp::MinusPrimary => SetOp::Except,
                    SubstraitSetOp::MinusPrimaryAll => SetOp::ExceptAll,
                    SubstraitSetOp::IntersectionMultiset => SetOp::Intersect,
                    SubstraitSetOp::IntersectionMultisetAll => SetOp::IntersectAll,
                    other => {
                        return Err(AdError::NotImplemented(format!(
                            "writing the set operation {} as SQL",
                            other.as_str_name()
                        )))
                    }
                };
                let mut width = None;
                let mut inputs = Vec::new();
                for _ in &s.inputs {
                    let Placed { from, columns, .. } = next(self);
                    if *width.get_or_insert(columns.len()) != columns.len() {
                        return Err(AdError::InvalidPlan(
                            "a set operation's inputs have different widths".into(),
                        ));
                    }
                    inputs.push(Select {
                        columns,
                        from,
                        ..Select::default()
                    });
                }
                let width = width
                    .ok_or_else(|| AdError::InvalidPlan("a set operation has no inputs".into()))?;
                let alias = self.alias();
                let select = Select {
                    columns: positions(alias, width),
                    from: From::Set { op, inputs, alias },
                    ..Select::default()
                };
                (select, s.common.as_ref(), none())
            }
            RelType::Window(w) => {
                let Placed { from, columns, .. } = next(self);
                let env = env(&columns);
                let partition_by = self.exprs(&w.partition_expressions, &env)?;
                let order_by = self.orders(&w.sorts, &env)?;
                let mut out = columns.clone();
                for f in &w.window_functions {
                    out.push(self.window(
                        &WindowParts {
                            function: f.function_reference,
                            arguments: &f.arguments,
                            invocation: f.invocation,
                            bounds_type: f.bounds_type,
                            lower: f.lower_bound.as_ref(),
                            upper: f.upper_bound.as_ref(),
                        },
                        partition_by.clone(),
                        order_by.clone(),
                        &env,
                    )?);
                }
                let select = Select {
                    columns: out,
                    from,
                    ..Select::default()
                };
                (select, w.common.as_ref(), none())
            }
            RelType::Reference(r) => return self.reference(r.subtree_ordinal, outer, placement),
            other => {
                return Err(AdError::NotImplemented(format!(
                    "writing a {} relation as SQL",
                    rel_name(other)
                )))
            }
        };
        let mut select = emitted(select, common)?;
        let width = select.columns.len();
        // Written as this query's ORDER BY, which orders its rows if it is
        // the whole query or picks rows; and kept as columns for a reader.
        select.order_by = order.clone();
        let order = carry(&mut select, order);
        Ok(Source {
            origin: Origin::Select(Box::new(select)),
            width,
            order,
        })
    }

    /// The shared relation at `ordinal` of the plan's relations: one CTE,
    /// read wherever it is shared, unless it is inside a subquery.
    fn reference(
        &mut self,
        ordinal: i32,
        outer: Option<&Env>,
        placement: Placement,
    ) -> Result<Source> {
        if placement == Placement::Cte {
            if let Some((index, width, order)) = self.shared.get(&ordinal) {
                return Ok(Source {
                    origin: Origin::Cte(*index),
                    width: *width,
                    order: order.clone(),
                });
            }
        }
        let plan = self.plan;
        let rel = usize::try_from(ordinal)
            .ok()
            .and_then(|i| plan.relations.get(i))
            .and_then(|r| match &r.rel_type {
                Some(PlanRelType::Rel(rel)) => Some(rel),
                Some(PlanRelType::Root(root)) => root.input.as_ref(),
                None => None,
            })
            .ok_or_else(|| {
                AdError::InvalidPlan(format!(
                    "a reference names relation {ordinal}, which the plan does not have"
                ))
            })?;
        // A relation that reads itself, directly or through others, has no
        // SQL: refused, rather than read until the stack runs out. Shared
        // relations nest shallowly otherwise, so this recursion is bounded
        // by how many read each other, not by depth.
        if !self.reading.insert(ordinal) {
            return Err(AdError::InvalidPlan(format!(
                "shared relation {ordinal} reads itself"
            )));
        }
        let source = self.relation(rel, outer, placement);
        self.reading.remove(&ordinal);
        let source = source?;
        if placement == Placement::InPlace {
            return Ok(source);
        }
        let index = match source.origin {
            Origin::Cte(index) => index,
            Origin::Select(select) => self.name(select),
        };
        self.shared
            .insert(ordinal, (index, source.width, source.order.clone()));
        Ok(Source {
            origin: Origin::Cte(index),
            ..source
        })
    }

    fn read(&mut self, read: &ReadRel, outer: Option<&Env>) -> Result<Select> {
        let schema = required(read.base_schema.as_ref(), "a read's base schema")?;
        let types: Vec<Option<&SubstraitType>> = match &schema.r#struct {
            Some(s) => s.types.iter().map(Some).collect(),
            None => schema.names.iter().map(|_| None).collect(),
        };
        if types.len() != schema.names.len() {
            return Err(AdError::NotImplemented(
                "writing a read of nested columns as SQL".into(),
            ));
        }
        let alias = self.alias();
        let (from, base) = match &read.read_type {
            Some(ReadType::NamedTable(t)) => {
                let columns = schema
                    .names
                    .iter()
                    .map(|n| Expr::Column {
                        table: alias,
                        column: Column::Named(n.clone()),
                    })
                    .collect();
                let from = From::Table {
                    name: t.names.clone(),
                    alias,
                };
                (from, columns)
            }
            Some(ReadType::VirtualTable(v)) => {
                let types: Vec<Type> = types
                    .iter()
                    .map(|t| ty(required(*t, "a virtual table's column type")?))
                    .collect::<Result<_>>()?;
                let empty = Env {
                    columns: Vec::new(),
                    outer,
                };
                let mut rows: Vec<Vec<Expr>> = Vec::new();
                #[allow(deprecated)]
                for row in &v.values {
                    let values = row
                        .fields
                        .iter()
                        .map(|l| literal(l).map(Expr::Literal))
                        .collect::<Result<_>>()?;
                    rows.push(values);
                }
                for row in &v.expressions {
                    rows.push(self.exprs(&row.fields, &empty)?);
                }
                if rows.iter().any(|r| r.len() != types.len()) {
                    return Err(AdError::InvalidPlan(
                        "a virtual table's row has the wrong width".into(),
                    ));
                }
                // Each value cast to its column's type, which VALUES would
                // otherwise infer from the rows.
                let typed = |value: Expr, ty: Type| Expr::Cast {
                    expr: Box::new(value),
                    ty,
                    fallible: false,
                };
                let from = if rows.is_empty() {
                    // VALUES cannot write no rows: one row of typed NULLs,
                    // filtered out.
                    let nulls = types
                        .iter()
                        .map(|t| typed(Expr::Literal(Literal::Null(*t)), *t))
                        .collect();
                    let query = Box::new(Select {
                        columns: nulls,
                        filter: vec![Expr::Literal(Literal::Bool(false))],
                        ..Select::default()
                    });
                    From::Subquery { query, alias }
                } else if types.is_empty() {
                    // Nor no columns: a placeholder in each row, which
                    // nothing reads, keeps the row count.
                    let one = || vec![Expr::Literal(Literal::Int(1, Type::Integer))];
                    let rows = rows.iter().map(|_| one()).collect();
                    From::Values { rows, alias }
                } else {
                    let rows = rows
                        .into_iter()
                        .map(|r| {
                            r.into_iter()
                                .zip(&types)
                                .map(|(v, t)| typed(v, *t))
                                .collect()
                        })
                        .collect();
                    From::Values { rows, alias }
                };
                (from, positions(alias, types.len()))
            }
            _ => {
                return Err(AdError::NotImplemented(
                    "writing a read of anything but a named or virtual table as SQL".into(),
                ))
            }
        };
        let env = Env {
            columns: base.clone(),
            outer,
        };
        let mut filter = Vec::new();
        for f in [read.filter.as_deref(), read.best_effort_filter.as_deref()]
            .into_iter()
            .flatten()
        {
            filter.push(self.expr(f, &env)?);
        }
        let columns = match &read.projection {
            None => base,
            Some(mask) => {
                let items = mask
                    .select
                    .as_ref()
                    .map_or(&[][..], |s| &s.struct_items[..]);
                items
                    .iter()
                    .map(|item| {
                        if item.child.is_some() {
                            return Err(AdError::NotImplemented(
                                "writing a read's nested projection as SQL".into(),
                            ));
                        }
                        usize::try_from(item.field)
                            .ok()
                            .and_then(|i| base.get(i).cloned())
                            .ok_or_else(|| {
                                AdError::InvalidPlan(format!(
                                    "a read projects field {}, past its schema",
                                    item.field
                                ))
                            })
                    })
                    .collect::<Result<_>>()?
            }
        };
        Ok(Select {
            columns,
            from,
            filter,
            ..Select::default()
        })
    }

    fn join(
        &mut self,
        Placed {
            from: left,
            columns: left_columns,
            ..
        }: Placed,
        Placed {
            from: right,
            columns: right_columns,
            ..
        }: Placed,
        condition: Option<&Expression>,
        post: Option<&Expression>,
        kind: JoinType,
        outer: Option<&Env>,
    ) -> Result<Select> {
        let env = Env {
            columns: left_columns.iter().chain(&right_columns).cloned().collect(),
            outer,
        };
        let on = condition.map(|c| self.expr(c, &env)).transpose()?;
        let keyword = match kind {
            JoinType::Inner => Some(JoinKind::Inner),
            JoinType::Left => Some(JoinKind::Left),
            JoinType::Right => Some(JoinKind::Right),
            JoinType::Outer => Some(JoinKind::Full),
            _ => None,
        };
        if let Some(keyword) = keyword {
            let filter = post.map(|p| self.expr(p, &env)).transpose()?;
            let from = From::Join {
                left: Box::new(left),
                right: Box::new(right),
                kind: keyword,
                on: Some(on.unwrap_or(Expr::Literal(Literal::Bool(true)))),
            };
            return Ok(Select {
                columns: env.columns,
                from,
                filter: filter.into_iter().collect(),
                ..Select::default()
            });
        }
        if post.is_some() {
            return Err(AdError::NotImplemented(format!(
                "writing a {} join's post-join filter as SQL",
                kind.as_str_name()
            )));
        }
        // A join that keeps one side's rows: an EXISTS over the other side,
        // whose select needs no columns.
        let exists = |other: From| {
            Expr::Exists(Box::new(Select {
                columns: Vec::new(),
                from: other,
                filter: on.clone().into_iter().collect(),
                ..Select::default()
            }))
        };
        // A mark join's column: whether a row has a match. As a count, not
        // `EXISTS`: DataFusion 55 cannot run an `EXISTS` as a column.
        let matched = |other: From| Expr::Op {
            op: Op::Gt,
            args: vec![
                Expr::Scalar(Box::new(Select {
                    columns: vec![Expr::Call(Call {
                        function: "count",
                        args: Vec::new(),
                        distinct: false,
                        order_by: Vec::new(),
                        filter: None,
                    })],
                    from: other,
                    filter: on.clone().into_iter().collect(),
                    ..Select::default()
                })),
                Expr::Literal(Literal::Int(0, Type::BigInt)),
            ],
        };
        let not = |e: Expr| Expr::Not(Box::new(e));
        let (from, mut columns, test, mark) = match kind {
            JoinType::LeftSemi => (left, left_columns, exists(right), false),
            JoinType::LeftAnti => (left, left_columns, not(exists(right)), false),
            JoinType::RightSemi => (right, right_columns, exists(left), false),
            JoinType::RightAnti => (right, right_columns, not(exists(left)), false),
            JoinType::LeftMark => (left, left_columns, matched(right), true),
            JoinType::RightMark => (right, right_columns, matched(left), true),
            other => {
                return Err(AdError::NotImplemented(format!(
                    "writing a {} join as SQL",
                    other.as_str_name()
                )))
            }
        };
        // A mark join keeps every row, with the test as a column.
        let filter = if mark {
            columns.push(test);
            Vec::new()
        } else {
            vec![test]
        };
        Ok(Select {
            columns,
            from,
            filter,
            ..Select::default()
        })
    }

    fn aggregate(
        &mut self,
        a: &AggregateRel,
        Placed { from, columns, .. }: Placed,
        outer: Option<&Env>,
    ) -> Result<Select> {
        if a.groupings.len() > 1 {
            return Err(AdError::NotImplemented(
                "writing an aggregate with several grouping sets as SQL".into(),
            ));
        }
        let env = Env { columns, outer };
        // The groupings: the top-level list the current spec uses, or the
        // deprecated per-grouping copy.
        let keys: Vec<&Expression> = match a.groupings.first() {
            None => Vec::new(),
            Some(g) if !a.grouping_expressions.is_empty() => g
                .expression_references
                .iter()
                .map(|&i| {
                    a.grouping_expressions.get(i as usize).ok_or_else(|| {
                        AdError::InvalidPlan(format!("a grouping refers to expression {i}"))
                    })
                })
                .collect::<Result<_>>()?,
            #[allow(deprecated)]
            Some(g) => g.grouping_expressions.iter().collect(),
        };
        let keys = keys
            .into_iter()
            .map(|k| self.expr(k, &env))
            .collect::<Result<Vec<_>>>()?;
        // Its output has each distinct key once, as DataFusion's aggregate
        // has: `GROUP BY g, g` gives one `g` (DataFusion writes both).
        let mut distinct: Vec<Expr> = Vec::new();
        for k in &keys {
            if !distinct.contains(k) {
                distinct.push(k.clone());
            }
        }
        let mut out = distinct;
        for m in &a.measures {
            let f = required(m.measure.as_ref(), "a measure's function")?;
            let call = Call {
                function: self.call_name(f.function_reference)?,
                args: self.arguments(&f.arguments, &env)?,
                distinct: f.invocation == AggregationInvocation::Distinct as i32,
                order_by: self.orders(&f.sorts, &env)?,
                filter: m
                    .filter
                    .as_ref()
                    .map(|c| self.expr(c, &env).map(Box::new))
                    .transpose()?,
            };
            out.push(Expr::Call(call));
        }
        Ok(Select {
            columns: out,
            from,
            group_by: Some(keys),
            ..Select::default()
        })
    }

    fn exprs(&mut self, es: &[Expression], env: &Env) -> Result<Vec<Expr>> {
        es.iter().map(|e| self.expr(e, env)).collect()
    }

    fn orders(&mut self, sorts: &[SortField], env: &Env) -> Result<Vec<Order>> {
        sorts
            .iter()
            .map(|s| {
                let expr =
                    self.expr(required(s.expr.as_ref(), "a sort field's expression")?, env)?;
                let (descending, nulls_first) = match s.sort_kind {
                    Some(SortKind::Direction(d)) => {
                        match SortDirection::try_from(d).unwrap_or(SortDirection::Unspecified) {
                            SortDirection::AscNullsFirst => (false, true),
                            SortDirection::AscNullsLast => (false, false),
                            SortDirection::DescNullsFirst => (true, true),
                            SortDirection::DescNullsLast => (true, false),
                            other => {
                                return Err(AdError::NotImplemented(format!(
                                    "writing the sort direction {} as SQL",
                                    other.as_str_name()
                                )))
                            }
                        }
                    }
                    _ => {
                        return Err(AdError::NotImplemented(
                            "writing a sort by a comparison function as SQL".into(),
                        ))
                    }
                };
                Ok(Order {
                    expr,
                    descending,
                    nulls_first,
                })
            })
            .collect()
    }

    /// An expression. Recursive: unlike a plan's relations, its expressions
    /// nest only as deep as a query writes them, and ddx's own steps are cut
    /// before protobuf's nesting limit (`opt::split`).
    fn expr(&mut self, e: &Expression, env: &Env) -> Result<Expr> {
        let rex = required(e.rex_type.as_ref(), "an expression's type")?;
        Ok(match rex {
            RexType::Literal(l) => Expr::Literal(literal(l)?),
            RexType::Selection(r) => {
                let field = match &r.reference_type {
                    Some(ReferenceType::DirectReference(s)) => match &s.reference_type {
                        Some(Segment::StructField(f)) if f.child.is_none() => f.field,
                        _ => {
                            return Err(AdError::NotImplemented(
                                "writing a reference into a nested column as SQL".into(),
                            ))
                        }
                    },
                    _ => {
                        return Err(AdError::NotImplemented(
                            "writing a masked field reference as SQL".into(),
                        ))
                    }
                };
                let steps_out = match &r.root_type {
                    None | Some(RootType::RootReference(_)) => 0,
                    Some(RootType::OuterReference(o)) => o.steps_out as usize,
                    Some(_) => {
                        return Err(AdError::NotImplemented(
                            "writing a field of an expression or a lambda as SQL".into(),
                        ))
                    }
                };
                env.field(steps_out, field)?
            }
            RexType::ScalarFunction(f) => {
                let args = self.arguments(&f.arguments, env)?;
                let integer = f.output_type.as_ref().is_some_and(is_integer);
                self.scalar(f.function_reference, args, integer)?
            }
            RexType::WindowFunction(f) => {
                let partition_by = self.exprs(&f.partitions, env)?;
                let order_by = self.orders(&f.sorts, env)?;
                self.window(
                    &WindowParts {
                        function: f.function_reference,
                        arguments: &f.arguments,
                        invocation: f.invocation,
                        bounds_type: f.bounds_type,
                        lower: f.lower_bound.as_ref(),
                        upper: f.upper_bound.as_ref(),
                    },
                    partition_by,
                    order_by,
                    env,
                )?
            }
            RexType::IfThen(i) => {
                // DataFusion's producer writes `CASE x WHEN v THEN …` as a
                // first clause holding `x` and no result.
                let (operand, clauses) = match i.ifs.split_first() {
                    Some((first, rest)) if first.then.is_none() => {
                        let x = required(first.r#if.as_ref(), "a CASE operand")?;
                        (Some(Box::new(self.expr(x, env)?)), rest)
                    }
                    _ => (None, &i.ifs[..]),
                };
                let whens = clauses
                    .iter()
                    .map(|c| {
                        let when = required(c.r#if.as_ref(), "a CASE clause's condition")?;
                        let then = required(c.then.as_ref(), "a CASE clause's result")?;
                        Ok((self.expr(when, env)?, self.expr(then, env)?))
                    })
                    .collect::<Result<_>>()?;
                let otherwise = i
                    .r#else
                    .as_deref()
                    .map(|x| self.expr(x, env).map(Box::new))
                    .transpose()?;
                Expr::Case {
                    operand,
                    whens,
                    otherwise,
                }
            }
            RexType::SwitchExpression(s) => {
                let value = required(s.r#match.as_deref(), "a switch's value")?;
                let whens = s
                    .ifs
                    .iter()
                    .map(|c| {
                        let when = required(c.r#if.as_ref(), "a switch clause's value")?;
                        let then = required(c.then.as_ref(), "a switch clause's result")?;
                        Ok((Expr::Literal(literal(when)?), self.expr(then, env)?))
                    })
                    .collect::<Result<_>>()?;
                let otherwise = s
                    .r#else
                    .as_deref()
                    .map(|x| self.expr(x, env).map(Box::new))
                    .transpose()?;
                Expr::Case {
                    operand: Some(Box::new(self.expr(value, env)?)),
                    whens,
                    otherwise,
                }
            }
            RexType::SingularOrList(s) => {
                let value = self.expr(required(s.value.as_deref(), "an IN list's value")?, env)?;
                if s.options.is_empty() {
                    return Ok(Expr::Literal(Literal::Bool(false)));
                }
                Expr::InList {
                    expr: Box::new(value),
                    list: self.exprs(&s.options, env)?,
                }
            }
            RexType::MultiOrList(m) => {
                // (a, b) IN ((x, y), …), as ORs of ANDs, which every dialect
                // reads.
                let values = self.exprs(&m.value, env)?;
                let mut options = Vec::new();
                for record in &m.options {
                    if record.fields.len() != values.len() {
                        return Err(AdError::InvalidPlan(
                            "a multi-column IN list's record has the wrong width".into(),
                        ));
                    }
                    let terms = values
                        .iter()
                        .zip(self.exprs(&record.fields, env)?)
                        .map(|(v, f)| Expr::Op {
                            op: Op::Eq,
                            args: vec![v.clone(), f],
                        })
                        .collect();
                    options.push(and(terms));
                }
                if options.is_empty() {
                    return Ok(Expr::Literal(Literal::Bool(false)));
                }
                or(options)
            }
            RexType::Cast(c) => Expr::Cast {
                expr: Box::new(self.expr(required(c.input.as_deref(), "a cast's input")?, env)?),
                ty: ty(required(c.r#type.as_ref(), "a cast's type")?)?,
                fallible: c.failure_behavior == cast::FailureBehavior::ReturnNull as i32,
            },
            RexType::Subquery(s) => self.subquery(s, env)?,
            other => {
                return Err(AdError::NotImplemented(format!(
                    "writing a {} expression as SQL",
                    crate::expr::rex_name(other)
                )))
            }
        })
    }

    /// A subquery's relation, read in place with `env` enclosing it.
    fn subquery_select(&mut self, rel: Option<&Rel>, env: &Env) -> Result<Box<Select>> {
        let rel = required(rel, "a subquery's relation")?;
        let source = self.relation(rel, Some(env), Placement::InPlace)?;
        Ok(Box::new(self.select_of(source)))
    }

    fn subquery(&mut self, s: &Subquery, env: &Env) -> Result<Expr> {
        let kind = required(s.subquery_type.as_ref(), "a subquery's type")?;
        Ok(match kind {
            SubqueryType::Scalar(q) => Expr::Scalar(self.subquery_select(q.input.as_deref(), env)?),
            SubqueryType::InPredicate(p) => {
                let [needle] = &p.needles[..] else {
                    return Err(AdError::NotImplemented(
                        "writing an IN subquery of several columns as SQL".into(),
                    ));
                };
                Expr::InQuery {
                    expr: Box::new(self.expr(needle, env)?),
                    query: self.subquery_select(p.haystack.as_deref(), env)?,
                }
            }
            SubqueryType::SetPredicate(p) => {
                if p.predicate_op != PredicateOp::Exists as i32 {
                    return Err(AdError::NotImplemented(
                        "writing a UNIQUE subquery as SQL".into(),
                    ));
                }
                Expr::Exists(self.subquery_select(p.tuples.as_deref(), env)?)
            }
            SubqueryType::SetComparison(c) => {
                let all = match ReductionOp::try_from(c.reduction_op) {
                    Ok(ReductionOp::Any) => false,
                    Ok(ReductionOp::All) => true,
                    _ => {
                        return Err(AdError::InvalidPlan(
                            "a set comparison has no reduction".into(),
                        ))
                    }
                };
                let op = match ComparisonOp::try_from(c.comparison_op) {
                    Ok(ComparisonOp::Eq) => Op::Eq,
                    Ok(ComparisonOp::Ne) => Op::NotEq,
                    Ok(ComparisonOp::Lt) => Op::Lt,
                    Ok(ComparisonOp::Gt) => Op::Gt,
                    Ok(ComparisonOp::Le) => Op::Lte,
                    Ok(ComparisonOp::Ge) => Op::Gte,
                    _ => {
                        return Err(AdError::InvalidPlan(
                            "a set comparison has no comparison".into(),
                        ))
                    }
                };
                let left = required(c.left.as_deref(), "a set comparison's left side")?;
                Expr::Quantified {
                    left: Box::new(self.expr(left, env)?),
                    op,
                    all,
                    query: self.subquery_select(c.right.as_deref(), env)?,
                }
            }
        })
    }

    fn arguments(&mut self, args: &[FunctionArgument], env: &Env) -> Result<Vec<Expr>> {
        args.iter()
            .map(|a| match &a.arg_type {
                Some(ArgType::Value(e)) => self.expr(e, env),
                _ => Err(AdError::NotImplemented(
                    "writing a function's type or enum argument as SQL".into(),
                )),
            })
            .collect()
    }

    /// The canonical name of the function `anchor` names, which must be one
    /// written as a call (an aggregate or a window function).
    fn call_name(&self, anchor: u32) -> Result<&'static str> {
        let name = self.functions.name(anchor)?;
        match syntax(name) {
            Some(Syntax::Function(f)) => Ok(f),
            _ => Err(unknown_function(name)),
        }
    }

    /// The scalar function `anchor` of `args`. `integer`: its output is an
    /// integer, which makes a `divide` integer division.
    fn scalar(&self, anchor: u32, mut args: Vec<Expr>, integer: bool) -> Result<Expr> {
        let name = self.functions.name(anchor)?;
        let syntax = syntax(name).ok_or_else(|| unknown_function(name))?;
        let one = |args: &mut Vec<Expr>| match args.len() {
            1 => Ok(Box::new(args.remove(0))),
            n => Err(AdError::InvalidPlan(format!(
                "`{name}` takes one argument, not {n}"
            ))),
        };
        Ok(match syntax {
            Syntax::Operator(op) => {
                if args.len() < 2 {
                    return Err(AdError::InvalidPlan(format!(
                        "`{name}` takes two arguments or more, not {}",
                        args.len()
                    )));
                }
                let op = match op {
                    Op::Divide { .. } => Op::Divide { integer },
                    op => op,
                };
                Expr::Op { op, args }
            }
            Syntax::Not => Expr::Not(one(&mut args)?),
            Syntax::Negate => Expr::Negate(one(&mut args)?),
            Syntax::Is(test) => Expr::Is {
                expr: one(&mut args)?,
                test,
            },
            Syntax::Identity => *one(&mut args)?,
            Syntax::Function(function) => Expr::Call(Call {
                function,
                args,
                distinct: false,
                order_by: Vec::new(),
                filter: None,
            }),
        })
    }

    fn window(
        &mut self,
        w: &WindowParts<'_>,
        partition_by: Vec<Expr>,
        order_by: Vec<Order>,
        env: &Env,
    ) -> Result<Expr> {
        let function = self.call_name(w.function)?;
        let frame = if RANKING.contains(&function) {
            None
        } else {
            // An unspecified bounds type is read as DataFusion's consumer
            // reads it.
            let units = match BoundsType::try_from(w.bounds_type).unwrap_or(BoundsType::Unspecified)
            {
                BoundsType::Rows => Units::Rows,
                BoundsType::Range => Units::Range,
                BoundsType::Unspecified if order_by.is_empty() => Units::Rows,
                BoundsType::Unspecified => Units::Range,
            };
            Some(Frame {
                units,
                lower: bound(w.lower, true),
                upper: bound(w.upper, false),
            })
        };
        Ok(Expr::Window {
            call: Call {
                function,
                args: self.arguments(w.arguments, env)?,
                distinct: w.invocation == AggregationInvocation::Distinct as i32,
                order_by: Vec::new(),
                filter: None,
            },
            partition_by,
            order_by,
            frame,
        })
    }
}

/// The relations whose SQL a relation is made from, in order: its inputs.
/// A shared relation is read by [`Reader::reference`], not as an input.
fn inputs<'r>(rel: &'r Rel, found: &mut Vec<&'r Rel>) -> Result<()> {
    let one = |input: &'r Option<Box<Rel>>| required(input.as_deref(), "a relation's input");
    match &rel.rel_type {
        Some(RelType::Project(p)) => found.push(one(&p.input)?),
        Some(RelType::Filter(f)) => found.push(one(&f.input)?),
        Some(RelType::Sort(s)) => found.push(one(&s.input)?),
        Some(RelType::Aggregate(a)) => found.push(one(&a.input)?),
        Some(RelType::Window(w)) => found.push(one(&w.input)?),
        Some(RelType::Fetch(f)) => found.push(one(&f.input)?),
        Some(RelType::Join(j)) => {
            found.push(one(&j.left)?);
            found.push(one(&j.right)?);
        }
        Some(RelType::Cross(c)) => {
            found.push(one(&c.left)?);
            found.push(one(&c.right)?);
        }
        Some(RelType::Set(s)) => found.extend(&s.inputs),
        // A read has no input, a shared relation is read by its position,
        // and anything else is refused when it is reached.
        _ => {}
    }
    Ok(())
}

/// A fetch's `OFFSET` and `LIMIT`, if it has them. The deprecated integer
/// fields are read too, as older producers write them; a count of -1 is
/// "all", as they wrote it.
#[allow(deprecated)]
fn fetch(f: &FetchRel) -> Result<(Option<i64>, Option<i64>)> {
    let offset = match &f.offset_mode {
        Some(OffsetMode::Offset(n)) => Some(*n),
        Some(OffsetMode::OffsetExpr(e)) => count(e)?,
        None => None,
    };
    let limit = match &f.count_mode {
        Some(CountMode::Count(n)) => Some(*n),
        Some(CountMode::CountExpr(e)) => count(e)?,
        None => None,
    };
    Ok((offset.filter(|n| *n > 0), limit.filter(|n| *n >= 0)))
}

/// A fetch's offset or count, which must be a literal integer.
fn count(e: &Expression) -> Result<Option<i64>> {
    match &e.rex_type {
        Some(RexType::Literal(SubstraitLiteral {
            literal_type: Some(t),
            ..
        })) => match t {
            LiteralType::I8(n) | LiteralType::I16(n) | LiteralType::I32(n) => {
                Ok(Some(i64::from(*n)))
            }
            LiteralType::I64(n) => Ok(Some(*n)),
            LiteralType::Null(_) => Ok(None),
            _ => Err(AdError::NotImplemented(
                "writing a fetch whose count is not an integer as SQL".into(),
            )),
        },
        _ => Err(AdError::NotImplemented(
            "writing a fetch whose count is not a literal as SQL".into(),
        )),
    }
}

/// `select` with its columns picked by the relation's emit.
fn emitted(mut select: Select, common: Option<&RelCommon>) -> Result<Select> {
    let Some(EmitKind::Emit(emit)) = common.and_then(|c| c.emit_kind.as_ref()) else {
        return Ok(select);
    };
    select.columns = emit
        .output_mapping
        .iter()
        .map(|&i| {
            select.columns.get(i as usize).cloned().ok_or_else(|| {
                AdError::InvalidPlan(format!(
                    "an emit picks column {i} of {}",
                    select.columns.len()
                ))
            })
        })
        .collect::<Result<_>>()?;
    Ok(select)
}

/// `order`, keys over `select`'s `FROM` item, as positions of `select`'s
/// columns: each key one of its columns, or appended after them, a column
/// only a reader that writes the order reads.
fn carry(select: &mut Select, order: Vec<Order>) -> Vec<Sorted> {
    order
        .into_iter()
        .map(|o| {
            let column = match select.columns.iter().position(|c| *c == o.expr) {
                Some(i) => i,
                None => {
                    select.columns.push(o.expr);
                    select.columns.len() - 1
                }
            };
            Sorted {
                column,
                descending: o.descending,
                nulls_first: o.nulls_first,
            }
        })
        .collect()
}

/// The columns `c0` to `c{width - 1}` of the item read under `alias`.
fn positions(alias: Alias, width: usize) -> Vec<Expr> {
    (0..width)
        .map(|i| Expr::Column {
            table: alias,
            column: Column::Position(i),
        })
        .collect()
}

fn required<'a, T: ?Sized>(x: Option<&'a T>, what: &str) -> Result<&'a T> {
    x.ok_or_else(|| AdError::InvalidPlan(format!("{what} is missing")))
}

fn and(terms: Vec<Expr>) -> Expr {
    match <[Expr; 1]>::try_from(terms) {
        Ok([one]) => one,
        Err(terms) => Expr::Op {
            op: Op::And,
            args: terms,
        },
    }
}

fn or(terms: Vec<Expr>) -> Expr {
    match <[Expr; 1]>::try_from(terms) {
        Ok([one]) => one,
        Err(terms) => Expr::Op {
            op: Op::Or,
            args: terms,
        },
    }
}

/// A window function's parts, from either a window expression or a window
/// relation.
struct WindowParts<'a> {
    function: u32,
    arguments: &'a [FunctionArgument],
    invocation: i32,
    bounds_type: i32,
    lower: Option<&'a SubstraitBound>,
    upper: Option<&'a SubstraitBound>,
}

/// Window functions that take no frame.
const RANKING: &[&str] = &[
    "row_number",
    "rank",
    "dense_rank",
    "percent_rank",
    "cume_dist",
    "ntile",
    "lag",
    "lead",
];

fn bound(b: Option<&SubstraitBound>, lower: bool) -> Bound {
    match b.and_then(|b| b.kind.as_ref()) {
        None | Some(BoundKind::Unbounded(_)) if lower => Bound::UnboundedPreceding,
        None | Some(BoundKind::Unbounded(_)) => Bound::UnboundedFollowing,
        Some(BoundKind::CurrentRow(_)) => Bound::CurrentRow,
        Some(BoundKind::Preceding(p)) => Bound::Preceding(p.offset),
        Some(BoundKind::Following(f)) => Bound::Following(f.offset),
    }
}

/// How a function is written in SQL: as an operator, or a call.
enum Syntax {
    Operator(Op),
    Not,
    Negate,
    Is(IsTest),
    /// The function's argument itself: ddx's stop-gradient, the identity
    /// at run time.
    Identity,
    /// A function call, by its canonical name.
    Function(&'static str),
}

/// How the function with the normalized name `name` is written, or `None`
/// if ddx does not know it.
fn syntax(name: &str) -> Option<Syntax> {
    use Syntax::*;
    Some(match name {
        "add" => Operator(ir::Op::Add),
        "subtract" => Operator(ir::Op::Subtract),
        "multiply" => Operator(ir::Op::Multiply),
        "divide" => Operator(ir::Op::Divide { integer: false }),
        "modulus" | "mod" => Operator(ir::Op::Modulus),
        "equal" => Operator(ir::Op::Eq),
        "not_equal" => Operator(ir::Op::NotEq),
        "lt" => Operator(ir::Op::Lt),
        "lte" => Operator(ir::Op::Lte),
        "gt" => Operator(ir::Op::Gt),
        "gte" => Operator(ir::Op::Gte),
        "and" => Operator(ir::Op::And),
        "or" => Operator(ir::Op::Or),
        "is_distinct_from" => Operator(ir::Op::IsDistinctFrom),
        "is_not_distinct_from" => Operator(ir::Op::IsNotDistinctFrom),
        "not" => Not,
        "negate" | "negative" => Negate,
        "is_null" => Is(IsTest::Null),
        "is_not_null" => Is(IsTest::NotNull),
        "is_true" => Is(IsTest::True),
        "is_false" => Is(IsTest::False),
        "is_not_true" => Is(IsTest::NotTrue),
        "is_not_false" => Is(IsTest::NotFalse),
        STOP_GRADIENT => Identity,
        "isnan" | "is_nan" => Function("isnan"),
        "sign" | "signum" => Function("sign"),
        _ => return call_name(name).map(Function),
    })
}

/// Scalar functions named alike in every dialect, by their canonical names.
const SCALARS: &[&str] = &[
    "abs", "sqrt", "cbrt", "exp", "ln", "log10", "log2", "power", "sin", "cos", "tan", "asin",
    "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "floor", "ceil",
    "round", "trunc", "greatest", "least", "coalesce", "nullif", "pi", "degrees", "radians",
    "random",
];

/// Aggregate functions named alike in every dialect.
const AGGREGATES: &[&str] = &[
    "sum",
    "count",
    "avg",
    "max",
    "min",
    "stddev",
    "stddev_samp",
    "stddev_pop",
    "var_samp",
    "var_pop",
    "bool_and",
    "bool_or",
];

/// Window functions named alike in every dialect.
const WINDOWS: &[&str] = &[
    "row_number",
    "rank",
    "dense_rank",
    "percent_rank",
    "cume_dist",
    "ntile",
    "lag",
    "lead",
    "first_value",
    "last_value",
    "nth_value",
];

/// The canonical name of a function named alike in every dialect.
fn call_name(name: &str) -> Option<&'static str> {
    [SCALARS, AGGREGATES, WINDOWS]
        .into_iter()
        .flatten()
        .copied()
        .find(|&f| f == name)
}

fn unknown_function(name: &str) -> AdError {
    AdError::NotImplemented(format!(
        "writing the function `{name}` as SQL: ddx does not know its name in SQL"
    ))
}

fn literal(l: &SubstraitLiteral) -> Result<Literal> {
    let t = required(l.literal_type.as_ref(), "a literal's value")?;
    Ok(match t {
        LiteralType::Boolean(b) => Literal::Bool(*b),
        LiteralType::I8(n) => Literal::Int(i64::from(*n), Type::TinyInt),
        LiteralType::I16(n) => Literal::Int(i64::from(*n), Type::SmallInt),
        LiteralType::I32(n) => Literal::Int(i64::from(*n), Type::Integer),
        LiteralType::I64(n) => Literal::Int(*n, Type::BigInt),
        LiteralType::Fp32(x) => Literal::Float(f64::from(*x), Type::Real),
        LiteralType::Fp64(x) => Literal::Float(*x, Type::Double),
        LiteralType::String(s) | LiteralType::FixedChar(s) => Literal::String(s.clone()),
        LiteralType::VarChar(v) => Literal::String(v.value.clone()),
        LiteralType::Date(days) => Literal::Date(i64::from(*days)),
        #[allow(deprecated)]
        LiteralType::Timestamp(micros) => Literal::Timestamp {
            value: *micros,
            precision: 6,
        },
        LiteralType::PrecisionTimestamp(p) => Literal::Timestamp {
            value: p.value,
            precision: timestamp_precision(p.precision)?,
        },
        LiteralType::Decimal(d) => {
            let bytes: [u8; 16] =
                d.value.as_slice().try_into().map_err(|_| {
                    AdError::InvalidPlan("a decimal literal is not 16 bytes".into())
                })?;
            let (precision, scale) = decimal(d.precision, d.scale)?;
            Literal::Decimal {
                value: i128::from_le_bytes(bytes),
                precision,
                scale,
            }
        }
        LiteralType::Null(t) => Literal::Null(ty(t)?),
        other => {
            return Err(AdError::NotImplemented(format!(
                "writing a {} literal as SQL",
                literal_name(other)
            )))
        }
    })
}

fn ty(t: &SubstraitType) -> Result<Type> {
    let kind = required(t.kind.as_ref(), "a type's kind")?;
    Ok(match kind {
        Kind::Bool(_) => Type::Boolean,
        Kind::I8(_) => Type::TinyInt,
        Kind::I16(_) => Type::SmallInt,
        Kind::I32(_) => Type::Integer,
        Kind::I64(_) => Type::BigInt,
        Kind::Fp32(_) => Type::Real,
        Kind::Fp64(_) => Type::Double,
        Kind::String(_) | Kind::Varchar(_) | Kind::FixedChar(_) => Type::Varchar,
        Kind::Date(_) => Type::Date,
        #[allow(deprecated)]
        Kind::Timestamp(_) => Type::Timestamp { precision: 6 },
        Kind::PrecisionTimestamp(p) => Type::Timestamp {
            precision: timestamp_precision(p.precision)?,
        },
        Kind::Decimal(d) => {
            let (precision, scale) = decimal(d.precision, d.scale)?;
            Type::Decimal { precision, scale }
        }
        Kind::Binary(_) => Type::Binary,
        _ => {
            return Err(AdError::NotImplemented(
                "writing a nested, interval or user-defined type as SQL".into(),
            ))
        }
    })
}

/// A timestamp's precision, which SQL types write only as seconds,
/// milliseconds, microseconds or nanoseconds; Substrait's go to
/// picoseconds.
fn timestamp_precision(precision: i32) -> Result<i32> {
    match precision {
        0 | 3 | 6 | 9 => Ok(precision),
        _ => Err(AdError::NotImplemented(format!(
            "writing a timestamp of precision {precision} as SQL: only 0, 3, 6 and 9 have SQL types"
        ))),
    }
}

/// A decimal's precision and scale, which Substrait bounds by
/// 1 ≤ precision ≤ 38 and 0 ≤ scale ≤ precision. A negative scale (a
/// multiple of a power of ten) has no SQL type in either dialect.
fn decimal(precision: i32, scale: i32) -> Result<(i32, i32)> {
    if scale < 0 {
        return Err(AdError::NotImplemented(format!(
            "writing a decimal of negative scale {scale} as SQL"
        )));
    }
    if !(1..=38).contains(&precision) || scale > precision {
        return Err(AdError::InvalidPlan(format!(
            "a decimal of precision {precision} and scale {scale}, outside Substrait's 1 to 38 and 0 to the precision"
        )));
    }
    Ok((precision, scale))
}

fn is_integer(t: &SubstraitType) -> bool {
    matches!(
        t.kind,
        Some(Kind::I8(_)) | Some(Kind::I16(_)) | Some(Kind::I32(_)) | Some(Kind::I64(_))
    )
}

fn rel_name(r: &RelType) -> &'static str {
    match r {
        RelType::ExtensionLeaf(_) | RelType::ExtensionSingle(_) | RelType::ExtensionMulti(_) => {
            "extension"
        }
        RelType::Write(_) => "write",
        RelType::Ddl(_) => "DDL",
        RelType::Update(_) => "update",
        RelType::HashJoin(_) | RelType::MergeJoin(_) | RelType::NestedLoopJoin(_) => {
            "physical join"
        }
        RelType::Expand(_) => "expand",
        RelType::Exchange(_) => "exchange",
        _ => "physical",
    }
}

#[allow(deprecated)]
fn literal_name(l: &LiteralType) -> &'static str {
    match l {
        LiteralType::Binary(_) | LiteralType::FixedBinary(_) => "binary",
        LiteralType::Time(_) | LiteralType::PrecisionTime(_) => "time",
        LiteralType::IntervalYearToMonth(_)
        | LiteralType::IntervalDayToSecond(_)
        | LiteralType::IntervalCompound(_) => "interval",
        LiteralType::PrecisionTimestampTz(_) | LiteralType::TimestampTz(_) => {
            "timestamp with time zone"
        }
        LiteralType::Struct(_) | LiteralType::List(_) | LiteralType::EmptyList(_) => "nested",
        LiteralType::Map(_) | LiteralType::EmptyMap(_) => "map",
        LiteralType::Uuid(_) => "UUID",
        LiteralType::UserDefined(_) => "user-defined",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_canonical_name_is_one_ddx_writes() {
        assert!(syntax("frobnicate").is_none());
        for name in ["sign", "signum", "isnan", "is_nan", "sum", "row_number"] {
            assert!(matches!(syntax(name), Some(Syntax::Function(_))), "{name}");
        }
        assert!(matches!(syntax(STOP_GRADIENT), Some(Syntax::Identity)));
    }

    #[test]
    fn a_reference_steps_out_through_its_environment() {
        let column = |t| Expr::Column {
            table: Alias(t),
            column: Column::Position(0),
        };
        let outer = Env {
            columns: vec![column(0)],
            outer: None,
        };
        let inner = Env {
            columns: vec![column(1)],
            outer: Some(&outer),
        };
        assert!(matches!(
            inner.field(1, 0),
            Ok(Expr::Column {
                table: Alias(0),
                ..
            })
        ));
        assert!(matches!(
            inner.field(0, 0),
            Ok(Expr::Column {
                table: Alias(1),
                ..
            })
        ));
        assert!(inner.field(2, 0).is_err());
        assert!(inner.field(0, 1).is_err());
    }

    #[test]
    fn a_shared_relation_that_reads_itself_is_refused() {
        use substrait::proto::{PlanRel, ReferenceRel, RelRoot};
        let reference = |ordinal| Rel {
            rel_type: Some(RelType::Reference(ReferenceRel {
                subtree_ordinal: ordinal,
            })),
        };
        // Relation 0 reads relation 1, which reads relation 0.
        let plan = Plan {
            relations: vec![
                PlanRel {
                    rel_type: Some(PlanRelType::Rel(reference(1))),
                },
                PlanRel {
                    rel_type: Some(PlanRelType::Rel(reference(0))),
                },
                PlanRel {
                    rel_type: Some(PlanRelType::Root(RelRoot {
                        input: Some(reference(0)),
                        names: vec![],
                    })),
                },
            ],
            ..Default::default()
        };
        for dialect in [Dialect::DataFusion, Dialect::DuckDb] {
            let err = to_sql(&plan, dialect).unwrap_err();
            assert!(
                matches!(&err, AdError::InvalidPlan(m) if m.contains("reads itself")),
                "{err}"
            );
        }
    }

    #[test]
    fn what_sql_has_no_type_for_is_refused() {
        for precision in [0, 3, 6, 9] {
            assert_eq!(timestamp_precision(precision), Ok(precision));
        }
        for precision in [1, 12] {
            assert!(matches!(
                timestamp_precision(precision),
                Err(AdError::NotImplemented(_))
            ));
        }
        assert!(matches!(decimal(10, -2), Err(AdError::NotImplemented(_))));
        assert_eq!(decimal(38, 38), Ok((38, 38)));
        for (precision, scale) in [(0, 0), (39, 2), (10, 11), (10, i32::MAX)] {
            assert!(
                matches!(decimal(precision, scale), Err(AdError::InvalidPlan(_))),
                "{precision}, {scale}"
            );
        }
    }
}
