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
//!   enclosing query's columns, which a CTE cannot see.
//! - **The plan's meaning is written out, not left to the dialect.** Every
//!   literal is written with its type, so `0.1` is a `DOUBLE` on an engine
//!   that would read it as a `DECIMAL`; a `divide` whose output is an integer
//!   is written as the dialect's integer division; and a window's frame is
//!   always written, since SQL's default frame for an ordered window is not
//!   Substrait's.
//! - **A function is written in the dialect's spelling**, looked up by its
//!   normalized name (`sign` is `signum` on DataFusion). A function with no
//!   spelling in the dialect is refused, naming it, rather than written as a
//!   guess.
//! - **A join that keeps one side's rows** (semi, anti, mark) is a correlated
//!   `EXISTS`, which every engine reads the same way.
//!
//! Anything else (an extension relation, a grouping set, an interval
//! literal) is refused with [`AdError::NotImplemented`].

use std::fmt::Write as _;

use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::expression::field_reference::{ReferenceType, RootType};
use substrait::proto::expression::literal::LiteralType;
use substrait::proto::expression::nested::NestedType;
use substrait::proto::expression::reference_segment::ReferenceType as Segment;
use substrait::proto::expression::subquery::set_comparison::{ComparisonOp, ReductionOp};
use substrait::proto::expression::subquery::set_predicate::PredicateOp;
use substrait::proto::expression::subquery::SubqueryType;
use substrait::proto::expression::window_function::bound::Kind as BoundKind;
use substrait::proto::expression::window_function::{Bound, BoundsType};
use substrait::proto::expression::{cast, Literal, RexType};
use substrait::proto::fetch_rel::{CountMode, OffsetMode};
use substrait::proto::function_argument::ArgType;
use substrait::proto::join_rel::JoinType;
use substrait::proto::plan_rel::RelType as PlanRelType;
use substrait::proto::r#type::Kind;
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::EmitKind;
use substrait::proto::set_rel::SetOp;
use substrait::proto::sort_field::{SortDirection, SortKind};
use substrait::proto::{
    AggregateRel, Expression, FunctionArgument, Plan, ReadRel, Rel, RelCommon, SortField, Type,
};

use crate::error::{AdError, Result};
use crate::functions::{Functions, STOP_GRADIENT};

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

impl Dialect {
    fn name(self) -> &'static str {
        match self {
            Dialect::DataFusion => "DataFusion",
            Dialect::DuckDb => "DuckDB",
        }
    }
}

/// `plan` as one SQL query in `dialect`, its columns named as the plan's
/// root names them.
///
/// The plan's reads must be bound (each with its `base_schema`), as a
/// program's steps are when [`crate::run`] hands them to a [`crate::Backend`].
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
    let mut writer = Writer {
        plan,
        functions: &functions,
        dialect,
        aliases: 0,
        ctes: Vec::new(),
        shared: std::collections::HashMap::new(),
        subqueries: 0,
        scopes: Vec::new(),
    };
    // The root relation is the query's own SELECT, under the root's names,
    // so its ORDER BY and LIMIT order the result.
    let body = match writer.built(input)? {
        Built::Select(select, emit) => finish(select, emit, Some(&root.names))?.0,
        Built::Query(q) => {
            let t = writer.alias();
            let columns = (0..q.width).map(|i| format!("{t}.c{i}")).collect();
            let select = Select {
                columns,
                from: format!("{} AS {t}", q.from),
                ..Select::default()
            };
            finish(select, None, Some(&root.names))?.0
        }
    };
    let mut sql = String::new();
    if !writer.ctes.is_empty() {
        write!(sql, "WITH {} ", writer.ctes.join(", ")).ok();
    }
    sql.push_str(&body);
    Ok(sql)
}

/// A relation being written: a `SELECT` and its emit, or a query already
/// written.
enum Built {
    Select(Select, Option<Vec<usize>>),
    Query(Query),
}

/// A relation written as SQL, with columns `c0` to `c{width - 1}`: a
/// `FROM` item, either the name of its CTE or a parenthesized query.
#[derive(Clone)]
struct Query {
    from: String,
    width: usize,
}

impl Query {
    /// It as a parenthesized query, for a subquery expression.
    fn subquery(&self) -> String {
        if self.from.starts_with('(') {
            self.from.clone()
        } else {
            format!("(SELECT * FROM {})", self.from)
        }
    }
}

/// A `SELECT` being written: its columns before the relation's emit picks
/// them, and its clauses.
#[derive(Default)]
struct Select {
    columns: Vec<String>,
    from: String,
    filter: Vec<String>,
    group_by: Option<Vec<String>>,
    order_by: Vec<String>,
    limit: Option<String>,
    offset: Option<String>,
}

struct Writer<'a> {
    plan: &'a Plan,
    functions: &'a Functions,
    dialect: Dialect,
    aliases: usize,
    /// Each relation outside a subquery, as `name AS (query)`, in the order
    /// they are written: the query's `WITH` clause.
    ctes: Vec<String>,
    /// The CTE of each shared relation (`ReferenceRel`) written so far.
    shared: std::collections::HashMap<i32, Query>,
    /// How many subqueries the relation being written is inside. A relation
    /// inside one is written in place, since it may read the enclosing
    /// query's columns, which a CTE cannot see.
    subqueries: usize,
    /// The columns each enclosing relation's expressions read, innermost
    /// last: an outer reference `steps_out` levels up reads the scope that
    /// many from the end.
    scopes: Vec<Vec<String>>,
}

impl Writer<'_> {
    fn alias(&mut self) -> String {
        let a = format!("t{}", self.aliases);
        self.aliases += 1;
        a
    }

    /// `rel` as a query, aliased: its `FROM` item and the columns it gives.
    fn input(&mut self, rel: Option<&Rel>) -> Result<(String, Vec<String>)> {
        let rel = rel.ok_or_else(|| AdError::InvalidPlan("a relation has no input".into()))?;
        let q = self.rel(rel)?;
        let a = self.alias();
        let columns = (0..q.width).map(|i| format!("{a}.c{i}")).collect();
        Ok((format!("{} AS {a}", q.from), columns))
    }

    /// Write `e` reading `scope`, with `scope` the innermost scope.
    fn scoped<T>(&mut self, scope: &[String], f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.scopes.push(scope.to_vec());
        let out = f(self);
        self.scopes.pop();
        out
    }

    fn exprs(&mut self, scope: &[String], es: &[Expression]) -> Result<Vec<String>> {
        self.scoped(scope, |w| es.iter().map(|e| w.expr(e)).collect())
    }

    fn rel(&mut self, rel: &Rel) -> Result<Query> {
        let (select, emit) = match self.built(rel)? {
            Built::Query(q) => return Ok(q),
            Built::Select(select, emit) => (select, emit),
        };
        let (sql, width) = finish(select, emit, None)?;
        if self.subqueries > 0 {
            return Ok(Query {
                from: format!("({sql})"),
                width,
            });
        }
        let name = format!("__ddx_q{}", self.ctes.len());
        self.ctes.push(format!("{name} AS ({sql})"));
        Ok(Query { from: name, width })
    }

    /// `rel` as a `SELECT` and the emit that picks its columns, or, for a
    /// shared relation, the query already written for it.
    fn built(&mut self, rel: &Rel) -> Result<Built> {
        let kind = rel
            .rel_type
            .as_ref()
            .ok_or_else(|| AdError::InvalidPlan("a relation has no type".into()))?;
        let (select, common) = match kind {
            RelType::Read(r) => (self.read(r)?, r.common.as_ref()),
            RelType::Project(p) => {
                let (from, scope) = self.input(p.input.as_deref())?;
                let computed = self.exprs(&scope, &p.expressions)?;
                let mut columns = scope;
                columns.extend(computed);
                let select = Select {
                    columns,
                    from,
                    ..Select::default()
                };
                (select, p.common.as_ref())
            }
            RelType::Filter(f) => {
                let (from, scope) = self.input(f.input.as_deref())?;
                let condition = f
                    .condition
                    .as_deref()
                    .ok_or_else(|| AdError::InvalidPlan("a filter has no condition".into()))?;
                let condition = self.scoped(&scope, |w| w.expr(condition))?;
                let select = Select {
                    columns: scope,
                    from,
                    filter: vec![condition],
                    ..Select::default()
                };
                (select, f.common.as_ref())
            }
            RelType::Join(j) => {
                let kind = JoinType::try_from(j.r#type).unwrap_or(JoinType::Unspecified);
                let select = self.join(
                    j.left.as_deref(),
                    j.right.as_deref(),
                    j.expression.as_deref(),
                    j.post_join_filter.as_deref(),
                    kind,
                )?;
                (select, j.common.as_ref())
            }
            RelType::Cross(c) => {
                let (left, mut columns) = self.input(c.left.as_deref())?;
                let (right, right_columns) = self.input(c.right.as_deref())?;
                columns.extend(right_columns);
                let select = Select {
                    columns,
                    from: format!("{left} CROSS JOIN {right}"),
                    ..Select::default()
                };
                (select, c.common.as_ref())
            }
            RelType::Aggregate(a) => (self.aggregate(a)?, a.common.as_ref()),
            RelType::Sort(s) => {
                let (from, scope) = self.input(s.input.as_deref())?;
                let order_by = self.sorts(&scope, &s.sorts)?;
                let select = Select {
                    columns: scope,
                    from,
                    order_by,
                    ..Select::default()
                };
                (select, s.common.as_ref())
            }
            RelType::Fetch(f) => {
                // A fetch over a sort is one query: an ORDER BY in a
                // subquery does not order the query reading it.
                let input = f
                    .input
                    .as_deref()
                    .ok_or_else(|| AdError::InvalidPlan("a fetch has no input".into()))?;
                let mut select = match &input.rel_type {
                    Some(RelType::Sort(s)) if emit_of(s.common.as_ref()).is_none() => {
                        let (from, scope) = self.input(s.input.as_deref())?;
                        let order_by = self.sorts(&scope, &s.sorts)?;
                        Select {
                            columns: scope,
                            from,
                            order_by,
                            ..Select::default()
                        }
                    }
                    _ => {
                        let (from, scope) = self.input(Some(input))?;
                        Select {
                            columns: scope,
                            from,
                            ..Select::default()
                        }
                    }
                };
                (select.offset, select.limit) = self.fetch(f)?;
                (select, f.common.as_ref())
            }
            RelType::Set(s) => {
                let op = match SetOp::try_from(s.op).unwrap_or(SetOp::Unspecified) {
                    SetOp::UnionAll => "UNION ALL",
                    SetOp::UnionDistinct => "UNION",
                    SetOp::MinusPrimary => "EXCEPT",
                    SetOp::MinusPrimaryAll => "EXCEPT ALL",
                    SetOp::IntersectionMultiset => "INTERSECT",
                    SetOp::IntersectionMultisetAll => "INTERSECT ALL",
                    other => {
                        return Err(AdError::NotImplemented(format!(
                            "writing the set operation {} as SQL",
                            other.as_str_name()
                        )))
                    }
                };
                let mut width = None;
                let mut parts = Vec::new();
                for input in &s.inputs {
                    let q = self.rel(input)?;
                    if *width.get_or_insert(q.width) != q.width {
                        return Err(AdError::InvalidPlan(
                            "a set operation's inputs have different widths".into(),
                        ));
                    }
                    parts.push(format!("SELECT * FROM {} AS {}", q.from, self.alias()));
                }
                let width = width
                    .ok_or_else(|| AdError::InvalidPlan("a set operation has no inputs".into()))?;
                let a = self.alias();
                let select = Select {
                    columns: (0..width).map(|i| format!("{a}.c{i}")).collect(),
                    from: format!("({}) AS {a}", parts.join(&format!(" {op} "))),
                    ..Select::default()
                };
                (select, s.common.as_ref())
            }
            RelType::Window(w) => {
                let (from, scope) = self.input(w.input.as_deref())?;
                let partitions = self.exprs(&scope, &w.partition_expressions)?;
                let order_by = self.sorts(&scope, &w.sorts)?;
                let mut columns = scope.clone();
                for f in &w.window_functions {
                    let call = self.scoped(&scope, |w| {
                        w.window(&Window {
                            function: f.function_reference,
                            arguments: &f.arguments,
                            invocation: f.invocation,
                            partitions: partitions.clone(),
                            order_by: order_by.clone(),
                            bounds_type: f.bounds_type,
                            lower: f.lower_bound.as_ref(),
                            upper: f.upper_bound.as_ref(),
                        })
                    })?;
                    columns.push(call);
                }
                let select = Select {
                    columns,
                    from,
                    ..Select::default()
                };
                (select, w.common.as_ref())
            }
            RelType::Reference(r) => {
                if self.subqueries == 0 {
                    if let Some(q) = self.shared.get(&r.subtree_ordinal) {
                        return Ok(Built::Query(q.clone()));
                    }
                }
                let rel = self
                    .plan
                    .relations
                    .get(r.subtree_ordinal as usize)
                    .and_then(|r| match &r.rel_type {
                        Some(PlanRelType::Rel(rel)) => Some(rel),
                        Some(PlanRelType::Root(root)) => root.input.as_ref(),
                        None => None,
                    })
                    .ok_or_else(|| {
                        AdError::InvalidPlan(format!(
                            "a reference names relation {}, which the plan does not have",
                            r.subtree_ordinal
                        ))
                    })?;
                let q = self.rel(rel)?;
                if self.subqueries == 0 {
                    self.shared.insert(r.subtree_ordinal, q.clone());
                }
                return Ok(Built::Query(q));
            }
            other => {
                return Err(AdError::NotImplemented(format!(
                    "writing a {} relation as SQL",
                    rel_name(other)
                )))
            }
        };
        Ok(Built::Select(select, emit_of(common)))
    }

    fn read(&mut self, read: &ReadRel) -> Result<Select> {
        let schema = read
            .base_schema
            .as_ref()
            .ok_or_else(|| AdError::InvalidPlan("a read has no base schema".into()))?;
        let types: Vec<Option<&Type>> = match &schema.r#struct {
            Some(s) => s.types.iter().map(Some).collect(),
            None => schema.names.iter().map(|_| None).collect(),
        };
        if types.len() != schema.names.len() {
            return Err(AdError::NotImplemented(
                "writing a read of nested columns as SQL".into(),
            ));
        }
        let a = self.alias();
        let (from, base): (String, Vec<String>) = match &read.read_type {
            Some(ReadType::NamedTable(t)) => {
                let name: Vec<String> = t.names.iter().map(|n| ident(n)).collect();
                let columns = schema
                    .names
                    .iter()
                    .map(|n| format!("{a}.{}", ident(n)))
                    .collect();
                (format!("{} AS {a}", name.join(".")), columns)
            }
            Some(ReadType::VirtualTable(v)) => {
                let types: Vec<&Type> = types
                    .iter()
                    .map(|t| {
                        t.ok_or_else(|| {
                            AdError::InvalidPlan("a virtual table's schema has no types".into())
                        })
                    })
                    .collect::<Result<_>>()?;
                let mut rows = Vec::new();
                #[allow(deprecated)]
                for row in &v.values {
                    let values = row
                        .fields
                        .iter()
                        .map(|l| self.literal(l))
                        .collect::<Result<Vec<_>>>()?;
                    rows.push(values);
                }
                for row in &v.expressions {
                    rows.push(self.exprs(&[], &row.fields)?);
                }
                let width = types.len();
                if rows.iter().any(|r| r.len() != width) {
                    return Err(AdError::InvalidPlan(
                        "a virtual table's row has the wrong width".into(),
                    ));
                }
                let typed = |i: usize, value: &str| -> Result<String> {
                    Ok(format!("CAST({value} AS {})", self.type_name(types[i])?))
                };
                let columns: Vec<String> = (0..width).map(|i| format!("{a}.c{i}")).collect();
                let from = if rows.is_empty() || width == 0 {
                    // No rows (or no columns, which VALUES cannot write):
                    // one row of typed NULLs, filtered out below.
                    let nulls = (0..width)
                        .map(|i| Ok(format!("{} AS c{i}", typed(i, "NULL")?)))
                        .collect::<Result<Vec<_>>>()?;
                    let mut sql = format!("SELECT {}", select_list(nulls));
                    if rows.is_empty() {
                        sql.push_str(" WHERE FALSE");
                    }
                    format!("({sql}) AS {a}")
                } else {
                    let rows: Vec<String> = rows
                        .iter()
                        .map(|r| {
                            let values = r
                                .iter()
                                .enumerate()
                                .map(|(i, v)| typed(i, v))
                                .collect::<Result<Vec<_>>>()?;
                            Ok(format!("({})", values.join(", ")))
                        })
                        .collect::<Result<_>>()?;
                    let names: Vec<String> = (0..width).map(|i| format!("c{i}")).collect();
                    format!("(VALUES {}) AS {a}({})", rows.join(", "), names.join(", "))
                };
                (from, columns)
            }
            _ => {
                return Err(AdError::NotImplemented(
                    "writing a read of anything but a named or virtual table as SQL".into(),
                ))
            }
        };
        let mut filter = Vec::new();
        for f in [read.filter.as_deref(), read.best_effort_filter.as_deref()]
            .into_iter()
            .flatten()
        {
            filter.push(self.scoped(&base, |w| w.expr(f))?);
        }
        let columns = match &read.projection {
            None => base,
            Some(mask) => {
                let items = mask
                    .select
                    .as_ref()
                    .map(|s| &s.struct_items[..])
                    .unwrap_or(&[]);
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
        left: Option<&Rel>,
        right: Option<&Rel>,
        condition: Option<&Expression>,
        post: Option<&Expression>,
        kind: JoinType,
    ) -> Result<Select> {
        let (left, left_columns) = self.input(left)?;
        let (right, right_columns) = self.input(right)?;
        let mut both = left_columns.clone();
        both.extend(right_columns.iter().cloned());
        let on = match condition {
            Some(c) => self.scoped(&both, |w| w.expr(c))?,
            None => "TRUE".to_string(),
        };
        let keyword = match kind {
            JoinType::Inner => Some("INNER JOIN"),
            JoinType::Left => Some("LEFT JOIN"),
            JoinType::Right => Some("RIGHT JOIN"),
            JoinType::Outer => Some("FULL JOIN"),
            _ => None,
        };
        if let Some(keyword) = keyword {
            let filter = match post {
                Some(p) => vec![self.scoped(&both, |w| w.expr(p))?],
                None => Vec::new(),
            };
            return Ok(Select {
                columns: both,
                from: format!("{left} {keyword} {right} ON {on}"),
                filter,
                ..Select::default()
            });
        }
        if post.is_some() {
            return Err(AdError::NotImplemented(format!(
                "writing a {} join's post-join filter as SQL",
                kind.as_str_name()
            )));
        }
        // A join that keeps one side's rows: an EXISTS over the other side.
        let exists = |other: &str| format!("EXISTS (SELECT 1 FROM {other} WHERE {on})");
        let (from, columns, test) = match kind {
            JoinType::LeftSemi => (left.clone(), left_columns, exists(&right)),
            JoinType::LeftAnti => (
                left.clone(),
                left_columns,
                format!("NOT {}", exists(&right)),
            ),
            JoinType::RightSemi => (right.clone(), right_columns, exists(&left)),
            JoinType::RightAnti => (
                right.clone(),
                right_columns,
                format!("NOT {}", exists(&left)),
            ),
            JoinType::LeftMark => {
                let mut columns = left_columns;
                columns.push(exists(&right));
                return Ok(Select {
                    columns,
                    from: left,
                    ..Select::default()
                });
            }
            JoinType::RightMark => {
                let mut columns = right_columns;
                columns.push(exists(&left));
                return Ok(Select {
                    columns,
                    from: right,
                    ..Select::default()
                });
            }
            other => {
                return Err(AdError::NotImplemented(format!(
                    "writing a {} join as SQL",
                    other.as_str_name()
                )))
            }
        };
        Ok(Select {
            columns,
            from,
            filter: vec![test],
            ..Select::default()
        })
    }

    fn aggregate(&mut self, a: &AggregateRel) -> Result<Select> {
        let (from, scope) = self.input(a.input.as_deref())?;
        if a.groupings.len() > 1 {
            return Err(AdError::NotImplemented(
                "writing an aggregate with several grouping sets as SQL".into(),
            ));
        }
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
        let keys: Vec<String> =
            self.scoped(&scope, |w| keys.iter().map(|e| w.expr(e)).collect())?;
        let mut columns = keys.clone();
        for m in &a.measures {
            let f = m
                .measure
                .as_ref()
                .ok_or_else(|| AdError::InvalidPlan("a measure has no function".into()))?;
            let call = self.scoped(&scope, |w| {
                let name = w.function_name(f.function_reference)?;
                let args = w.arguments(&f.arguments)?;
                let distinct = f.invocation == AggregationInvocation::Distinct as i32;
                let mut call = if args.is_empty() && name == "count" {
                    "count(*)".to_string()
                } else {
                    format!(
                        "{name}({}{})",
                        if distinct { "DISTINCT " } else { "" },
                        args.join(", ")
                    )
                };
                if !f.sorts.is_empty() {
                    let sorts = w.sort_list(&f.sorts)?;
                    // ORDER BY inside the call's parentheses.
                    call.pop();
                    write!(call, " ORDER BY {})", sorts.join(", ")).ok();
                }
                if let Some(filter) = &m.filter {
                    write!(call, " FILTER (WHERE {})", w.expr(filter)?).ok();
                }
                Ok(call)
            })?;
            columns.push(call);
        }
        Ok(Select {
            columns,
            from,
            group_by: Some(keys),
            ..Select::default()
        })
    }

    fn sorts(&mut self, scope: &[String], sorts: &[SortField]) -> Result<Vec<String>> {
        self.scoped(scope, |w| w.sort_list(sorts))
    }

    fn sort_list(&mut self, sorts: &[SortField]) -> Result<Vec<String>> {
        sorts
            .iter()
            .map(|s| {
                let e = s
                    .expr
                    .as_ref()
                    .ok_or_else(|| AdError::InvalidPlan("a sort field has no expression".into()))?;
                let direction = match s.sort_kind {
                    Some(SortKind::Direction(d)) => {
                        match SortDirection::try_from(d).unwrap_or(SortDirection::Unspecified) {
                            SortDirection::AscNullsFirst => "ASC NULLS FIRST",
                            SortDirection::AscNullsLast => "ASC NULLS LAST",
                            SortDirection::DescNullsFirst => "DESC NULLS FIRST",
                            SortDirection::DescNullsLast => "DESC NULLS LAST",
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
                Ok(format!("{} {direction}", self.expr(e)?))
            })
            .collect()
    }

    /// A fetch's `OFFSET` and `LIMIT`, if it has them. The deprecated
    /// integer fields are read too, as older producers write them.
    #[allow(deprecated)]
    fn fetch(
        &mut self,
        f: &substrait::proto::FetchRel,
    ) -> Result<(Option<String>, Option<String>)> {
        let offset = match &f.offset_mode {
            Some(OffsetMode::Offset(n)) => Some(*n),
            Some(OffsetMode::OffsetExpr(e)) => self.count(e)?,
            None => None,
        };
        let count = match &f.count_mode {
            Some(CountMode::Count(n)) => Some(*n),
            Some(CountMode::CountExpr(e)) => self.count(e)?,
            None => None,
        };
        // A count of -1 is "all", as the deprecated field wrote it.
        Ok((
            offset.filter(|n| *n > 0).map(|n| n.to_string()),
            count.filter(|n| *n >= 0).map(|n| n.to_string()),
        ))
    }

    /// A fetch's offset or count, which must be a literal integer.
    fn count(&mut self, e: &Expression) -> Result<Option<i64>> {
        match &e.rex_type {
            Some(RexType::Literal(Literal {
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

    fn expr(&mut self, e: &Expression) -> Result<String> {
        let rex = e
            .rex_type
            .as_ref()
            .ok_or_else(|| AdError::InvalidPlan("an expression has no type".into()))?;
        match rex {
            RexType::Literal(l) => self.literal(l),
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
                let scope = self
                    .scopes
                    .len()
                    .checked_sub(1 + steps_out)
                    .and_then(|i| self.scopes.get(i))
                    .ok_or_else(|| {
                        AdError::InvalidPlan(format!(
                            "a field reference steps out {steps_out} levels, past the outermost"
                        ))
                    })?;
                usize::try_from(field)
                    .ok()
                    .and_then(|i| scope.get(i))
                    .cloned()
                    .ok_or_else(|| {
                        AdError::InvalidPlan(format!(
                            "field {field} is past its input's {} columns",
                            scope.len()
                        ))
                    })
            }
            RexType::ScalarFunction(f) => {
                let args = self.arguments(&f.arguments)?;
                self.call(f.function_reference, args, f.output_type.as_ref())
            }
            RexType::WindowFunction(f) => {
                let partitions = f
                    .partitions
                    .iter()
                    .map(|p| self.expr(p))
                    .collect::<Result<_>>()?;
                let order_by = self.sort_list(&f.sorts)?;
                self.window(&Window {
                    function: f.function_reference,
                    arguments: &f.arguments,
                    invocation: f.invocation,
                    partitions,
                    order_by,
                    bounds_type: f.bounds_type,
                    lower: f.lower_bound.as_ref(),
                    upper: f.upper_bound.as_ref(),
                })
            }
            RexType::IfThen(i) => {
                // DataFusion's producer writes `CASE x WHEN v THEN …` as a
                // first clause holding `x` and no result.
                let (mut sql, clauses) = match i.ifs.split_first() {
                    Some((first, rest)) if first.then.is_none() => {
                        let operand = first
                            .r#if
                            .as_ref()
                            .ok_or_else(|| AdError::InvalidPlan("a CASE clause is empty".into()))?;
                        (format!("CASE {}", self.expr(operand)?), rest)
                    }
                    _ => ("CASE".to_string(), &i.ifs[..]),
                };
                for clause in clauses {
                    let (Some(c), Some(t)) = (&clause.r#if, &clause.then) else {
                        return Err(AdError::InvalidPlan("a CASE clause is incomplete".into()));
                    };
                    write!(sql, " WHEN {} THEN {}", self.expr(c)?, self.expr(t)?).ok();
                }
                if let Some(e) = &i.r#else {
                    write!(sql, " ELSE {}", self.expr(e)?).ok();
                }
                sql.push_str(" END");
                Ok(sql)
            }
            RexType::SwitchExpression(s) => {
                let value = s
                    .r#match
                    .as_deref()
                    .ok_or_else(|| AdError::InvalidPlan("a switch has no value".into()))?;
                let mut sql = format!("CASE {}", self.expr(value)?);
                for clause in &s.ifs {
                    let (Some(c), Some(t)) = (&clause.r#if, &clause.then) else {
                        return Err(AdError::InvalidPlan("a switch clause is incomplete".into()));
                    };
                    write!(sql, " WHEN {} THEN {}", self.literal(c)?, self.expr(t)?).ok();
                }
                if let Some(e) = &s.r#else {
                    write!(sql, " ELSE {}", self.expr(e)?).ok();
                }
                sql.push_str(" END");
                Ok(sql)
            }
            RexType::SingularOrList(s) => {
                let value = s
                    .value
                    .as_deref()
                    .ok_or_else(|| AdError::InvalidPlan("an IN list has no value".into()))?;
                let value = self.expr(value)?;
                if s.options.is_empty() {
                    return Ok("FALSE".into());
                }
                let options = s
                    .options
                    .iter()
                    .map(|o| self.expr(o))
                    .collect::<Result<Vec<_>>>()?;
                Ok(format!("({value} IN ({}))", options.join(", ")))
            }
            RexType::MultiOrList(m) => {
                // (a, b) IN ((x, y), …), as ORs of ANDs, which every
                // dialect reads.
                let values = m
                    .value
                    .iter()
                    .map(|v| self.expr(v))
                    .collect::<Result<Vec<_>>>()?;
                let mut options = Vec::new();
                for record in &m.options {
                    if record.fields.len() != values.len() {
                        return Err(AdError::InvalidPlan(
                            "a multi-column IN list's record has the wrong width".into(),
                        ));
                    }
                    let terms = values
                        .iter()
                        .zip(&record.fields)
                        .map(|(v, f)| Ok(format!("({v} = {})", self.expr(f)?)))
                        .collect::<Result<Vec<_>>>()?;
                    options.push(format!("({})", terms.join(" AND ")));
                }
                if options.is_empty() {
                    return Ok("FALSE".into());
                }
                Ok(format!("({})", options.join(" OR ")))
            }
            RexType::Cast(c) => {
                let input = c
                    .input
                    .as_deref()
                    .ok_or_else(|| AdError::InvalidPlan("a cast has no input".into()))?;
                let ty = c
                    .r#type
                    .as_ref()
                    .ok_or_else(|| AdError::InvalidPlan("a cast has no type".into()))?;
                let function = if c.failure_behavior == cast::FailureBehavior::ReturnNull as i32 {
                    "TRY_CAST"
                } else {
                    "CAST"
                };
                Ok(format!(
                    "{function}({} AS {})",
                    self.expr(input)?,
                    self.type_name(ty)?
                ))
            }
            RexType::Subquery(s) => self.subquery(s),
            RexType::Nested(n) => match &n.nested_type {
                Some(NestedType::Struct(_))
                | Some(NestedType::List(_))
                | Some(NestedType::Map(_))
                | None => Err(AdError::NotImplemented(
                    "writing a nested value as SQL".into(),
                )),
            },
            other => Err(AdError::NotImplemented(format!(
                "writing a {} expression as SQL",
                crate::expr::rex_name(other)
            ))),
        }
    }

    fn subquery(&mut self, s: &substrait::proto::expression::Subquery) -> Result<String> {
        self.subqueries += 1;
        let out = self.subquery_in(s);
        self.subqueries -= 1;
        out
    }

    fn subquery_in(&mut self, s: &substrait::proto::expression::Subquery) -> Result<String> {
        let kind = s
            .subquery_type
            .as_ref()
            .ok_or_else(|| AdError::InvalidPlan("a subquery has no type".into()))?;
        let missing = || AdError::InvalidPlan("a subquery has no relation".into());
        match kind {
            SubqueryType::Scalar(q) => {
                let q = self.rel(q.input.as_deref().ok_or_else(missing)?)?;
                Ok(q.subquery())
            }
            SubqueryType::InPredicate(p) => {
                let [needle] = &p.needles[..] else {
                    return Err(AdError::NotImplemented(
                        "writing an IN subquery of several columns as SQL".into(),
                    ));
                };
                let needle = self.expr(needle)?;
                let q = self.rel(p.haystack.as_deref().ok_or_else(missing)?)?;
                Ok(format!("({needle} IN {})", q.subquery()))
            }
            SubqueryType::SetPredicate(p) => {
                if p.predicate_op != PredicateOp::Exists as i32 {
                    return Err(AdError::NotImplemented(
                        "writing a UNIQUE subquery as SQL".into(),
                    ));
                }
                let q = self.rel(p.tuples.as_deref().ok_or_else(missing)?)?;
                Ok(format!("EXISTS {}", q.subquery()))
            }
            SubqueryType::SetComparison(c) => {
                let reduction = match ReductionOp::try_from(c.reduction_op) {
                    Ok(ReductionOp::Any) => "ANY",
                    Ok(ReductionOp::All) => "ALL",
                    _ => {
                        return Err(AdError::InvalidPlan(
                            "a set comparison has no reduction".into(),
                        ))
                    }
                };
                let op = match ComparisonOp::try_from(c.comparison_op) {
                    Ok(ComparisonOp::Eq) => "=",
                    Ok(ComparisonOp::Ne) => "<>",
                    Ok(ComparisonOp::Lt) => "<",
                    Ok(ComparisonOp::Gt) => ">",
                    Ok(ComparisonOp::Le) => "<=",
                    Ok(ComparisonOp::Ge) => ">=",
                    _ => {
                        return Err(AdError::InvalidPlan(
                            "a set comparison has no comparison".into(),
                        ))
                    }
                };
                let left = c
                    .left
                    .as_deref()
                    .ok_or_else(|| AdError::InvalidPlan("a set comparison has no left".into()))?;
                let left = self.expr(left)?;
                let q = self.rel(c.right.as_deref().ok_or_else(missing)?)?;
                Ok(format!("({left} {op} {reduction} {})", q.subquery()))
            }
        }
    }

    fn arguments(&mut self, args: &[FunctionArgument]) -> Result<Vec<String>> {
        args.iter()
            .map(|a| match &a.arg_type {
                Some(ArgType::Value(e)) => self.expr(e),
                _ => Err(AdError::NotImplemented(
                    "writing a function's type or enum argument as SQL".into(),
                )),
            })
            .collect()
    }

    /// The dialect's spelling of the function `anchor` names.
    fn function_name(&self, anchor: u32) -> Result<&'static str> {
        let name = self.functions.name(anchor)?;
        match spelling(name) {
            Some(Spelling::Call(f)) => f(self.dialect).ok_or_else(|| self.no_spelling(name)),
            _ => Err(self.no_spelling(name)),
        }
    }

    fn no_spelling(&self, name: &str) -> AdError {
        AdError::NotImplemented(format!(
            "writing the function `{name}` as {} SQL: ddx knows no spelling of it there",
            self.dialect.name()
        ))
    }

    fn call(&mut self, anchor: u32, args: Vec<String>, output: Option<&Type>) -> Result<String> {
        let name = self.functions.name(anchor)?.to_string();
        if name == STOP_GRADIENT {
            let [x] = &args[..] else {
                return Err(AdError::InvalidPlan(format!(
                    "{STOP_GRADIENT} takes one argument"
                )));
            };
            return Ok(x.clone());
        }
        let arity = |n: usize| -> Result<()> {
            if args.len() == n {
                Ok(())
            } else {
                Err(AdError::InvalidPlan(format!(
                    "`{name}` takes {n} arguments, not {}",
                    args.len()
                )))
            }
        };
        match spelling(&name) {
            Some(Spelling::Infix(op)) => {
                if args.len() < 2 {
                    arity(2)?;
                }
                // `divide` of integers is the dialect's integer division:
                // DuckDB's `/` of two integers is a DOUBLE.
                let op = if name == "divide"
                    && self.dialect == Dialect::DuckDb
                    && output.is_some_and(is_integer)
                {
                    "//"
                } else {
                    op
                };
                Ok(format!("({})", args.join(&format!(" {op} "))))
            }
            Some(Spelling::Prefix(op)) => {
                arity(1)?;
                Ok(format!("({op} {})", args[0]))
            }
            Some(Spelling::Postfix(op)) => {
                arity(1)?;
                Ok(format!("({} {op})", args[0]))
            }
            Some(Spelling::Call(f)) => {
                let f = f(self.dialect).ok_or_else(|| self.no_spelling(&name))?;
                Ok(format!("{f}({})", args.join(", ")))
            }
            None => Err(self.no_spelling(&name)),
        }
    }

    fn window(&mut self, w: &Window<'_>) -> Result<String> {
        let name = self.function_name(w.function)?;
        let args = self.arguments(w.arguments)?;
        let distinct = w.invocation == AggregationInvocation::Distinct as i32;
        let call = if args.is_empty() && name == "count" {
            "count(*)".to_string()
        } else {
            format!(
                "{name}({}{})",
                if distinct { "DISTINCT " } else { "" },
                args.join(", ")
            )
        };
        let mut over = Vec::new();
        if !w.partitions.is_empty() {
            over.push(format!("PARTITION BY {}", w.partitions.join(", ")));
        }
        if !w.order_by.is_empty() {
            over.push(format!("ORDER BY {}", w.order_by.join(", ")));
        }
        if !RANKING.contains(&name) {
            // Always written: SQL's default frame for an ordered window ends
            // at the current row, Substrait's at the end of the partition.
            // An unspecified bounds type is read as DataFusion's consumer
            // reads it.
            let units = match BoundsType::try_from(w.bounds_type).unwrap_or(BoundsType::Unspecified)
            {
                BoundsType::Rows => "ROWS",
                BoundsType::Range => "RANGE",
                BoundsType::Unspecified if w.order_by.is_empty() => "ROWS",
                BoundsType::Unspecified => "RANGE",
            };
            over.push(format!(
                "{units} BETWEEN {} AND {}",
                bound(w.lower, true)?,
                bound(w.upper, false)?
            ));
        }
        Ok(format!("{call} OVER ({})", over.join(" ")))
    }

    fn literal(&self, l: &Literal) -> Result<String> {
        let t = l
            .literal_type
            .as_ref()
            .ok_or_else(|| AdError::InvalidPlan("a literal has no value".into()))?;
        Ok(match t {
            LiteralType::Boolean(b) => if *b { "TRUE" } else { "FALSE" }.into(),
            LiteralType::I8(n) => format!("CAST({n} AS TINYINT)"),
            LiteralType::I16(n) => format!("CAST({n} AS SMALLINT)"),
            LiteralType::I32(n) => format!("CAST({n} AS INTEGER)"),
            LiteralType::I64(n) => format!("CAST({n} AS BIGINT)"),
            LiteralType::Fp32(x) => float(f64::from(*x), &format!("{x:e}"), "REAL"),
            LiteralType::Fp64(x) => float(*x, &format!("{x:e}"), "DOUBLE"),
            LiteralType::String(s) | LiteralType::FixedChar(s) => string(s),
            LiteralType::VarChar(v) => string(&v.value),
            LiteralType::Date(days) => format!("CAST('{}' AS DATE)", date(i64::from(*days))),
            #[allow(deprecated)]
            LiteralType::Timestamp(micros) => {
                format!("CAST('{}' AS TIMESTAMP)", timestamp(*micros, 6))
            }
            LiteralType::PrecisionTimestamp(p) => {
                format!("CAST('{}' AS TIMESTAMP)", timestamp(p.value, p.precision))
            }
            LiteralType::Decimal(d) => {
                let bytes: [u8; 16] = d.value.as_slice().try_into().map_err(|_| {
                    AdError::InvalidPlan("a decimal literal is not 16 bytes".into())
                })?;
                let digits = decimal(i128::from_le_bytes(bytes), d.scale);
                format!("CAST('{digits}' AS DECIMAL({}, {}))", d.precision, d.scale)
            }
            LiteralType::Null(ty) => format!("CAST(NULL AS {})", self.type_name(ty)?),
            other => {
                return Err(AdError::NotImplemented(format!(
                    "writing a {} literal as SQL",
                    literal_name(other)
                )))
            }
        })
    }

    fn type_name(&self, ty: &Type) -> Result<String> {
        let kind = ty
            .kind
            .as_ref()
            .ok_or_else(|| AdError::InvalidPlan("a type has no kind".into()))?;
        Ok(match kind {
            Kind::Bool(_) => "BOOLEAN".into(),
            Kind::I8(_) => "TINYINT".into(),
            Kind::I16(_) => "SMALLINT".into(),
            Kind::I32(_) => "INTEGER".into(),
            Kind::I64(_) => "BIGINT".into(),
            Kind::Fp32(_) => "REAL".into(),
            Kind::Fp64(_) => "DOUBLE".into(),
            Kind::String(_) | Kind::Varchar(_) | Kind::FixedChar(_) => "VARCHAR".into(),
            Kind::Date(_) => "DATE".into(),
            #[allow(deprecated)]
            Kind::Timestamp(_) => "TIMESTAMP".into(),
            Kind::PrecisionTimestamp(_) => "TIMESTAMP".into(),
            Kind::Decimal(d) => format!("DECIMAL({}, {})", d.precision, d.scale),
            Kind::Binary(_) => match self.dialect {
                Dialect::DataFusion => "BYTEA".into(),
                Dialect::DuckDb => "BLOB".into(),
            },
            _ => {
                return Err(AdError::NotImplemented(
                    "writing a nested, interval or user-defined type as SQL".into(),
                ))
            }
        })
    }
}

/// A window function's parts, from either a window expression or a window
/// relation.
struct Window<'a> {
    function: u32,
    arguments: &'a [FunctionArgument],
    invocation: i32,
    partitions: Vec<String>,
    order_by: Vec<String>,
    bounds_type: i32,
    lower: Option<&'a Bound>,
    upper: Option<&'a Bound>,
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

fn bound(b: Option<&Bound>, lower: bool) -> Result<String> {
    let unbounded = if lower {
        "UNBOUNDED PRECEDING"
    } else {
        "UNBOUNDED FOLLOWING"
    };
    Ok(match b.and_then(|b| b.kind.as_ref()) {
        None | Some(BoundKind::Unbounded(_)) => unbounded.into(),
        Some(BoundKind::CurrentRow(_)) => "CURRENT ROW".into(),
        Some(BoundKind::Preceding(p)) => format!("{} PRECEDING", p.offset),
        Some(BoundKind::Following(f)) => format!("{} FOLLOWING", f.offset),
    })
}

/// How a function is written.
enum Spelling {
    /// Between its arguments: `(a + b)`, `(a AND b AND c)`.
    Infix(&'static str),
    /// Before its one argument: `(NOT a)`.
    Prefix(&'static str),
    /// After its one argument: `(a IS NULL)`.
    Postfix(&'static str),
    /// A call, named per dialect, or `None` where the dialect has no
    /// function of that meaning.
    Call(fn(Dialect) -> Option<&'static str>),
}

/// How the function with the normalized name `name` is written, or `None`
/// if ddx knows no spelling of it.
fn spelling(name: &str) -> Option<Spelling> {
    use Spelling::*;
    Some(match name {
        "add" => Infix("+"),
        "subtract" => Infix("-"),
        "multiply" => Infix("*"),
        "divide" => Infix("/"),
        "modulus" | "mod" => Infix("%"),
        "equal" => Infix("="),
        "not_equal" => Infix("<>"),
        "lt" => Infix("<"),
        "lte" => Infix("<="),
        "gt" => Infix(">"),
        "gte" => Infix(">="),
        "and" => Infix("AND"),
        "or" => Infix("OR"),
        "is_distinct_from" => Infix("IS DISTINCT FROM"),
        "is_not_distinct_from" => Infix("IS NOT DISTINCT FROM"),
        "not" => Prefix("NOT"),
        "negate" | "negative" => Prefix("-"),
        "is_null" => Postfix("IS NULL"),
        "is_not_null" => Postfix("IS NOT NULL"),
        "is_true" => Postfix("IS TRUE"),
        "is_false" => Postfix("IS FALSE"),
        "is_not_true" => Postfix("IS NOT TRUE"),
        "is_not_false" => Postfix("IS NOT FALSE"),
        "isnan" | "is_nan" => Call(|_| Some("isnan")),
        "sign" | "signum" => Call(|d| match d {
            Dialect::DataFusion => Some("signum"),
            Dialect::DuckDb => Some("sign"),
        }),
        _ => return call_spelling(name).map(Call),
    })
}

/// Functions spelled alike in every dialect.
fn call_spelling(name: &str) -> Option<fn(Dialect) -> Option<&'static str>> {
    macro_rules! alike {
        ($($n:literal),* $(,)?) => {
            match name {
                $($n => Some(|_| Some($n)),)*
                _ => None,
            }
        };
    }
    alike!(
        // Scalars.
        "abs",
        "sqrt",
        "cbrt",
        "exp",
        "ln",
        "log10",
        "log2",
        "power",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "floor",
        "ceil",
        "round",
        "trunc",
        "greatest",
        "least",
        "coalesce",
        "nullif",
        "pi",
        "degrees",
        "radians",
        "random",
        // Aggregates.
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
        // Window functions.
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
    )
}

/// The columns `select` gives, picked by `emit`, as a query: its SQL and its
/// width. The columns are named `names`, or `c0`, `c1`, … without them.
fn finish(
    select: Select,
    emit: Option<Vec<usize>>,
    names: Option<&[String]>,
) -> Result<(String, usize)> {
    let columns = match emit {
        None => select.columns,
        Some(mapping) => mapping
            .iter()
            .map(|&i| {
                select.columns.get(i).cloned().ok_or_else(|| {
                    AdError::InvalidPlan(format!(
                        "an emit picks column {i} of {}",
                        select.columns.len()
                    ))
                })
            })
            .collect::<Result<_>>()?,
    };
    let width = columns.len();
    if let Some(names) = names {
        if names.len() != width {
            return Err(AdError::InvalidPlan(format!(
                "the root names {} columns, but its relation has {width}",
                names.len()
            )));
        }
    }
    let named = columns
        .iter()
        .enumerate()
        .map(|(i, c)| match names {
            Some(names) => format!("{c} AS {}", ident(&names[i])),
            None => format!("{c} AS c{i}"),
        })
        .collect();
    let mut sql = format!("SELECT {} FROM {}", select_list(named), select.from);
    if !select.filter.is_empty() {
        write!(sql, " WHERE {}", select.filter.join(" AND ")).ok();
    }
    if let Some(keys) = select.group_by {
        if !keys.is_empty() {
            write!(sql, " GROUP BY {}", keys.join(", ")).ok();
        }
    }
    if !select.order_by.is_empty() {
        write!(sql, " ORDER BY {}", select.order_by.join(", ")).ok();
    }
    if let Some(n) = select.limit {
        write!(sql, " LIMIT {n}").ok();
    }
    if let Some(n) = select.offset {
        write!(sql, " OFFSET {n}").ok();
    }
    Ok((sql, width))
}

/// A select list, which SQL does not allow empty: a relation of no columns
/// still gives one, which nothing reads.
fn select_list(columns: Vec<String>) -> String {
    if columns.is_empty() {
        "1 AS __ddx_unit".into()
    } else {
        columns.join(", ")
    }
}

fn emit_of(common: Option<&RelCommon>) -> Option<Vec<usize>> {
    match common.and_then(|c| c.emit_kind.as_ref()) {
        Some(EmitKind::Emit(e)) => Some(e.output_mapping.iter().map(|&i| i as usize).collect()),
        _ => None,
    }
}

fn is_integer(ty: &Type) -> bool {
    matches!(
        ty.kind,
        Some(Kind::I8(_)) | Some(Kind::I16(_)) | Some(Kind::I32(_)) | Some(Kind::I64(_))
    )
}

/// A quoted identifier.
fn ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn string(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// A floating-point literal of the SQL type `ty`: in exponent notation, which
/// every dialect reads as a floating-point number and parses correctly
/// rounded, or a string for a value no literal writes.
fn float(x: f64, text: &str, ty: &str) -> String {
    if x.is_nan() {
        format!("CAST('NaN' AS {ty})")
    } else if x.is_infinite() {
        let sign = if x < 0.0 { "-" } else { "" };
        format!("CAST('{sign}Infinity' AS {ty})")
    } else {
        format!("CAST({text} AS {ty})")
    }
}

/// `value` with `scale` digits after the point.
fn decimal(value: i128, scale: i32) -> String {
    let negative = value < 0;
    let digits = value.unsigned_abs().to_string();
    let scale = scale.max(0) as usize;
    let digits = if digits.len() <= scale {
        format!("{}{digits}", "0".repeat(scale + 1 - digits.len()))
    } else {
        digits
    };
    let (whole, fraction) = digits.split_at(digits.len() - scale);
    let sign = if negative { "-" } else { "" };
    if fraction.is_empty() {
        format!("{sign}{whole}")
    } else {
        format!("{sign}{whole}.{fraction}")
    }
}

/// The civil date `days` after 1970-01-01 (Howard Hinnant's algorithm).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

fn date(days: i64) -> String {
    let (y, m, d) = civil(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// A timestamp `value` units of 10^-`precision` seconds after the epoch.
fn timestamp(value: i64, precision: i32) -> String {
    let per_second = 10_i64.pow(precision.clamp(0, 9) as u32);
    let seconds = value.div_euclid(per_second);
    let fraction = value.rem_euclid(per_second);
    let (days, second) = (seconds.div_euclid(86_400), seconds.rem_euclid(86_400));
    let mut text = format!(
        "{} {:02}:{:02}:{:02}",
        date(days),
        second / 3600,
        second / 60 % 60,
        second % 60
    );
    if precision > 0 {
        write!(text, ".{fraction:0width$}", width = precision as usize).ok();
    }
    text
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
    fn dates_and_timestamps_are_civil() {
        assert_eq!(date(0), "1970-01-01");
        assert_eq!(date(-1), "1969-12-31");
        assert_eq!(date(19_723), "2024-01-01");
        assert_eq!(date(11_016), "2000-02-29");
        assert_eq!(timestamp(1_500_000, 6), "1970-01-01 00:00:01.500000");
        assert_eq!(timestamp(-1, 3), "1969-12-31 23:59:59.999");
        assert_eq!(timestamp(86_400 + 3_661, 0), "1970-01-02 01:01:01");
    }

    #[test]
    fn decimals_keep_their_scale() {
        assert_eq!(decimal(12_345, 2), "123.45");
        assert_eq!(decimal(-5, 3), "-0.005");
        assert_eq!(decimal(7, 0), "7");
    }

    #[test]
    fn floats_round_trip() {
        for x in [
            0.1,
            0.30000000000000004,
            -1e-300,
            1.7976931348623157e308,
            5e-324,
        ] {
            let text = format!("{x:e}");
            assert_eq!(text.parse::<f64>().unwrap(), x, "{text}");
        }
        assert_eq!(float(f64::NAN, "NaN", "DOUBLE"), "CAST('NaN' AS DOUBLE)");
        assert_eq!(
            float(f64::NEG_INFINITY, "-inf", "DOUBLE"),
            "CAST('-Infinity' AS DOUBLE)"
        );
    }

    #[test]
    fn identifiers_and_strings_are_quoted() {
        assert_eq!(ident("a\"b"), "\"a\"\"b\"");
        assert_eq!(string("it's"), "'it''s'");
    }

    #[test]
    fn an_unknown_function_has_no_spelling() {
        assert!(spelling("frobnicate").is_none());
        assert!(matches!(spelling("sign"), Some(Spelling::Call(_))));
    }
}
