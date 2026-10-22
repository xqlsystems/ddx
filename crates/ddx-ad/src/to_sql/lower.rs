// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A [`Query`] written in one dialect, by [polyglot-sql]'s generators.
//!
//! The query is lowered to polyglot's SQL tree, node for node, and polyglot
//! writes it in the dialect: its quoting, escaping and keywords. This module
//! is the only one that knows polyglot, so an upgrade of it touches nothing
//! else; most of polyglot's nodes have no `Default`, so each is built here
//! with every field.
//!
//! What ddx decides rather than polyglot:
//!
//! - **Every compound expression is parenthesized** (`Paren`), so no
//!   dialect's precedence comes into it.
//! - **Every literal is cast to its type**, as the tree says.
//! - **A few names**, where polyglot's dialect does not rename them: see
//!   [`function`] and [`Lowering::ty`].
//!
//! [polyglot-sql]: https://github.com/tobilg/polyglot

use std::fmt::Write as _;

use polyglot_sql::expressions::Function;
use polyglot_sql::expressions::{
    self as pg, AggregateFunction, BinaryFunc, BinaryOp, BooleanLiteral, Case as PgCase,
    Cast as PgCast, Cte, DataType, Exists, Expression, From as PgFrom, GroupBy, Identifier, In,
    IsNull, IsTrueFalse, Join, JoinKind as PgJoinKind, Limit, Offset, OrderBy, Ordered, Over,
    Paren, QuantifiedExpr, QuantifiedOp, Star, Subquery, TableRef, Tuple, UnaryOp, Values, Where,
    WindowFrame, WindowFrameBound, WindowFrameKind, WindowFunction, With,
};
use polyglot_sql::{DialectType, Generator};

use super::ir::{
    Alias, Bound, Call, Column, Expr, Frame, From, IsTest, JoinKind, Literal, Op, Order, Query,
    Select, SetOp, Type, Units,
};
use super::Dialect;
use crate::error::{AdError, Result};

/// `query` in `dialect`.
pub(super) fn query(query: &Query, dialect: Dialect) -> Result<String> {
    let lowering = Lowering { dialect };
    let mut body = lowering.select(&query.body, "c");
    if !query.ctes.is_empty() {
        body.with = Some(With {
            ctes: query
                .ctes
                .iter()
                .enumerate()
                .map(|(i, s)| Cte {
                    alias: Identifier::new(cte(i)),
                    this: Expression::Select(Box::new(lowering.select(s, "c"))),
                    columns: Vec::new(),
                    materialized: None,
                    key_expressions: Vec::new(),
                    alias_first: true,
                    comments: Vec::new(),
                })
                .collect(),
            recursive: false,
            leading_comments: Vec::new(),
            search: None,
        });
    }
    let target = match dialect {
        Dialect::DataFusion => DialectType::DataFusion,
        Dialect::DuckDb => DialectType::DuckDB,
    };
    // The dialect's generator, without its complexity guard: a walk of the
    // whole tree, which guards against hostile input, before writing it.
    // ddx builds the tree, and deep plans are its business (`opt::split`).
    let mut config = polyglot_sql::Dialect::get(target)
        .generator_config()
        .clone();
    config.complexity_guard.max_ast_nodes = None;
    config.complexity_guard.max_ast_depth = None;
    Generator::with_config(config)
        .generate(&Expression::Select(Box::new(body)))
        .map_err(|e| AdError::NotImplemented(format!("writing SQL for {dialect:?}: {e}")))
}

/// The name of the canonical function `function` in `dialect`. The names
/// [`super::syntax`] gives are the same in every dialect but these, which
/// polyglot does not rename.
pub(super) fn function(function: &'static str, dialect: Dialect) -> &'static str {
    match (function, dialect) {
        ("sign", Dialect::DataFusion) => "signum",
        _ => function,
    }
}

fn cte(index: usize) -> String {
    format!("__ddx_q{index}")
}

fn alias_name(a: Alias) -> Identifier {
    Identifier::new(format!("t{}", a.0))
}

struct Lowering {
    dialect: Dialect,
}

impl Lowering {
    /// `s`, its columns named `{prefix}0`, `{prefix}1`, … unless it has
    /// names of its own.
    fn select(&self, s: &Select, prefix: &str) -> pg::Select {
        let mut columns: Vec<Expression> = s
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let name = match &s.names {
                    Some(names) => Identifier::quoted(names[i].clone()),
                    None => Identifier::new(format!("{prefix}{i}")),
                };
                Expression::Alias(Box::new(pg::Alias::new(self.expr(c), name)))
            })
            .collect();
        if columns.is_empty() {
            // SQL has no empty select list: a relation of no columns gives
            // one, which nothing reads. An aggregate of no keys and no
            // measures still gives one row, as `count(*)` does.
            let unit = if s.group_by.as_ref().is_some_and(Vec::is_empty) {
                count_star()
            } else {
                number("1")
            };
            columns.push(Expression::Alias(Box::new(pg::Alias::new(
                unit,
                Identifier::new("__ddx_unit"),
            ))));
        }
        let mut select = pg::Select {
            expressions: columns,
            ..pg::Select::default()
        };
        match &s.from {
            From::Nothing => {}
            From::Join {
                left,
                right,
                kind,
                on,
            } => {
                select.from = Some(PgFrom {
                    expressions: vec![self.from(left)],
                });
                select.joins = vec![self.join(right, *kind, on.as_ref())];
            }
            from => {
                select.from = Some(PgFrom {
                    expressions: vec![self.from(from)],
                })
            }
        }
        if let Some(first) = s.filter.first() {
            let condition = s.filter[1..].iter().fold(self.expr(first), |acc, f| {
                Expression::And(Box::new(BinaryOp::new(acc, self.expr(f))))
            });
            select.where_clause = Some(Where { this: condition });
        }
        if let Some(keys) = s.group_by.as_ref().filter(|k| !k.is_empty()) {
            select.group_by = Some(GroupBy {
                expressions: keys.iter().map(|k| self.expr(k)).collect(),
                all: None,
                totals: false,
                comments: Vec::new(),
            });
        }
        if !s.order_by.is_empty() {
            select.order_by = Some(OrderBy {
                expressions: self.orders(&s.order_by),
                siblings: false,
                comments: Vec::new(),
            });
        }
        if let Some(n) = s.limit {
            select.limit = Some(Limit {
                this: number(&n.to_string()),
                percent: false,
                comments: Vec::new(),
            });
        }
        if let Some(n) = s.offset {
            select.offset = Some(Offset {
                this: number(&n.to_string()),
                rows: None,
            });
        }
        select
    }

    fn join(&self, right: &From, kind: JoinKind, on: Option<&Expr>) -> Join {
        Join {
            this: self.from(right),
            on: on.map(|e| self.expr(e)),
            using: Vec::new(),
            kind: match kind {
                JoinKind::Inner => PgJoinKind::Inner,
                JoinKind::Left => PgJoinKind::Left,
                JoinKind::Right => PgJoinKind::Right,
                JoinKind::Full => PgJoinKind::Full,
                JoinKind::Cross => PgJoinKind::Cross,
            },
            use_inner_keyword: kind == JoinKind::Inner,
            use_outer_keyword: false,
            deferred_condition: false,
            join_hint: None,
            match_condition: None,
            pivots: Vec::new(),
            comments: Vec::new(),
            nesting_group: 0,
            directed: false,
        }
    }

    /// A `FROM` item: a table, a CTE or a subquery, under its alias.
    fn from(&self, f: &From) -> Expression {
        match f {
            From::Nothing | From::Join { .. } => {
                unreachable!("a select's FROM, not an item of it")
            }
            From::Table { name, alias } => {
                let mut parts = name.iter().rev().map(|n| Identifier::quoted(n.clone()));
                let mut table =
                    TableRef::from_identifier(parts.next().unwrap_or_else(Identifier::empty));
                table.schema = parts.next();
                table.catalog = parts.next();
                table.alias = Some(alias_name(*alias));
                table.alias_explicit_as = true;
                Expression::Table(Box::new(table))
            }
            From::Cte { index, alias } => {
                let mut table = TableRef::new(cte(*index));
                table.alias = Some(alias_name(*alias));
                table.alias_explicit_as = true;
                Expression::Table(Box::new(table))
            }
            From::Subquery { query, alias } => subquery(
                Expression::Select(Box::new(self.select(query, "c"))),
                Some(*alias),
                Vec::new(),
            ),
            From::Values { rows, alias } => {
                let width = rows.first().map_or(0, Vec::len);
                let values = Values {
                    expressions: rows
                        .iter()
                        .map(|r| Tuple {
                            expressions: r.iter().map(|e| self.expr(e)).collect(),
                        })
                        .collect(),
                    alias: None,
                    column_aliases: Vec::new(),
                };
                subquery(
                    Expression::Values(Box::new(values)),
                    Some(*alias),
                    (0..width)
                        .map(|i| Identifier::new(format!("c{i}")))
                        .collect(),
                )
            }
            From::Set { op, inputs, alias } => {
                let mut inputs = inputs
                    .iter()
                    .map(|s| Expression::Select(Box::new(self.select(s, "c"))));
                let first = inputs.next().expect("a set operation has inputs");
                let set = inputs.fold(first, |left, right| set_op(*op, left, right));
                subquery(set, Some(*alias), Vec::new())
            }
        }
    }

    fn orders(&self, orders: &[Order]) -> Vec<Ordered> {
        orders
            .iter()
            .map(|o| {
                let mut ordered = if o.descending {
                    Ordered::desc(self.expr(&o.expr))
                } else {
                    Ordered::asc(self.expr(&o.expr))
                };
                ordered.explicit_asc = !o.descending;
                ordered.nulls_first = Some(o.nulls_first);
                ordered
            })
            .collect()
    }

    /// An expression's subquery. Its columns are named `__ddx_v0`, …, which
    /// no column it reads is: DataFusion 55's decorrelation fails on a
    /// subquery column named like one it reads (`c0` over `t3.c0`), as an
    /// ambiguous name.
    fn subquery_select(&self, q: &Select) -> Expression {
        Expression::Select(Box::new(self.select(q, "__ddx_v")))
    }

    fn expr(&self, e: &Expr) -> Expression {
        match e {
            Expr::Column { table, column } => Expression::Column(Box::new(pg::Column {
                name: match column {
                    Column::Position(i) => Identifier::new(format!("c{i}")),
                    Column::Named(name) => Identifier::quoted(name.clone()),
                },
                table: Some(alias_name(*table)),
                join_mark: false,
                trailing_comments: Vec::new(),
                span: None,
                inferred_type: None,
            })),
            Expr::Literal(l) => self.literal(l),
            Expr::Op { op, args } => {
                let mut args = args.iter().map(|a| self.expr(a));
                let first = args
                    .next()
                    .expect("an operator takes two arguments or more");
                paren(args.fold(first, |left, right| self.op(*op, left, right)))
            }
            Expr::Not(x) => paren(Expression::Not(Box::new(UnaryOp::new(self.expr(x))))),
            Expr::Negate(x) => paren(Expression::Neg(Box::new(UnaryOp::new(self.expr(x))))),
            Expr::Is { expr, test } => {
                let this = self.expr(expr);
                let truth = |not| {
                    Box::new(IsTrueFalse {
                        this: this.clone(),
                        not,
                    })
                };
                paren(match test {
                    IsTest::Null | IsTest::NotNull => Expression::IsNull(Box::new(IsNull {
                        this,
                        not: *test == IsTest::NotNull,
                        postfix_form: false,
                    })),
                    IsTest::True => Expression::IsTrue(truth(false)),
                    IsTest::NotTrue => Expression::IsTrue(truth(true)),
                    IsTest::False => Expression::IsFalse(truth(false)),
                    IsTest::NotFalse => Expression::IsFalse(truth(true)),
                })
            }
            Expr::Call(call) => self.call(call),
            Expr::Window {
                call,
                partition_by,
                order_by,
                frame,
            } => Expression::WindowFunction(Box::new(WindowFunction {
                this: self.call(call),
                over: Over {
                    window_name: None,
                    partition_by: partition_by.iter().map(|p| self.expr(p)).collect(),
                    order_by: self.orders(order_by),
                    frame: frame.as_ref().map(|f| self.frame(f)),
                    alias: None,
                },
                keep: None,
                inferred_type: None,
            })),
            Expr::Case {
                operand,
                whens,
                otherwise,
            } => Expression::Case(Box::new(PgCase {
                operand: operand.as_deref().map(|x| self.expr(x)),
                whens: whens
                    .iter()
                    .map(|(w, t)| (self.expr(w), self.expr(t)))
                    .collect(),
                else_: otherwise.as_deref().map(|x| self.expr(x)),
                comments: Vec::new(),
                inferred_type: None,
            })),
            Expr::Cast { expr, ty, fallible } => cast(self.expr(expr), self.ty(*ty), *fallible),
            Expr::InList { expr, list } => paren(Expression::In(Box::new(In {
                this: self.expr(expr),
                expressions: list.iter().map(|x| self.expr(x)).collect(),
                query: None,
                not: false,
                global: false,
                unnest: None,
                is_field: false,
            }))),
            Expr::Exists(q) => Expression::Exists(Box::new(Exists {
                this: self.subquery_select(q),
                not: false,
            })),
            Expr::InQuery { expr, query } => paren(Expression::In(Box::new(In {
                this: self.expr(expr),
                expressions: Vec::new(),
                query: Some(self.subquery_select(query)),
                not: false,
                global: false,
                unnest: None,
                is_field: false,
            }))),
            Expr::Scalar(q) => subquery(self.subquery_select(q), None, Vec::new()),
            Expr::Quantified {
                left,
                op,
                all,
                query,
            } => {
                let quantified = QuantifiedExpr {
                    this: self.expr(left),
                    subquery: subquery(self.subquery_select(query), None, Vec::new()),
                    op: Some(match op {
                        Op::Eq => QuantifiedOp::Eq,
                        Op::NotEq => QuantifiedOp::Neq,
                        Op::Lt => QuantifiedOp::Lt,
                        Op::Lte => QuantifiedOp::Lte,
                        Op::Gt => QuantifiedOp::Gt,
                        Op::Gte => QuantifiedOp::Gte,
                        other => unreachable!("a set comparison by {other:?}"),
                    }),
                };
                paren(if *all {
                    Expression::All(Box::new(quantified))
                } else {
                    Expression::Any(Box::new(quantified))
                })
            }
        }
    }

    fn op(&self, op: Op, left: Expression, right: Expression) -> Expression {
        if op == (Op::Divide { integer: true }) && self.dialect == Dialect::DuckDb {
            // Truncating division: DuckDB's `/` of two integers is a DOUBLE,
            // and polyglot writes `IntDiv` as each dialect truncates.
            return Expression::IntDiv(Box::new(BinaryFunc {
                this: left,
                expression: right,
                original_name: None,
                inferred_type: None,
            }));
        }
        let b = Box::new(BinaryOp::new(left, right));
        match op {
            Op::Add => Expression::Add(b),
            Op::Subtract => Expression::Sub(b),
            Op::Multiply => Expression::Mul(b),
            Op::Divide { .. } => Expression::Div(b),
            Op::Modulus => Expression::Mod(b),
            Op::Eq => Expression::Eq(b),
            Op::NotEq => Expression::Neq(b),
            Op::Lt => Expression::Lt(b),
            Op::Lte => Expression::Lte(b),
            Op::Gt => Expression::Gt(b),
            Op::Gte => Expression::Gte(b),
            Op::And => Expression::And(b),
            Op::Or => Expression::Or(b),
            Op::IsDistinctFrom => Expression::NullSafeNeq(b),
            Op::IsNotDistinctFrom => Expression::NullSafeEq(b),
        }
    }

    fn call(&self, c: &Call) -> Expression {
        let args: Vec<Expression> = if c.args.is_empty() && c.function == "count" {
            vec![Expression::Star(star())]
        } else {
            c.args.iter().map(|a| self.expr(a)).collect()
        };
        let name = function(c.function, self.dialect).to_string();
        let aggregate = c.distinct || c.filter.is_some() || !c.order_by.is_empty();
        if !aggregate && !super::AGGREGATES.contains(&c.function) {
            let mut f = Function::new(name, args);
            f.distinct = c.distinct;
            return Expression::Function(Box::new(f));
        }
        Expression::AggregateFunction(Box::new(AggregateFunction {
            name,
            args,
            distinct: c.distinct,
            filter: c.filter.as_deref().map(|f| self.expr(f)),
            order_by: self.orders(&c.order_by),
            ..AggregateFunction::default()
        }))
    }

    fn frame(&self, f: &Frame) -> WindowFrame {
        let bound = |b: Bound| match b {
            Bound::UnboundedPreceding => WindowFrameBound::UnboundedPreceding,
            Bound::Preceding(n) => WindowFrameBound::Preceding(Box::new(number(&n.to_string()))),
            Bound::CurrentRow => WindowFrameBound::CurrentRow,
            Bound::Following(n) => WindowFrameBound::Following(Box::new(number(&n.to_string()))),
            Bound::UnboundedFollowing => WindowFrameBound::UnboundedFollowing,
        };
        WindowFrame {
            kind: match f.units {
                Units::Rows => WindowFrameKind::Rows,
                Units::Range => WindowFrameKind::Range,
            },
            start: bound(f.lower),
            end: Some(bound(f.upper)),
            exclude: None,
            kind_text: None,
            start_side_text: None,
            end_side_text: None,
        }
    }

    fn literal(&self, l: &Literal) -> Expression {
        let typed = |value: Expression, ty: Type| cast(value, self.ty(ty), false);
        match l {
            Literal::Bool(b) => Expression::Boolean(BooleanLiteral { value: *b }),
            Literal::Int(n, ty) => typed(number(&n.to_string()), *ty),
            Literal::Float(x, ty) => {
                let value = if x.is_nan() {
                    string("NaN")
                } else if x.is_infinite() {
                    string(if *x < 0.0 { "-Infinity" } else { "Infinity" })
                } else {
                    // Exponent notation, the shortest that parses back to
                    // `x`: every dialect reads it as a floating-point
                    // number, correctly rounded.
                    number(&format!("{x:e}"))
                };
                typed(value, *ty)
            }
            Literal::String(s) => string(s),
            Literal::Date(days) => typed(string(&date(*days)), Type::Date),
            Literal::Timestamp { value, precision } => typed(
                string(&timestamp(*value, *precision)),
                Type::Timestamp {
                    precision: *precision,
                },
            ),
            Literal::Decimal {
                value,
                precision,
                scale,
            } => typed(
                string(&decimal(*value, *scale)),
                Type::Decimal {
                    precision: *precision,
                    scale: *scale,
                },
            ),
            Literal::Null(ty) => typed(Expression::Null(pg::Null), *ty),
        }
    }

    fn ty(&self, ty: Type) -> DataType {
        let u = |n: i32| Some(n as u32);
        match ty {
            Type::Boolean => DataType::Boolean,
            Type::TinyInt => DataType::TinyInt { length: None },
            Type::SmallInt => DataType::SmallInt { length: None },
            Type::Integer => DataType::Int {
                length: None,
                integer_spelling: true,
            },
            Type::BigInt => DataType::BigInt { length: None },
            Type::Real => DataType::Float {
                precision: None,
                scale: None,
                real_spelling: true,
            },
            Type::Double => DataType::Double {
                precision: None,
                scale: None,
            },
            Type::Varchar => DataType::VarChar {
                length: None,
                parenthesized_length: false,
            },
            Type::Date => DataType::Date,
            Type::Timestamp { precision } => DataType::Timestamp {
                precision: u(precision),
                timezone: false,
            },
            Type::Decimal { precision, scale } => DataType::Decimal {
                precision: u(precision),
                scale: u(scale),
            },
            Type::Binary => match self.dialect {
                Dialect::DataFusion => DataType::VarBinary { length: None },
                Dialect::DuckDb => DataType::Blob,
            },
        }
    }
}

fn paren(this: Expression) -> Expression {
    Expression::Paren(Box::new(Paren {
        this,
        trailing_comments: Vec::new(),
    }))
}

fn cast(this: Expression, to: DataType, fallible: bool) -> Expression {
    let cast = Box::new(PgCast {
        this,
        to,
        trailing_comments: Vec::new(),
        double_colon_syntax: false,
        format: None,
        default: None,
        inferred_type: None,
    });
    if fallible {
        Expression::TryCast(cast)
    } else {
        Expression::Cast(cast)
    }
}

fn number(text: &str) -> Expression {
    Expression::Literal(Box::new(pg::Literal::Number(text.to_string())))
}

fn string(text: &str) -> Expression {
    Expression::Literal(Box::new(pg::Literal::String(text.to_string())))
}

fn star() -> Star {
    Star {
        table: None,
        except: None,
        replace: None,
        rename: None,
        trailing_comments: Vec::new(),
        span: None,
    }
}

fn count_star() -> Expression {
    Expression::AggregateFunction(Box::new(AggregateFunction {
        name: "count".into(),
        args: vec![Expression::Star(star())],
        ..AggregateFunction::default()
    }))
}

fn subquery(this: Expression, alias: Option<Alias>, column_aliases: Vec<Identifier>) -> Expression {
    Expression::Subquery(Box::new(Subquery {
        this,
        alias: alias.map(alias_name),
        column_aliases,
        alias_explicit_as: alias.is_some(),
        alias_keyword: None,
        order_by: None,
        limit: None,
        offset: None,
        distribute_by: None,
        sort_by: None,
        cluster_by: None,
        lateral: false,
        modifiers_inside: false,
        trailing_comments: Vec::new(),
        inferred_type: None,
    }))
}

fn set_op(op: SetOp, left: Expression, right: Expression) -> Expression {
    let all = matches!(op, SetOp::UnionAll | SetOp::ExceptAll | SetOp::IntersectAll);
    match op {
        SetOp::UnionAll | SetOp::Union => Expression::Union(Box::new(pg::Union {
            left,
            right,
            all,
            distinct: false,
            with: None,
            order_by: None,
            limit: None,
            offset: None,
            distribute_by: None,
            sort_by: None,
            cluster_by: None,
            by_name: false,
            side: None,
            kind: None,
            corresponding: false,
            strict: false,
            on_columns: Vec::new(),
        })),
        SetOp::Intersect | SetOp::IntersectAll => Expression::Intersect(Box::new(pg::Intersect {
            left,
            right,
            all,
            distinct: false,
            with: None,
            order_by: None,
            limit: None,
            offset: None,
            distribute_by: None,
            sort_by: None,
            cluster_by: None,
            by_name: false,
            side: None,
            kind: None,
            corresponding: false,
            strict: false,
            on_columns: Vec::new(),
        })),
        SetOp::Except | SetOp::ExceptAll => Expression::Except(Box::new(pg::Except {
            left,
            right,
            all,
            distinct: false,
            with: None,
            order_by: None,
            limit: None,
            offset: None,
            distribute_by: None,
            sort_by: None,
            cluster_by: None,
            by_name: false,
            side: None,
            kind: None,
            corresponding: false,
            strict: false,
            on_columns: Vec::new(),
        })),
    }
}

/// `value` with `scale` (≥ 0) digits after the point.
fn decimal(value: i128, scale: i32) -> String {
    let negative = value < 0;
    let digits = value.unsigned_abs().to_string();
    let scale = scale as usize;
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

/// A timestamp `value` units of 10^-`precision` seconds after the epoch,
/// `precision` one of 0, 3, 6 and 9.
fn timestamp(value: i64, precision: i32) -> String {
    debug_assert!(matches!(precision, 0 | 3 | 6 | 9), "read as one of these");
    let per_second = 10_i64.pow(precision as u32);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn written(e: Expr, dialect: Dialect) -> String {
        query(
            &Query {
                ctes: Vec::new(),
                body: Select {
                    columns: vec![e],
                    names: Some(vec!["x".into()]),
                    ..Select::default()
                },
            },
            dialect,
        )
        .unwrap()
    }

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
        let float = |x| {
            written(
                Expr::Literal(Literal::Float(x, Type::Double)),
                Dialect::DuckDb,
            )
        };
        assert!(
            float(f64::NAN).contains("CAST('NaN' AS DOUBLE)"),
            "{}",
            float(f64::NAN)
        );
        assert!(
            float(0.1).contains("CAST(1e-1 AS DOUBLE)"),
            "{}",
            float(0.1)
        );
    }

    #[test]
    fn identifiers_and_strings_are_quoted() {
        let sql = written(
            Expr::Literal(Literal::String("it's".into())),
            Dialect::DuckDb,
        );
        assert!(sql.contains("'it''s'"), "{sql}");
        let sql = query(
            &Query {
                ctes: Vec::new(),
                body: Select {
                    columns: vec![Expr::Literal(Literal::Bool(true))],
                    names: Some(vec!["a\"b".into()]),
                    ..Select::default()
                },
            },
            Dialect::DataFusion,
        )
        .unwrap();
        assert!(sql.contains("\"a\"\"b\""), "{sql}");
    }

    #[test]
    fn each_timestamp_precision_is_its_own_type() {
        // A plain TIMESTAMP is nanoseconds on DataFusion and microseconds on
        // DuckDB; both read `TIMESTAMP(p)` as the type of precision `p`
        // (DuckDB 1.5.5: `TIMESTAMP_S`, `_MS`, `TIMESTAMP`, `_NS`).
        for dialect in [Dialect::DuckDb, Dialect::DataFusion] {
            for precision in [0, 3, 6, 9] {
                let sql = written(
                    Expr::Literal(Literal::Timestamp {
                        value: 1,
                        precision,
                    }),
                    dialect,
                );
                assert!(
                    sql.contains(&format!("AS TIMESTAMP({precision}))")),
                    "{dialect:?} {precision}: {sql}"
                );
            }
        }
    }

    #[test]
    fn an_aggregate_of_nothing_is_one_row() {
        let sql = query(
            &Query {
                ctes: Vec::new(),
                body: Select {
                    group_by: Some(Vec::new()),
                    ..Select::default()
                },
            },
            Dialect::DuckDb,
        )
        .unwrap();
        assert!(
            sql.to_lowercase().contains("count(*) as __ddx_unit"),
            "{sql}"
        );
    }

    #[test]
    fn integer_division_is_each_dialect_s_own() {
        let divide = |dialect| {
            written(
                Expr::Op {
                    op: Op::Divide { integer: true },
                    args: vec![
                        Expr::Literal(Literal::Int(7, Type::BigInt)),
                        Expr::Literal(Literal::Int(2, Type::BigInt)),
                    ],
                },
                dialect,
            )
        };
        assert!(
            divide(Dialect::DuckDb).contains("//"),
            "{}",
            divide(Dialect::DuckDb)
        );
        assert!(
            divide(Dialect::DataFusion).contains("CAST(7 AS BIGINT) / CAST(2 AS BIGINT)"),
            "{}",
            divide(Dialect::DataFusion)
        );
    }
}
