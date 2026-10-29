// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A [`Query`] written in one dialect: the only place a dialect matters.
//!
//! Every compound expression is parenthesized, so no dialect's precedence
//! rules come into it; every identifier the plan names is quoted, and every
//! name ddx makes up (`__ddx_q0`, `t0`, `c0`) needs no quoting.

use std::fmt::Write as _;

use super::ir::{
    Alias, Bound, Call, Column, Expr, Frame, From, IsTest, JoinKind, Literal, Op, Order, Query,
    Select, SetOp, Type, Units,
};
use super::Dialect;

/// `query` in `dialect`.
pub(super) fn query(query: &Query, dialect: Dialect) -> String {
    let mut p = Printer {
        dialect,
        sql: String::new(),
    };
    for (i, s) in query.ctes.iter().enumerate() {
        p.sql.push_str(if i == 0 { "WITH " } else { ", " });
        write!(p.sql, "__ddx_q{i} AS (").ok();
        p.select(s);
        p.sql.push(')');
    }
    if !query.ctes.is_empty() {
        p.sql.push(' ');
    }
    p.select(&query.body);
    p.sql
}

/// The name of the canonical function `function` in `dialect`. The names
/// [`super::syntax`] gives are the same in every dialect but these.
pub(super) fn function(function: &'static str, dialect: Dialect) -> &'static str {
    match (function, dialect) {
        ("sign", Dialect::DataFusion) => "signum",
        _ => function,
    }
}

/// Prints into one buffer, `sql`, as it walks the tree.
struct Printer {
    dialect: Dialect,
    sql: String,
}

impl Printer {
    /// `items`, each printed by `each`, with `separator` between them.
    fn each<T>(&mut self, items: &[T], separator: &str, mut each: impl FnMut(&mut Self, &T)) {
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                self.sql.push_str(separator);
            }
            each(self, item);
        }
    }

    fn exprs(&mut self, es: &[Expr], separator: &str) {
        self.each(es, separator, |p, e| p.expr(e));
    }

    fn select(&mut self, s: &Select) {
        self.select_named(s, "c");
    }

    /// `s`, its columns named `{prefix}0`, `{prefix}1`, … unless it has
    /// names of its own.
    fn select_named(&mut self, s: &Select, prefix: &str) {
        self.sql.push_str("SELECT ");
        if s.columns.is_empty() {
            // SQL has no empty select list: a relation of no columns gives
            // one, which nothing reads. An aggregate of no keys and no
            // measures still gives one row, as `count(*)` does.
            self.sql
                .push_str(if s.group_by.as_ref().is_some_and(Vec::is_empty) {
                    "count(*) AS __ddx_unit"
                } else {
                    "1 AS __ddx_unit"
                });
        }
        for (i, c) in s.columns.iter().enumerate() {
            if i > 0 {
                self.sql.push_str(", ");
            }
            self.expr(c);
            match &s.names {
                Some(names) => {
                    self.sql.push_str(" AS ");
                    ident(&mut self.sql, &names[i]);
                }
                None => {
                    write!(self.sql, " AS {prefix}{i}").ok();
                }
            }
        }
        if !matches!(s.from, From::Nothing) {
            self.sql.push_str(" FROM ");
            self.from(&s.from);
        }
        if !s.filter.is_empty() {
            self.sql.push_str(" WHERE ");
            self.exprs(&s.filter, " AND ");
        }
        if let Some(keys) = s.group_by.as_ref().filter(|k| !k.is_empty()) {
            self.sql.push_str(" GROUP BY ");
            self.exprs(keys, ", ");
        }
        if !s.order_by.is_empty() {
            self.sql.push_str(" ORDER BY ");
            self.orders(&s.order_by);
        }
        if let Some(n) = s.limit {
            write!(self.sql, " LIMIT {n}").ok();
        }
        if let Some(n) = s.offset {
            write!(self.sql, " OFFSET {n}").ok();
        }
    }

    fn alias(&mut self, a: Alias) {
        write!(self.sql, " AS t{}", a.0).ok();
    }

    fn from(&mut self, f: &From) {
        match f {
            From::Nothing => unreachable!("a select with no FROM prints none"),
            From::Table { name, alias } => {
                for (i, part) in name.iter().enumerate() {
                    if i > 0 {
                        self.sql.push('.');
                    }
                    ident(&mut self.sql, part);
                }
                self.alias(*alias);
            }
            From::Cte { index, alias } => {
                write!(self.sql, "__ddx_q{index}").ok();
                self.alias(*alias);
            }
            From::Subquery { query, alias } => {
                self.sql.push('(');
                self.select(query);
                self.sql.push(')');
                self.alias(*alias);
            }
            From::Values { rows, alias } => {
                self.sql.push_str("(VALUES ");
                self.each(rows, ", ", |p, row| {
                    p.sql.push('(');
                    p.exprs(row, ", ");
                    p.sql.push(')');
                });
                self.sql.push(')');
                self.alias(*alias);
                let width = rows.first().map_or(0, Vec::len);
                self.sql.push('(');
                for i in 0..width {
                    if i > 0 {
                        self.sql.push_str(", ");
                    }
                    write!(self.sql, "c{i}").ok();
                }
                self.sql.push(')');
            }
            From::Join {
                left,
                right,
                kind,
                on,
            } => {
                self.from(left);
                self.sql.push_str(match kind {
                    JoinKind::Inner => " INNER JOIN ",
                    JoinKind::Left => " LEFT JOIN ",
                    JoinKind::Right => " RIGHT JOIN ",
                    JoinKind::Full => " FULL JOIN ",
                    JoinKind::Cross => " CROSS JOIN ",
                });
                self.from(right);
                if let Some(on) = on {
                    self.sql.push_str(" ON ");
                    self.expr(on);
                }
            }
            From::Set { op, inputs, alias } => {
                let op = match op {
                    SetOp::UnionAll => " UNION ALL ",
                    SetOp::Union => " UNION ",
                    SetOp::Except => " EXCEPT ",
                    SetOp::ExceptAll => " EXCEPT ALL ",
                    SetOp::Intersect => " INTERSECT ",
                    SetOp::IntersectAll => " INTERSECT ALL ",
                };
                self.sql.push('(');
                self.each(inputs, op, |p, input| p.select(input));
                self.sql.push(')');
                self.alias(*alias);
            }
        }
    }

    fn orders(&mut self, orders: &[Order]) {
        self.each(orders, ", ", |p, o| {
            p.expr(&o.expr);
            p.sql.push_str(if o.descending { " DESC" } else { " ASC" });
            p.sql.push_str(if o.nulls_first {
                " NULLS FIRST"
            } else {
                " NULLS LAST"
            });
        });
    }

    /// An expression's subquery, parenthesized. Its columns are named
    /// `__ddx_v0`, …, which no column it reads is: DataFusion 55's
    /// decorrelation fails on a subquery column named like one it reads
    /// (`c0` over `t3.c0`), as an ambiguous name.
    fn parenthesized(&mut self, q: &Select) {
        self.sql.push('(');
        self.select_named(q, "__ddx_v");
        self.sql.push(')');
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Column { table, column } => {
                write!(self.sql, "t{}.", table.0).ok();
                match column {
                    Column::Position(i) => {
                        write!(self.sql, "c{i}").ok();
                    }
                    Column::Named(name) => ident(&mut self.sql, name),
                }
            }
            Expr::Literal(l) => self.literal(l),
            Expr::Op { op, args } => {
                let op = format!(" {} ", self.op(*op));
                self.sql.push('(');
                self.exprs(args, &op);
                self.sql.push(')');
            }
            Expr::Not(x) => {
                self.sql.push_str("(NOT ");
                self.expr(x);
                self.sql.push(')');
            }
            Expr::Negate(x) => {
                self.sql.push_str("(- ");
                self.expr(x);
                self.sql.push(')');
            }
            Expr::Is { expr, test } => {
                self.sql.push('(');
                self.expr(expr);
                self.sql.push_str(match test {
                    IsTest::Null => " IS NULL)",
                    IsTest::NotNull => " IS NOT NULL)",
                    IsTest::True => " IS TRUE)",
                    IsTest::False => " IS FALSE)",
                    IsTest::NotTrue => " IS NOT TRUE)",
                    IsTest::NotFalse => " IS NOT FALSE)",
                });
            }
            Expr::Call(call) => self.call(call),
            Expr::Window {
                call,
                partition_by,
                order_by,
                frame,
            } => {
                self.call(call);
                self.sql.push_str(" OVER (");
                let mut space = "";
                if !partition_by.is_empty() {
                    self.sql.push_str("PARTITION BY ");
                    self.exprs(partition_by, ", ");
                    space = " ";
                }
                if !order_by.is_empty() {
                    write!(self.sql, "{space}ORDER BY ").ok();
                    self.orders(order_by);
                    space = " ";
                }
                if let Some(frame) = frame {
                    self.sql.push_str(space);
                    self.frame(frame);
                }
                self.sql.push(')');
            }
            Expr::Case {
                operand,
                whens,
                otherwise,
            } => {
                self.sql.push_str("CASE");
                if let Some(x) = operand {
                    self.sql.push(' ');
                    self.expr(x);
                }
                for (when, then) in whens {
                    self.sql.push_str(" WHEN ");
                    self.expr(when);
                    self.sql.push_str(" THEN ");
                    self.expr(then);
                }
                if let Some(x) = otherwise {
                    self.sql.push_str(" ELSE ");
                    self.expr(x);
                }
                self.sql.push_str(" END");
            }
            Expr::Cast { expr, ty, fallible } => {
                self.sql
                    .push_str(if *fallible { "TRY_CAST(" } else { "CAST(" });
                self.expr(expr);
                self.sql.push_str(" AS ");
                self.ty(*ty);
                self.sql.push(')');
            }
            Expr::InList { expr, list } => {
                self.sql.push('(');
                self.expr(expr);
                self.sql.push_str(" IN (");
                self.exprs(list, ", ");
                self.sql.push_str("))");
            }
            Expr::Exists(q) => {
                self.sql.push_str("EXISTS ");
                self.parenthesized(q);
            }
            Expr::InQuery { expr, query } => {
                self.sql.push('(');
                self.expr(expr);
                self.sql.push_str(" IN ");
                self.parenthesized(query);
                self.sql.push(')');
            }
            Expr::Scalar(q) => self.parenthesized(q),
            Expr::Quantified {
                left,
                op,
                all,
                query,
            } => {
                self.sql.push('(');
                self.expr(left);
                write!(
                    self.sql,
                    " {} {} ",
                    self.op(*op),
                    if *all { "ALL" } else { "ANY" }
                )
                .ok();
                self.parenthesized(query);
                self.sql.push(')');
            }
        }
    }

    fn op(&self, op: Op) -> &'static str {
        match op {
            Op::Add => "+",
            Op::Subtract => "-",
            Op::Multiply => "*",
            // DuckDB's `/` of two integers is a DOUBLE; `//` truncates.
            Op::Divide { integer: true } if self.dialect == Dialect::DuckDb => "//",
            Op::Divide { .. } => "/",
            Op::Modulus => "%",
            Op::Eq => "=",
            Op::NotEq => "<>",
            Op::Lt => "<",
            Op::Lte => "<=",
            Op::Gt => ">",
            Op::Gte => ">=",
            Op::And => "AND",
            Op::Or => "OR",
            Op::IsDistinctFrom => "IS DISTINCT FROM",
            Op::IsNotDistinctFrom => "IS NOT DISTINCT FROM",
        }
    }

    fn call(&mut self, c: &Call) {
        if c.args.is_empty() && c.function == "count" {
            self.sql.push_str("count(*");
        } else {
            write!(self.sql, "{}(", function(c.function, self.dialect)).ok();
            if c.distinct {
                self.sql.push_str("DISTINCT ");
            }
            self.exprs(&c.args, ", ");
        }
        if !c.order_by.is_empty() {
            self.sql.push_str(" ORDER BY ");
            self.orders(&c.order_by);
        }
        self.sql.push(')');
        if let Some(filter) = &c.filter {
            self.sql.push_str(" FILTER (WHERE ");
            self.expr(filter);
            self.sql.push(')');
        }
    }

    fn frame(&mut self, f: &Frame) {
        self.sql.push_str(match f.units {
            Units::Rows => "ROWS BETWEEN ",
            Units::Range => "RANGE BETWEEN ",
        });
        self.bound(f.lower);
        self.sql.push_str(" AND ");
        self.bound(f.upper);
    }

    fn bound(&mut self, b: Bound) {
        match b {
            Bound::UnboundedPreceding => self.sql.push_str("UNBOUNDED PRECEDING"),
            Bound::Preceding(n) => {
                write!(self.sql, "{n} PRECEDING").ok();
            }
            Bound::CurrentRow => self.sql.push_str("CURRENT ROW"),
            Bound::Following(n) => {
                write!(self.sql, "{n} FOLLOWING").ok();
            }
            Bound::UnboundedFollowing => self.sql.push_str("UNBOUNDED FOLLOWING"),
        }
    }

    fn literal(&mut self, l: &Literal) {
        match l {
            Literal::Bool(b) => self.sql.push_str(if *b { "TRUE" } else { "FALSE" }),
            Literal::Int(n, ty) => {
                write!(self.sql, "CAST({n} AS ").ok();
                self.ty(*ty);
                self.sql.push(')');
            }
            Literal::Float(x, ty) => {
                if x.is_nan() {
                    self.sql.push_str("CAST('NaN' AS ");
                } else if x.is_infinite() {
                    let sign = if *x < 0.0 { "-" } else { "" };
                    write!(self.sql, "CAST('{sign}Infinity' AS ").ok();
                } else {
                    // Exponent notation, the shortest that parses back to
                    // `x`: every dialect reads it as a floating-point
                    // number, correctly rounded.
                    write!(self.sql, "CAST({x:e} AS ").ok();
                }
                self.ty(*ty);
                self.sql.push(')');
            }
            Literal::String(s) => string(&mut self.sql, s),
            Literal::Date(days) => {
                write!(self.sql, "CAST('{}' AS DATE)", date(*days)).ok();
            }
            Literal::Timestamp { value, precision } => {
                write!(self.sql, "CAST('{}' AS ", timestamp(*value, *precision)).ok();
                self.ty(Type::Timestamp {
                    precision: *precision,
                });
                self.sql.push(')');
            }
            Literal::Decimal {
                value,
                precision,
                scale,
            } => {
                write!(
                    self.sql,
                    "CAST('{}' AS DECIMAL({precision}, {scale}))",
                    decimal(*value, *scale)
                )
                .ok();
            }
            Literal::Null(ty) => {
                self.sql.push_str("CAST(NULL AS ");
                self.ty(*ty);
                self.sql.push(')');
            }
            Literal::UntypedNull => self.sql.push_str("NULL"),
        }
    }

    fn ty(&mut self, ty: Type) {
        match ty {
            Type::Boolean => self.sql.push_str("BOOLEAN"),
            Type::TinyInt => self.sql.push_str("TINYINT"),
            Type::SmallInt => self.sql.push_str("SMALLINT"),
            Type::Integer => self.sql.push_str("INTEGER"),
            Type::BigInt => self.sql.push_str("BIGINT"),
            Type::Real => self.sql.push_str("REAL"),
            Type::Double => self.sql.push_str("DOUBLE"),
            Type::Varchar => self.sql.push_str("VARCHAR"),
            Type::Date => self.sql.push_str("DATE"),
            // DataFusion's plain TIMESTAMP is nanoseconds, DuckDB's
            // microseconds: each precision written as its own type.
            Type::Timestamp { precision } => match self.dialect {
                Dialect::DataFusion => {
                    write!(self.sql, "TIMESTAMP({precision})").ok();
                }
                Dialect::DuckDb => self.sql.push_str(match precision {
                    0 => "TIMESTAMP_S",
                    3 => "TIMESTAMP_MS",
                    9 => "TIMESTAMP_NS",
                    _ => "TIMESTAMP",
                }),
            },
            Type::Decimal { precision, scale } => {
                write!(self.sql, "DECIMAL({precision}, {scale})").ok();
            }
            Type::Binary => self.sql.push_str(match self.dialect {
                Dialect::DataFusion => "BYTEA",
                Dialect::DuckDb => "BLOB",
            }),
        }
    }
}

/// A quoted identifier.
fn ident(sql: &mut String, name: &str) {
    sql.push('"');
    sql.push_str(&name.replace('"', "\"\""));
    sql.push('"');
}

fn string(sql: &mut String, s: &str) {
    sql.push('\'');
    sql.push_str(&s.replace('\'', "''"));
    sql.push('\'');
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
        let print = |x| {
            let mut p = Printer {
                dialect: Dialect::DuckDb,
                sql: String::new(),
            };
            p.literal(&Literal::Float(x, Type::Double));
            p.sql
        };
        assert_eq!(print(f64::NAN), "CAST('NaN' AS DOUBLE)");
        assert_eq!(print(f64::NEG_INFINITY), "CAST('-Infinity' AS DOUBLE)");
        assert_eq!(print(0.1), "CAST(1e-1 AS DOUBLE)");
    }

    #[test]
    fn identifiers_and_strings_are_quoted() {
        let mut sql = String::new();
        ident(&mut sql, "a\"b");
        sql.push(' ');
        string(&mut sql, "it's");
        assert_eq!(sql, "\"a\"\"b\" 'it''s'");
    }

    #[test]
    fn each_timestamp_precision_is_its_own_type() {
        let print = |dialect, precision| {
            let mut p = Printer {
                dialect,
                sql: String::new(),
            };
            p.literal(&Literal::Timestamp {
                value: 1,
                precision,
            });
            p.sql
        };
        assert_eq!(
            print(Dialect::DuckDb, 9),
            "CAST('1970-01-01 00:00:00.000000001' AS TIMESTAMP_NS)"
        );
        assert_eq!(
            print(Dialect::DataFusion, 6),
            "CAST('1970-01-01 00:00:00.000001' AS TIMESTAMP(6))"
        );
        assert_eq!(
            print(Dialect::DuckDb, 0),
            "CAST('1970-01-01 00:00:01' AS TIMESTAMP_S)"
        );
    }

    #[test]
    fn an_aggregate_of_nothing_is_one_row() {
        let mut p = Printer {
            dialect: Dialect::DuckDb,
            sql: String::new(),
        };
        p.select(&Select {
            group_by: Some(Vec::new()),
            ..Select::default()
        });
        assert_eq!(p.sql, "SELECT count(*) AS __ddx_unit");
    }

    #[test]
    fn integer_division_is_each_dialect_s_own() {
        let divide = |dialect| {
            let e = Expr::Op {
                op: Op::Divide { integer: true },
                args: vec![
                    Expr::Literal(Literal::Int(7, Type::BigInt)),
                    Expr::Literal(Literal::Int(2, Type::BigInt)),
                ],
            };
            let mut p = Printer {
                dialect,
                sql: String::new(),
            };
            p.expr(&e);
            p.sql
        };
        assert_eq!(
            divide(Dialect::DuckDb),
            "(CAST(7 AS BIGINT) // CAST(2 AS BIGINT))"
        );
        assert_eq!(
            divide(Dialect::DataFusion),
            "(CAST(7 AS BIGINT) / CAST(2 AS BIGINT))"
        );
    }
}
