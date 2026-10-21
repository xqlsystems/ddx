// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! SQL as a typed tree: what a query means, before a dialect spells it.
//!
//! [`super::to_sql`] reads a Substrait plan into this tree, and
//! [`super::print`] spells the tree in one dialect. The tree says what the
//! plan means and nothing about spelling: a `divide` knows whether it is
//! integer division, a literal knows its type, a window its frame, and a
//! function is named by ddx's canonical name. Every choice a dialect makes
//! (`//` or `/`, `signum` or `sign`, `BLOB` or `BYTEA`, where parentheses
//! go) is the printer's alone.

/// A query: CTEs, each named by its position (`__ddx_q{i}`), then the
/// select that reads them.
pub(super) struct Query {
    pub ctes: Vec<Select>,
    pub body: Select,
}

/// One `SELECT`. Its columns are named by position (`c0`, `c1`, …), or by
/// `names` at the root.
#[derive(Clone, Default)]
pub(super) struct Select {
    pub columns: Vec<Expr>,
    pub names: Option<Vec<String>>,
    pub from: From,
    /// Conditions, all of which a row must meet.
    pub filter: Vec<Expr>,
    /// `Some` for an aggregate: its grouping keys, perhaps none.
    pub group_by: Option<Vec<Expr>>,
    pub order_by: Vec<Order>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// The alias a `FROM` item is read under, `t{n}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Alias(pub usize);

#[derive(Clone, Default)]
pub(super) enum From {
    /// No `FROM` clause: one row.
    #[default]
    Nothing,
    /// A table, by its name's parts.
    Table { name: Vec<String>, alias: Alias },
    /// The query's CTE `index`.
    Cte { index: usize, alias: Alias },
    /// A query written in place.
    Subquery { query: Box<Select>, alias: Alias },
    /// Rows of values, at least one, with columns `c0`, `c1`, ….
    Values { rows: Vec<Vec<Expr>>, alias: Alias },
    Join {
        left: Box<From>,
        right: Box<From>,
        kind: JoinKind,
        /// `None` only for a cross join.
        on: Option<Expr>,
    },
    /// A set operation over `inputs`, each read whole (`SELECT *`).
    Set {
        op: SetOp,
        inputs: Vec<From>,
        alias: Alias,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum JoinKind {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SetOp {
    UnionAll,
    Union,
    Except,
    ExceptAll,
    Intersect,
    IntersectAll,
}

#[derive(Clone)]
pub(super) enum Expr {
    /// Column `column` of the `FROM` item read under `table`.
    Column {
        table: Alias,
        column: Column,
    },
    Literal(Literal),
    /// `args` joined by `op`, which takes two or more.
    Op {
        op: Op,
        args: Vec<Expr>,
    },
    Not(Box<Expr>),
    Negate(Box<Expr>),
    Is {
        expr: Box<Expr>,
        test: IsTest,
    },
    /// A scalar or aggregate function, by its canonical name. `count` of no
    /// arguments is `count(*)`.
    Call(Call),
    /// A window function and its window. `frame` is `None` only for the
    /// functions that take none (ranking and offset functions).
    Window {
        call: Call,
        partition_by: Vec<Expr>,
        order_by: Vec<Order>,
        frame: Option<Frame>,
    },
    Case {
        operand: Option<Box<Expr>>,
        whens: Vec<(Expr, Expr)>,
        otherwise: Option<Box<Expr>>,
    },
    /// `CAST`, or `TRY_CAST` (NULL where the cast fails) if `fallible`.
    Cast {
        expr: Box<Expr>,
        ty: Type,
        fallible: bool,
    },
    /// `expr IN (list)`, `list` not empty.
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
    },
    /// `EXISTS (query)`.
    Exists(Box<Select>),
    /// `expr IN (query)`.
    InQuery {
        expr: Box<Expr>,
        query: Box<Select>,
    },
    /// `(query)`, a one-column query giving at most one row.
    Scalar(Box<Select>),
    /// `left op ANY (query)`, or `ALL` if `all`.
    Quantified {
        left: Box<Expr>,
        op: Op,
        all: bool,
        query: Box<Select>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Column {
    /// `c{n}`, a column of a query this tree writes.
    Position(usize),
    /// A table's column, by name.
    Named(String),
}

#[derive(Clone)]
pub(super) struct Call {
    /// ddx's canonical name: one the printer has a spelling of.
    pub function: &'static str,
    pub args: Vec<Expr>,
    pub distinct: bool,
    /// An aggregate's ordering of its arguments.
    pub order_by: Vec<Order>,
    /// An aggregate's `FILTER (WHERE …)`.
    pub filter: Option<Box<Expr>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Op {
    Add,
    Subtract,
    Multiply,
    /// Division, truncating if `integer` (its output is an integer).
    Divide {
        integer: bool,
    },
    Modulus,
    Eq,
    NotEq,
    Lt,
    Lte,
    Gt,
    Gte,
    And,
    Or,
    IsDistinctFrom,
    IsNotDistinctFrom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IsTest {
    Null,
    NotNull,
    True,
    False,
    NotTrue,
    NotFalse,
}

#[derive(Clone)]
pub(super) struct Order {
    pub expr: Expr,
    pub descending: bool,
    pub nulls_first: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Frame {
    pub units: Units,
    pub lower: Bound,
    pub upper: Bound,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Units {
    Rows,
    Range,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Bound {
    UnboundedPreceding,
    Preceding(i64),
    CurrentRow,
    Following(i64),
    UnboundedFollowing,
}

/// A literal, with its type.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Literal {
    Bool(bool),
    /// An integer of an integer type.
    Int(i64, Type),
    /// A floating-point number of `REAL` or `DOUBLE`.
    Float(f64, Type),
    String(String),
    /// Days since 1970-01-01.
    Date(i64),
    /// `value` units of 10^-`precision` seconds since the epoch.
    Timestamp {
        value: i64,
        precision: i32,
    },
    /// `value` × 10^-`scale`, of `DECIMAL(precision, scale)`.
    Decimal {
        value: i128,
        precision: i32,
        scale: i32,
    },
    Null(Type),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Type {
    Boolean,
    TinyInt,
    SmallInt,
    Integer,
    BigInt,
    Real,
    Double,
    Varchar,
    Date,
    Timestamp,
    Decimal { precision: i32, scale: i32 },
    Binary,
}
