// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx_ad::to_sql` on random queries: whatever DataFusion plans, the SQL
//! `to_sql` writes of the plan must mean the plan.
//!
//! Each seed builds random tables (NULLs, NaN, ±0, ±∞, extreme integers,
//! strings with quotes, backslashes and non-ASCII, column names like ddx's
//! own `c0` and `t1` and like SQL keywords) and a random query over them:
//! joins of every kind, derived tables, `VALUES`, aggregates with
//! `DISTINCT` and `FILTER`, `HAVING`, windows with frames, set operations,
//! correlated subqueries of every kind, `CASE`, casts, `IN` lists, and an
//! `ORDER BY` and `LIMIT`. DataFusion plans it, optimized and not (the
//! unoptimized plan keeps subqueries as expressions, so it reaches
//! `to_sql`'s environment of enclosing columns), and these must hold:
//!
//! - **Meaning.** The written SQL gives the rows, and the column names and
//!   types, of the query, or of the plan run through DataFusion's Substrait
//!   consumer (`from_substrait_plan`): one of the two, since the consumer
//!   has round-trip bugs of its own (#104), which are not `to_sql`'s.
//!   An `ORDER BY` at the root orders the rows: the query's ends with every
//!   output column, so its order is determined, and the rows are compared
//!   in order.
//! - **Sharing.** With every repeated subtree shared through `ReferenceRel`
//!   (`common::shared`), the SQL gives the same rows.
//! - **A fixed point.** The written SQL, planned again and written again,
//!   gives the same rows.
//! - **Determinism.** Writing a plan twice gives the same text.
//! - **Refusals.** `to_sql` refuses only with `NotImplemented`, in both
//!   dialects alike, and never panics: a DataFusion plan is never an
//!   `InvalidPlan`.
//!
//! `DDX_TO_SQL_SEEDS` (default 300) sets the number of seeds and
//! `DDX_TO_SQL_BASE` the first; `DDX_TO_SQL_VERBOSE=1` prints each failure's
//! written SQL.

mod common;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use datafusion::arrow::array::{Array, AsArray, RecordBatch};
use datafusion::arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt64Type,
};
use datafusion::prelude::{SessionConfig, SessionContext};
use ddx_ad::substrait::proto::Plan;
use ddx_ad::{to_sql, AdError, Dialect};
use ddx_core::test_utils::{seeded, Rng};

// ---------------------------------------------------------------- values

/// A cell, compared with a tolerance for floats: two plans of one query may
/// add in different orders.
#[derive(Clone, Debug)]
enum Cell {
    Null,
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Other(String),
}

impl Cell {
    fn same(&self, other: &Cell) -> bool {
        match (self, other) {
            (Cell::Float(a), Cell::Float(b)) => {
                (a.is_nan() && b.is_nan())
                    || a == b && a.is_sign_negative() == b.is_sign_negative()
                    || (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
            }
            (Cell::Null, Cell::Null) => true,
            (Cell::Int(a), Cell::Int(b)) => a == b,
            (Cell::Str(a), Cell::Str(b)) | (Cell::Other(a), Cell::Other(b)) => a == b,
            (Cell::Bool(a), Cell::Bool(b)) => a == b,
            _ => false,
        }
    }

    /// A sort key that floats within the tolerance share, mostly.
    fn key(&self) -> String {
        match self {
            Cell::Null => "0".into(),
            Cell::Int(n) => format!("1{n:+021}"),
            Cell::Float(x) if x.is_nan() => "2nan".into(),
            Cell::Float(x) => format!("2{x:+.6e}"),
            Cell::Str(s) => format!("3{s}"),
            Cell::Bool(b) => format!("4{b}"),
            Cell::Other(s) => format!("5{s}"),
        }
    }
}

type Row = Vec<Cell>;

/// What a query gave: its columns' names and types, and its rows.
#[derive(Debug)]
struct Result_ {
    columns: Vec<(String, DataType)>,
    rows: Vec<Row>,
}

fn cells(batches: &[RecordBatch]) -> Vec<Row> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let row = (0..b.num_columns())
                .map(|c| {
                    let a = b.column(c);
                    if a.is_null(r) {
                        return Cell::Null;
                    }
                    match a.data_type() {
                        DataType::Int8 => Cell::Int(a.as_primitive::<Int8Type>().value(r).into()),
                        DataType::Int16 => Cell::Int(a.as_primitive::<Int16Type>().value(r).into()),
                        DataType::Int32 => Cell::Int(a.as_primitive::<Int32Type>().value(r).into()),
                        DataType::Int64 => Cell::Int(a.as_primitive::<Int64Type>().value(r)),
                        DataType::UInt64 => {
                            Cell::Int(a.as_primitive::<UInt64Type>().value(r) as i64)
                        }
                        DataType::Float32 => {
                            Cell::Float(a.as_primitive::<Float32Type>().value(r).into())
                        }
                        DataType::Float64 => Cell::Float(a.as_primitive::<Float64Type>().value(r)),
                        DataType::Utf8 => Cell::Str(a.as_string::<i32>().value(r).to_string()),
                        DataType::Utf8View => Cell::Str(a.as_string_view().value(r).to_string()),
                        DataType::LargeUtf8 => Cell::Str(a.as_string::<i64>().value(r).to_string()),
                        DataType::Boolean => Cell::Bool(a.as_boolean().value(r)),
                        _ => Cell::Other(
                            datafusion::arrow::util::display::array_value_to_string(a, r)
                                .unwrap_or_default(),
                        ),
                    }
                })
                .collect();
            out.push(row);
        }
    }
    out
}

async fn run_sql(ctx: &SessionContext, sql: &str) -> Result<Result_, String> {
    let df = ctx.sql(sql).await.map_err(|e| e.to_string())?;
    collect(df).await
}

async fn collect(df: datafusion::dataframe::DataFrame) -> Result<Result_, String> {
    let columns = df
        .schema()
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    let batches = df.collect().await.map_err(|e| e.to_string())?;
    Ok(Result_ {
        columns,
        rows: cells(&batches),
    })
}

async fn run_plan(ctx: &SessionContext, plan: &Plan) -> Result<Result_, String> {
    let lp = datafusion_substrait::logical_plan::consumer::from_substrait_plan(&ctx.state(), plan)
        .await
        .map_err(|e| e.to_string())?;
    let df = ctx
        .execute_logical_plan(lp)
        .await
        .map_err(|e| e.to_string())?;
    collect(df).await
}

/// Where `got` and `want` differ, if they do: in order if `ordered`.
fn differs(want: &Result_, got: &Result_, ordered: bool) -> Option<String> {
    let names = |r: &Result_| r.columns.iter().map(|c| c.0.clone()).collect::<Vec<_>>();
    if names(want) != names(got) {
        return Some(format!("columns {:?}, want {:?}", names(got), names(want)));
    }
    // Compared by family: Substrait has no unsigned integers, and a string's
    // Arrow representation (`Utf8`, `Utf8View`) is the engine's choice.
    let family = |t: &DataType| match t {
        t if t.is_integer() => "integer".to_string(),
        t if t.is_floating() => "float".to_string(),
        DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => "string".to_string(),
        t => t.to_string(),
    };
    let types = |r: &Result_| r.columns.iter().map(|c| family(&c.1)).collect::<Vec<_>>();
    if types(want) != types(got) {
        return Some(format!(
            "column types {:?}, want {:?}",
            types(got),
            types(want)
        ));
    }
    if want.rows.len() != got.rows.len() {
        return Some(format!("{} rows, want {}", got.rows.len(), want.rows.len()));
    }
    let sorted = |rows: &[Row]| {
        let mut rows = rows.to_vec();
        if !ordered {
            rows.sort_by_cached_key(|r| r.iter().map(Cell::key).collect::<Vec<_>>());
        }
        rows
    };
    let (w, g) = (sorted(&want.rows), sorted(&got.rows));
    for (i, (a, b)) in w.iter().zip(&g).enumerate() {
        if a.len() != b.len() || a.iter().zip(b).any(|(x, y)| !x.same(y)) {
            return Some(format!(
                "{} row {i} is {b:?}, want {a:?}",
                if ordered { "in order," } else { "sorted," }
            ));
        }
    }
    None
}

// ---------------------------------------------------------------- tables

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    Int,
    Float,
    Str,
    Bool,
}

const TYS: [Ty; 4] = [Ty::Int, Ty::Float, Ty::Str, Ty::Bool];

/// A column in scope: `qualifier.name`, quoted, of type `ty`.
#[derive(Clone, Debug)]
struct Col {
    q: String,
    name: String,
    ty: Ty,
}

impl Col {
    fn sql(&self) -> String {
        format!("{}.{}", self.q, quote(&self.name))
    }
}

fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

struct Table {
    name: String,
    /// `id` first: unique per row.
    columns: Vec<(String, Ty)>,
    /// The statements that make it in DuckDB, for the cross-engine check.
    duckdb: Vec<String>,
}

/// Column names, some like the names `to_sql` makes up or like keywords.
const NAMES: &[&[&str]] = &[
    &["k", "x", "g", "b", "n"],
    &["c0", "t1", "Select", "c1", "A b"],
    &["k", "c2", "x \"y\"", "t0", "n"],
];

fn int_value(rng: &mut Rng) -> String {
    match rng.below(12) {
        0 => "NULL".into(),
        1 => "9223372036854775807".into(),
        2 => "-9223372036854775807".into(),
        3 => "0".into(),
        _ => (rng.below(7) as i64 - 3).to_string(),
    }
}

fn float_value(rng: &mut Rng) -> String {
    match rng.below(16) {
        0 => "NULL".into(),
        1 => "CAST('NaN' AS DOUBLE)".into(),
        2 => "CAST('inf' AS DOUBLE)".into(),
        3 => "CAST('-inf' AS DOUBLE)".into(),
        4 => "CAST('-0.0' AS DOUBLE)".into(),
        5 => "1e300".into(),
        6 => "5e-324".into(),
        7 => "0.1".into(),
        _ => format!("{:.2}", rng.range(-3.0, 3.0)),
    }
}

const STRINGS: &[&str] = &[
    "'a'",
    "'b'",
    "'it''s'",
    r"'back\slash'",
    "'é😈'",
    "''",
    "' '",
    "NULL",
    "'A'",
    "'--'",
];

fn str_value(rng: &mut Rng) -> String {
    rng.pick(STRINGS).to_string()
}

fn bool_value(rng: &mut Rng) -> String {
    rng.pick(&["TRUE", "FALSE", "NULL"]).to_string()
}

fn value(rng: &mut Rng, ty: Ty) -> String {
    match ty {
        Ty::Int => int_value(rng),
        Ty::Float => float_value(rng),
        Ty::Str => str_value(rng),
        Ty::Bool => bool_value(rng),
    }
}

fn sql_type(ty: Ty) -> &'static str {
    match ty {
        Ty::Int => "BIGINT",
        Ty::Float => "DOUBLE",
        Ty::Str => "VARCHAR",
        Ty::Bool => "BOOLEAN",
    }
}

async fn tables(rng: &mut Rng, ctx: &SessionContext) -> Vec<Table> {
    let mut out = Vec::new();
    for (i, name) in ["a", "B b", "c"].iter().enumerate() {
        let pool = NAMES[(i + rng.below(3) as usize) % NAMES.len()];
        let mut columns = vec![("id".to_string(), Ty::Int)];
        // Small key columns, so joins match.
        columns.push((pool[0].to_string(), Ty::Int));
        for n in &pool[1..] {
            columns.push((n.to_string(), *rng.pick(&TYS)));
        }
        let rows = rng.below(7) as usize;
        let create = format!(
            "CREATE OR REPLACE TABLE {} ({})",
            quote(name),
            columns
                .iter()
                .map(|(n, t)| format!("{} {}", quote(n), sql_type(*t)))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut sql = create.clone();
        let mut duckdb = vec![create];
        if rows > 0 {
            sql.push_str(" AS VALUES ");
            for r in 0..rows {
                if r > 0 {
                    sql.push_str(", ");
                }
                let mut vals = vec![r.to_string()];
                vals.push(match rng.below(5) {
                    0 => "NULL".into(),
                    _ => rng.below(3).to_string(),
                });
                for (_, t) in &columns[2..] {
                    vals.push(value(rng, *t));
                }
                write!(sql, "({})", vals.join(", ")).ok();
            }
            let values = &sql[sql.find(" AS VALUES ").unwrap() + " AS VALUES ".len()..];
            duckdb.push(format!("INSERT INTO {} VALUES {values}", quote(name)));
        }
        ctx.sql(&sql)
            .await
            .unwrap_or_else(|e| panic!("{e}\n{sql}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{e}\n{sql}"));
        out.push(Table {
            name: name.to_string(),
            columns,
            duckdb,
        });
    }
    out
}

// ---------------------------------------------------------------- queries

struct Gen<'t> {
    rng: Rng,
    tables: &'t [Table],
    aliases: usize,
    /// Whether the query's root orders its rows completely.
    ordered: bool,
}

/// A `FROM` clause and the columns it brings into scope; `keys` are columns
/// whose values together tell its rows apart, if it has them.
struct Scope {
    from: String,
    cols: Vec<Col>,
    keys: Vec<Col>,
}

impl Gen<'_> {
    fn alias(&mut self) -> String {
        self.aliases += 1;
        format!("s{}", self.aliases)
    }

    fn chance(&mut self, n: u64) -> bool {
        self.rng.below(n) == 0
    }

    fn table_scope(&mut self) -> Scope {
        let t = &self.tables[self.rng.below(self.tables.len() as u64) as usize];
        let q = self.alias();
        let cols: Vec<Col> = t
            .columns
            .iter()
            .map(|(n, ty)| Col {
                q: q.clone(),
                name: n.clone(),
                ty: *ty,
            })
            .collect();
        Scope {
            from: format!("{} AS {q}", quote(&t.name)),
            keys: vec![cols[0].clone()],
            cols,
        }
    }

    fn values_scope(&mut self) -> Scope {
        let q = self.alias();
        let tys: Vec<Ty> = (0..1 + self.rng.below(3))
            .map(|_| *self.rng.pick(&TYS))
            .collect();
        let rows = 1 + self.rng.below(3);
        let mut body = Vec::new();
        for r in 0..rows {
            let mut vals = vec![format!("CAST({r} AS BIGINT)")];
            for t in &tys {
                let v = value(&mut self.rng, *t);
                vals.push(format!("CAST({v} AS {})", sql_type(*t)));
            }
            body.push(format!("({})", vals.join(", ")));
        }
        let mut names = vec!["id".to_string()];
        names.extend((0..tys.len()).map(|i| format!("v{i}")));
        let mut cols = vec![Col {
            q: q.clone(),
            name: "id".into(),
            ty: Ty::Int,
        }];
        for (i, t) in tys.iter().enumerate() {
            cols.push(Col {
                q: q.clone(),
                name: format!("v{i}"),
                ty: *t,
            });
        }
        Scope {
            from: format!(
                "(VALUES {}) AS {q}({})",
                body.join(", "),
                names
                    .iter()
                    .map(|n| quote(n))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            keys: vec![cols[0].clone()],
            cols,
        }
    }

    fn derived_scope(&mut self, depth: u32) -> Scope {
        let (sql, out) = self.select(depth - 1, &[], None, false);
        let q = self.alias();
        let cols = out
            .iter()
            .enumerate()
            .map(|(i, ty)| Col {
                q: q.clone(),
                name: format!("o{i}"),
                ty: *ty,
            })
            .collect();
        Scope {
            from: format!("({sql}) AS {q}"),
            cols,
            keys: Vec::new(),
        }
    }

    fn base_scope(&mut self, depth: u32) -> Scope {
        match self.rng.below(10) {
            0 => self.values_scope(),
            1 | 2 if depth > 0 => self.derived_scope(depth),
            _ => self.table_scope(),
        }
    }

    fn from(&mut self, depth: u32) -> Scope {
        let left = self.base_scope(depth);
        if self.chance(2) {
            return left;
        }
        let right = self.base_scope(depth);
        let kind = *self
            .rng
            .pick(&["JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "CROSS JOIN"]);
        let mut on = String::new();
        if kind != "CROSS JOIN" {
            let pairs: Vec<(Col, Col)> = left
                .cols
                .iter()
                .flat_map(|l| {
                    right
                        .cols
                        .iter()
                        .filter(|r| r.ty == l.ty)
                        .map(move |r| (l.clone(), r.clone()))
                })
                .collect();
            let cond = if pairs.is_empty() {
                "TRUE".to_string()
            } else {
                let (l, r) = self.rng.pick(&pairs).clone();
                let op = *self.rng.pick(&["=", "=", "IS NOT DISTINCT FROM", "<"]);
                format!("{} {op} {}", l.sql(), r.sql())
            };
            let mut all = left.cols.clone();
            all.extend(right.cols.clone());
            let extra = if self.chance(3) {
                format!(" AND {}", self.boolean(&all, 1, 0))
            } else {
                String::new()
            };
            on = format!(" ON {cond}{extra}");
        }
        let mut cols = left.cols;
        cols.extend(right.cols);
        let mut keys = Vec::new();
        if !left.keys.is_empty() && !right.keys.is_empty() {
            keys = left.keys;
            keys.extend(right.keys);
        }
        Scope {
            from: format!("{} {kind} {}{on}", left.from, right.from),
            cols,
            keys,
        }
    }

    fn col_of(&mut self, cols: &[Col], ty: Ty) -> Option<String> {
        let of: Vec<&Col> = cols.iter().filter(|c| c.ty == ty).collect();
        if of.is_empty() {
            None
        } else {
            Some(of[self.rng.below(of.len() as u64) as usize].sql())
        }
    }

    fn literal(&mut self, ty: Ty) -> String {
        match ty {
            Ty::Int => match self.rng.below(8) {
                0 => "CAST(-9223372036854775807 AS BIGINT) - 1".into(),
                1 => "CAST(NULL AS BIGINT)".into(),
                _ => (self.rng.below(9) as i64 - 4).to_string(),
            },
            Ty::Float => float_value(&mut self.rng).replace("NULL", "CAST(NULL AS DOUBLE)"),
            Ty::Str => str_value(&mut self.rng).replace("NULL", "CAST(NULL AS VARCHAR)"),
            Ty::Bool => bool_value(&mut self.rng).replace("NULL", "CAST(NULL AS BOOLEAN)"),
        }
    }

    /// An expression of type `ty` over `cols`; `subq` > 0 allows subqueries.
    fn expr(&mut self, cols: &[Col], ty: Ty, depth: u32, subq: u32) -> String {
        if depth == 0 || self.chance(3) {
            if !self.chance(4) {
                if let Some(c) = self.col_of(cols, ty) {
                    return c;
                }
            }
            return self.literal(ty);
        }
        let d = depth - 1;
        if self.chance(8) {
            let c = self.boolean(cols, d, subq);
            let a = self.expr(cols, ty, d, subq);
            let b = self.expr(cols, ty, d, subq);
            return if self.chance(2) {
                format!("CASE WHEN {c} THEN {a} ELSE {b} END")
            } else {
                let x = self.expr(cols, Ty::Int, d, 0);
                format!("CASE {x} WHEN 0 THEN {a} WHEN 1 THEN {b} END")
            };
        }
        if subq > 0 && self.chance(10) {
            if let Some(s) = self.scalar_subquery(cols, ty, subq - 1) {
                return s;
            }
        }
        match ty {
            Ty::Int => {
                let a = self.expr(cols, Ty::Int, d, subq);
                match self.rng.below(9) {
                    0 => format!("({a} + {})", self.expr(cols, Ty::Int, d, subq)),
                    1 => format!("({a} - {})", self.expr(cols, Ty::Int, d, subq)),
                    2 => format!("({a} * {})", self.literal(Ty::Int)),
                    3 => format!("({a} / {})", *self.rng.pick(&["2", "-3", "7"])),
                    4 => format!("({a} % {})", *self.rng.pick(&["2", "-3"])),
                    5 => format!("(- {a})"),
                    6 => format!("abs({a})"),
                    7 => {
                        let f = self.expr(cols, Ty::Float, d, subq);
                        format!("TRY_CAST({f} AS BIGINT)")
                    }
                    _ => format!("coalesce({a}, {})", self.literal(Ty::Int)),
                }
            }
            Ty::Float => {
                let a = self.expr(cols, Ty::Float, d, subq);
                match self.rng.below(11) {
                    0 => format!("({a} + {})", self.expr(cols, Ty::Float, d, subq)),
                    1 => format!("({a} * {})", self.expr(cols, Ty::Float, d, subq)),
                    2 => format!("({a} / {})", self.expr(cols, Ty::Float, d, subq)),
                    3 => format!("sqrt(abs({a}))"),
                    4 => format!("round({a})"),
                    5 => format!("signum({a})"),
                    6 => {
                        let i = self.expr(cols, Ty::Int, d, subq);
                        format!("CAST({i} AS DOUBLE)")
                    }
                    7 => format!("greatest({a}, {})", self.literal(Ty::Float)),
                    8 => format!("nullif({a}, 0.0)"),
                    9 => format!("CAST(CAST({a} AS REAL) AS DOUBLE)"),
                    _ => format!("(- {a})"),
                }
            }
            Ty::Str => {
                let a = self.expr(cols, Ty::Str, d, subq);
                match self.rng.below(3) {
                    0 => format!("coalesce({a}, {})", self.literal(Ty::Str)),
                    1 => {
                        let i = self.expr(cols, Ty::Int, d, subq);
                        format!("CAST({i} AS VARCHAR)")
                    }
                    _ => a,
                }
            }
            Ty::Bool => self.boolean(cols, d, subq),
        }
    }

    fn boolean(&mut self, cols: &[Col], depth: u32, subq: u32) -> String {
        if subq > 0 && self.chance(4) {
            if let Some(s) = self.predicate_subquery(cols, subq - 1) {
                return s;
            }
        }
        let d = depth.saturating_sub(1);
        match self.rng.below(9) {
            0 if depth > 0 => format!(
                "({} AND {})",
                self.boolean(cols, d, subq),
                self.boolean(cols, d, subq)
            ),
            1 if depth > 0 => format!(
                "({} OR {})",
                self.boolean(cols, d, subq),
                self.boolean(cols, d, subq)
            ),
            2 if depth > 0 => format!("(NOT {})", self.boolean(cols, d, subq)),
            3 => {
                let ty = *self.rng.pick(&TYS);
                let e = self.expr(cols, ty, d, subq);
                format!("({e} IS {}NULL)", if self.chance(2) { "NOT " } else { "" })
            }
            4 => {
                let ty = *self.rng.pick(&[Ty::Int, Ty::Str]);
                let e = self.expr(cols, ty, d, subq);
                let list: Vec<String> = (0..1 + self.rng.below(3))
                    .map(|_| self.literal(ty))
                    .collect();
                format!(
                    "({e} {}IN ({}))",
                    if self.chance(3) { "NOT " } else { "" },
                    list.join(", ")
                )
            }
            5 => {
                let f = self.expr(cols, Ty::Float, d, subq);
                format!("isnan({f})")
            }
            6 => {
                if let Some(c) = self.col_of(cols, Ty::Bool) {
                    c
                } else {
                    self.literal(Ty::Bool)
                }
            }
            _ => {
                let ty = *self.rng.pick(&[Ty::Int, Ty::Float, Ty::Str]);
                let a = self.expr(cols, ty, d, subq);
                let b = self.expr(cols, ty, d, subq);
                let op = *self
                    .rng
                    .pick(&["=", "<>", "<", "<=", ">", ">=", "IS DISTINCT FROM"]);
                format!("({a} {op} {b})")
            }
        }
    }

    /// `EXISTS`, `IN` or `ANY`/`ALL` over a table, perhaps correlated with
    /// `outer`.
    fn predicate_subquery(&mut self, outer: &[Col], subq: u32) -> Option<String> {
        let inner = self.table_scope();
        let corr = self.correlation(outer, &inner.cols);
        let mut wh = Vec::new();
        if let Some(c) = corr {
            wh.push(c);
        }
        if self.chance(2) {
            let mut all = inner.cols.clone();
            all.extend(outer.iter().cloned());
            wh.push(self.boolean(&inner.cols, 1, subq));
        }
        let wh = if wh.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", wh.join(" AND "))
        };
        Some(match self.rng.below(4) {
            0 => format!(
                "({}EXISTS (SELECT 1 FROM {}{wh}))",
                if self.chance(3) { "NOT " } else { "" },
                inner.from
            ),
            1 | 2 => {
                let ty = *self.rng.pick(&[Ty::Int, Ty::Str]);
                let needle = self.col_of(outer, ty)?;
                let hay = self.col_of(&inner.cols, ty)?;
                format!(
                    "({needle} {}IN (SELECT {hay} FROM {}{wh}))",
                    if self.chance(3) { "NOT " } else { "" },
                    inner.from
                )
            }
            _ => {
                let left = self.col_of(outer, Ty::Int)?;
                let right = self.col_of(&inner.cols, Ty::Int)?;
                let op = *self.rng.pick(&["=", "<", ">="]);
                let q = *self.rng.pick(&["ANY", "ALL"]);
                format!("({left} {op} {q} (SELECT {right} FROM {}{wh}))", inner.from)
            }
        })
    }

    /// A correlated scalar aggregate over a table.
    fn scalar_subquery(&mut self, outer: &[Col], ty: Ty, _subq: u32) -> Option<String> {
        let inner = self.table_scope();
        let arg = self.col_of(&inner.cols, ty)?;
        let agg = match ty {
            Ty::Int | Ty::Float => *self.rng.pick(&["max", "min", "sum", "count"]),
            Ty::Str => *self.rng.pick(&["max", "min"]),
            Ty::Bool => "bool_and",
        };
        let corr = self
            .correlation(outer, &inner.cols)
            .map(|c| format!(" WHERE {c}"))
            .unwrap_or_default();
        let call = if agg == "count" {
            format!("CAST(count({arg}) AS {})", sql_type(ty))
        } else {
            format!("{agg}({arg})")
        };
        Some(format!("(SELECT {call} FROM {}{corr})", inner.from))
    }

    fn correlation(&mut self, outer: &[Col], inner: &[Col]) -> Option<String> {
        if outer.is_empty() || self.chance(3) {
            return None;
        }
        let ty = *self.rng.pick(&[Ty::Int, Ty::Str]);
        let o = self.col_of(outer, ty)?;
        let i = self.col_of(inner, ty)?;
        Some(format!("{i} = {o}"))
    }

    fn window(&mut self, scope: &Scope) -> Option<(String, Ty)> {
        let part = if self.chance(2) {
            let ty = *self.rng.pick(&[Ty::Int, Ty::Str]);
            self.col_of(&scope.cols, ty)
                .map(|c| format!("PARTITION BY {c} "))
                .unwrap_or_default()
        } else {
            String::new()
        };
        // An order of the keys tells rows apart, so order-sensitive
        // functions are deterministic.
        let unique = !scope.keys.is_empty();
        let order_keys = if unique {
            scope
                .keys
                .iter()
                .map(|k| {
                    format!(
                        "{} {}",
                        k.sql(),
                        *self
                            .rng
                            .pick(&["ASC", "DESC", "ASC NULLS FIRST", "DESC NULLS LAST"])
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            String::new()
        };
        let f = self.col_of(&scope.cols, Ty::Float);
        let i = self.col_of(&scope.cols, Ty::Int);
        Some(match self.rng.below(9) {
            0 if unique => (
                format!("row_number() OVER ({part}ORDER BY {order_keys})"),
                Ty::Int,
            ),
            1 => {
                let i = i?;
                (format!("rank() OVER ({part}ORDER BY {i})"), Ty::Int)
            }
            2 => {
                let i = i?;
                (
                    format!("dense_rank() OVER ({part}ORDER BY {i} DESC)"),
                    Ty::Int,
                )
            }
            3 => {
                let f = f?;
                (format!("sum({f}) OVER ({part})"), Ty::Float)
            }
            4 if unique => {
                let f = f?;
                let frame = *self.rng.pick(&[
                    "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING",
                    "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
                    "ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
                    "ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING",
                ]);
                let agg = *self.rng.pick(&["sum", "min", "max", "avg", "count"]);
                let ty = if agg == "count" { Ty::Int } else { Ty::Float };
                (
                    format!("{agg}({f}) OVER ({part}ORDER BY {order_keys} {frame})"),
                    ty,
                )
            }
            5 => {
                let i = i?;
                let frame = *self.rng.pick(&[
                    "",
                    " RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING",
                    " RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
                    " RANGE BETWEEN CURRENT ROW AND 2 FOLLOWING",
                ]);
                let dir = *self.rng.pick(&["", " DESC"]);
                (
                    format!("count(*) OVER ({part}ORDER BY {i}{dir}{frame})"),
                    Ty::Int,
                )
            }
            6 if unique => {
                let f = f?;
                let fun = *self.rng.pick(&["lag", "lead"]);
                (
                    format!("{fun}({f}, 1, -1.0) OVER ({part}ORDER BY {order_keys})"),
                    Ty::Float,
                )
            }
            7 if unique => {
                let f = f?;
                let fun = *self.rng.pick(&["first_value", "last_value"]);
                (
                    format!("{fun}({f}) OVER ({part}ORDER BY {order_keys})"),
                    Ty::Float,
                )
            }
            _ => {
                let f = f?;
                (format!("max({f}) OVER ({part})"), Ty::Float)
            }
        })
    }

    fn aggregate(&mut self, cols: &[Col]) -> (String, Ty) {
        let ty = *self.rng.pick(&TYS);
        let Some(arg) = self.col_of(cols, ty) else {
            return ("count(*)".into(), Ty::Int);
        };
        let filter = if self.chance(5) {
            format!(" FILTER (WHERE {})", self.boolean(cols, 1, 0))
        } else {
            String::new()
        };
        let distinct = if self.chance(5) { "DISTINCT " } else { "" };
        let (call, out) = match (ty, self.rng.below(4)) {
            (Ty::Int, 0) => (format!("sum({distinct}{arg})"), Ty::Int),
            (Ty::Float, 0) => (format!("sum({distinct}{arg})"), Ty::Float),
            (Ty::Int | Ty::Float, 1) => (format!("avg({arg})"), Ty::Float),
            (Ty::Bool, 0) => (format!("bool_and({arg})"), Ty::Bool),
            (Ty::Bool, 1) => (format!("bool_or({arg})"), Ty::Bool),
            (_, 2) => (format!("count({distinct}{arg})"), Ty::Int),
            (t, _) => (format!("{}({arg})", *self.rng.pick(&["min", "max"])), t),
        };
        (format!("{call}{filter}"), out)
    }

    /// A `SELECT`, its output columns named `o0`, `o1`, …; of the types
    /// `want` if given. `top` may order and limit by every output column.
    fn select(
        &mut self,
        depth: u32,
        outer: &[Col],
        want: Option<&[Ty]>,
        top: bool,
    ) -> (String, Vec<Ty>) {
        if depth > 0 && want.is_none() && self.chance(6) {
            // A set operation over two selects of the same types.
            let (a, tys) = self.select(depth - 1, outer, None, false);
            let (b, _) = self.select(depth - 1, outer, Some(&tys), false);
            let op = *self.rng.pick(&[
                "UNION ALL",
                "UNION",
                "EXCEPT",
                "INTERSECT",
                "EXCEPT ALL",
                "INTERSECT ALL",
            ]);
            let sql = format!(
                "SELECT * FROM ({a}) AS {} {op} SELECT * FROM ({b}) AS {}",
                self.alias(),
                self.alias()
            );
            return (self.ordered(sql, &tys, top), tys);
        }
        let scope = self.from(depth);
        let mut cols = scope.cols.clone();
        cols.extend(outer.iter().cloned());
        let subq = if depth > 0 { 1 } else { 0 };
        let wh = if self.chance(2) {
            format!(" WHERE {}", self.boolean(&cols, 2, subq))
        } else {
            String::new()
        };
        let mut items: Vec<(String, Ty)> = Vec::new();
        let mut tail = String::new();
        let grouped = self.chance(3);
        if grouped {
            let keys: Vec<String> = (0..self.rng.below(3))
                .filter_map(|_| {
                    let ty = *self.rng.pick(&[Ty::Int, Ty::Str, Ty::Bool]);
                    self.col_of(&scope.cols, ty).map(|c| (c, ty))
                })
                .map(|(c, ty)| {
                    items.push((c.clone(), ty));
                    c
                })
                .collect();
            for _ in 0..1 + self.rng.below(3) {
                items.push(self.aggregate(&scope.cols));
            }
            if !keys.is_empty() {
                write!(tail, " GROUP BY {}", keys.join(", ")).ok();
            }
            if self.chance(3) {
                let (agg, ty) = self.aggregate(&scope.cols);
                if ty != Ty::Bool && ty != Ty::Str {
                    write!(tail, " HAVING {agg} IS NOT NULL").ok();
                }
            }
        } else {
            for _ in 0..1 + self.rng.below(4) {
                if self.chance(5) {
                    if let Some(w) = self.window(&scope) {
                        items.push(w);
                        continue;
                    }
                }
                let ty = *self.rng.pick(&TYS);
                let e = self.expr(&cols, ty, 2, subq);
                items.push((e, ty));
            }
        }
        // Fit to the wanted types, casting or replacing.
        if let Some(want) = want {
            let mut fitted = Vec::new();
            for (i, ty) in want.iter().enumerate() {
                match items.get(i) {
                    Some((e, t)) if t == ty => fitted.push((e.clone(), *t)),
                    Some(_) | None if grouped => {
                        fitted.push((format!("CAST(NULL AS {})", sql_type(*ty)), *ty))
                    }
                    _ => {
                        let e = self.expr(&cols, *ty, 1, 0);
                        fitted.push((e, *ty));
                    }
                }
            }
            items = fitted;
        }
        let distinct = if !grouped && self.chance(8) {
            "DISTINCT "
        } else {
            ""
        };
        let list = items
            .iter()
            .enumerate()
            .map(|(i, (e, _))| format!("{e} AS o{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let tys: Vec<Ty> = items.iter().map(|i| i.1).collect();
        let mut sql = format!("SELECT {distinct}{list} FROM {}{wh}{tail}", scope.from);
        if !top && distinct.is_empty() && !grouped && self.chance(6) {
            // A derived table's top rows, by every column: the set is
            // determined.
            let by = (0..tys.len())
                .map(|i| format!("o{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            write!(sql, " ORDER BY {by} LIMIT {}", 1 + self.rng.below(3)).ok();
        }
        (self.ordered(sql, &tys, top), tys)
    }

    /// `sql` ordered, perhaps limited, by every output column, if `top`.
    fn ordered(&mut self, sql: String, tys: &[Ty], top: bool) -> String {
        if !top || self.chance(2) {
            return sql;
        }
        self.ordered = true;
        // Sometimes the first column orders but is not selected: a
        // projection over the sort.
        let hide = tys.len() > 1 && self.chance(2);
        let by = (0..tys.len())
            .map(|i| {
                format!(
                    "o{i} {}",
                    *self
                        .rng
                        .pick(&["", "DESC", "ASC NULLS FIRST", "DESC NULLS LAST"])
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let limit = match self.rng.below(4) {
            0 => format!(" LIMIT {}", self.rng.below(4)),
            1 => format!(" LIMIT {} OFFSET 1", 1 + self.rng.below(3)),
            _ => String::new(),
        };
        let list = if hide {
            (1..tys.len())
                .map(|i| format!("o{i}"))
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            "*".to_string()
        };
        format!(
            "SELECT {list} FROM ({sql}) AS {} ORDER BY {by}{limit}",
            self.alias()
        )
    }
}

// ---------------------------------------------------------------- DuckDB

/// Cases for the cross-engine check (`to_sql_fuzz_duckdb.py`): one JSON
/// line each, with the statements that make the tables in DuckDB, the
/// DuckDB SQL `to_sql` wrote, and the rows DataFusion gives the query.
struct Cases<'a> {
    out: std::fs::File,
    setup: &'a [String],
}

impl Cases<'_> {
    fn push(&mut self, seed: u64, optimized: bool, sql: &str, ordered: bool, want: &Result_) {
        use std::io::Write as _;
        let setup = self
            .setup
            .iter()
            .map(|s| json(s))
            .collect::<Vec<_>>()
            .join(", ");
        let names = want
            .columns
            .iter()
            .map(|c| json(&c.0))
            .collect::<Vec<_>>()
            .join(", ");
        let rows = want
            .rows
            .iter()
            .map(|r| {
                format!(
                    "[{}]",
                    r.iter().map(cell_json).collect::<Vec<_>>().join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            self.out,
            "{{\"seed\": {seed}, \"optimized\": {optimized}, \"setup\": [{setup}], \"sql\": {}, \
             \"ordered\": {ordered}, \"columns\": [{names}], \"rows\": [{rows}]}}",
            json(sql)
        )
        .ok();
    }
}

fn json(s: &str) -> String {
    let mut out = String::from('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                write!(out, "\\u{:04x}", c as u32).ok();
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn cell_json(c: &Cell) -> String {
    match c {
        Cell::Null => "null".into(),
        Cell::Int(n) => format!("{{\"i\": \"{n}\"}}"),
        Cell::Float(x) if x.is_nan() => "{\"f\": \"nan\"}".into(),
        Cell::Float(x) if x.is_infinite() => {
            format!("{{\"f\": \"{}inf\"}}", if *x < 0.0 { "-" } else { "" })
        }
        Cell::Float(x) => format!("{{\"f\": \"{x:e}\"}}"),
        Cell::Str(s) => format!("{{\"s\": {}}}", json(s)),
        Cell::Bool(b) => b.to_string(),
        Cell::Other(s) => format!("{{\"o\": {}}}", json(s)),
    }
}

// ---------------------------------------------------------------- checks

#[derive(Default)]
struct Tally {
    cases: u64,
    planned: u64,
    written: u64,
    compared: u64,
    refused: BTreeMap<String, u64>,
    skipped: BTreeMap<String, u64>,
    failures: Vec<String>,
}

fn short(m: &str) -> String {
    m.split_whitespace().take(9).collect::<Vec<_>>().join(" ")
}

fn verbose() -> bool {
    std::env::var("DDX_TO_SQL_VERBOSE").is_ok_and(|v| v == "1")
}

/// `to_sql`, with a panic caught as an error.
fn write(plan: &Plan, dialect: Dialect) -> Result<Result<String, AdError>, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| to_sql(plan, dialect))).map_err(|p| {
        p.downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    })
}

async fn check(
    ctx: &SessionContext,
    seed: u64,
    sql: &str,
    ordered: bool,
    optimized: bool,
    t: &mut Tally,
    cases: &mut Option<Cases<'_>>,
) {
    let label = format!(
        "seed {seed} ({})",
        if optimized {
            "optimized"
        } else {
            "unoptimized"
        }
    );
    let plan = {
        let df = match ctx.sql(sql).await {
            Ok(df) => df,
            Err(e) => {
                *t.skipped
                    .entry(format!("plan: {}", short(&e.to_string())))
                    .or_default() += 1;
                return;
            }
        };
        let lp = if optimized {
            match df.into_optimized_plan() {
                Ok(lp) => lp,
                Err(e) => {
                    *t.skipped
                        .entry(format!("optimize: {}", short(&e.to_string())))
                        .or_default() += 1;
                    return;
                }
            }
        } else {
            df.into_unoptimized_plan()
        };
        match datafusion_substrait::logical_plan::producer::to_substrait_plan(&lp, &ctx.state()) {
            Ok(p) => *p,
            Err(e) => {
                *t.skipped
                    .entry(format!("produce: {}", short(&e.to_string())))
                    .or_default() += 1;
                return;
            }
        }
    };
    t.planned += 1;
    let original = run_sql(ctx, sql).await;
    let Ok(original) = original else {
        *t.skipped.entry("the query fails".into()).or_default() += 1;
        return;
    };
    let consumer = run_plan(ctx, &plan).await;
    // Where DataFusion's own round trip changes the meaning, there is no
    // one meaning to hold `to_sql` to.
    if consumer
        .as_ref()
        .is_ok_and(|c| differs(&original, c, ordered).is_some())
    {
        *t.skipped
            .entry("the consumer's round trip differs from the query".into())
            .or_default() += 1;
        return;
    }

    // Refusals: NotImplemented only, both dialects alike, no panic.
    let written = match write(&plan, Dialect::DataFusion) {
        Err(panic) => {
            t.failures
                .push(format!("[panic] {label}: {panic}\n  {sql}"));
            return;
        }
        Ok(w) => w,
    };
    match write(&plan, Dialect::DuckDb) {
        Err(panic) => t
            .failures
            .push(format!("[panic duckdb] {label}: {panic}\n  {sql}")),
        Ok(duck) if duck.is_ok() != written.is_ok() => t.failures.push(format!(
            "[dialects] {label}: DataFusion {:?}, DuckDB {:?}\n  {sql}",
            written.as_ref().err(),
            duck.err()
        )),
        Ok(_) => {}
    }
    let written = match written {
        Ok(w) => w,
        Err(AdError::NotImplemented(m)) => {
            *t.refused.entry(short(&m)).or_default() += 1;
            return;
        }
        Err(e) => {
            t.failures.push(format!("[refused] {label}: {e}\n  {sql}"));
            return;
        }
    };
    t.written += 1;
    if let (Some(c), Ok(Ok(duck))) = (cases.as_mut(), write(&plan, Dialect::DuckDb)) {
        c.push(seed, optimized, &duck, ordered, &original);
    }
    if to_sql(&plan, Dialect::DataFusion).ok().as_deref() != Some(written.as_str()) {
        t.failures
            .push(format!("[determinism] {label}: two writes differ\n  {sql}"));
    }

    // Meaning: the query's, or the consumer's.
    let show = |w: &str| {
        if verbose() {
            format!("\n  written: {w}")
        } else {
            String::new()
        }
    };
    let meaning = |got: &Result_| -> Option<String> {
        let vs_query = differs(&original, got, ordered)?;
        match &consumer {
            Ok(c) => {
                let vs_consumer = differs(c, got, ordered)?;
                Some(format!("{vs_query} (the consumer's: {vs_consumer})"))
            }
            Err(_) => Some(vs_query),
        }
    };
    match run_sql(ctx, &written).await {
        Err(e) => {
            // If the consumer cannot run the plan either, DataFusion reads
            // the plan's SQL as it reads the plan.
            if consumer.is_ok() {
                t.failures.push(format!(
                    "[runs] {label}: the written SQL fails: {}\n  {sql}{}",
                    short(&e),
                    show(&written)
                ));
            } else {
                *t.skipped
                    .entry("neither the consumer nor the SQL runs".into())
                    .or_default() += 1;
            }
            return;
        }
        Ok(got) => {
            t.compared += 1;
            if let Some(d) = meaning(&got) {
                // The rows themselves, or only their order?
                let rows_too = differs(&original, &got, false).is_some()
                    && consumer
                        .as_ref()
                        .map_or(true, |c| differs(c, &got, false).is_some());
                let kind = if rows_too { "meaning" } else { "order" };
                t.failures
                    .push(format!("[{kind}] {label}: {d}\n  {sql}{}", show(&written)));
                return;
            }
        }
    }

    // Sharing.
    let shared = common::shared(&plan);
    if common::shared_count(&shared) > 0 {
        match write(&shared, Dialect::DataFusion) {
            Ok(Ok(w)) => match run_sql(ctx, &w).await {
                Ok(got) => {
                    if let Some(d) = meaning(&got) {
                        t.failures
                            .push(format!("[shared] {label}: {d}\n  {sql}{}", show(&w)));
                    }
                }
                Err(e) => t.failures.push(format!(
                    "[shared runs] {label}: {}\n  {sql}{}",
                    short(&e),
                    show(&w)
                )),
            },
            Ok(Err(e)) => t
                .failures
                .push(format!("[shared refused] {label}: {e}\n  {sql}")),
            Err(p) => t
                .failures
                .push(format!("[shared panic] {label}: {p}\n  {sql}")),
        }
    }

    // A fixed point: the written SQL, planned and written again.
    if let Ok(df) = ctx.sql(&written).await {
        let lp = if optimized {
            df.into_optimized_plan().ok()
        } else {
            Some(df.into_unoptimized_plan())
        };
        if let Some(Ok(again)) = lp.map(|lp| {
            datafusion_substrait::logical_plan::producer::to_substrait_plan(&lp, &ctx.state())
        }) {
            match write(&again, Dialect::DataFusion) {
                Ok(Ok(w)) => {
                    if let Ok(got) = run_sql(ctx, &w).await {
                        if let Some(d) = meaning(&got) {
                            t.failures
                                .push(format!("[fixed point] {label}: {d}\n  {sql}{}", show(&w)));
                        }
                    }
                }
                Ok(Err(AdError::NotImplemented(_))) => {}
                Ok(Err(e)) => t
                    .failures
                    .push(format!("[fixed point refused] {label}: {e}\n  {sql}")),
                Err(p) => t
                    .failures
                    .push(format!("[fixed point panic] {label}: {p}\n  {sql}")),
            }
        }
    }
}

#[test]
fn to_sql_writes_what_the_plan_means() {
    let env = |k: &str, d: u64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (n, base) = (env("DDX_TO_SQL_SEEDS", 300), env("DDX_TO_SQL_BASE", 0));
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut t = Tally::default();
    let cases_path = std::env::var("DDX_TO_SQL_DUCKDB_CASES").ok();
    let mut cases_file = cases_path
        .as_ref()
        .map(|p| std::fs::File::create(p).unwrap_or_else(|e| panic!("{p}: {e}")));
    for seed in base..base + n {
        let mut rng = seeded(seed, 0x7051_5C11);
        // One partition: rows a plan reads in one order it gives in one
        // order, so only an ORDER BY the SQL loses can reorder them.
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
        let tables = rt.block_on(tables(&mut rng, &ctx));
        let mut g = Gen {
            rng,
            tables: &tables,
            aliases: 0,
            ordered: false,
        };
        let depth = 1 + g.rng.below(2) as u32;
        let (sql, _) = g.select(depth, &[], None, true);
        let ordered = g.ordered;
        t.cases += 1;
        let setup: Vec<String> = tables.iter().flat_map(|t| t.duckdb.clone()).collect();
        for optimized in [true, false] {
            let mut cases = cases_file.as_mut().map(|f| Cases {
                out: f.try_clone().unwrap(),
                setup: &setup,
            });
            rt.block_on(check(
                &ctx, seed, &sql, ordered, optimized, &mut t, &mut cases,
            ));
        }
    }
    eprintln!(
        "{} queries: {} plans, {} written, {} compared",
        t.cases, t.planned, t.written, t.compared
    );
    let top = |m: &BTreeMap<String, u64>| {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        v.into_iter()
            .take(8)
            .map(|(k, n)| format!("  {n}× {k}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    eprintln!(
        "refused:\n{}\nskipped:\n{}",
        top(&t.refused),
        top(&t.skipped)
    );
    let mut kinds: BTreeMap<String, u64> = BTreeMap::new();
    for f in &t.failures {
        let k = f.split(']').next().unwrap_or("").to_string() + "]";
        *kinds.entry(k).or_default() += 1;
    }
    eprintln!("failures by kind: {kinds:?}");
    for f in t.failures.iter().take(30) {
        eprintln!("\nFAILURE {f}");
    }
    assert!(t.failures.is_empty(), "{} failures", t.failures.len());
}
