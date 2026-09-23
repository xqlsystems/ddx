// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The Rust half of `ddxdb` — a thin PyO3 surface over `ddx-core` and `ddx-ad`.
//!
//! Thin is the point. All the calculus lives in `ddx-core`; this file moves
//! strings across the FFI boundary and turns a `DiffError` into an exception a
//! Python caller can actually branch on. Nothing here decides anything about
//! differentiation, and nothing here should grow to.

use std::collections::HashMap;

use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::{NamedStruct, Plan, Rel};
use ddx_ad::{AdError, ColumnRef};
use ddx_core::sqlparser::dialect::{dialect_from_str, Dialect, GenericDialect};
use ddx_core::{Ddx, DiffError, IdentCasing};
use prost::Message;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// The first argument names the module the class claims to live in, and it must
// be the *importable* path (`ddxdb._ddxdb`), not the bare crate name. Python
// looks the class up by that path to reconstruct it, so a bare `_ddxdb` — which
// is importable nowhere — makes these exceptions unpicklable, and a rewrite
// failure inside a multiprocessing / joblib / pytest-xdist worker would reach
// the parent as a PicklingError instead of the typed error. It is also what
// `repr()` prints.
create_exception!(
    ddxdb._ddxdb,
    DdxError,
    PyException,
    "Base class for every error ddx raises."
);
create_exception!(
    ddxdb._ddxdb,
    UnsupportedExpression,
    DdxError,
    "The expression contains something ddx has no differentiation rule for."
);
create_exception!(
    ddxdb._ddxdb,
    InvalidMarker,
    DdxError,
    "A `grad`/`jvp` call is malformed — wrong argument count, or a `wrt` that is not a bare column."
);
create_exception!(
    ddxdb._ddxdb,
    AmbiguousColumn,
    DdxError,
    "An occurrence of the differentiation variable could not be pinned to one column; qualify it."
);
create_exception!(
    ddxdb._ddxdb,
    ProjectionBoundary,
    DdxError,
    "A marker references a column computed upstream (in a CTE or subquery), where differentiation would silently drop terms."
);
create_exception!(
    ddxdb._ddxdb,
    SqlParseError,
    DdxError,
    "The statement did not parse under the chosen dialect."
);

create_exception!(
    ddxdb._ddxdb,
    NotScalar,
    DdxError,
    "grad was asked for the gradient of something that is not a loss: one row and one column."
);
create_exception!(
    ddxdb._ddxdb,
    UnknownColumn,
    DdxError,
    "A wrt column names a table or column the query does not read."
);

/// Map a [`DiffError`] onto a Python exception, one class per variant.
///
/// A single exception type carrying a message would force callers to match on
/// prose to tell "ddx cannot differentiate this yet" from "your query is
/// ambiguous, qualify the column" — one is a limitation to route around, the
/// other is a fix the caller makes. Distinct classes make that a normal `except`
/// clause. This mirrors the Rust side, which keeps the typed error downcastable
/// rather than stringifying it.
fn to_py_err(e: DiffError) -> PyErr {
    let msg = e.to_string();
    match e {
        DiffError::NotImplemented(_) => UnsupportedExpression::new_err(msg),
        DiffError::InvalidMarker(_) => InvalidMarker::new_err(msg),
        DiffError::AmbiguousColumn(_) => AmbiguousColumn::new_err(msg),
        DiffError::ProjectionBoundary(_) => ProjectionBoundary::new_err(msg),
        DiffError::Parse(_) => SqlParseError::new_err(msg),
        DiffError::Internal(_) => DdxError::new_err(msg),
    }
}

/// Map an [`AdError`] onto a Python exception. Variants that mean the same
/// thing as a v1 error share its class.
fn ad_to_py_err(e: AdError) -> PyErr {
    let msg = e.to_string();
    match e {
        AdError::NotImplemented(_) => UnsupportedExpression::new_err(msg),
        AdError::NotScalar(_) => NotScalar::new_err(msg),
        AdError::UnknownWrt(_) => UnknownColumn::new_err(msg),
        AdError::Diff(inner) => to_py_err(inner),
        AdError::InvalidPlan(_) | AdError::Internal(_) => DdxError::new_err(msg),
    }
}

/// How each engine resolves an identifier to a column.
///
/// This is the one thing `sqlparser` does not carry, and it cannot be guessed
/// from the parser: parsing tells us `"X"` is a quoted identifier, not which
/// column `"X"` *is*. Engines disagree in three incompatible ways —
///
/// * fold unquoted to lower, quoted keeps case (Postgres, DataFusion);
/// * fold unquoted to UPPER, quoted keeps case (Snowflake, Oracle);
/// * fold everything (DuckDB, Spark, MySQL) or nothing (ClickHouse).
///
/// — and getting it wrong is silent. `grad("X" * "X", X)` under the Postgres
/// rule matches nothing and differentiates to `0`; under the Snowflake rule it
/// is `2*X`. The inverse is worse: the wrong rule can match the *other* column
/// and return a confident, wrong, nonzero derivative.
///
/// So the table is exhaustive over what `sqlparser` parses rather than a list of
/// exceptions to a default. A default would mean any dialect added upstream
/// silently inherits Postgres semantics, which is how six of these came to be
/// wrong; an unmapped dialect raises instead (see [`engine_for`]).
const IDENTIFIER_FOLDING: &[(&str, IdentCasing)] = &[
    // Unquoted folds to lowercase; quoting pins the case.
    ("generic", IdentCasing::FoldUnquoted),
    ("datafusion", IdentCasing::FoldUnquoted),
    ("postgres", IdentCasing::FoldUnquoted),
    ("postgresql", IdentCasing::FoldUnquoted),
    ("ansi", IdentCasing::FoldUnquoted),
    // Unquoted folds to uppercase; quoting pins the case. Same shape as above,
    // opposite target, so `X` means `"X"` here and `"x"` there.
    ("snowflake", IdentCasing::FoldUnquotedUpper),
    ("oracle", IdentCasing::FoldUnquotedUpper),
    // Case-insensitive throughout: quoting does not make an identifier
    // case-sensitive, so `x`, `X`, `"x"` and `"X"` are all one column.
    ("duckdb", IdentCasing::FoldAll),
    ("mysql", IdentCasing::FoldAll),
    ("sqlite", IdentCasing::FoldAll),
    ("bigquery", IdentCasing::FoldAll),
    ("redshift", IdentCasing::FoldAll),
    ("hive", IdentCasing::FoldAll),
    ("spark", IdentCasing::FoldAll),
    ("sparksql", IdentCasing::FoldAll),
    ("databricks", IdentCasing::FoldAll),
    // Collation-dependent, but case-insensitive under the default collation
    // these ship with. A case-sensitive collation would need FoldNone.
    ("mssql", IdentCasing::FoldAll),
    ("teradata", IdentCasing::FoldAll),
    // Case-sensitive throughout: `x` and `X` are simply different columns, so
    // `grad(X*X, x)` really is 0 here.
    ("clickhouse", IdentCasing::FoldNone),
];

/// The engine and parser for a dialect name.
///
/// Parsing is delegated wholesale to `sqlparser::dialect_from_str` — ddx never
/// enumerates parsers. `"datafusion"` is the one alias ddx adds, since
/// DataFusion has no dialect of its own and parses as generic SQL.
///
/// Folding, by contrast, is ddx's own knowledge and is looked up in
/// [`IDENTIFIER_FOLDING`]. The two are chosen together because they must agree
/// with the engine that will actually run the SQL, and a name that parses but
/// has no established folding rule is refused rather than guessed at.
fn engine_for(dialect: &str) -> PyResult<(Ddx, Box<dyn Dialect>)> {
    let name = dialect.to_ascii_lowercase();
    // DataFusion parses as generic SQL and has no `sqlparser` dialect of its own.
    let lookup = if name == "datafusion" {
        "generic"
    } else {
        &name
    };

    let parser = dialect_from_str(lookup).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown SQL dialect {dialect:?}. Accepts: {}",
            known_dialects()
        ))
    })?;

    let casing = IDENTIFIER_FOLDING
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, c)| *c)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "SQL dialect {dialect:?} parses, but ddx has not established how \
                 it resolves identifiers to columns — and guessing would silently \
                 differentiate with respect to the wrong column. Use one of: {}",
                known_dialects()
            ))
        })?;

    Ok((Ddx::with_casing(casing), parser))
}

/// The dialect names ddx accepts, for error messages.
///
/// Derived from the folding table rather than written out, so it cannot drift
/// from what `engine_for` will actually accept.
fn known_dialects() -> String {
    IDENTIFIER_FOLDING
        .iter()
        .map(|(n, _)| *n)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Rewrite every `grad`/`jvp` marker in `sql` to derivative SQL.
///
/// A statement with no marker comes back byte-identical, and is never parsed —
/// so wrapping every query costs essentially nothing.
#[pyfunction]
#[pyo3(signature = (sql, dialect = "datafusion"))]
fn rewrite_sql(sql: &str, dialect: &str) -> PyResult<String> {
    let (ddx, d) = engine_for(dialect)?;
    ddx.rewrite_sql(sql, d.as_ref()).map_err(to_py_err)
}

/// Differentiate a bare scalar expression, returning the derivative as SQL text.
///
/// The "calculus compiler" escape hatch, for building an update rule somewhere a
/// marker cannot reach — inside a recursive term, say. `expr` is an expression,
/// not a statement, and `wrt` must be a bare column name.
#[pyfunction]
#[pyo3(signature = (expr, wrt, dialect = "datafusion"))]
fn differentiate_sql(expr: &str, wrt: &str, dialect: &str) -> PyResult<String> {
    let (ddx, d) = engine_for(dialect)?;
    ddx.differentiate_sql(expr, wrt, d.as_ref())
        .map_err(to_py_err)
}

/// The unary function names ddx can differentiate, sorted.
///
/// Read out of the engine's rule registry rather than restated here, so it
/// cannot fall behind what is implemented. Useful for deciding whether to hand
/// ddx an expression at all, and for a test asking "is every rule covered?" to
/// ask the engine instead of a second list someone has to remember to update.
#[pyfunction]
fn supported_functions() -> Vec<String> {
    Ddx::new().unary_rule_names()
}

fn decode(plan: &[u8]) -> PyResult<Plan> {
    Plan::decode(plan).map_err(|e| PyValueError::new_err(format!("not a Substrait plan: {e}")))
}

fn bytes<'py>(py: Python<'py>, plan: &Plan) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &plan.encode_to_vec())
}

/// A step as `(name, plan bytes)`.
type PyStep<'py> = (String, Bound<'py, PyBytes>);

/// A program as `(forward_steps, value, cotangent, backward_steps, gradients)`:
/// a step is `(name, plan bytes)`, `cotangent` the columns a vjp program's
/// cotangent table needs, and a gradient `(table, step, columns)`.
type PyProgram<'py> = (
    Vec<PyStep<'py>>,
    String,
    Vec<String>,
    Vec<PyStep<'py>>,
    Vec<(String, String, Vec<String>)>,
);

fn to_py_program<'py>(py: Python<'py>, program: ddx_ad::BackwardProgram) -> PyProgram<'py> {
    let steps = |steps: &[ddx_ad::Step]| {
        steps
            .iter()
            .map(|s| (s.name.clone(), bytes(py, &s.plan)))
            .collect::<Vec<_>>()
    };
    let gradients = program
        .gradients
        .iter()
        .map(|g| (g.table.join("."), g.step.clone(), g.columns.clone()))
        .collect();
    (
        steps(&program.forward_steps),
        program.value.clone(),
        program.cotangent.clone(),
        steps(&program.backward_steps),
        gradients,
    )
}

fn wrt_of(wrt: Vec<(String, String)>) -> Vec<ColumnRef> {
    wrt.into_iter().map(|(t, c)| ColumnRef::new(t, c)).collect()
}

/// `grad` of a serialized Substrait plan (see `ddxdb.ad`).
#[pyfunction]
fn _grad<'py>(
    py: Python<'py>,
    plan: &[u8],
    wrt: Vec<(String, String)>,
) -> PyResult<PyProgram<'py>> {
    let program = ddx_ad::grad(&decode(plan)?, &wrt_of(wrt)).map_err(ad_to_py_err)?;
    Ok(to_py_program(py, program))
}

/// `vjp` of a serialized Substrait plan (see `ddxdb.ad`).
#[pyfunction]
fn _vjp<'py>(py: Python<'py>, plan: &[u8], wrt: Vec<(String, String)>) -> PyResult<PyProgram<'py>> {
    let program = ddx_ad::vjp(&decode(plan)?, &wrt_of(wrt)).map_err(ad_to_py_err)?;
    Ok(to_py_program(py, program))
}

/// The `grad(loss, table.column, …)` calls in a SQL statement, or `None`.
///
/// Returns `(losses, calls)`: a loss is `(name, query, wrt)`, a call
/// `(loss index, table, columns)`, in source order.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn _find_grad_calls(
    sql: &str,
) -> PyResult<
    Option<(
        Vec<(String, String, Vec<(String, String)>)>,
        Vec<(usize, String, Vec<String>)>,
    )>,
> {
    let Some(found) = ddx_ad::GradCalls::find(sql, &GenericDialect {}).map_err(ad_to_py_err)?
    else {
        return Ok(None);
    };
    let losses = found
        .losses
        .iter()
        .map(|l| {
            let wrt = l
                .wrt
                .iter()
                .map(|w| (w.table.clone(), w.column.clone()))
                .collect();
            (l.name.clone(), l.query.clone(), wrt)
        })
        .collect();
    let calls = found
        .calls
        .iter()
        .map(|c| (c.loss, c.table.clone(), c.columns.clone()))
        .collect();
    Ok(Some((losses, calls)))
}

/// `sql` with its `grad` calls replaced, in source order, by `relations`.
#[pyfunction]
fn _rewrite_grad_calls(sql: &str, relations: Vec<String>) -> PyResult<String> {
    let found = ddx_ad::GradCalls::find(sql, &GenericDialect {})
        .map_err(ad_to_py_err)?
        .ok_or_else(|| PyValueError::new_err("the statement has no grad(loss, …) call"))?;
    if relations.len() != found.calls.len() {
        return Err(PyValueError::new_err(format!(
            "{} replacements for {} grad calls",
            relations.len(),
            found.calls.len()
        )));
    }
    let mut next = relations.into_iter();
    Ok(found.rewrite(&mut |_| next.next().expect("counted above")))
}

/// The tables a step's plan reads without their types: the earlier steps it
/// depends on.
#[pyfunction]
fn _unbound_reads(plan: &[u8]) -> PyResult<Vec<String>> {
    Ok(ddx_ad::emit::unbound_reads(&decode(plan)?))
}

/// Bind a step's reads. `schemas` maps each table it reads to the serialized
/// plan of `SELECT * FROM table` on the engine, whose read states the table's
/// types in the engine's own terms.
#[pyfunction]
fn _bind_reads<'py>(
    py: Python<'py>,
    plan: &[u8],
    schemas: HashMap<String, Vec<u8>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut plan = decode(plan)?;
    let mut structs = HashMap::new();
    for (name, select_all) in schemas {
        let s = base_schema(&decode(&select_all)?).ok_or_else(|| {
            PyValueError::new_err(format!("no table read in the plan given for `{name}`"))
        })?;
        structs.insert(name, s);
    }
    ddx_ad::emit::bind_reads(&mut plan, &mut |n| structs.get(n).cloned()).map_err(ad_to_py_err)?;
    Ok(bytes(py, &plan))
}

/// The base schema of the table read at the bottom of `SELECT * FROM t`.
fn base_schema(plan: &Plan) -> Option<NamedStruct> {
    let mut rel: Option<&Rel> = plan.relations.iter().find_map(|r| match &r.rel_type {
        Some(PlanRelType::Root(root)) => root.input.as_ref(),
        _ => None,
    });
    while let Some(r) = rel {
        match &r.rel_type {
            Some(RelType::Read(read)) => return read.base_schema.clone(),
            Some(RelType::Project(p)) => rel = p.input.as_deref(),
            Some(RelType::Filter(f)) => rel = f.input.as_deref(),
            _ => return None,
        }
    }
    None
}

#[pymodule]
fn _ddxdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rewrite_sql, m)?)?;
    m.add_function(wrap_pyfunction!(differentiate_sql, m)?)?;
    m.add_function(wrap_pyfunction!(supported_functions, m)?)?;
    m.add_function(wrap_pyfunction!(_grad, m)?)?;
    m.add_function(wrap_pyfunction!(_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(_find_grad_calls, m)?)?;
    m.add_function(wrap_pyfunction!(_rewrite_grad_calls, m)?)?;
    m.add_function(wrap_pyfunction!(_unbound_reads, m)?)?;
    m.add_function(wrap_pyfunction!(_bind_reads, m)?)?;

    m.add("DdxError", m.py().get_type::<DdxError>())?;
    m.add(
        "UnsupportedExpression",
        m.py().get_type::<UnsupportedExpression>(),
    )?;
    m.add("InvalidMarker", m.py().get_type::<InvalidMarker>())?;
    m.add("AmbiguousColumn", m.py().get_type::<AmbiguousColumn>())?;
    m.add(
        "ProjectionBoundary",
        m.py().get_type::<ProjectionBoundary>(),
    )?;
    m.add("SqlParseError", m.py().get_type::<SqlParseError>())?;
    m.add("NotScalar", m.py().get_type::<NotScalar>())?;
    m.add("UnknownColumn", m.py().get_type::<UnknownColumn>())?;
    Ok(())
}
