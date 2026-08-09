// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The Rust half of `ddxdb` — a thin PyO3 surface over `ddx-core`.
//!
//! Thin is the point. All the calculus lives in `ddx-core`; this file moves
//! strings across the FFI boundary and turns a `DiffError` into an exception a
//! Python caller can actually branch on. Nothing here decides anything about
//! differentiation, and nothing here should grow to.

use ddx_core::sqlparser::dialect::{dialect_from_str, Dialect};
use ddx_core::{Ddx, DiffError};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

create_exception!(
    _ddxdb,
    DdxError,
    PyException,
    "Base class for every error ddx raises."
);
create_exception!(
    _ddxdb,
    UnsupportedExpression,
    DdxError,
    "The expression contains something ddx has no differentiation rule for."
);
create_exception!(
    _ddxdb,
    InvalidMarker,
    DdxError,
    "A `grad`/`jvp` call is malformed — wrong argument count, or a `wrt` that is not a bare column."
);
create_exception!(
    _ddxdb,
    AmbiguousColumn,
    DdxError,
    "An occurrence of the differentiation variable could not be pinned to one column; qualify it."
);
create_exception!(
    _ddxdb,
    ProjectionBoundary,
    DdxError,
    "A marker references a column computed upstream (in a CTE or subquery), where differentiation would silently drop terms."
);
create_exception!(
    _ddxdb,
    SqlParseError,
    DdxError,
    "The statement did not parse under the chosen dialect."
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

/// Dialects that fold **quoted** identifiers, not just unquoted ones.
///
/// This is the one thing `sqlparser` does not tell us, and it cannot be
/// guessed: SQL folds unquoted identifiers everywhere, but whether `"X"` and
/// `"x"` are the same column is per-engine. DuckDB says yes; Postgres,
/// DataFusion and the rest say no. Getting it wrong does not raise — it
/// silently differentiates to zero, because the `wrt` stops matching its own
/// occurrences.
///
/// Deliberately an exception list of one rather than an enumeration of all
/// dialects: everything else takes the standard rule, so a dialect `sqlparser`
/// adds tomorrow is handled correctly without touching this file. Add an entry
/// only for another fully case-insensitive engine.
const FOLDS_QUOTED_IDENTIFIERS: &[&str] = &["duckdb"];

/// The engine and parser for a dialect name.
///
/// Parser selection is delegated wholesale to `sqlparser::dialect_from_str`, so
/// every dialect it knows works here — `snowflake`, `bigquery`, `mysql`,
/// `spark`, and the rest — and a dialect added upstream needs no change here.
/// `"datafusion"` is the one alias ddx adds, since DataFusion has no dialect of
/// its own and parses as generic SQL.
///
/// The identifier-folding policy is chosen alongside the parser rather than
/// separately, because the two must agree with the engine that will actually
/// run the SQL.
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
            "unknown SQL dialect {dialect:?}. Accepts any dialect sqlparser \
             knows — generic, datafusion, duckdb, postgres, mysql, sqlite, \
             snowflake, bigquery, redshift, clickhouse, mssql, hive, spark, \
             databricks, ansi, oracle, teradata"
        ))
    })?;

    let engine = if FOLDS_QUOTED_IDENTIFIERS.contains(&name.as_str()) {
        Ddx::for_duckdb()
    } else {
        Ddx::for_datafusion()
    };
    Ok((engine, parser))
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

#[pymodule]
fn _ddxdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rewrite_sql, m)?)?;
    m.add_function(wrap_pyfunction!(differentiate_sql, m)?)?;

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
    Ok(())
}
