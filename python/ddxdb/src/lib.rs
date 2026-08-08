// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The Rust half of `ddxdb` — a thin PyO3 surface over `ddx-core`.
//!
//! Thin is the point. All the calculus lives in `ddx-core`; this file moves
//! strings across the FFI boundary and turns a `DiffError` into an exception a
//! Python caller can actually branch on. Nothing here decides anything about
//! differentiation, and nothing here should grow to.

use ddx_core::sqlparser::dialect::{Dialect, DuckDbDialect, GenericDialect};
use ddx_core::{Ddx, DiffError};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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

/// The engine and parser dialect for a dialect name.
///
/// The two must be chosen together: the identifier-folding policy has to match
/// the engine that will run the SQL, or column matching silently diverges from
/// the engine's own. DuckDB folds quoted identifiers; DataFusion and Postgres
/// do not.
fn engine_for(dialect: &str) -> PyResult<(Ddx, Box<dyn Dialect>)> {
    match dialect.to_ascii_lowercase().as_str() {
        "datafusion" | "generic" | "postgres" | "postgresql" => {
            Ok((Ddx::for_datafusion(), Box::new(GenericDialect {})))
        }
        "duckdb" => Ok((Ddx::for_duckdb(), Box::new(DuckDbDialect {}))),
        other => Err(PyValueError::new_err(format!(
            "unknown dialect {other:?}; expected one of \
             'datafusion', 'duckdb', 'postgres', 'generic'"
        ))),
    }
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

/// Preview what [`rewrite_sql`] would do, without running anything.
///
/// Returns the rewritten statement plus one entry per marker, each giving the
/// marker as written and the derivative it becomes — for logging, for a
/// notebook, or for understanding why a result looks the way it does.
#[pyfunction]
#[pyo3(signature = (sql, dialect = "datafusion"))]
fn explain<'py>(py: Python<'py>, sql: &str, dialect: &str) -> PyResult<Bound<'py, PyDict>> {
    let (ddx, d) = engine_for(dialect)?;
    let ex = ddx.explain(sql, d.as_ref()).map_err(to_py_err)?;

    let steps = PyList::empty(py);
    for step in &ex.steps {
        let entry = PyDict::new(py);
        entry.set_item("marker", &step.marker)?;
        entry.set_item("derivative", &step.derivative)?;
        steps.append(entry)?;
    }
    let out = PyDict::new(py);
    out.set_item("rewritten", &ex.rewritten)?;
    out.set_item("steps", steps)?;
    Ok(out)
}

#[pymodule]
fn _ddxdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rewrite_sql, m)?)?;
    m.add_function(wrap_pyfunction!(differentiate_sql, m)?)?;
    m.add_function(wrap_pyfunction!(explain, m)?)?;

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
