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
use ddx_core::sqlparser::dialect::{dialect_from_str, Dialect};
use ddx_core::{Ddx, DiffError, IdentCasing};
use prost::Message;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

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
create_exception!(
    ddxdb._ddxdb,
    InvalidColumn,
    DdxError,
    "A wrt column cannot be differentiated: it is not a float, its table has no dims, or two of its rows share their dims."
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
        AdError::InvalidWrt(_) => InvalidColumn::new_err(msg),
        AdError::Diff(inner) => to_py_err(inner),
        // InvalidPlan, Internal, InvalidOptions, and any kind of refusal
        // ddx-ad adds later.
        _ => DdxError::new_err(msg),
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

/// One step of a program: a plan, and the table name to materialize its
/// result under (`ddx_ad::Step`).
#[pyclass(frozen, eq, skip_from_py_object, module = "ddxdb", name = "Step")]
#[derive(Clone, PartialEq)]
struct Step {
    #[pyo3(get)]
    name: String,
    plan: Vec<u8>,
}

#[pymethods]
impl Step {
    /// The plan, as serialized Substrait.
    #[getter]
    fn plan<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.plan)
    }

    fn __repr__(&self) -> String {
        format!(
            "Step(name={:?}, plan=<{} bytes>)",
            self.name,
            self.plan.len()
        )
    }
}

/// A plan that must return no rows, and what a row means (`ddx_ad::Check`).
#[pyclass(frozen, eq, skip_from_py_object, module = "ddxdb", name = "Check")]
#[derive(Clone, PartialEq)]
struct Check {
    plan: Vec<u8>,
    #[pyo3(get)]
    message: String,
}

#[pymethods]
impl Check {
    /// The plan, as serialized Substrait.
    #[getter]
    fn plan<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.plan)
    }

    fn __repr__(&self) -> String {
        format!("Check(message={:?})", self.message)
    }
}

/// Where a `wrt` table's gradient lands (a `ddx_ad::OutputTable` of a `wrt`
/// table): the step's
/// table has the table's dims, then its `wrt` values, named as in the table;
/// each value column holds the gradient, `0` where none reached and `NULL`
/// where the value itself is `NULL`.
#[pyclass(frozen, eq, skip_from_py_object, module = "ddxdb", name = "Gradient")]
#[derive(Clone, PartialEq)]
struct Gradient {
    #[pyo3(get)]
    table: String,
    #[pyo3(get)]
    step: String,
    columns: Vec<String>,
}

#[pymethods]
impl Gradient {
    #[getter]
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.columns)
    }

    fn __repr__(&self) -> String {
        format!(
            "Gradient(table={:?}, step={:?}, columns={:?})",
            self.table, self.step, self.columns
        )
    }
}

/// The steps that compute a query's value and its gradient: the Rust
/// `ddx_ad::BackwardProgram` itself, read from Python.
#[pyclass(frozen, module = "ddxdb", name = "BackwardProgram")]
struct BackwardProgram(ddx_ad::BackwardProgram);

fn to_steps(steps: &[ddx_ad::Step]) -> Vec<Step> {
    steps
        .iter()
        .map(|s| Step {
            name: s.name.clone(),
            plan: s.plan_bytes(),
        })
        .collect()
}

#[pymethods]
impl BackwardProgram {
    /// The saved aggregates, then the query's value.
    #[getter]
    fn forward_steps(&self) -> Vec<Step> {
        to_steps(&self.0.forward_steps)
    }

    /// The cotangents, then the gradients.
    #[getter]
    fn backward_steps(&self) -> Vec<Step> {
        to_steps(&self.0.backward_steps)
    }

    /// The step holding the query's own result.
    #[getter]
    fn value(&self) -> &str {
        &self.0.value.step
    }

    /// For a vjp program: the table the caller registers the cotangent as,
    /// its one input table; empty for a grad program, which has none.
    #[getter]
    fn cotangent_table(&self) -> &str {
        self.0.inputs.first().map_or("", |i| i.name.as_str())
    }

    /// For a vjp program: that table's columns.
    #[getter]
    fn cotangent<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let columns: &[String] = self.0.inputs.first().map_or(&[], |i| &i.columns);
        PyTuple::new(py, columns)
    }

    /// Plans that must return no rows, run before the steps.
    #[getter]
    fn checks(&self) -> Vec<Check> {
        self.0
            .checks
            .iter()
            .map(|c| Check {
                plan: c.plan_bytes(),
                message: c.message.clone(),
            })
            .collect()
    }

    /// One per `wrt` table.
    #[getter]
    fn gradients(&self) -> Vec<Gradient> {
        self.0
            .gradients
            .iter()
            .map(|g| Gradient {
                table: g.of.table().map(|t| t.join(".")).unwrap_or_default(),
                step: g.step.clone(),
                columns: g.columns.clone(),
            })
            .collect()
    }

    /// Every step, in the order they must run.
    fn steps(&self) -> Vec<Step> {
        let mut out = self.forward_steps();
        out.extend(self.backward_steps());
        out
    }

    /// The steps only other steps read: the saved aggregates and the
    /// cotangents. `run` drops them once the gradients are written.
    fn intermediate_steps(&self) -> Vec<Step> {
        self.0
            .intermediate_steps()
            .map(|s| Step {
                name: s.name.clone(),
                plan: s.plan_bytes(),
            })
            .collect()
    }

    fn __eq__(&self, other: &Self) -> bool {
        let key = |p: &Self| {
            (
                p.steps(),
                p.checks(),
                p.gradients(),
                p.0.value.step.clone(),
                p.cotangent_table().to_string(),
                p.0.inputs.first().map(|i| i.columns.clone()),
            )
        };
        key(self) == key(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "BackwardProgram(value={:?}, steps={}, checks={}, gradients={:?})",
            self.0.value.step,
            self.0.steps().count(),
            self.0.checks.len(),
            self.gradients()
                .iter()
                .map(|g| g.table.clone())
                .collect::<Vec<_>>()
        )
    }
}

/// `ddx_ad::Options` with an optional fixed namespace and restrictions:
/// `(table, serialized plan of SELECT * FROM table WHERE predicate)`.
fn options(
    namespace: Option<String>,
    restrict: Option<Vec<(String, Vec<u8>)>>,
) -> PyResult<ddx_ad::Options> {
    let mut options = match namespace {
        Some(ns) => ddx_ad::Options::new().namespace(ns),
        None => ddx_ad::Options::new(),
    };
    for (table, select) in restrict.unwrap_or_default() {
        options = options.restrict(table, decode(&select)?);
    }
    Ok(options)
}

fn wrt_of(wrt: Vec<(String, String)>) -> Vec<ColumnRef> {
    wrt.into_iter().map(|(t, c)| ColumnRef::new(t, c)).collect()
}

/// The gradient of the number the serialized Substrait `plan` computes, with
/// respect to the `(table, column)` pairs in `wrt` (`ddx_ad::grad`).
///
/// `namespace` names every table the program writes `{namespace}…` instead of
/// under a fresh prefix, so the same plan gives the same program. It must
/// start with `__ddx_`, end with `_`, and hold only lower-case ASCII letters,
/// digits and `_`.
///
/// `restrict` is `[(table, select), …]`, each `select` the engine's serialized
/// plan of `SELECT * FROM table WHERE predicate` over the table's dims: only
/// those rows of the table's gradient are computed (`ddx_ad::Options::restrict`).
#[pyfunction]
#[pyo3(signature = (plan, wrt, *, namespace = None, restrict = None))]
fn grad_plan(
    plan: &[u8],
    wrt: Vec<(String, String)>,
    namespace: Option<String>,
    restrict: Option<Vec<(String, Vec<u8>)>>,
) -> PyResult<BackwardProgram> {
    ddx_ad::grad_with(&decode(plan)?, &wrt_of(wrt), &options(namespace, restrict)?)
        .map(BackwardProgram)
        .map_err(ad_to_py_err)
}

/// The vector-Jacobian product of the query the serialized Substrait `plan`
/// computes (`ddx_ad::vjp`; see `grad_plan`).
#[pyfunction]
#[pyo3(signature = (plan, wrt, *, namespace = None, restrict = None))]
fn vjp_plan(
    plan: &[u8],
    wrt: Vec<(String, String)>,
    namespace: Option<String>,
    restrict: Option<Vec<(String, Vec<u8>)>>,
) -> PyResult<BackwardProgram> {
    ddx_ad::vjp_with(&decode(plan)?, &wrt_of(wrt), &options(namespace, restrict)?)
        .map(BackwardProgram)
        .map_err(ad_to_py_err)
}

/// A Python engine object as a `ddx_ad::Backend`: the four primitives, with
/// plans crossing as serialized Substrait.
struct PyBackend<'py>(Bound<'py, PyAny>);

impl ddx_ad::Backend for PyBackend<'_> {
    type Error = PyErr;

    fn table_schema(&mut self, name: &str) -> PyResult<NamedStruct> {
        let plan: Vec<u8> = self.0.call_method1("select_all", (name,))?.extract()?;
        base_schema(&decode(&plan)?).ok_or_else(|| {
            PyValueError::new_err(format!(
                "select_all({name:?}) returned a plan with no table read in it"
            ))
        })
    }

    fn returns_rows(&mut self, plan: &Plan) -> PyResult<bool> {
        let py = self.0.py();
        self.0
            .call_method1("returns_rows", (bytes(py, plan),))?
            .extract()
    }

    fn materialize(&mut self, name: &str, plan: &Plan) -> PyResult<()> {
        let py = self.0.py();
        self.0
            .call_method1("materialize", (name, bytes(py, plan)))?;
        Ok(())
    }

    fn drop_table(&mut self, name: &str) -> PyResult<()> {
        self.0.call_method1("drop_table", (name,))?;
        Ok(())
    }
}

/// Run `program` on `backend`: `ddx_ad::run`, the same protocol the Rust
/// adapters follow. Its checks run first, then every step. A run that
/// succeeds drops the intermediate tables and leaves the value and the
/// gradients; one that fails leaves none of the program's tables.
///
/// `backend` is any object with four methods (see `ddxdb.Backend`):
/// `select_all(name) -> bytes` (the engine's serialized Substrait plan of
/// `SELECT * FROM name`, whose read gives the table's schema; asked once per
/// table, and again after the table is rewritten), `returns_rows(plan) ->
/// bool`, `materialize(name, plan)` and `drop_table(name)`. Plans arrive as
/// serialized Substrait with their reads of earlier steps bound. Raises
/// `InvalidColumn` when a check returns a row, or what the backend raised.
#[pyfunction]
fn run(backend: Bound<'_, PyAny>, program: PyRef<'_, BackwardProgram>) -> PyResult<()> {
    ddx_ad::run(&mut PyBackend(backend), &program.0).map_err(|e| match e {
        ddx_ad::RunError::Refused(e) => ad_to_py_err(e),
        ddx_ad::RunError::Engine(e) => e,
        other => DdxError::new_err(other.to_string()),
    })
}

/// `ddx_ad::sql::Statements`: several statements' `grad` calls, planned
/// together. A `jvp` call is refused for now.
#[pyclass(frozen, module = "ddxdb._ddxdb", name = "_Statements")]
struct Statements(ddx_ad::Statements);

#[pymethods]
impl Statements {
    /// Plan `statements`, parsed as `dialect` writes SQL (the names
    /// `rewrite_sql` accepts).
    #[new]
    #[pyo3(signature = (statements, dialect = "datafusion"))]
    fn new(statements: Vec<String>, dialect: &str) -> PyResult<Self> {
        let (_, parser) = engine_for(dialect)?;
        let refs: Vec<&str> = statements.iter().map(String::as_str).collect();
        let planned = ddx_ad::Statements::plan(&refs, parser.as_ref()).map_err(ad_to_py_err)?;
        if !planned.jvp_jobs().is_empty() {
            return Err(ad_to_py_err(ddx_ad::AdError::NotImplemented(
                "jvp(f, …) in SQL from Python; it runs from Rust, in ddx-datafusion's ad::sql"
                    .into(),
            )));
        }
        Ok(Statements(planned))
    }

    /// The objectives to differentiate, as `(query, [(table, column), …],
    /// [(table, predicate), …])`: the last, the rows of a table's gradient
    /// every statement reads, where they say (`ddx_ad::sql::Job::restrict`).
    #[allow(clippy::type_complexity)]
    fn jobs(&self) -> Vec<(String, Vec<(String, String)>, Vec<(String, String)>)> {
        self.0
            .jobs()
            .iter()
            .map(|j| {
                let wrt = j
                    .wrt
                    .iter()
                    .map(|w| (w.table.clone(), w.column.clone()))
                    .collect();
                (j.query.clone(), wrt, j.restrict.clone())
            })
            .collect()
    }

    /// Each statement over the gradients `programs` (one per job, run)
    /// computed.
    fn rewrite(&self, programs: Vec<PyRef<'_, BackwardProgram>>) -> PyResult<Vec<String>> {
        let programs: Vec<&ddx_ad::BackwardProgram> = programs.iter().map(|p| &p.0).collect();
        self.0.rewrite(&programs, &[]).map_err(ad_to_py_err)
    }
}

/// Does `sql` hold `x NOT IN (subquery)`? DataFusion's Substrait producer
/// writes its null-aware anti-join as a plain one (ddx issue #104), so a
/// gradient of such a query would be of a different query. The Rust adapter
/// sees DataFusion's plan and refuses only when a NULL is possible; from
/// Python only the SQL is at hand, so any `NOT IN` over a subquery is
/// refused. A statement that does not parse is let through.
#[pyfunction]
fn _not_in_subquery(sql: &str) -> bool {
    use ddx_core::sqlparser::ast::{Expr, Visit, Visitor};
    use ddx_core::sqlparser::dialect::GenericDialect;
    use ddx_core::sqlparser::parser::Parser;
    use std::ops::ControlFlow;
    struct Finder;
    impl Visitor for Finder {
        type Break = ();
        fn pre_visit_expr(&mut self, e: &Expr) -> ControlFlow<()> {
            match e {
                Expr::InSubquery { negated: true, .. } => ControlFlow::Break(()),
                _ => ControlFlow::Continue(()),
            }
        }
    }
    Parser::parse_sql(&GenericDialect {}, sql)
        .map(|stmts| stmts.visit(&mut Finder).is_break())
        .unwrap_or(false)
}

/// The tables a step's plan reads without their types: the earlier steps it
/// depends on.
#[pyfunction]
fn _unbound_reads(plan: &[u8]) -> PyResult<Vec<String>> {
    Ok(ddx_ad::unbound_reads(&decode(plan)?))
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
    ddx_ad::bind_reads(&mut plan, &mut |n| structs.get(n).cloned()).map_err(ad_to_py_err)?;
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
    m.add_function(wrap_pyfunction!(grad_plan, m)?)?;
    m.add_function(wrap_pyfunction!(vjp_plan, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(_not_in_subquery, m)?)?;
    m.add_class::<BackwardProgram>()?;
    m.add_class::<Step>()?;
    m.add_class::<Check>()?;
    m.add_class::<Gradient>()?;
    m.add_class::<Statements>()?;
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
    m.add("InvalidColumn", m.py().get_type::<InvalidColumn>())?;
    Ok(())
}
