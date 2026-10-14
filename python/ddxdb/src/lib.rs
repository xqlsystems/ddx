// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The Rust half of `ddxdb` — a thin PyO3 surface over `ddx-ad`.
//!
//! Thin is the point. All the calculus lives in `ddx-ad`; this file moves
//! plans across the FFI boundary and turns an `AdError` into an exception a
//! Python caller can actually branch on. Nothing here decides anything about
//! differentiation, and nothing here should grow to.

use std::future::{ready, Future, Ready};
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::{NamedStruct, Plan, Rel};
use ddx_ad::{AdError, ColumnRef};
use ddx_core::sqlparser::dialect::{dialect_from_str, Dialect};
use ddx_core::{Ddx, DiffError};
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

/// Map an [`AdError`] onto a Python exception.
///
/// A single exception type carrying a message would force callers to match on
/// prose to tell "ddx cannot differentiate this yet" from "this column cannot
/// be differentiated" — one is a limitation to route around, the other is a
/// fix the caller makes. Distinct classes make that a normal `except` clause.
/// This mirrors the Rust side, which keeps the typed error downcastable rather
/// than stringifying it.
fn ad_to_py_err(e: AdError) -> PyErr {
    let msg = e.to_string();
    match e {
        AdError::NotImplemented(_) | AdError::Diff(DiffError::NotImplemented(_)) => {
            UnsupportedExpression::new_err(msg)
        }
        AdError::NotScalar(_) => NotScalar::new_err(msg),
        AdError::UnknownWrt(_) => UnknownColumn::new_err(msg),
        AdError::InvalidWrt(_) => InvalidColumn::new_err(msg),
        // InvalidPlan, Internal, InvalidOptions, and any kind of refusal
        // ddx-ad adds later.
        _ => DdxError::new_err(msg),
    }
}

/// The parser for a dialect name, delegated wholesale to
/// `sqlparser::dialect_from_str`. `"datafusion"` is the one alias ddx adds,
/// since DataFusion has no dialect of its own and parses as generic SQL.
fn parser_for(dialect: &str) -> PyResult<Box<dyn Dialect>> {
    let name = dialect.to_ascii_lowercase();
    let lookup = if name == "datafusion" {
        "generic"
    } else {
        &name
    };
    dialect_from_str(lookup)
        .ok_or_else(|| PyValueError::new_err(format!("unknown SQL dialect {dialect:?}")))
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

/// What an `InputTable` or `OutputTable` is of, for Python: `None` for the
/// query's output, or a `wrt` table's name parts.
fn of_py(of: &ddx_ad::Of) -> Option<Vec<String>> {
    of.table().map(<[String]>::to_vec)
}

/// `of` as Python sees it: `None`, or a tuple of name parts.
fn of_tuple<'py>(
    py: Python<'py>,
    of: &Option<Vec<String>>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    of.as_ref().map(|t| PyTuple::new(py, t)).transpose()
}

/// A table the caller registers before a program runs (`ddx_ad::InputTable`):
/// a vjp's cotangent of the output (`of` is `None`), or a jvp's tangent of a
/// `wrt` table (`of` is its name parts). Its `columns` are its keys, the
/// first `keys` of them, then its values, named as in what it is of.
#[pyclass(frozen, eq, skip_from_py_object, module = "ddxdb", name = "InputTable")]
#[derive(Clone, PartialEq)]
struct InputTable {
    #[pyo3(get)]
    name: String,
    of: Option<Vec<String>>,
    columns: Vec<String>,
    #[pyo3(get)]
    keys: usize,
}

#[pymethods]
impl InputTable {
    #[getter]
    fn of<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        of_tuple(py, &self.of)
    }

    #[getter]
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.columns)
    }

    fn __repr__(&self) -> String {
        format!(
            "InputTable(name={:?}, of={:?}, columns={:?}, keys={})",
            self.name, self.of, self.columns, self.keys
        )
    }
}

impl From<&ddx_ad::InputTable> for InputTable {
    fn from(t: &ddx_ad::InputTable) -> Self {
        InputTable {
            name: t.name.clone(),
            of: of_py(&t.of),
            columns: t.columns.clone(),
            keys: t.keys,
        }
    }
}

/// Where a column's tangent is, in an `OutputTable` (`ddx_ad::Tangent`).
#[pyclass(frozen, eq, skip_from_py_object, module = "ddxdb", name = "Tangent")]
#[derive(Clone, PartialEq)]
struct Tangent {
    #[pyo3(get)]
    column: String,
    #[pyo3(get)]
    tangent: String,
}

#[pymethods]
impl Tangent {
    fn __repr__(&self) -> String {
        format!(
            "Tangent(column={:?}, tangent={:?})",
            self.column, self.tangent
        )
    }
}

/// A table a program leaves once it has run (`ddx_ad::OutputTable`): the
/// value (`of` is `None`), or a `wrt` table's gradient (`of` is its name
/// parts), written by `step`. A gradient's `columns` are the table's dims,
/// then its `wrt` values, named as in the table, each holding the gradient:
/// `0` where none reached, `NULL` where the value itself is `NULL`. For a jvp
/// program, `tangents` says which column holds each column's tangent.
#[pyclass(
    frozen,
    eq,
    skip_from_py_object,
    module = "ddxdb",
    name = "OutputTable"
)]
#[derive(Clone, PartialEq)]
struct OutputTable {
    #[pyo3(get)]
    step: String,
    of: Option<Vec<String>>,
    columns: Vec<String>,
    tangents: Vec<Tangent>,
    query: String,
}

#[pymethods]
impl OutputTable {
    /// SQL for its rows as a caller reads them once its program has run:
    /// its own columns, then each tangent as `{column}_tangent`
    /// (`ddx_ad::OutputTable::query`).
    fn query(&self) -> String {
        self.query.clone()
    }

    #[getter]
    fn of<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        of_tuple(py, &self.of)
    }

    #[getter]
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.columns)
    }

    #[getter]
    fn tangents(&self) -> Vec<Tangent> {
        self.tangents.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OutputTable(step={:?}, of={:?}, columns={:?})",
            self.step, self.of, self.columns
        )
    }
}

impl From<&ddx_ad::OutputTable> for OutputTable {
    fn from(t: &ddx_ad::OutputTable) -> Self {
        OutputTable {
            step: t.step.clone(),
            of: of_py(&t.of),
            columns: t.columns.clone(),
            tangents: t
                .tangents
                .iter()
                .map(|x| Tangent {
                    column: x.column.clone(),
                    tangent: x.tangent.clone(),
                })
                .collect(),
            query: t.query(),
        }
    }
}

fn to_checks(checks: &[ddx_ad::Check]) -> Vec<Check> {
    checks
        .iter()
        .map(|c| Check {
            plan: c.plan.to_bytes(),
            message: c.message.clone(),
        })
        .collect()
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
            plan: s.plan.to_bytes(),
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

    /// The tables to register before it runs: for a vjp program, the
    /// output's cotangent; none for a grad program. Of a program, that
    /// program's own inputs come first.
    #[getter]
    fn inputs(&self) -> Vec<InputTable> {
        self.0.inputs.iter().map(InputTable::from).collect()
    }

    /// Plans that must return no rows, run before the steps.
    #[getter]
    fn checks(&self) -> Vec<Check> {
        to_checks(&self.0.checks)
    }

    /// The query's own result.
    #[getter]
    fn value(&self) -> OutputTable {
        (&self.0.value).into()
    }

    /// One per `wrt` table.
    #[getter]
    fn gradients(&self) -> Vec<OutputTable> {
        self.0.gradients.iter().map(OutputTable::from).collect()
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
                plan: s.plan.to_bytes(),
            })
            .collect()
    }

    fn __eq__(&self, other: &Self) -> bool {
        let key = |p: &Self| (p.inputs(), p.checks(), p.steps(), p.value(), p.gradients());
        key(self) == key(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "BackwardProgram(value={:?}, steps={}, checks={}, gradients={:?})",
            self.0.value.step,
            self.0.steps().count(),
            self.0.checks.len(),
            self.0
                .gradients
                .iter()
                .filter_map(|g| g.of.table().map(|t| t.join(".")))
                .collect::<Vec<_>>()
        )
    }
}

/// The steps that compute a query's output and its tangent: the Rust
/// `ddx_ad::ForwardProgram` itself, read from Python. Its anatomy is a
/// `BackwardProgram`'s: inputs (each `wrt` table's tangent), checks, steps,
/// the value with its tangents, and, of a program, each of its gradients
/// with theirs (`H·v`, of a grad program).
#[pyclass(frozen, module = "ddxdb", name = "ForwardProgram")]
struct ForwardProgram(ddx_ad::ForwardProgram);

#[pymethods]
impl ForwardProgram {
    /// The tables to register before it runs: of a program, that program's
    /// own inputs first, then each `wrt` table's tangent.
    #[getter]
    fn inputs(&self) -> Vec<InputTable> {
        self.0.inputs.iter().map(InputTable::from).collect()
    }

    /// Plans that must return no rows, run before the steps.
    #[getter]
    fn checks(&self) -> Vec<Check> {
        to_checks(&self.0.checks)
    }

    /// The output, or a program's value, with its tangents.
    #[getter]
    fn value(&self) -> OutputTable {
        (&self.0.value).into()
    }

    /// Of a program: each of its gradients, with their tangents. Empty for
    /// jvp of a query.
    #[getter]
    fn gradients(&self) -> Vec<OutputTable> {
        self.0.gradients.iter().map(OutputTable::from).collect()
    }

    /// Every step, in the order they must run.
    fn steps(&self) -> Vec<Step> {
        to_steps(&self.0.steps)
    }

    fn __eq__(&self, other: &Self) -> bool {
        let key = |p: &Self| (p.inputs(), p.checks(), p.steps(), p.value(), p.gradients());
        key(self) == key(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "ForwardProgram(value={:?}, steps={}, checks={}, inputs={:?})",
            self.0.value.step,
            self.0.steps.len(),
            self.0.checks.len(),
            self.0
                .inputs
                .iter()
                .map(|i| i.name.clone())
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

/// `grad_plan`, `vjp_plan` and `jvp_plan` are one path, `derivative`, over
/// ddx-ad's own: `ddx_ad::grad` is `vjp` with the cotangent 1, built by the
/// same code with the 1 inlined (no input table to register, none to check).
fn derivative<T>(
    plan: &Bound<'_, PyAny>,
    wrt: Vec<(String, String)>,
    options: ddx_ad::Options,
    of: impl FnOnce(&dyn ddx_ad::Differentiable, &[ColumnRef], &ddx_ad::Options) -> ddx_ad::Result<T>,
) -> PyResult<T> {
    of(Subject::of(plan)?.get(), &wrt_of(wrt), &options).map_err(ad_to_py_err)
}

/// What `grad_plan`, `vjp_plan` and `jvp_plan` differentiate: a serialized
/// Substrait plan, or a program of either kind (`ddx_ad::Differentiable`).
enum Subject<'py> {
    Plan(Box<Plan>),
    Backward(PyRef<'py, BackwardProgram>),
    Forward(PyRef<'py, ForwardProgram>),
}

impl<'py> Subject<'py> {
    fn of(of: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(p) = of.extract::<PyRef<'py, BackwardProgram>>() {
            return Ok(Subject::Backward(p));
        }
        if let Ok(p) = of.extract::<PyRef<'py, ForwardProgram>>() {
            return Ok(Subject::Forward(p));
        }
        match of.extract::<Vec<u8>>() {
            Ok(bytes) if of.is_instance_of::<PyBytes>() => {
                Ok(Subject::Plan(Box::new(decode(&bytes)?)))
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "expected a serialized Substrait plan (bytes), a BackwardProgram or a \
                 ForwardProgram",
            )),
        }
    }

    fn get(&self) -> &dyn ddx_ad::Differentiable {
        match self {
            Subject::Plan(p) => &**p,
            Subject::Backward(p) => &p.0,
            Subject::Forward(p) => &p.0,
        }
    }
}

/// The gradient of the number `plan` computes, with respect to the `(table,
/// column)` pairs in `wrt` (`ddx_ad::grad`). `plan` is a serialized Substrait
/// plan, or a program of either kind, whose one output table is
/// differentiated: `grad`, `vjp` and `jvp` compose.
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
    plan: &Bound<'_, PyAny>,
    wrt: Vec<(String, String)>,
    namespace: Option<String>,
    restrict: Option<Vec<(String, Vec<u8>)>>,
) -> PyResult<BackwardProgram> {
    derivative(plan, wrt, options(namespace, restrict)?, |p, w, o| {
        ddx_ad::grad_with(p, w, o)
    })
    .map(BackwardProgram)
}

/// The vector-Jacobian product of what `plan` computes, a serialized
/// Substrait plan or a program (`ddx_ad::vjp`; see `grad_plan`).
#[pyfunction]
#[pyo3(signature = (plan, wrt, *, namespace = None, restrict = None))]
fn vjp_plan(
    plan: &Bound<'_, PyAny>,
    wrt: Vec<(String, String)>,
    namespace: Option<String>,
    restrict: Option<Vec<(String, Vec<u8>)>>,
) -> PyResult<BackwardProgram> {
    derivative(plan, wrt, options(namespace, restrict)?, |p, w, o| {
        ddx_ad::vjp_with(p, w, o)
    })
    .map(BackwardProgram)
}

/// The Jacobian-vector product of what `plan` computes, a serialized
/// Substrait plan or a program, with respect to the `(table, column)` pairs
/// in `wrt` (`ddx_ad::jvp`): its output with, beside it, its tangent along
/// the tangents the caller registers as the program's `inputs`. Of a grad
/// program, each gradient's tangent is the Hessian-vector product `H·v`.
/// `namespace` is as for `grad_plan`.
#[pyfunction]
#[pyo3(signature = (plan, wrt, *, namespace = None))]
fn jvp_plan(
    plan: &Bound<'_, PyAny>,
    wrt: Vec<(String, String)>,
    namespace: Option<String>,
) -> PyResult<ForwardProgram> {
    derivative(plan, wrt, options(namespace, None)?, |p, w, o| {
        ddx_ad::jvp_with(p, w, o)
    })
    .map(ForwardProgram)
}

/// A Python engine object as a `ddx_ad::Backend`: the four primitives, with
/// plans crossing as serialized Substrait. Each method calls Python and
/// returns a future already done, so a run is driven by [`now`].
struct PyBackend(Py<PyAny>);

impl PyBackend {
    fn call<T>(&self, f: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<T>) -> Ready<PyResult<T>> {
        ready(Python::attach(|py| f(self.0.bind(py))))
    }
}

impl ddx_ad::Backend for PyBackend {
    type Error = PyErr;

    fn table_schema(&mut self, name: &str) -> impl Future<Output = PyResult<NamedStruct>> + Send {
        self.call(|b| {
            let plan: Vec<u8> = b.call_method1("select_all", (name,))?.extract()?;
            base_schema(&decode(&plan)?).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "select_all({name:?}) returned a plan with no table read in it"
                ))
            })
        })
    }

    fn returns_rows(
        &mut self,
        plans: &[ddx_ad::Bound],
    ) -> impl Future<Output = PyResult<Vec<bool>>> + Send {
        self.call(|b| {
            plans
                .iter()
                .map(|plan| {
                    b.call_method1("returns_rows", (bytes(b.py(), plan),))?
                        .extract()
                })
                .collect()
        })
    }

    fn materialize(
        &mut self,
        name: &str,
        plan: &ddx_ad::Bound,
    ) -> impl Future<Output = PyResult<()>> + Send {
        self.call(|b| {
            b.call_method1("materialize", (name, bytes(b.py(), plan)))?;
            Ok(())
        })
    }

    fn drop_table(&mut self, name: &str) -> impl Future<Output = PyResult<()>> + Send {
        self.call(|b| {
            b.call_method1("drop_table", (name,))?;
            Ok(())
        })
    }

    fn plan_sql(&mut self, sql: &str) -> impl Future<Output = PyResult<Plan>> + Send {
        self.call(|b| decode(&b.call_method1("plan_sql", (sql,))?.extract::<Vec<u8>>()?))
    }
}

/// The output of a future whose every await is already done, as a run on a
/// [`PyBackend`] is.
fn now<F: Future>(f: F) -> F::Output {
    match pin!(f).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(out) => out,
        Poll::Pending => unreachable!("a Python backend's future is never pending"),
    }
}

/// Run `program` on `backend`: `ddx_ad::run`, the same protocol the Rust
/// adapters follow. Its checks run first, then every step. A run that
/// succeeds drops the intermediate tables and leaves the value and the
/// gradients (or the value with its tangents); one that fails leaves none of
/// the program's tables. `program` is a `BackwardProgram` or a
/// `ForwardProgram`, whose input tables must be registered first.
///
/// `backend` is any object with four methods (see `ddxdb.Backend`):
/// `select_all(name) -> bytes` (the engine's serialized Substrait plan of
/// `SELECT * FROM name`, whose read gives the table's schema; asked once per
/// table, and again after the table is rewritten), `returns_rows(plan) ->
/// bool`, `materialize(name, plan)` and `drop_table(name)`. Plans arrive as
/// serialized Substrait with their reads of earlier steps bound. Raises
/// `InvalidColumn` when a check returns a row, or what the backend raised.
#[pyfunction]
fn run(backend: Bound<'_, PyAny>, program: Bound<'_, PyAny>) -> PyResult<()> {
    let mut backend = PyBackend(backend.unbind());
    let result = if let Ok(p) = program.extract::<PyRef<'_, BackwardProgram>>() {
        now(ddx_ad::run(&mut backend, &p.0))
    } else if let Ok(p) = program.extract::<PyRef<'_, ForwardProgram>>() {
        now(ddx_ad::run(&mut backend, &p.0))
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "run takes a BackwardProgram or a ForwardProgram",
        ));
    };
    result.map_err(run_err)
}

/// Compute what the `grad`, `vjp` and `jvp` calls in `statements` read, on
/// `backend`, and rewrite each statement to read it (`ddx_ad::rewrite`).
/// `backend` is as for `run`, with a fifth method, `plan_sql(sql) -> bytes`:
/// the engine's serialized Substrait plan of a query. Returns the rewritten
/// statements and the tables they read, for the caller to drop once it has
/// planned them. `dialect` is how the statements are written (a name
/// `sqlparser` knows, or `"datafusion"`).
#[pyfunction]
#[pyo3(signature = (backend, statements, dialect = "datafusion"))]
fn rewrite(
    backend: Bound<'_, PyAny>,
    statements: Vec<String>,
    dialect: &str,
) -> PyResult<(Vec<String>, Vec<String>)> {
    let parser = parser_for(dialect)?;
    let refs: Vec<&str> = statements.iter().map(String::as_str).collect();
    let mut backend = PyBackend(backend.unbind());
    let out = now(ddx_ad::rewrite(&mut backend, &refs, parser.as_ref())).map_err(run_err)?;
    Ok((out.statements, out.tables))
}

/// A run's failure as the exception a Python caller branches on.
fn run_err(e: ddx_ad::RunError<PyErr>) -> PyErr {
    match e {
        ddx_ad::RunError::Refused(e) => ad_to_py_err(e),
        ddx_ad::RunError::Engine(e) => e,
        other => DdxError::new_err(other.to_string()),
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
    m.add_function(wrap_pyfunction!(supported_functions, m)?)?;
    m.add_function(wrap_pyfunction!(grad_plan, m)?)?;
    m.add_function(wrap_pyfunction!(vjp_plan, m)?)?;
    m.add_function(wrap_pyfunction!(jvp_plan, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(_not_in_subquery, m)?)?;
    m.add_class::<BackwardProgram>()?;
    m.add_class::<ForwardProgram>()?;
    m.add_class::<Step>()?;
    m.add_class::<Check>()?;
    m.add_class::<InputTable>()?;
    m.add_class::<OutputTable>()?;
    m.add_class::<Tangent>()?;
    m.add_function(wrap_pyfunction!(rewrite, m)?)?;

    m.add("DdxError", m.py().get_type::<DdxError>())?;
    m.add(
        "UnsupportedExpression",
        m.py().get_type::<UnsupportedExpression>(),
    )?;
    m.add("NotScalar", m.py().get_type::<NotScalar>())?;
    m.add("UnknownColumn", m.py().get_type::<UnknownColumn>())?;
    m.add("InvalidColumn", m.py().get_type::<InvalidColumn>())?;
    Ok(())
}
