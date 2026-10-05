// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad` and `vjp` of a query: the program of plans that computes them.
//!
//! As in JAX:
//!
//! - [`vjp`] pulls a cotangent of the query's output back to the `wrt`
//!   columns. The cotangent is a relation with the output's dims and values,
//!   which the caller registers as the program's one input table
//!   ([`BackwardProgram::inputs`]) before it runs.
//! - [`grad`] is `vjp` of a loss seeded with 1. The query's output must be one
//!   row and one column, as `jax.grad` requires a scalar.
//!
//! Either way the result is a [`BackwardProgram`]: steps the engine runs in
//! order, materializing each under its name (design.md §4.4).
//!
//! 1. **Forward**: one step per saved aggregate, `…saved_{n}`, then the
//!    query's own result, [`BackwardProgram::value`], so a program is
//!    `value_and_grad`.
//! 2. **Backward**: one step per saved aggregate that gradient reaches,
//!    `…cotangent_{n}`, with the aggregate's dims and the cotangent of each of
//!    its values that receives gradient, under the same names.
//! 3. **Gradients**: one step per `wrt` table, named in
//!    [`BackwardProgram::gradients`], shaped like the table: its dims and its
//!    `wrt` values, under the table's own column names and in their types,
//!    holding the gradient; `0` where none reached, NULL where the value is
//!    NULL.
//!
//! Every name starts with a prefix fresh to the program (`__ddx_{id}_`), so
//! programs never read or replace each other's tables. Before the steps, an
//! adapter runs [`BackwardProgram::checks`]: plans that must return no rows,
//! one per `wrt` table, confirming its dims identify its rows.
//!
//! # Fan-in
//!
//! Saved aggregates are processed parents first, so each one's cotangent is
//! complete before it is pushed further down. A relation read in more than one
//! place (a table joined twice, a weight table read by every layer) gets one
//! contribution per read, and they are added with `UNION ALL` and a grouped
//! `SUM`. A join would drop the rows one contribution lacks: cotangents are
//! sparse, and an inner join of nn.py's three per-layer contributions to
//! `weight` matches no rows at all.

use std::collections::BTreeSet;

use ddx_core::Ddx;
use substrait::proto::join_rel::JoinType;
use substrait::proto::{Expression, Plan, Rel};

use crate::compose::{output_plan, Differentiable, Subject};
use crate::emit::{aggregate, join, plan, project_emit, read_step, read_table, select, union_all};
use crate::error::{AdError, Result};
use crate::expr::{call, cast, field, if_then, lit_f64, map_fields, null_f64};
use crate::forward::{saved_name, step_columns, Forward, Input};
use crate::functions::{Extensions, Functions};
use crate::relation::ColumnRef;
use crate::relation::Table;
use crate::tables::{InputTable, Of, OutputTable};
use crate::transpose::{Contribution, Transposer};

/// One step of a program: a plan, and the name its result is materialized
/// under.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Step {
    /// The table name to materialize the result as.
    pub name: String,
    /// The plan. Its reads of earlier steps are unbound; see
    /// [`crate::emit::bind_reads`].
    pub plan: Plan,
}

/// The steps that compute a query's value and its gradient.
///
/// A program is data: plans, and the names to materialize their results
/// under. Its fields will grow, so it is read, not built, outside this crate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BackwardProgram {
    /// The tables the caller registers before the program runs: for
    /// [`vjp`], the output's cotangent, its keys the output's dims and its
    /// values the output's values that depend on `wrt`, named as in the
    /// output. None for [`grad`], which seeds 1 itself. Of a program, that
    /// program's own inputs come first.
    pub inputs: Vec<InputTable>,
    /// Plans that must return no rows, run before the steps. Each checks a
    /// promise the plan cannot show (that a `wrt` table's dims identify its
    /// rows, that a cotangent's keys do); a row back means the promise is
    /// broken, and the program must not run.
    pub checks: Vec<Check>,
    /// The saved aggregates, then the query's value.
    pub forward_steps: Vec<Step>,
    /// The cotangents, then the gradients.
    pub backward_steps: Vec<Step>,
    /// The query's output, under its own column names.
    pub value: OutputTable,
    /// One per `wrt` table, [`Of::Table`]: its dims, then each `wrt`
    /// value's gradient under the value's name.
    pub gradients: Vec<OutputTable>,
}

impl BackwardProgram {
    /// Every step, in the order they must run.
    pub fn steps(&self) -> impl Iterator<Item = &Step> {
        self.forward_steps.iter().chain(&self.backward_steps)
    }

    /// Step `i` of [`BackwardProgram::steps`], as [`crate::Action`] numbers
    /// them.
    pub fn step(&self, i: usize) -> &Step {
        self.steps().nth(i).expect("a step index from this program")
    }

    /// The steps whose tables only other steps read: the saved aggregates
    /// and the cotangents. An adapter may drop them once the program has run;
    /// the value and the gradients are what a caller reads.
    pub fn intermediate_steps(&self) -> impl Iterator<Item = &Step> {
        let keep: Vec<&str> = std::iter::once(self.value.step.as_str())
            .chain(self.gradients.iter().map(|g| g.step.as_str()))
            .collect();
        self.steps()
            .filter(move |s| !keep.contains(&s.name.as_str()))
    }
}

/// The steps that compute a query's output and its tangent.
///
/// Its anatomy is a [`BackwardProgram`]'s: input tables ([`InputTable`])
/// the caller registers, checks, steps, and output tables ([`OutputTable`]) it leaves. Like
/// it, a program is data: it is read, not built, outside this crate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ForwardProgram {
    /// One per `wrt` table, [`Of::Table`]: its tangent, keyed by the table's
    /// dims, to register before the program runs. Of a program, that
    /// program's own inputs come first.
    pub inputs: Vec<InputTable>,
    /// Plans that must return no rows, run before the steps: that each
    /// `wrt` table's dims identify its rows, and that each tangent table has
    /// one row per dim tuple.
    pub checks: Vec<Check>,
    /// The steps, in order.
    pub steps: Vec<Step>,
    /// The query's output, or a program's value, with its tangents.
    pub value: OutputTable,
    /// For [`jvp`](crate::jvp) of a program: each of the program's
    /// gradients, with its tangents. Of a [`grad`](crate::grad) program along
    /// `v`, those are the Hessian-vector product `H·v`, shaped like the
    /// gradient. Empty for [`jvp`](crate::jvp) of a query.
    pub gradients: Vec<OutputTable>,
}

/// A plan that must return no rows (see [`BackwardProgram::checks`]).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Check {
    /// The plan.
    pub plan: Plan,
    /// What a returned row means, for the error.
    pub message: String,
}

impl Step {
    /// The plan as Substrait protobuf bytes, for a host that holds plans as
    /// bytes rather than as this crate's `substrait` types (see
    /// [`decode_plan`]).
    pub fn plan_bytes(&self) -> Vec<u8> {
        prost::Message::encode_to_vec(&self.plan)
    }
}

impl Check {
    /// The plan as Substrait protobuf bytes (see [`Step::plan_bytes`]).
    pub fn plan_bytes(&self) -> Vec<u8> {
        prost::Message::encode_to_vec(&self.plan)
    }
}

/// A Substrait plan from its protobuf bytes.
///
/// [`grad`] and [`vjp`] take this crate's `substrait::proto::Plan`, so a host
/// that links `substrait` must link the same version (it is re-exported as
/// [`crate::substrait`]). Most of the Substrait ecosystem holds plans as
/// bytes instead; such a host never names the type:
/// `grad(&decode_plan(bytes)?, wrt)`, then [`Step::plan_bytes`].
pub fn decode_plan(bytes: &[u8]) -> Result<Plan> {
    <Plan as prost::Message>::decode(bytes)
        .map_err(|e| AdError::InvalidPlan(format!("not a Substrait plan: {e}")))
}

/// How [`grad_with`] and [`vjp_with`] build a program.
#[derive(Clone, Default)]
pub struct Options {
    ddx: Option<Ddx>,
    namespace: Option<String>,
    restrict: Vec<(String, Plan)>,
}

impl Options {
    /// The defaults: [`Ddx::new`], and a fresh namespace per program.
    pub fn new() -> Self {
        Options::default()
    }

    /// Differentiate elementwise expressions with `ddx`, which may carry a
    /// caller's own scalar rules.
    pub fn ddx(mut self, ddx: Ddx) -> Self {
        self.ddx = Some(ddx);
        self
    }

    /// Name every table the program writes `{namespace}…` instead of under a
    /// fresh prefix. The same plan and options then give the same program,
    /// byte for byte, which a golden test of emitted plans, or a cache of
    /// programs keyed by plan, needs. It must start with `__ddx_`, the prefix
    /// ddx reserves, end with `_` (so `{namespace}value` reads as two parts),
    /// and hold only lower-case ASCII letters, digits and `_` (an engine folds
    /// an unquoted name to lower case). Two programs on one engine must not
    /// share a namespace: they would write the same tables.
    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Compute `table`'s gradient only on the rows where a predicate holds,
    /// because nothing reads the others (a `grad(…)` in SQL whose statement
    /// filters it: [`crate::sql::Job::restrict`]). `select` is the engine's
    /// own plan of `SELECT * FROM table WHERE predicate`; the predicate may
    /// read only the table's dims, since a gradient row is a sum grouped by
    /// them. Rows outside it are left out of the gradient step, and every
    /// contribution to it is filtered before it is summed, which an engine
    /// pushes down into the region beneath.
    pub fn restrict(mut self, table: impl Into<String>, select: Plan) -> Self {
        self.restrict.push((table.into(), select));
        self
    }

    fn forward(&self, plan: &Plan, wrt: &[ColumnRef]) -> Result<Forward> {
        match &self.namespace {
            None => Forward::new(plan, wrt),
            Some(_) => Forward::in_namespace(plan, wrt, self.namespace_or_new()?),
        }
    }

    /// The namespace to write under: the caller's, checked, or a fresh one.
    pub(crate) fn namespace_or_new(&self) -> Result<String> {
        let Some(ns) = &self.namespace else {
            return Ok(crate::forward::new_namespace());
        };
        let allowed = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_';
        if !ns.starts_with("__ddx_") || !ns.ends_with('_') || !ns.chars().all(allowed) {
            return Err(AdError::InvalidOptions(format!(
                "namespace `{ns}` must start with `__ddx_`, end with `_`, and hold only \
                 lower-case ASCII letters, digits and `_`"
            )));
        }
        Ok(ns.clone())
    }

    /// Refuse [`Options::restrict`] for a program that cannot honour it:
    /// set and ignored, it would compute what the caller asked not to, or
    /// read as a restriction that was applied.
    pub(crate) fn refuse_restrict(&self, what: &str) -> Result<()> {
        if self.restrict.is_empty() {
            return Ok(());
        }
        Err(AdError::InvalidOptions(format!(
            "{what} computes every row's tangent; Options::restrict applies only to grad and \
             vjp"
        )))
    }

    /// The `ddx-core` engine to differentiate scalar expressions with.
    pub(crate) fn ddx_or_default(&self) -> Ddx {
        self.ddx.clone().unwrap_or_default()
    }
}

fn cotangent_name(namespace: &str, n: usize) -> String {
    format!("{namespace}cotangent_{n}")
}

/// Table `i`'s gradient step: numbered, so it is unique whatever the table is
/// called, with the table's last name part after it, for a reader.
pub(crate) fn gradient_name(namespace: &str, i: usize, table: &[String]) -> String {
    format!("{namespace}grad_{i}_{}", readable(table))
}

/// A table's last name part, lower case and ASCII, since an engine folds the
/// unquoted name a step is registered under.
pub(crate) fn readable(table: &[String]) -> String {
    table
        .last()
        .map(String::as_str)
        .unwrap_or("")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// The gradient of the loss `of` computes with respect to the `wrt`
/// columns: of a query's [`Plan`], which must return one row and one
/// column; of a program, of its one output table, which must too (see
/// [`Differentiable`]).
///
/// Each [`ColumnRef`] names a table as [`ColumnRef::table`] describes, and a
/// column of it, case-insensitively.
pub fn grad<D: Differentiable + ?Sized>(of: &D, wrt: &[ColumnRef]) -> Result<BackwardProgram> {
    grad_with(of, wrt, &Options::new())
}

/// [`grad`] with [`Options`]: a caller's `ddx-core` engine, for custom scalar
/// rules, or a fixed namespace.
pub fn grad_with<D: Differentiable + ?Sized>(
    of: &D,
    wrt: &[ColumnRef],
    options: &Options,
) -> Result<BackwardProgram> {
    backward(of, wrt, options, Seed::One)
}

/// The vector-Jacobian product of `of`, a query's [`Plan`] or a program's
/// one output table (see [`Differentiable`]), with respect to the `wrt`
/// columns: the program pulls back the cotangent the caller registers as its
/// input table ([`BackwardProgram::inputs`]).
pub fn vjp<D: Differentiable + ?Sized>(of: &D, wrt: &[ColumnRef]) -> Result<BackwardProgram> {
    vjp_with(of, wrt, &Options::new())
}

/// [`vjp`] with [`Options`].
pub fn vjp_with<D: Differentiable + ?Sized>(
    of: &D,
    wrt: &[ColumnRef],
    options: &Options,
) -> Result<BackwardProgram> {
    backward(of, wrt, options, Seed::Cotangent)
}

/// [`grad`] or [`vjp`] of `of`: of a program, of its output's plan, written
/// out (see [`crate::compose`]).
fn backward<D: Differentiable + ?Sized>(
    of: &D,
    wrt: &[ColumnRef],
    options: &Options,
    seed: Seed,
) -> Result<BackwardProgram> {
    let ddx = options.ddx_or_default();
    match of.subject() {
        Subject::Query(plan) => build(&ddx, &options.forward(plan, wrt)?, seed, &options.restrict),
        Subject::Program(parts) => {
            let plan = output_plan(&parts)?;
            let mut program = build(&ddx, &options.forward(&plan, wrt)?, seed, &options.restrict)?;
            parts.carry(&mut program.inputs, &mut program.checks);
            Ok(program)
        }
    }
}

enum Seed {
    One,
    Cotangent,
}

fn build(
    ddx: &Ddx,
    f: &Forward,
    seed: Seed,
    restrict: &[(String, Plan)],
) -> Result<BackwardProgram> {
    let mut t = Transposer::new(f, ddx);
    // Each wrt table's restriction, if any: a predicate over its columns.
    let mut restrictions: Vec<Option<Expression>> = vec![None; f.tables.len()];
    for (name, select) in restrict {
        let i = f
            .tables
            .iter()
            .position(|t| crate::relation::table_matches(name, &t.names))
            .ok_or_else(|| {
                AdError::InvalidOptions(format!(
                    "restrict names `{name}`, which is not a wrt table of the query"
                ))
            })?;
        let pred = restriction(select, &f.tables[i], &mut t.ext)?;
        restrictions[i] = Some(match restrictions[i].take() {
            // Two restrictions of one table: rows both allow.
            Some(prev) => {
                let and = t.ext.anchor("and");
                call(and, vec![prev, pred])
            }
            None => pred,
        });
    }

    let mut forward_steps = Vec::new();
    for (n, saved) in f.saved.iter().enumerate() {
        forward_steps.push(Step {
            name: saved_name(&f.namespace, n),
            plan: plan(saved.rel.clone(), step_columns(saved.outputs.len()), &t.ext),
        });
    }
    forward_steps.push(Step {
        name: format!("{}value", f.namespace),
        plan: plan(
            select(f.output.rel.clone(), f.output.outputs.clone()),
            f.output_names.clone(),
            &t.ext,
        ),
    });

    let mut backward_steps = Vec::new();
    // The region steps the transposes so far asked for, before anything that
    // reads their contributions.
    let drain = |t: &mut Transposer, steps: &mut Vec<Step>| {
        for (name, rel, names) in t.steps.drain(..) {
            steps.push(Step {
                name,
                plan: plan(rel, names, &t.ext),
            });
        }
    };
    let (inputs, cotangent_check) = match seed {
        Seed::One => {
            let col = scalar_output(f)?;
            // A NULL loss does not move with anything in it, as with vjp's
            // NULL output rows (seed_cotangent): its seed is NULL, so no
            // gradient flows back through it.
            let is_null = t.ext.anchor("is_null");
            let seed = if_then(
                vec![(call(is_null, vec![field(col)]), null_f64())],
                lit_f64(1.0),
            );
            t.region(&f.output, f.output.rel.clone(), vec![(col, seed)])?;
            (Vec::new(), None)
        }
        Seed::Cotangent => {
            let (names, dims) = seed_cotangent(&mut t, f)?;
            let check = cotangent_check(&mut t, f, &names, dims);
            let input = InputTable {
                name: format!("{}cotangent", f.namespace),
                of: Of::Output,
                columns: names,
                keys: dims,
            };
            (vec![input], Some(check))
        }
    };

    drain(&mut t, &mut backward_steps);

    // Parents first: every saved aggregate that reads saved aggregate n comes
    // after it in `f.saved`.
    for n in (0..f.saved.len()).rev() {
        let Some(contribs) = t.contributions.remove(&Input::Saved(n)) else {
            continue; // no gradient reaches it
        };
        let dims = f.saved[n].dims();
        let (rel, cols) = combine(&mut t, contribs, dims.len())?;
        let names: Vec<String> = dims.iter().chain(&cols).map(|c| format!("c{c}")).collect();
        backward_steps.push(Step {
            name: cotangent_name(&f.namespace, n),
            plan: plan(rel, names.clone(), &t.ext),
        });
        t.saved(n, &cols, read_step(&cotangent_name(&f.namespace, n), names))?;
        drain(&mut t, &mut backward_steps);
    }

    let mut gradients = Vec::new();
    for (i, table) in f.tables.iter().enumerate() {
        let contribs = t.contributions.remove(&Input::Table(i)).unwrap_or_default();
        let step = gradient_name(&f.namespace, i, &table.names);
        let rel = dense_gradient(&mut t, i, contribs, restrictions[i].as_ref())?;
        let columns: Vec<String> = table
            .dims
            .iter()
            .chain(&table.values)
            .map(|&c| table.columns()[c].clone())
            .collect();
        backward_steps.push(Step {
            name: step.clone(),
            plan: plan(rel, columns.clone(), &t.ext),
        });
        gradients.push(OutputTable {
            step,
            of: Of::Table(table.names.clone()),
            columns,
            tangents: Vec::new(),
        });
    }
    if !t.contributions.is_empty() {
        return Err(AdError::Internal(format!(
            "cotangents were left unconsumed: {:?}",
            t.contributions.keys().collect::<Vec<_>>()
        )));
    }
    let mut program = BackwardProgram {
        inputs,
        checks: f
            .tables
            .iter()
            .map(|table| dims_check(&mut t.ext, table))
            .chain(cotangent_check)
            .collect(),
        forward_steps,
        backward_steps,
        value: OutputTable {
            step: format!("{}value", f.namespace),
            of: Of::Output,
            columns: f.output_names.clone(),
            tangents: Vec::new(),
        },
        gradients,
    };
    prune_program(&mut program);
    Ok(program)
}

/// Drop the columns nothing reads (see [`crate::prune`]): within each plan,
/// then across steps, from the last back, so each step keeps only the columns
/// some later step reads. The value and the gradients keep all theirs.
fn prune_program(program: &mut BackwardProgram) {
    use std::collections::{BTreeSet, HashMap};
    for c in program.checks.iter_mut() {
        crate::prune::prune_plan(&mut c.plan);
    }
    let keep: BTreeSet<String> = std::iter::once(program.value.step.clone())
        .chain(program.gradients.iter().map(|g| g.step.clone()))
        .collect();
    // Every step's columns that plans after it read, by name.
    let mut read: HashMap<String, BTreeSet<String>> = HashMap::new();
    let note_reads = |plan: &Plan, read: &mut HashMap<String, BTreeSet<String>>| {
        crate::emit::for_each_unbound_read(plan, &mut |name, cols| {
            read.entry(name.to_string())
                .or_default()
                .extend(cols.iter().cloned());
        });
    };
    for c in &program.checks {
        note_reads(&c.plan, &mut read);
    }
    let n_forward = program.forward_steps.len();
    let total = n_forward + program.backward_steps.len();
    for i in (0..total).rev() {
        let step = if i < n_forward {
            &mut program.forward_steps[i]
        } else {
            &mut program.backward_steps[i - n_forward]
        };
        let narrowed = !keep.contains(&step.name) && {
            let used = read.get(&step.name).cloned().unwrap_or_default();
            crate::prune::prune_plan_to_names(&mut step.plan, &used)
        };
        if !narrowed {
            crate::prune::prune_plan(&mut step.plan);
        }
        note_reads(&step.plan, &mut read);
    }
}

/// The loss column for [`grad`]: the query must return one column, on one
/// row.
fn scalar_output(f: &Forward) -> Result<usize> {
    let names = &f.output_names;
    let [col] = f.output.outputs.as_slice() else {
        return Err(AdError::NotScalar(format!(
            "grad needs a loss, one row and one column, but the query returns {} columns: \
             {names:?}. Return only the loss, or use vjp",
            names.len()
        )));
    };
    // One row is certain only when every input the output reads is: a table
    // has a row per dim tuple, and constant data joined in can multiply rows
    // without having dims ddx knows about.
    if f.output
        .slots
        .iter()
        .any(|s| s.offset.is_some() && !s.at_most_one_row)
    {
        return Err(AdError::NotScalar(format!(
            "grad needs a loss, one row and one column, and ddx cannot show from the plan \
             that `{}` is one row: it reads a table, a grouped aggregate, or data the plan \
             does not prove is one row (ddx accepts an ungrouped aggregate). Sum it into one \
             row, or use vjp",
            names[0]
        )));
    }
    if !f.output.varied[*col] {
        return Err(AdError::NotScalar(format!(
            "the loss `{}` does not depend on any wrt column",
            names[0]
        )));
    }
    Ok(*col)
}

/// The output's region columns that are dims: the dims of every input whose
/// columns reach the output (a table, or a saved aggregate that grouped).
fn output_dims(f: &Forward) -> Vec<usize> {
    let mut dims = Vec::new();
    for s in &f.output.slots {
        let Some(offset) = s.offset else { continue };
        let input_dims = match s.input {
            Input::Table(i) => f.tables[i].dims.clone(),
            Input::Saved(n) => f.saved[n].dims(),
            Input::Const => continue,
        };
        dims.extend(input_dims.into_iter().map(|d| offset + d));
    }
    dims
}

/// Seed [`vjp`]: join the output to the caller's cotangent on the output's
/// dims, which must all be output columns so each output row is identified.
/// Returns the cotangent table's columns, and how many of them (the first)
/// are its keys.
fn seed_cotangent(t: &mut Transposer, f: &Forward) -> Result<(Vec<String>, usize)> {
    let out = &f.output;
    if out
        .slots
        .iter()
        .any(|s| s.offset.is_some() && s.input == Input::Const)
    {
        return Err(AdError::NotImplemented(
            "vjp of an output joined to constant data: ddx cannot tell which columns identify \
             its rows. Join the data before the last aggregate, or use grad on a loss"
                .into(),
        ));
    }
    let mut dim_cols = Vec::new();
    for d in output_dims(f) {
        match out.outputs.iter().position(|&o| o == d) {
            Some(i) => dim_cols.push(i),
            None => {
                return Err(AdError::NotImplemented(format!(
                    "vjp needs the output to keep every dim so each row is identified; it \
                     drops one of its inputs' dims (region column {d})"
                )))
            }
        }
    }
    let value_cols: Vec<usize> = (0..out.outputs.len())
        .filter(|i| !dim_cols.contains(i) && out.varied[out.outputs[*i]])
        .collect();
    if value_cols.is_empty() {
        return Err(AdError::NotScalar(
            "no output column depends on a wrt column".into(),
        ));
    }
    let names: Vec<String> = dim_cols
        .iter()
        .chain(&value_cols)
        .map(|&i| f.output_names[i].clone())
        .collect();
    let width = out.defs.len();
    let keys: Vec<Expression> = dim_cols.iter().map(|&i| field(out.outputs[i])).collect();
    let positions: Vec<usize> = (0..dim_cols.len()).collect();
    let base = t.join_on(
        out.rel.clone(),
        read_step(&format!("{}cotangent", f.namespace), names.clone()),
        keys,
        &positions,
        width,
    )?;
    // An output row whose value is NULL does not move with anything in it,
    // as an aggregate that skips it would say: its cotangent is NULL, so none
    // flows through it to its other inputs (the `p` of `p + q`, `q` NULL).
    let is_null = t.ext.anchor("is_null");
    let seeds = value_cols
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            let seed = if_then(
                vec![(call(is_null, vec![field(out.outputs[i])]), null_f64())],
                field(width + dim_cols.len() + k),
            );
            (out.outputs[i], seed)
        })
        .collect();
    t.region(out, base, seeds)?;
    Ok((names, dim_cols.len()))
}

/// Add up an input's contributions: its dims, then one cotangent column per
/// input column any contribution has, in column order.
/// The predicate of `select`, an engine's plan of `SELECT * FROM table
/// WHERE predicate` (see [`Options::restrict`]), over `table`'s columns, with
/// its functions declared in `ext`.
fn restriction(select: &Plan, table: &Table, ext: &mut Extensions) -> Result<Expression> {
    use substrait::proto::plan_rel::RelType as PlanRelType;
    use substrait::proto::read_rel::ReadType;
    use substrait::proto::rel::RelType;
    let refuse = |why: &str| {
        AdError::InvalidOptions(format!(
            "the restriction of `{}` {why}; it must be the plan of `SELECT * FROM table WHERE \
             predicate`",
            table.names.join(".")
        ))
    };
    let mut rel = select
        .relations
        .iter()
        .find_map(|r| match &r.rel_type {
            Some(PlanRelType::Root(root)) => root.input.as_ref(),
            _ => None,
        })
        .ok_or_else(|| refuse("has no root"))?;
    // Down through projections to the filter, then its read.
    let filter = loop {
        match &rel.rel_type {
            Some(RelType::Project(p)) => {
                rel = p.input.as_deref().ok_or_else(|| refuse("is malformed"))?;
            }
            Some(RelType::Filter(f)) => break f,
            _ => return Err(refuse("has no filter over the table")),
        }
    };
    let Some(RelType::Read(read)) = filter.input.as_deref().and_then(|r| r.rel_type.as_ref())
    else {
        return Err(refuse("filters something other than the table"));
    };
    let read_names = match (&read.read_type, &read.base_schema) {
        (Some(ReadType::NamedTable(t)), Some(schema)) if table_matches_names(table, &t.names) => {
            schema.names.clone()
        }
        _ => return Err(refuse("reads another table")),
    };
    let condition = filter
        .condition
        .as_deref()
        .ok_or_else(|| refuse("has an empty filter"))?;
    let functions = Functions::from_plan(select)?;
    let columns = table.columns();
    let on_table = map_fields(condition, &mut |i| {
        let name = read_names
            .get(i)
            .ok_or_else(|| refuse("reads past the table"))?;
        let c = columns
            .iter()
            .position(|c| c.eq_ignore_ascii_case(name))
            .ok_or_else(|| refuse("reads a column the table does not have"))?;
        if !table.dims.contains(&c) {
            return Err(refuse(&format!(
                "reads `{name}`, which is not one of the table's dims"
            )));
        }
        Ok(c)
    })?;
    redeclare_functions(on_table, &functions, ext)
}

fn table_matches_names(table: &Table, names: &[String]) -> bool {
    crate::relation::table_matches(&names.join("."), &table.names)
        || crate::relation::table_matches(&table.names.join("."), names)
}

/// `e` with each function anchor of `from` replaced by its anchor in `ext`,
/// by name. Literals, field references, casts, `CASE`, `IN` lists and scalar
/// functions are followed; anything else is refused.
fn redeclare_functions(
    mut e: Expression,
    from: &Functions,
    ext: &mut Extensions,
) -> Result<Expression> {
    use substrait::proto::expression::RexType;
    use substrait::proto::function_argument::ArgType;
    fn walk(e: &mut Expression, from: &Functions, ext: &mut Extensions) -> Result<()> {
        match e.rex_type.as_mut() {
            Some(RexType::Literal(_)) | Some(RexType::Selection(_)) => Ok(()),
            Some(RexType::ScalarFunction(f)) => {
                f.function_reference = ext.anchor(from.name(f.function_reference)?);
                for a in f.arguments.iter_mut() {
                    if let Some(ArgType::Value(x)) = a.arg_type.as_mut() {
                        walk(x, from, ext)?;
                    }
                }
                Ok(())
            }
            Some(RexType::Cast(c)) => match c.input.as_deref_mut() {
                Some(x) => walk(x, from, ext),
                None => Ok(()),
            },
            Some(RexType::IfThen(it)) => {
                for c in it.ifs.iter_mut() {
                    for x in c.r#if.iter_mut().chain(c.then.iter_mut()) {
                        walk(x, from, ext)?;
                    }
                }
                for x in it.r#else.iter_mut() {
                    walk(x, from, ext)?;
                }
                Ok(())
            }
            Some(RexType::SingularOrList(l)) => {
                for x in l.value.iter_mut() {
                    walk(x, from, ext)?;
                }
                for x in l.options.iter_mut() {
                    walk(x, from, ext)?;
                }
                Ok(())
            }
            _ => Err(AdError::InvalidOptions(
                "a restriction may only compare a table's dims with literals".into(),
            )),
        }
    }
    walk(&mut e, from, ext)?;
    Ok(e)
}

fn combine(
    t: &mut Transposer,
    contribs: Vec<Contribution>,
    dims: usize,
) -> Result<(Rel, Vec<usize>)> {
    let cols: Vec<usize> = contribs
        .iter()
        .flat_map(|c| c.cols.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut aligned: Vec<Rel> = contribs
        .into_iter()
        .map(|c| {
            // Pad the columns this contribution lacks with zeros.
            let mut emit: Vec<usize> = (0..dims).collect();
            let mut zeros = Vec::new();
            let width = dims + c.cols.len();
            for col in &cols {
                match c.cols.iter().position(|x| x == col) {
                    Some(i) => emit.push(dims + i),
                    None => {
                        emit.push(width + zeros.len());
                        zeros.push(lit_f64(0.0));
                    }
                }
            }
            project_emit(c.rel, zeros, Some(emit))
        })
        .collect();
    if aligned.len() == 1 {
        return Ok((aligned.pop().unwrap(), cols));
    }
    let sum = t.ext.anchor("sum");
    let rel = aggregate(
        union_all(aligned),
        (0..dims).map(field).collect(),
        (0..cols.len())
            .map(|i| (sum, vec![field(dims + i)]))
            .collect(),
    );
    Ok((rel, cols))
}

/// Table `i`'s gradient, dense: every row of the table, its dims, and each
/// `wrt` value's gradient, in the value's own type.
///
/// Two conventions are pinned here, whatever the loss:
/// - a row no gradient reached gets `0`;
/// - a row whose value is NULL gets NULL, as v1 does (#60): a missing value
///   has no gradient, not a zero one.
fn dense_gradient(
    t: &mut Transposer,
    i: usize,
    contribs: Vec<Contribution>,
    restriction: Option<&Expression>,
) -> Result<Rel> {
    let table = &t.f.tables[i];
    // Only the rows the restriction allows: in the table, and in every
    // contribution, whose leading columns are the table's dims in order.
    let contribs = match restriction {
        None => contribs,
        Some(pred) => {
            let on_dims = map_fields(pred, &mut |c| {
                table.dims.iter().position(|&d| d == c).ok_or_else(|| {
                    AdError::Internal("a restriction reads a column that is not a dim".into())
                })
            })?;
            contribs
                .into_iter()
                .map(|c| Contribution {
                    rel: crate::emit::filter(c.rel, on_dims.clone()),
                    cols: c.cols,
                })
                .collect()
        }
    };
    let dims = table.dims.clone();
    let values = table.values.clone();
    let types: Vec<substrait::proto::Type> = values
        .iter()
        .map(|&v| {
            let mut ty = table
                .schema
                .r#struct
                .as_ref()
                .and_then(|s| s.types.get(v).cloned())
                .unwrap_or_else(crate::expr::fp64);
            set_nullable(&mut ty);
            ty
        })
        .collect();
    // The table's dims, then its wrt values (for the NULL convention).
    let picked: Vec<usize> = dims.iter().chain(&values).copied().collect();
    let read = read_table(table.names.clone(), table.schema.clone());
    let read = match restriction {
        Some(pred) => crate::emit::filter(read, pred.clone()),
        None => read,
    };
    let rows = select(read, picked);
    let (k, v) = (dims.len(), values.len());
    let (joined, cols, right) = if contribs.is_empty() {
        (rows, Vec::new(), k + v)
    } else {
        let (summed, cols) = combine(t, contribs, k)?;
        let same = t.ext.anchor("is_not_distinct_from");
        let and = t.ext.anchor("and");
        let cond = (0..k)
            .map(|d| call(same, vec![field(d), field(k + v + d)]))
            .reduce(|a, b| call(and, vec![a, b]))
            .unwrap_or_else(lit_true);
        (join(rows, summed, cond, JoinType::Left), cols, k + v + k)
    };
    // CASE, not coalesce: DataFusion 54 runs coalesce only after its
    // simplifier has rewritten it to a CASE, and a context may not run that
    // rule.
    let is_null = t.ext.anchor("is_null");
    let grads: Vec<Expression> = values
        .iter()
        .enumerate()
        .map(|(n, val)| {
            let null_value = (call(is_null, vec![field(k + n)]), null_f64());
            let pinned = match cols.iter().position(|c| c == val) {
                Some(c) => if_then(
                    vec![
                        null_value,
                        (call(is_null, vec![field(right + c)]), lit_f64(0.0)),
                    ],
                    field(right + c),
                ),
                None => if_then(vec![null_value], lit_f64(0.0)),
            };
            cast(pinned, types[n].clone())
        })
        .collect();
    let width = right + cols.len();
    let mut emit: Vec<usize> = (0..k).collect();
    emit.extend((0..grads.len()).map(|g| width + g));
    Ok(project_emit(joined, grads, Some(emit)))
}

/// Make a type nullable: a gradient is NULL where its value is.
fn set_nullable(ty: &mut substrait::proto::Type) {
    use substrait::proto::r#type::{Kind, Nullability};
    let n = Nullability::Nullable as i32;
    match ty.kind.as_mut() {
        Some(Kind::Fp32(t)) => t.nullability = n,
        Some(Kind::Fp64(t)) => t.nullability = n,
        _ => {}
    }
}

/// The check that table's dims identify its rows: the dim tuples that occur
/// more than once. Must return no rows.
/// The key tuples of `rel` (its columns `keys`) that more than one row has.
pub(crate) fn repeated_keys(ext: &mut Extensions, rel: Rel, keys: &[usize]) -> Rel {
    let count = ext.anchor("count");
    let gt = ext.anchor("gt");
    let k = keys.len();
    let grouped = aggregate(
        rel,
        keys.iter().map(|&d| field(d)).collect(),
        vec![(count, vec![lit_f64(1.0)])],
    );
    select(
        crate::emit::filter(grouped, call(gt, vec![field(k), lit_f64(1.0)])),
        (0..k).collect(),
    )
}

/// A check that a vjp's cotangent has one row per output row: a key that
/// repeats would add its rows' cotangents together, and double that row's
/// gradient without a word.
fn cotangent_check(t: &mut Transposer, f: &Forward, names: &[String], keys: usize) -> Check {
    let table = format!("{}cotangent", f.namespace);
    let keys: Vec<usize> = (0..keys).collect();
    let repeated = repeated_keys(&mut t.ext, read_step(&table, names.to_vec()), &keys);
    Check {
        plan: plan(repeated, names[..keys.len()].to_vec(), &t.ext),
        message: format!(
            "the cotangent table `{table}` has rows that share their keys ({}); it needs one \
             row per output row",
            names[..keys.len()].join(", ")
        ),
    }
}

pub(crate) fn dims_check(ext: &mut Extensions, table: &Table) -> Check {
    let repeated = repeated_keys(
        ext,
        read_table(table.names.clone(), table.schema.clone()),
        &table.dims,
    );
    let names: Vec<String> = table
        .dims
        .iter()
        .map(|&d| table.columns()[d].clone())
        .collect();
    Check {
        plan: plan(repeated, names.clone(), ext),
        message: format!(
            "table `{}` has rows that share their dims ({}), so their gradients cannot be \
             told apart; ddx takes the columns not named in wrt as the table's dims, and \
             needs them to identify its rows",
            table.names.join("."),
            names.join(", ")
        ),
    }
}

fn lit_true() -> Expression {
    use substrait::proto::expression::literal::LiteralType;
    use substrait::proto::expression::{Literal, RexType};
    Expression {
        rex_type: Some(RexType::Literal(Literal {
            nullable: false,
            type_variation_reference: 0,
            literal_type: Some(LiteralType::Boolean(true)),
        })),
    }
}
