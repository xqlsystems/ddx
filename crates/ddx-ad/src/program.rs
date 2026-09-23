// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad` and `vjp` of a query: the program of plans that computes them.
//!
//! As in JAX:
//!
//! - [`vjp`] pulls a cotangent of the query's output back to the `wrt`
//!   columns. The cotangent is a relation with the output's dims and values,
//!   supplied by the caller as the table [`COTANGENT`] before the backward steps
//!   run.
//! - [`grad`] is `vjp` of a loss seeded with 1. The query's output must be one
//!   row and one column, as `jax.grad` requires a scalar.
//!
//! Either way the result is a [`BackwardProgram`]: steps the engine runs in
//! order, materializing each under its name (design.md §4.4).
//!
//! 1. **Forward**: one step per saved aggregate, `__ddx_saved_{n}`, then
//!    [`VALUE`], the query's own result, so a program is `value_and_grad`.
//! 2. **Backward**: one step per saved aggregate that gradient reaches,
//!    `__ddx_cotangent_{n}`, with the aggregate's dims and the cotangent of
//!    each of its values that receives gradient, under the same names.
//! 3. **Gradients**: one step per `wrt` table, `__ddx_grad_{table}`, shaped like
//!    the table: its dims and its `wrt` values, under the table's own column
//!    names, holding the gradient, with `0` where none reached.
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

use crate::emit::{
    aggregate, join, plan, project, project_emit, read_step, read_table, select, union_all,
};
use crate::error::{AdError, Result};
use crate::expr::{call, field, lit_f64};
use crate::forward::{saved_name, step_columns, Forward, Input};
use crate::relation::ColumnRef;
use crate::transpose::{Contribution, Transposer};

/// One step of a program: a plan, and the name its result is materialized
/// under.
#[derive(Debug, Clone)]
pub struct Step {
    /// The table name to materialize the result as.
    pub name: String,
    /// The plan. Its reads of earlier steps are unbound; see
    /// [`crate::emit::bind_reads`].
    pub plan: Plan,
}

/// A `wrt` table's gradient.
#[derive(Debug, Clone)]
pub struct Gradient {
    /// The table, as the plan names it.
    pub table: Vec<String>,
    /// The step that computes it.
    pub step: String,
    /// Its columns: the table's dims, then its `wrt` values, named as in the
    /// table.
    pub columns: Vec<String>,
}

/// The steps that compute a query's value and its gradient.
#[derive(Debug, Clone)]
pub struct BackwardProgram {
    /// The saved aggregates, then the query's value.
    pub forward_steps: Vec<Step>,
    /// The name of the step holding the query's value: [`VALUE`].
    pub value: String,
    /// For [`vjp`]: the columns the caller's [`COTANGENT`] table must have,
    /// the output's dims then its values that depend on `wrt`, named as in
    /// the output. Empty for [`grad`], which seeds 1 itself.
    pub cotangent: Vec<String>,
    /// The cotangents, then the gradients.
    pub backward_steps: Vec<Step>,
    /// One per `wrt` table.
    pub gradients: Vec<Gradient>,
}

impl BackwardProgram {
    /// Every step, in the order they must run.
    pub fn steps(&self) -> impl Iterator<Item = &Step> {
        self.forward_steps.iter().chain(&self.backward_steps)
    }
}

/// The step holding the query's value.
pub const VALUE: &str = "__ddx_value";

/// The table a [`vjp`] program reads the output's cotangent from.
pub const COTANGENT: &str = "__ddx_cotangent";

fn cotangent_name(n: usize) -> String {
    format!("__ddx_cotangent_{n}")
}

fn gradient_name(table: &[String]) -> String {
    format!("__ddx_grad_{}", table.join("_"))
}

/// The gradient of the loss `plan` computes with respect to the `wrt`
/// columns. The query must return one row and one column.
pub fn grad(plan: &Plan, wrt: &[ColumnRef]) -> Result<BackwardProgram> {
    grad_with(&Ddx::new(), plan, wrt)
}

/// [`grad`] with a caller's `ddx-core` engine, for custom scalar rules.
pub fn grad_with(ddx: &Ddx, plan: &Plan, wrt: &[ColumnRef]) -> Result<BackwardProgram> {
    let f = Forward::new(plan, wrt)?;
    build(ddx, &f, Seed::One)
}

/// The vector-Jacobian product of the query `plan` with respect to the `wrt`
/// columns: the program pulls the cotangent in table [`COTANGENT`] back to
/// them. [`BackwardProgram::cotangent`] lists the columns that table needs.
pub fn vjp(plan: &Plan, wrt: &[ColumnRef]) -> Result<BackwardProgram> {
    vjp_with(&Ddx::new(), plan, wrt)
}

/// [`vjp`] with a caller's `ddx-core` engine.
pub fn vjp_with(ddx: &Ddx, plan: &Plan, wrt: &[ColumnRef]) -> Result<BackwardProgram> {
    let f = Forward::new(plan, wrt)?;
    build(ddx, &f, Seed::Cotangent)
}

enum Seed {
    One,
    Cotangent,
}

fn build(ddx: &Ddx, f: &Forward, seed: Seed) -> Result<BackwardProgram> {
    let mut t = Transposer::new(f, ddx);

    let mut forward_steps = Vec::new();
    for (n, saved) in f.saved.iter().enumerate() {
        forward_steps.push(Step {
            name: saved_name(n),
            plan: plan(saved.rel.clone(), step_columns(saved.outputs.len()), &t.ext),
        });
    }
    forward_steps.push(Step {
        name: VALUE.into(),
        plan: plan(
            select(f.output.rel.clone(), f.output.outputs.clone()),
            f.output_names.clone(),
            &t.ext,
        ),
    });

    let cotangent = match seed {
        Seed::One => {
            let col = scalar_output(f)?;
            t.region(&f.output, f.output.rel.clone(), vec![(col, lit_f64(1.0))])?;
            Vec::new()
        }
        Seed::Cotangent => seed_cotangent(&mut t, f)?,
    };

    // Parents first: every saved aggregate that reads saved aggregate n comes
    // after it in `f.saved`.
    let mut backward_steps = Vec::new();
    for n in (0..f.saved.len()).rev() {
        let Some(contribs) = t.contributions.remove(&Input::Saved(n)) else {
            continue; // no gradient reaches it
        };
        let dims = f.saved[n].dims();
        let (rel, cols) = combine(&mut t, contribs, dims.len())?;
        let names: Vec<String> = dims.iter().chain(&cols).map(|c| format!("c{c}")).collect();
        backward_steps.push(Step {
            name: cotangent_name(n),
            plan: plan(rel, names.clone(), &t.ext),
        });
        t.saved(n, &cols, read_step(&cotangent_name(n), names))?;
    }

    let mut gradients = Vec::new();
    for (i, table) in f.tables.iter().enumerate() {
        let contribs = t.contributions.remove(&Input::Table(i)).unwrap_or_default();
        let step = gradient_name(&table.names);
        let rel = dense_gradient(&mut t, i, contribs)?;
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
        gradients.push(Gradient {
            table: table.names.clone(),
            step,
            columns,
        });
    }
    if !t.contributions.is_empty() {
        return Err(AdError::Internal(format!(
            "cotangents were left unconsumed: {:?}",
            t.contributions.keys().collect::<Vec<_>>()
        )));
    }
    Ok(BackwardProgram {
        forward_steps,
        value: VALUE.into(),
        cotangent,
        backward_steps,
        gradients,
    })
}

/// The loss column for [`grad`]: the query must return one column, on one
/// row. One row is certain when no table or grouped aggregate with dims feeds
/// the output.
fn scalar_output(f: &Forward) -> Result<usize> {
    let names = &f.output_names;
    let [col] = f.output.outputs.as_slice() else {
        return Err(AdError::NotScalar(format!(
            "grad needs a loss, one row and one column, but the query returns {} columns: \
             {names:?}. Return only the loss, or use vjp",
            names.len()
        )));
    };
    let dims = output_dims(f);
    if !dims.is_empty() {
        return Err(AdError::NotScalar(format!(
            "grad needs a loss, one row and one column, but `{}` has a row per value of its \
             dims. Sum it into one row, or use vjp",
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
/// Returns the cotangent table's columns.
fn seed_cotangent(t: &mut Transposer, f: &Forward) -> Result<Vec<String>> {
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
    let base = t.join_on(
        out.rel.clone(),
        read_step(COTANGENT, names.clone()),
        keys,
        width,
    )?;
    let seeds = value_cols
        .iter()
        .enumerate()
        .map(|(k, &i)| (out.outputs[i], field(width + dim_cols.len() + k)))
        .collect();
    t.region(out, base, seeds)?;
    Ok(names)
}

/// Add up an input's contributions: its dims, then one cotangent column per
/// input column any contribution has, in column order.
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
/// `wrt` value's gradient, zero where none reached it.
fn dense_gradient(t: &mut Transposer, i: usize, contribs: Vec<Contribution>) -> Result<Rel> {
    let table = &t.f.tables[i];
    let dims = table.dims.clone();
    let values = table.values.clone();
    let rows = select(
        read_table(table.names.clone(), table.schema.clone()),
        dims.clone(),
    );
    if contribs.is_empty() {
        let zeros = values.iter().map(|_| lit_f64(0.0)).collect();
        return Ok(project(rows, zeros));
    }
    let (summed, cols) = combine(t, contribs, dims.len())?;
    let same = t.ext.anchor("is_not_distinct_from");
    let and = t.ext.anchor("and");
    let coalesce = t.ext.anchor("coalesce");
    let cond = (0..dims.len())
        .map(|k| call(same, vec![field(k), field(dims.len() + k)]))
        .reduce(|a, b| call(and, vec![a, b]))
        .unwrap_or_else(lit_true);
    let joined = join(rows, summed, cond, JoinType::Left);
    let right = 2 * dims.len();
    let grads: Vec<Expression> = values
        .iter()
        .map(|v| match cols.iter().position(|c| c == v) {
            Some(i) => call(coalesce, vec![field(right + i), lit_f64(0.0)]),
            None => lit_f64(0.0),
        })
        .collect();
    let width = right + cols.len();
    let mut emit: Vec<usize> = (0..dims.len()).collect();
    emit.extend((0..grads.len()).map(|i| width + i));
    Ok(project_emit(joined, grads, Some(emit)))
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
