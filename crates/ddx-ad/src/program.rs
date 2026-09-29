// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad` and `vjp` of a query: the program of plans that computes them.
//!
//! As in JAX:
//!
//! - [`vjp`] pulls a cotangent of the query's output back to the `wrt`
//!   columns. The cotangent is a relation with the output's dims and values,
//!   supplied by the caller as the table [`BackwardProgram::cotangent_table`]
//!   names, before the backward steps run.
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

use crate::emit::{aggregate, join, plan, project_emit, read_step, read_table, select, union_all};
use crate::error::{AdError, Result};
use crate::expr::{call, cast, field, if_then, lit_f64, null_f64};
use crate::forward::{saved_name, step_columns, Forward, Input};
use crate::relation::ColumnRef;
use crate::relation::Table;
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
    /// The name of the step holding the query's value.
    pub value: String,
    /// For [`vjp`]: the table the caller registers the output's cotangent as,
    /// before the backward steps run.
    pub cotangent_table: String,
    /// For [`vjp`]: the columns that table must have, the output's dims then
    /// its values that depend on `wrt`, named as in the output. Empty for
    /// [`grad`], which seeds 1 itself.
    pub cotangent: Vec<String>,
    /// Plans that must return no rows, run before the steps. Each checks a
    /// promise the plan cannot show, that a `wrt` table's dims identify its
    /// rows; a row back means the promise is broken, and the program must not
    /// run.
    pub checks: Vec<Check>,
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

    /// The steps whose tables only other steps read: the saved aggregates
    /// and the cotangents. An adapter may drop them once the program has run;
    /// the value and the gradients are what a caller reads.
    pub fn intermediate_steps(&self) -> impl Iterator<Item = &Step> {
        let keep: Vec<&str> = std::iter::once(self.value.as_str())
            .chain(self.gradients.iter().map(|g| g.step.as_str()))
            .collect();
        self.steps()
            .filter(move |s| !keep.contains(&s.name.as_str()))
    }
}

/// A plan that must return no rows (see [`BackwardProgram::checks`]).
#[derive(Debug, Clone)]
pub struct Check {
    /// The plan.
    pub plan: Plan,
    /// What a returned row means, for the error.
    pub message: String,
}

fn cotangent_name(namespace: &str, n: usize) -> String {
    format!("{namespace}cotangent_{n}")
}

/// Table `i`'s gradient step: numbered, so it is unique whatever the table is
/// called, with the table's last name part after it, for a reader: lower
/// case and ASCII, since an engine folds the unquoted name a step is
/// registered under.
fn gradient_name(namespace: &str, i: usize, table: &[String]) -> String {
    let readable: String = table
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
        .collect();
    format!("{namespace}grad_{i}_{readable}")
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
/// columns: the program pulls back the cotangent the caller registers as
/// [`BackwardProgram::cotangent_table`], with the columns
/// [`BackwardProgram::cotangent`] lists.
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
            name: cotangent_name(&f.namespace, n),
            plan: plan(rel, names.clone(), &t.ext),
        });
        t.saved(n, &cols, read_step(&cotangent_name(&f.namespace, n), names))?;
    }

    let mut gradients = Vec::new();
    for (i, table) in f.tables.iter().enumerate() {
        let contribs = t.contributions.remove(&Input::Table(i)).unwrap_or_default();
        let step = gradient_name(&f.namespace, i, &table.names);
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
        value: format!("{}value", f.namespace),
        cotangent_table: format!("{}cotangent", f.namespace),
        checks: f
            .tables
            .iter()
            .map(|table| dims_check(&mut t, table))
            .collect(),
        cotangent,
        backward_steps,
        gradients,
    })
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
        read_step(&format!("{}cotangent", f.namespace), names.clone()),
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
/// `wrt` value's gradient, in the value's own type.
///
/// Two conventions are pinned here, whatever the loss:
/// - a row no gradient reached gets `0`;
/// - a row whose value is NULL gets NULL, as v1 does (#60): a missing value
///   has no gradient, not a zero one.
fn dense_gradient(t: &mut Transposer, i: usize, contribs: Vec<Contribution>) -> Result<Rel> {
    let table = &t.f.tables[i];
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
    let rows = select(
        read_table(table.names.clone(), table.schema.clone()),
        picked,
    );
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
fn dims_check(t: &mut Transposer, table: &Table) -> Check {
    let count = t.ext.anchor("count");
    let gt = t.ext.anchor("gt");
    let k = table.dims.len();
    let grouped = aggregate(
        read_table(table.names.clone(), table.schema.clone()),
        table.dims.iter().map(|&d| field(d)).collect(),
        vec![(count, vec![lit_f64(1.0)])],
    );
    let repeated = select(
        crate::emit::filter(grouped, call(gt, vec![field(k), lit_f64(1.0)])),
        (0..k).collect(),
    );
    let names: Vec<String> = table
        .dims
        .iter()
        .map(|&d| table.columns()[d].clone())
        .collect();
    Check {
        plan: plan(repeated, names.clone(), &t.ext),
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
