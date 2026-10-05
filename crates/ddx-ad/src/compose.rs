// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! What ddx differentiates, and how a program is differentiated again.
//!
//! [`grad`](crate::grad), [`vjp`](crate::vjp) and [`jvp`](crate::jvp) each
//! take a query's [`Plan`] or a program, of either kind, so they compose as
//! JAX's do: `jvp` of a `grad` program is forward over reverse (`H·v`),
//! `vjp` of a `jvp` program is reverse over forward, `jvp` of a `jvp`
//! program is forward over forward.
//!
//! A program's **output** is its gradients if it has any, and otherwise its
//! value: what the function it computes returns.
//!
//! - **`jvp` of a program** rewrites its steps in order, each reading the
//!   rewritten steps before it ([`crate::jvp`]), so every intermediate table
//!   carries its tangent, and every output.
//! - **`grad` and `vjp` of a program** differentiate its one output table.
//!   Reverse mode needs the whole computation in one plan, to save what the
//!   backward pass reads and to send each cotangent back along the way it
//!   came, so the output step's plan is written out with every read of an
//!   earlier step replaced by that step's plan ([`output_plan`]). That plan
//!   is pruned before it is differentiated, which drops whatever of the
//!   program its output does not need (another gradient's steps, a column no
//!   later step reads), and the program it gives is pruned again, as every
//!   program is. A step read twice is written out twice; reverse mode saves
//!   each distinct aggregate once whichever way it is written.
//!
//! Either way the new program reads the old one's input tables (its tangent,
//! its cotangent) as constants, and the caller registers them as before: they
//! come first in the new program's [`inputs`](crate::BackwardProgram::inputs),
//! and the old program's checks first in its checks.

use std::collections::HashMap;

use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::{Plan, ReadRel, Rel};

use crate::emit::{plan as plan_of, select, unbound_reads};
use crate::error::{AdError, Result};
use crate::forward::{inline_references, rel_inputs_mut, root_of};
use crate::functions::{Extensions, Functions};
use crate::program::{BackwardProgram, Check, ForwardProgram, Step};
use crate::prune::prune_plan;
use crate::tables::{InputTable, OutputTable};

pub(crate) mod sealed {
    /// What a [`Differentiable`](super::Differentiable) is, to this crate.
    pub trait Sealed {
        fn subject(&self) -> super::Subject<'_>;
    }
}

/// What [`grad`](crate::grad), [`vjp`](crate::vjp) and [`jvp`](crate::jvp)
/// differentiate: a query's [`Plan`], or a program, a [`BackwardProgram`] or
/// a [`ForwardProgram`], so they compose as JAX's do. `jvp` of a `grad`
/// program is forward over reverse (`H·v`), `vjp` of a `jvp` program is
/// reverse over forward, `jvp` of a `jvp` program forward over forward.
///
/// A program's output is its gradients if it has any, and otherwise its
/// value. `jvp` of a program rewrites its steps in order, so every one
/// carries its tangent. `grad` and `vjp` of a program differentiate its one
/// output table: its step's plan, with each read of an earlier step replaced
/// by that step's plan, pruned to what the output needs. Reverse over
/// reverse (`vjp` of a `grad` program) is refused for now. The new program
/// reads the old one's input tables as constants: they come first in its
/// inputs, and the old program's checks first in its checks.
///
/// Sealed: these are the things ddx differentiates.
pub trait Differentiable: sealed::Sealed {}

/// A query's plan, or a program's parts.
pub enum Subject<'a> {
    Query(&'a Plan),
    Program(Parts<'a>),
}

/// A program's parts, whichever its kind.
pub struct Parts<'a> {
    pub inputs: &'a [InputTable],
    pub checks: &'a [Check],
    pub steps: Vec<&'a Step>,
    pub value: &'a OutputTable,
    pub gradients: &'a [OutputTable],
}

impl sealed::Sealed for Plan {
    fn subject(&self) -> Subject<'_> {
        Subject::Query(self)
    }
}
impl Differentiable for Plan {}

impl sealed::Sealed for BackwardProgram {
    fn subject(&self) -> Subject<'_> {
        Subject::Program(Parts {
            inputs: &self.inputs,
            checks: &self.checks,
            steps: self.steps().collect(),
            value: &self.value,
            gradients: &self.gradients,
        })
    }
}
impl Differentiable for BackwardProgram {}

impl sealed::Sealed for ForwardProgram {
    fn subject(&self) -> Subject<'_> {
        Subject::Program(Parts {
            inputs: &self.inputs,
            checks: &self.checks,
            steps: self.steps.iter().collect(),
            value: &self.value,
            gradients: &self.gradients,
        })
    }
}
impl Differentiable for ForwardProgram {}

impl Parts<'_> {
    /// Every plan of the program: its checks' and its steps'.
    pub fn plans(&self) -> impl Iterator<Item = &Plan> {
        self.checks
            .iter()
            .map(|c| &c.plan)
            .chain(self.steps.iter().map(|s| &s.plan))
    }

    /// The namespace the program writes under: its value step's name up to
    /// its last `_` (`…value`, `…jvp`).
    pub fn namespace(&self) -> Result<&str> {
        let step = &self.value.step;
        step.rfind('_')
            .map(|i| &step[..=i])
            .ok_or_else(|| AdError::Internal(format!("a program's value step `{step}`")))
    }

    /// The program's one output table: its gradient, or its value if it has
    /// none.
    fn output(&self) -> Result<&OutputTable> {
        match self.gradients {
            [] => Ok(self.value),
            [g] => Ok(g),
            gs => Err(AdError::NotImplemented(format!(
                "grad and vjp of a program differentiate its one output table, and it has {} \
                 gradients; build it with respect to one table",
                gs.len()
            ))),
        }
    }

    /// `inputs` and `checks` after the program's own, as a program built
    /// from it needs: it reads the program's input tables, and the
    /// program's promises still hold it up. A check already there is kept
    /// once.
    pub fn carry(&self, inputs: &mut Vec<InputTable>, checks: &mut Vec<Check>) {
        inputs.splice(0..0, self.inputs.iter().cloned());
        let mine = std::mem::replace(checks, self.checks.to_vec());
        for c in mine {
            if !checks.iter().any(|k| k.message == c.message) {
                checks.push(c);
            }
        }
    }
}

/// The plan of `parts`' output table (see [`Parts::output`]): the step that
/// writes it, with every read of an earlier step replaced by that step's
/// plan, pruned to the output's columns.
pub(crate) fn output_plan(parts: &Parts) -> Result<Plan> {
    let output = parts.output()?;
    let functions = Functions::union(parts.plans())?;
    // Each step so far, written out: its relation and its columns.
    let mut steps: HashMap<&str, (Rel, Vec<String>)> = HashMap::new();
    for step in &parts.steps {
        let (root, names) = root_of(&step.plan)?;
        let mut rel = inline_references(root, &step.plan.relations)?;
        substitute(&mut rel, &steps)?;
        if step.name != output.step {
            steps.insert(&step.name, (rel, names));
            continue;
        }
        let mut plan = plan_of(rel, names, &Extensions::new(&functions));
        if let Some(read) = unbound_reads(&plan)
            .into_iter()
            .find(|r| steps.contains_key(r.as_str()))
        {
            return Err(AdError::NotImplemented(format!(
                "a program whose step `{}` reads step `{read}` inside an expression",
                step.name
            )));
        }
        prune_plan(&mut plan);
        return Ok(plan);
    }
    Err(AdError::Internal(format!(
        "no step writes the output `{}`",
        output.step
    )))
}

/// `rel` with every read of a step in `steps` replaced by the step's
/// relation, its columns picked by name. A loop, not recursion: a program's
/// steps can chain hundreds of relations deep.
fn substitute(rel: &mut Rel, steps: &HashMap<&str, (Rel, Vec<String>)>) -> Result<()> {
    let mut stack = vec![rel];
    while let Some(r) = stack.pop() {
        if let Some(RelType::Read(read)) = &r.rel_type {
            if let Some(step) = step_of(read, steps)? {
                *r = step;
            }
        } else if let Some(kind) = r.rel_type.as_mut() {
            stack.extend(rel_inputs_mut(kind));
        }
    }
    Ok(())
}

/// The step `read` reads, written out, if it reads one of `steps`.
fn step_of(read: &ReadRel, steps: &HashMap<&str, (Rel, Vec<String>)>) -> Result<Option<Rel>> {
    let (Some(ReadType::NamedTable(t)), Some(schema)) = (&read.read_type, &read.base_schema) else {
        return Ok(None);
    };
    let ([name], None) = (t.names.as_slice(), &schema.r#struct) else {
        return Ok(None);
    };
    let Some((rel, columns)) = steps.get(name.as_str()) else {
        return Ok(None);
    };
    if read.filter.is_some() || read.projection.is_some() || read.common.is_some() {
        return Err(AdError::Internal(format!(
            "a read of step `{name}` that filters or projects"
        )));
    }
    let picked = schema
        .names
        .iter()
        .map(|c| {
            columns
                .iter()
                .position(|n| n == c)
                .ok_or_else(|| AdError::Internal(format!("step `{name}` has no column `{c}`")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(select(rel.clone(), picked)))
}
