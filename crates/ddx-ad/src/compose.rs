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
//!   earlier step a read of that step's plan ([`output_plan`]). Each step is
//!   written once, among the plan's relations, and read through a
//!   `ReferenceRel` wherever a step reads it, so the plan is the program's
//!   size however often its steps read one another; reverse mode lowers
//!   each once. A step the output does not reach is not read, and the
//!   program the plan gives is pruned, as every program is.
//!
//! Either way the new program reads the old one's input tables (its tangent,
//! its cotangent) as constants, and the caller registers them as before: they
//! come first in the new program's [`inputs`](crate::BackwardProgram::inputs),
//! and the old program's checks first in its checks.

use std::collections::HashMap;

use substrait::proto::expression::RexType;
use substrait::proto::function_argument::ArgType;
use substrait::proto::plan_rel::RelType as PlanRelType;
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::EmitKind;
use substrait::proto::{Expression, FunctionArgument, Plan, PlanRel, ReadRel, ReferenceRel, Rel};

use crate::emit::{filter, is_identity, plan as plan_of, read_step, select, unbound_reads};
use crate::error::{AdError, Result};
use crate::forward::{
    inline_references, read_outputs, referenced, rel_inputs_mut, root_of, shared_relations,
    step_columns, width,
};
use crate::functions::{Extensions, Functions};
use crate::program::{BackwardProgram, Check, ForwardProgram, Step, Unbound};
use crate::tables::{InputTable, Of, OutputTable};

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
            .map(|c| &c.plan.0)
            .chain(self.steps.iter().map(|s| &s.plan.0))
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

/// A relation of a plan, as a step: its name, relation and columns.
pub(crate) struct DagStep {
    pub name: String,
    pub rel: Rel,
    pub columns: Vec<String>,
}

/// `plan`, which shares relations through `ReferenceRel`s, as steps: each
/// relation it shares, after those it reads ([`shared_relations`]), as the
/// step `{namespace}shared_{ordinal}` with columns `c0, c1, …`, then its
/// root, under the name `root`. Each reads the shared relations before it
/// by name where the plan reads them through a reference, as a program's
/// steps read each other, so each is computed once and an engine, which
/// reads no `ReferenceRel`, can run every one. `None` if the plan shares
/// nothing.
pub(crate) fn dag_steps(
    plan: &Plan,
    namespace: &str,
    root_name: &str,
) -> Result<Option<Vec<DagStep>>> {
    let (root, names) = root_of(plan)?;
    let order = shared_relations(root, &plan.relations)?;
    if order.is_empty() {
        return Ok(None);
    }
    let mut read: HashMap<i32, (String, Vec<String>)> = HashMap::new();
    let mut steps = Vec::with_capacity(order.len() + 1);
    for ordinal in order {
        let mut rel = referenced(&plan.relations, ordinal)?.clone();
        read_references(&mut rel, &read)?;
        let columns = step_columns(width(&rel)?);
        let name = format!("{namespace}shared_{ordinal}");
        read.insert(ordinal, (name.clone(), columns.clone()));
        steps.push(DagStep { name, rel, columns });
    }
    let mut rel = root.clone();
    read_references(&mut rel, &read)?;
    steps.push(DagStep {
        name: root_name.to_string(),
        rel,
        columns: names,
    });
    Ok(Some(steps))
}

/// `rel` with each `ReferenceRel` replaced by a read of the step `read`
/// names for its ordinal. A loop, not recursion.
fn read_references(rel: &mut Rel, read: &HashMap<i32, (String, Vec<String>)>) -> Result<()> {
    let mut stack = vec![rel];
    while let Some(r) = stack.pop() {
        if let Some(RelType::Reference(x)) = &r.rel_type {
            let (name, columns) = read.get(&x.subtree_ordinal).ok_or_else(|| {
                AdError::Internal(format!(
                    "relation {} read before it was made a step",
                    x.subtree_ordinal
                ))
            })?;
            *r = read_step(name, columns.clone());
        } else if let Some(kind) = r.rel_type.as_mut() {
            stack.extend(rel_inputs_mut(kind));
        }
    }
    Ok(())
}

/// The plan of `parts`' output table (see [`Parts::output`]): the step that
/// writes it, with every read of an earlier step a read of that step's
/// plan, pruned to the output's columns. Each earlier step is written once,
/// in the plan's relations, and read through a `ReferenceRel` wherever a
/// step reads it, so a program whose steps read one another more than once
/// gives a plan of its own size, not of its steps written out at every read.
/// A step the output does not read, directly or through other steps, is
/// left out.
pub(crate) fn output_plan(parts: &Parts) -> Result<Plan> {
    let output = parts.output()?;
    let functions = Functions::union(parts.plans())?;
    // Each step so far: the ordinal of its relation in the plan, and its
    // columns.
    let mut steps: HashMap<&str, (i32, Vec<String>)> = HashMap::new();
    let mut shared: Vec<PlanRel> = Vec::new();
    // Each step's relation, from its root: what it computes.
    let ext = Extensions::new(&functions);
    let mut rels = Vec::with_capacity(parts.steps.len());
    for step in &parts.steps {
        let (root, names) = root_of(&step.plan)?;
        rels.push((inline_references(root, &step.plan.relations)?, names));
    }
    // Only the steps the output reads, directly or through each other, are
    // written: another gradient's steps are not, nor anything in them.
    let mut needed: Vec<String> = vec![output.step.clone()];
    for (step, (rel, names)) in parts.steps.iter().zip(&rels).rev() {
        if needed.contains(&step.name) {
            needed.extend(unbound_reads(&plan_of(rel.clone(), names.clone(), &ext)));
        }
    }
    for (step, (rel, names)) in parts.steps.iter().zip(rels) {
        if !needed.contains(&step.name) {
            continue;
        }
        let mut rel = rel;
        substitute(&mut rel, &steps)?;
        if step.name != output.step {
            // After the root, which a plan holds first.
            steps.insert(&step.name, (shared.len() as i32 + 1, names));
            shared.push(PlanRel {
                rel_type: Some(PlanRelType::Rel(rel)),
            });
            continue;
        }
        let mut plan = plan_of(rel, names, &ext);
        plan.relations.extend(shared);
        if let Some(read) = unbound_reads(&plan)
            .into_iter()
            .find(|r| steps.contains_key(r.as_str()))
        {
            return Err(AdError::NotImplemented(format!(
                "a program whose step `{}` reads step `{read}` inside an expression",
                step.name
            )));
        }
        crate::opt::plan(&mut plan);
        return Ok(plan);
    }
    Err(AdError::Internal(format!(
        "no step writes the output `{}`",
        output.step
    )))
}

/// `rel` with every read of a step in `steps` replaced by a reference to
/// the step's relation, its columns picked by name. A loop, not recursion:
/// a program's steps can chain hundreds of relations deep.
fn substitute(rel: &mut Rel, steps: &HashMap<&str, (i32, Vec<String>)>) -> Result<()> {
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

/// A reference to the step `read` reads, its columns picked, if it reads
/// one of `steps`.
fn step_of(read: &ReadRel, steps: &HashMap<&str, (i32, Vec<String>)>) -> Result<Option<Rel>> {
    let (Some(ReadType::NamedTable(t)), Some(schema)) = (&read.read_type, &read.base_schema) else {
        return Ok(None);
    };
    let ([name], None) = (t.names.as_slice(), &schema.r#struct) else {
        return Ok(None);
    };
    let Some((ordinal, columns)) = steps.get(name.as_str()) else {
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
    let reference = Rel {
        rel_type: Some(RelType::Reference(ReferenceRel {
            subtree_ordinal: *ordinal,
        })),
    };
    Ok(Some(select(reference, picked)))
}

/// A query over the tables of earlier programs, as one program: theirs
/// steps, renamed into one namespace, then the query, as the step that
/// writes its value. [`grad`](crate::grad), [`vjp`](crate::vjp) and
/// [`jvp`](crate::jvp) of it differentiate through the programs as of any
/// program, so a call in SQL can take a query over another call: `jvp` of
/// a query over `grad(loss, w.val)` is `H·v`.
///
/// The query is the engine's plan: it reads the programs' tables as the
/// engine registered them, with their types and any filter or projection
/// the engine pushed into the read, and declares functions under its own
/// anchors. Each such read becomes a read of the step, as a program's steps
/// read each other, and every plan is declared from one set of functions.
pub(crate) struct After {
    inputs: Vec<InputTable>,
    checks: Vec<Check>,
    steps: Vec<Step>,
    value: OutputTable,
}

impl After {
    pub(crate) fn new(programs: &[&dyn Differentiable], query: &Plan) -> Result<After> {
        let namespace = crate::forward::new_namespace();
        let mut ext = Extensions::new(&Functions::from_plan(&Plan::default())?);
        let mut renamed: HashMap<String, String> = HashMap::new();
        let mut inputs: Vec<InputTable> = Vec::new();
        let mut checks: Vec<Check> = Vec::new();
        let mut steps: Vec<Step> = Vec::new();
        for (k, program) in programs.iter().enumerate() {
            let Subject::Program(parts) = program.subject() else {
                return Err(AdError::Internal(
                    "a query over a query, not a program".into(),
                ));
            };
            let old = parts.namespace()?;
            for step in &parts.steps {
                let rest = step.name.strip_prefix(old).ok_or_else(|| {
                    AdError::Internal(format!(
                        "step `{}` is outside its program's namespace",
                        step.name
                    ))
                })?;
                renamed.insert(step.name.clone(), format!("{namespace}{k}_{rest}"));
            }
            for input in parts.inputs {
                if !inputs.iter().any(|i| i.name == input.name) {
                    inputs.push(input.clone());
                }
            }
            for check in parts.checks {
                let mut check = check.clone();
                redeclare(&mut check.plan.0, &HashMap::new(), &mut ext)?;
                if !checks.iter().any(|c| c.message == check.message) {
                    checks.push(check);
                }
            }
            for step in &parts.steps {
                let mut plan = step.plan.clone();
                redeclare(&mut plan.0, &renamed, &mut ext)?;
                steps.push(Step {
                    name: renamed[&step.name].clone(),
                    plan,
                });
            }
        }
        let (root, names) = root_of(query)?;
        let from = Functions::from_plan(query)?;
        let value = OutputTable {
            step: format!("{namespace}value"),
            of: Of::Output,
            columns: names.clone(),
            tangents: Vec::new(),
        };
        // A relation the query shares is a step of its own, read by name
        // wherever the query reads it; the query's root writes the value.
        let query_steps = match dag_steps(query, &namespace, &value.step)? {
            Some(dag) => dag,
            None => vec![DagStep {
                name: value.step.clone(),
                rel: root.clone(),
                columns: names,
            }],
        };
        for step in query_steps {
            let mut rel = step.rel;
            redeclare_rel(&mut rel, &from, &renamed, &mut ext)?;
            let mut plan = plan_of(Rel::default(), step.columns, &ext);
            set_root(&mut plan, rel);
            steps.push(Step {
                name: step.name,
                plan: Unbound(plan),
            });
        }
        // Every plan declares the whole set, which only grew, so an anchor
        // names one function in all of them.
        for plan in steps
            .iter_mut()
            .map(|s| &mut s.plan.0)
            .chain(checks.iter_mut().map(|c| &mut c.plan.0))
        {
            plan.extensions = ext.declarations();
            plan.extension_urns = ext.urns();
        }
        Ok(After {
            inputs,
            checks,
            steps,
            value,
        })
    }
}

impl sealed::Sealed for After {
    fn subject(&self) -> Subject<'_> {
        Subject::Program(Parts {
            inputs: &self.inputs,
            checks: &self.checks,
            steps: self.steps.iter().collect(),
            value: &self.value,
            gradients: &[],
        })
    }
}
impl Differentiable for After {}

fn set_root(plan: &mut Plan, rel: Rel) {
    for r in plan.relations.iter_mut() {
        if let Some(PlanRelType::Root(root)) = r.rel_type.as_mut() {
            root.input = Some(rel);
            return;
        }
    }
}

/// A read of a table `renamed` names, made a read of that step under its
/// new name. The engine wrote the read typed, maybe filtered and projected:
/// it becomes an unbound read of every column the step has, then that
/// filter, then that projection, then its emit.
fn step_read(read: &ReadRel, renamed: &HashMap<String, String>) -> Result<Option<Rel>> {
    let (Some(ReadType::NamedTable(t)), Some(schema)) = (&read.read_type, &read.base_schema) else {
        return Ok(None);
    };
    let Some(new) = t.names.last().and_then(|n| renamed.get(n)) else {
        return Ok(None);
    };
    let mut rel = read_step(new, schema.names.clone());
    if let Some(f) = &read.filter {
        rel = filter(rel, (**f).clone());
    }
    let outputs = read_outputs(read)?;
    if !is_identity(&outputs, schema.names.len()) {
        rel = select(rel, outputs.clone());
    }
    if let Some(EmitKind::Emit(e)) = read.common.as_ref().and_then(|c| c.emit_kind.as_ref()) {
        let emit = e
            .output_mapping
            .iter()
            .map(|&i| {
                usize::try_from(i)
                    .ok()
                    .filter(|&i| i < outputs.len())
                    .ok_or_else(|| AdError::InvalidPlan(format!("a read emits column {i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        rel = select(rel, emit);
    }
    Ok(Some(rel))
}

/// `plan`'s function anchors, re-declared by name in `ext`, and its reads of
/// the tables `renamed` names made reads of those steps (see
/// [`redeclare_rel`]).
fn redeclare(
    plan: &mut Plan,
    renamed: &HashMap<String, String>,
    ext: &mut Extensions,
) -> Result<()> {
    let from = Functions::from_plan(plan)?;
    for r in plan.relations.iter_mut() {
        match r.rel_type.as_mut() {
            Some(PlanRelType::Root(root)) => {
                if let Some(rel) = root.input.as_mut() {
                    redeclare_rel(rel, &from, renamed, ext)?;
                }
            }
            Some(PlanRelType::Rel(rel)) => redeclare_rel(rel, &from, renamed, ext)?,
            None => {}
        }
    }
    Ok(())
}

/// Every function anchor in `rel` and beneath it, `from`'s, re-declared by
/// name in `ext`, and every read of a table `renamed` names made a read of
/// that step ([`step_read`]), subqueries included: a read left behind would
/// read the table as data, and the derivative through it would be lost. A
/// relation this crate does not know is refused, since a function it calls
/// would keep an anchor that names another.
pub(crate) fn redeclare_rel(
    rel: &mut Rel,
    from: &Functions,
    renamed: &HashMap<String, String>,
    ext: &mut Extensions,
) -> Result<()> {
    let mut stack = vec![rel];
    while let Some(r) = stack.pop() {
        if let Some(RelType::Read(read)) = &r.rel_type {
            if let Some(step) = step_read(read, renamed)? {
                // Walk what replaced it, for the filter's functions: its
                // read names a step under a new name, which `renamed` lacks.
                *r = step;
                stack.push(r);
                continue;
            }
        }
        let Some(kind) = r.rel_type.as_mut() else {
            continue;
        };
        match kind {
            RelType::Read(_)
            | RelType::Filter(_)
            | RelType::Fetch(_)
            | RelType::Sort(_)
            | RelType::Project(_)
            | RelType::Join(_)
            | RelType::Cross(_)
            | RelType::Set(_) => {}
            RelType::Aggregate(a) => {
                for m in a.measures.iter_mut() {
                    if let Some(f) = m.measure.as_mut() {
                        f.function_reference = ext.anchor_named(from.name(f.function_reference)?);
                        for s in f.sorts.iter_mut() {
                            if let Some(e) = s.expr.as_mut() {
                                redeclare_expr(e, from, renamed, ext)?;
                            }
                        }
                    }
                }
            }
            other => {
                return Err(AdError::NotImplemented(format!(
                    "a {} in a query over another call",
                    crate::forward::rel_name(other)
                )))
            }
        }
        for e in crate::forward::rel_expressions_mut(kind) {
            redeclare_expr(e, from, renamed, ext)?;
        }
        stack.extend(rel_inputs_mut(kind));
    }
    Ok(())
}

fn redeclare_args(
    args: &mut [FunctionArgument],
    from: &Functions,
    renamed: &HashMap<String, String>,
    ext: &mut Extensions,
) -> Result<()> {
    for a in args.iter_mut() {
        if let Some(ArgType::Value(e)) = a.arg_type.as_mut() {
            redeclare_expr(e, from, renamed, ext)?;
        }
    }
    Ok(())
}

fn redeclare_expr(
    e: &mut Expression,
    from: &Functions,
    renamed: &HashMap<String, String>,
    ext: &mut Extensions,
) -> Result<()> {
    use substrait::proto::expression::nested::NestedType;
    use substrait::proto::expression::subquery::SubqueryType;
    let Some(rex) = e.rex_type.as_mut() else {
        return Ok(());
    };
    let mut each = |es: &mut dyn Iterator<Item = &mut Expression>| -> Result<()> {
        for x in es {
            redeclare_expr(x, from, renamed, ext)?;
        }
        Ok(())
    };
    match rex {
        RexType::Literal(_) | RexType::Selection(_) | RexType::DynamicParameter(_) => Ok(()),
        RexType::ScalarFunction(f) => {
            f.function_reference = ext.anchor_named(from.name(f.function_reference)?);
            #[allow(deprecated)]
            for x in f.args.iter_mut() {
                redeclare_expr(x, from, renamed, ext)?;
            }
            redeclare_args(&mut f.arguments, from, renamed, ext)
        }
        RexType::WindowFunction(w) => {
            w.function_reference = ext.anchor_named(from.name(w.function_reference)?);
            #[allow(deprecated)]
            for x in w.args.iter_mut() {
                redeclare_expr(x, from, renamed, ext)?;
            }
            redeclare_args(&mut w.arguments, from, renamed, ext)?;
            for x in w.partitions.iter_mut() {
                redeclare_expr(x, from, renamed, ext)?;
            }
            for s in w.sorts.iter_mut() {
                if let Some(x) = s.expr.as_mut() {
                    redeclare_expr(x, from, renamed, ext)?;
                }
            }
            Ok(())
        }
        RexType::IfThen(it) => {
            for c in it.ifs.iter_mut() {
                each(&mut c.r#if.iter_mut().chain(c.then.iter_mut()))?;
            }
            each(&mut it.r#else.iter_mut().map(|b| &mut **b))
        }
        RexType::SwitchExpression(sw) => {
            each(&mut sw.r#match.iter_mut().map(|b| &mut **b))?;
            each(&mut sw.ifs.iter_mut().filter_map(|c| c.then.as_mut()))?;
            each(&mut sw.r#else.iter_mut().map(|b| &mut **b))
        }
        RexType::SingularOrList(l) => {
            each(&mut l.value.iter_mut().map(|b| &mut **b))?;
            each(&mut l.options.iter_mut())
        }
        RexType::MultiOrList(l) => {
            each(&mut l.value.iter_mut())?;
            each(&mut l.options.iter_mut().flat_map(|r| r.fields.iter_mut()))
        }
        RexType::Cast(c) => each(&mut c.input.iter_mut().map(|b| &mut **b)),
        RexType::Nested(n) => match n.nested_type.as_mut() {
            Some(NestedType::Struct(s)) => each(&mut s.fields.iter_mut()),
            Some(NestedType::List(l)) => each(&mut l.values.iter_mut()),
            Some(NestedType::Map(m)) => each(
                &mut m
                    .key_values
                    .iter_mut()
                    .flat_map(|kv| kv.key.iter_mut().chain(kv.value.iter_mut())),
            ),
            None => Ok(()),
        },
        RexType::Subquery(sq) => match sq.subquery_type.as_mut() {
            Some(SubqueryType::Scalar(s)) => match s.input.as_deref_mut() {
                Some(r) => redeclare_rel(r, from, renamed, ext),
                None => Ok(()),
            },
            Some(SubqueryType::InPredicate(p)) => {
                each(&mut p.needles.iter_mut())?;
                match p.haystack.as_deref_mut() {
                    Some(r) => redeclare_rel(r, from, renamed, ext),
                    None => Ok(()),
                }
            }
            Some(SubqueryType::SetPredicate(p)) => match p.tuples.as_deref_mut() {
                Some(r) => redeclare_rel(r, from, renamed, ext),
                None => Ok(()),
            },
            Some(SubqueryType::SetComparison(c)) => {
                each(&mut c.left.iter_mut().map(|b| &mut **b))?;
                match c.right.as_deref_mut() {
                    Some(r) => redeclare_rel(r, from, renamed, ext),
                    None => Ok(()),
                }
            }
            None => Ok(()),
        },
        #[allow(unreachable_patterns)]
        _ => Err(AdError::NotImplemented(
            "this kind of expression in a query over another call".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::project;
    use crate::expr::field;

    fn step(name: &str, rel: Rel) -> Step {
        let ext = Extensions::new(&Functions::from_plan(&Plan::default()).unwrap());
        Step {
            name: name.into(),
            plan: Unbound(plan_of(rel, vec!["c0".into()], &ext)),
        }
    }

    #[test]
    fn a_step_the_output_does_not_read_is_left_out_of_its_plan() {
        // The output reads `p_a`; `p_other` (another gradient's step, say)
        // reads `p_a` too, but nothing reads it, so it is not written.
        let a = step("p_a", read_step("t", vec!["c0".into()]));
        let other = step(
            "p_other",
            select(
                project(read_step("p_a", vec!["c0".into()]), vec![field(0)]),
                vec![1],
            ),
        );
        let out = step(
            "p_value",
            select(
                project(read_step("p_a", vec!["c0".into()]), vec![field(0)]),
                vec![1],
            ),
        );
        let value = OutputTable {
            step: "p_value".into(),
            of: Of::Output,
            columns: vec!["c0".into()],
            tangents: Vec::new(),
        };
        let parts = Parts {
            inputs: &[],
            checks: &[],
            steps: vec![&a, &other, &out],
            value: &value,
            gradients: &[],
        };
        let plan = output_plan(&parts).unwrap();
        // The root, and `p_a` once.
        assert_eq!(plan.relations.len(), 2, "{plan:?}");
    }

    #[test]
    fn a_step_only_a_relation_the_root_never_reaches_reads_is_left_out() {
        // The output's plan holds, beside its root, a relation nothing
        // reaches that reads `p_other`: what the output computes reads
        // only `p_a`.
        let a = step("p_a", read_step("t", vec!["c0".into()]));
        let other = step("p_other", read_step("t", vec!["c0".into()]));
        let mut out = step(
            "p_value",
            select(
                project(read_step("p_a", vec!["c0".into()]), vec![field(0)]),
                vec![1],
            ),
        );
        out.plan.0.relations.push(PlanRel {
            rel_type: Some(PlanRelType::Rel(read_step("p_other", vec!["c0".into()]))),
        });
        let value = OutputTable {
            step: "p_value".into(),
            of: Of::Output,
            columns: vec!["c0".into()],
            tangents: Vec::new(),
        };
        let parts = Parts {
            inputs: &[],
            checks: &[],
            steps: vec![&a, &other, &out],
            value: &value,
            gradients: &[],
        };
        let plan = output_plan(&parts).unwrap();
        assert_eq!(plan.relations.len(), 2, "{plan:?}");
    }
}
