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

use substrait::proto::expression::RexType;
use substrait::proto::function_argument::ArgType;
use substrait::proto::plan_rel::RelType as PlanRelType;
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::EmitKind;
use substrait::proto::{Expression, FunctionArgument, Plan, ReadRel, Rel};

use crate::emit::{filter, is_identity, plan as plan_of, read_step, select, unbound_reads};
use crate::error::{AdError, Result};
use crate::forward::{inline_references, read_outputs, rel_inputs_mut, root_of};
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
        crate::opt::plan(&mut plan);
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
        let mut rel = inline_references(root, &query.relations)?;
        let mut plan = plan_of(Rel::default(), names.clone(), &ext);
        let from = Functions::from_plan(query)?;
        redeclare_rel(&mut rel, &from, &renamed, &mut ext)?;
        set_root(&mut plan, rel);
        let value = OutputTable {
            step: format!("{namespace}value"),
            of: Of::Output,
            columns: names,
            tangents: Vec::new(),
        };
        steps.push(Step {
            name: value.step.clone(),
            plan: Unbound(plan),
        });
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
fn redeclare_rel(
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
                        redeclare_args(&mut f.arguments, from, renamed, ext)?;
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
