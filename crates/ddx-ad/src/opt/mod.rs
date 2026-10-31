// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Optimizing the plans and programs ddx writes, without changing what they
//! compute.
//!
//! - [`prune`]: drop the columns nothing reads, within a plan and across a
//!   program's steps.
//! - [`fuse`]: fold a projection that only picks columns into its
//!   neighbour.
//! - [`inline`]: compute a step that one later step reads, once, inside
//!   that step.
//! - [`split`]: split a step too deep for a consumer into steps.
//! - [`flatten_unions`]: splice a `UNION ALL` that is an input of a union,
//!   beneath only projections, into it.
//!
//! Every plan ddx writes goes through [`plan`]; every program's steps,
//! through [`steps`] once they are final.

pub(crate) mod fuse;
pub(crate) mod inline;
pub(crate) mod prune;
pub(crate) mod split;

use std::collections::{BTreeSet, HashMap};

use substrait::proto::expression::RexType;
use substrait::proto::rel::RelType;
use substrait::proto::set_rel::SetOp;
use substrait::proto::{Plan, ProjectRel, Rel};

use crate::optics::{Compose, Deep, PlanRels, RelKind, Setter};
use crate::program::{BackwardProgram, Step};

/// Optimize one plan: prune it to its output, then fuse its projections,
/// then flatten its unions.
pub(crate) fn plan(plan: &mut Plan) {
    prune::prune_plan(plan);
    fuse::fuse_plan(plan);
    flatten_unions(plan);
}

/// Optimize a program's final steps: split each too deep for a consumer,
/// then flatten their unions.
pub(crate) fn steps(steps: &mut Vec<Step>) {
    split::split_steps(steps);
    for step in steps.iter_mut() {
        flatten_unions(&mut step.plan.0);
    }
}

/// `plan` with each `UNION ALL` that is an input of a union (`UNION ALL` or
/// `UNION`), beneath only projections computed row by row, spliced into it:
/// each of its inputs, with its emit and those projections over it, is an
/// input of the union above. The same rows, since such a projection of a
/// `UNION ALL` is the `UNION ALL` of the projection of each input.
///
/// Pruning and inlining can leave one union directly beneath another, and
/// the plans engines give have them beneath a projection. DataFusion's
/// optimizer splices them itself (`optimize_unions`), and fails when the
/// inner union's inputs name their columns otherwise than its first does:
/// it looks the union's names up in each input, both for a union directly
/// beneath (ddx issue #236) and for a projection it pushes onto each input
/// (#249).
pub(crate) fn flatten_unions(plan: &mut Plan) {
    PlanRels.then(Deep).then(RelKind).modify(plan, |kind| {
        let RelType::Set(set) = kind else { return };
        let union = [SetOp::UnionAll as i32, SetOp::UnionDistinct as i32];
        if !union.contains(&set.op) || !set.inputs.iter().any(spliced) {
            return;
        }
        let mut pending = std::mem::take(&mut set.inputs);
        pending.reverse();
        while let Some(input) = pending.pop() {
            if spliced(&input) {
                pending.extend(splice(input).into_iter().rev());
            } else {
                set.inputs.push(input);
            }
        }
    });
}

/// Whether `rel` is a `UNION ALL` beneath only projections computed row by
/// row, which a union above splices.
fn spliced(rel: &Rel) -> bool {
    let window =
        |e: &substrait::proto::Expression| matches!(e.rex_type, Some(RexType::WindowFunction(_)));
    match &rel.rel_type {
        Some(RelType::Set(set)) => set.op == SetOp::UnionAll as i32,
        Some(RelType::Project(p)) => {
            !p.expressions
                .iter()
                .any(|e| crate::expr::contains(e, &window))
                && p.input.as_deref().is_some_and(spliced)
        }
        _ => false,
    }
}

/// The inputs of the `UNION ALL` that `rel` ([`spliced`]) is, each with the
/// union's emit and the projections above it over it.
fn splice(rel: Rel) -> Vec<Rel> {
    let over = |input: Rel, p: &ProjectRel| Rel {
        rel_type: Some(RelType::Project(Box::new(ProjectRel {
            common: p.common.clone(),
            input: Some(Box::new(input)),
            expressions: p.expressions.clone(),
            advanced_extension: p.advanced_extension.clone(),
        }))),
    };
    match rel.rel_type {
        Some(RelType::Set(set)) if set.common.as_ref().is_none_or(|c| c.emit_kind.is_none()) => {
            set.inputs
        }
        Some(RelType::Set(set)) => {
            let picks = ProjectRel {
                common: set.common,
                ..Default::default()
            };
            set.inputs.into_iter().map(|i| over(i, &picks)).collect()
        }
        Some(RelType::Project(mut p)) => {
            let input = p.input.take().expect("a spliced projection has an input");
            splice(*input).into_iter().map(|i| over(i, &p)).collect()
        }
        _ => unreachable!("only a spliced relation is spliced"),
    }
}

/// Optimize a backward program: prune each plan, then across steps, from the
/// last back, so each step keeps only the columns some later step reads (the
/// value and the gradients keep all theirs); fuse each; inline each step
/// that one later step reads once; then split the steps too deep for a
/// consumer.
pub(crate) fn backward_program(program: &mut BackwardProgram) {
    for c in program.checks.iter_mut() {
        plan(&mut c.plan.0);
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
            prune::prune_plan_to_names(&mut step.plan.0, &used)
        };
        if narrowed {
            fuse::fuse_plan(&mut step.plan.0);
        } else {
            plan(&mut step.plan.0);
        }
        note_reads(&step.plan, &mut read);
    }
    inline_program(program, &keep);
    steps(&mut program.forward_steps);
    steps(&mut program.backward_steps);
}

/// [`inline::inline_steps`] over a program's forward and backward steps,
/// in the order they run; each step that stays keeps its place.
fn inline_program(program: &mut BackwardProgram, keep: &BTreeSet<String>) {
    let forward: BTreeSet<String> = program
        .forward_steps
        .iter()
        .map(|s| s.name.clone())
        .collect();
    let mut all = std::mem::take(&mut program.forward_steps);
    all.append(&mut program.backward_steps);
    inline::inline_steps(&mut all, keep);
    for s in all {
        if forward.contains(&s.name) {
            program.forward_steps.push(s);
        } else {
            program.backward_steps.push(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{read_step, union_all};
    use crate::functions::{Extensions, Functions};

    #[test]
    fn a_union_all_beneath_projections_in_a_union_is_spliced_into_it() {
        use crate::emit::project_emit;
        use crate::expr::{field, window};
        use substrait::proto::rel_common::{Emit, EmitKind};
        use substrait::proto::RelCommon;
        let t = |name: &str| read_step(name, vec!["x".into()]);
        let functions = Functions::from_plan(&Plan::default()).unwrap();
        let ext = Extensions::new(&functions);
        let flattened = |rel: Rel| {
            let mut plan = crate::emit::plan(rel, vec!["x".into()], &ext);
            flatten_unions(&mut plan);
            plan
        };
        let mut picking = union_all(vec![t("d"), t("e")]);
        if let Some(RelType::Set(s)) = &mut picking.rel_type {
            s.common = Some(RelCommon {
                emit_kind: Some(EmitKind::Emit(Emit {
                    output_mapping: vec![0],
                })),
                ..Default::default()
            });
        }
        let doubled = |rel: Rel| project_emit(rel, vec![field(0)], Some(vec![1]));
        let nested = union_all(vec![
            t("a"),
            union_all(vec![t("b"), union_all(vec![t("c")])]),
            picking,
            doubled(union_all(vec![t("f"), t("g")])),
        ]);
        let picked = |name: &str| project_emit(t(name), vec![], Some(vec![0]));
        let flat = union_all(vec![
            t("a"),
            t("b"),
            t("c"),
            picked("d"),
            picked("e"),
            doubled(t("f")),
            doubled(t("g")),
        ]);
        assert_eq!(
            flattened(nested),
            crate::emit::plan(flat, vec!["x".into()], &ext)
        );
        // A window over a union isn't the union of the window over each.
        let windowed = project_emit(
            union_all(vec![t("f"), t("g")]),
            vec![window(0, vec![field(0)], vec![])],
            Some(vec![1]),
        );
        let kept = union_all(vec![t("a"), windowed]);
        assert_eq!(
            flattened(kept.clone()),
            crate::emit::plan(kept, vec!["x".into()], &ext)
        );
        // A UNION ALL in a UNION is spliced too; a UNION is not.
        let distinct = |inputs: Vec<Rel>| {
            let mut u = union_all(inputs);
            if let Some(RelType::Set(s)) = &mut u.rel_type {
                s.op = SetOp::UnionDistinct as i32;
            }
            u
        };
        let nested = distinct(vec![
            t("a"),
            union_all(vec![t("b"), t("c")]),
            distinct(vec![t("d"), t("e")]),
        ]);
        let flat = distinct(vec![t("a"), t("b"), t("c"), distinct(vec![t("d"), t("e")])]);
        assert_eq!(
            flattened(nested),
            crate::emit::plan(flat, vec!["x".into()], &ext)
        );
    }
}
