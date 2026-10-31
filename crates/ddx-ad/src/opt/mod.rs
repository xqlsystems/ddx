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
//! - [`flatten_unions`]: splice a `UNION ALL` that is an input of a
//!   `UNION ALL` into it.
//!
//! Every plan ddx writes goes through [`plan`]; every program's steps,
//! through [`steps`] once they are final.

pub(crate) mod fuse;
pub(crate) mod inline;
pub(crate) mod prune;
pub(crate) mod split;

use std::collections::{BTreeSet, HashMap};

use substrait::proto::rel::RelType;
use substrait::proto::set_rel::SetOp;
use substrait::proto::{Plan, Rel, SetRel};

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

/// `plan` with each `UNION ALL` that is an input of a `UNION ALL`, and picks
/// no columns of its own (has no emit), spliced into it: the same rows, in
/// the columns of the outer union's first input, which name a union's
/// columns.
///
/// Pruning and inlining can leave one union directly beneath another,
/// where the plan an engine gave had a projection between them. DataFusion's
/// optimizer splices them itself (`optimize_unions`), and fails when the
/// inner union names its columns otherwise than the outer one: it looks the
/// outer names up in the inner union (ddx issue #236).
pub(crate) fn flatten_unions(plan: &mut Plan) {
    fn union_all(set: &SetRel) -> bool {
        set.op == SetOp::UnionAll as i32
            && set.common.as_ref().is_none_or(|c| c.emit_kind.is_none())
    }
    let spliced = |rel: &Rel| matches!(&rel.rel_type, Some(RelType::Set(s)) if union_all(s));
    PlanRels.then(Deep).then(RelKind).modify(plan, |kind| {
        let RelType::Set(set) = kind else { return };
        if set.op != SetOp::UnionAll as i32 || !set.inputs.iter().any(spliced) {
            return;
        }
        let mut pending = std::mem::take(&mut set.inputs);
        pending.reverse();
        while let Some(input) = pending.pop() {
            match input.rel_type {
                Some(RelType::Set(inner)) if union_all(&inner) => {
                    pending.extend(inner.inputs.into_iter().rev());
                }
                rel_type => set.inputs.push(Rel { rel_type }),
            }
        }
    });
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
    fn a_union_all_beneath_a_union_all_is_spliced_into_it_unless_it_picks_columns() {
        use substrait::proto::rel_common::{Emit, EmitKind};
        use substrait::proto::RelCommon;
        let t = |name: &str| read_step(name, vec!["x".into()]);
        let mut picking = union_all(vec![t("d"), t("e")]);
        if let Some(RelType::Set(s)) = &mut picking.rel_type {
            s.common = Some(RelCommon {
                emit_kind: Some(EmitKind::Emit(Emit {
                    output_mapping: vec![0],
                })),
                ..Default::default()
            });
        }
        let nested = union_all(vec![
            t("a"),
            union_all(vec![t("b"), union_all(vec![t("c")])]),
            picking.clone(),
        ]);
        let functions = Functions::from_plan(&Plan::default()).unwrap();
        let ext = Extensions::new(&functions);
        let mut plan = crate::emit::plan(nested, vec!["x".into()], &ext);
        flatten_unions(&mut plan);
        let flat = union_all(vec![t("a"), t("b"), t("c"), picking]);
        assert_eq!(plan, crate::emit::plan(flat, vec!["x".into()], &ext));
    }
}
