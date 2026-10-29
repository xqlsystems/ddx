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
//!
//! Every plan ddx writes goes through [`plan`]; every program's steps,
//! through [`steps`] once they are final.

pub(crate) mod fuse;
pub(crate) mod inline;
pub(crate) mod prune;
pub(crate) mod split;

use std::collections::{BTreeSet, HashMap};

use substrait::proto::Plan;

use crate::program::{BackwardProgram, Step};

/// Optimize one plan: prune it to its output, then fuse its projections.
pub(crate) fn plan(plan: &mut Plan) {
    prune::prune_plan(plan);
    fuse::fuse_plan(plan);
}

/// Optimize a program's final steps: split each too deep for a consumer.
pub(crate) fn steps(steps: &mut Vec<Step>) {
    split::split_steps(steps);
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
