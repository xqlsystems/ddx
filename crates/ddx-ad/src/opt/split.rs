// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Splitting a step too deep for a consumer into steps.
//!
//! A plan is a tree, and a consumer decodes and walks it from the root:
//! datafusion-python decodes Substrait with protobuf's default limit of 100
//! nested messages (two per relation, more for a deep expression), and a
//! consumer recurses once per relation. A step whose query is deep (a model
//! of many layers, or the `jvp` of one, which carries a tangent beside each
//! relation) passes those limits. So every subtree [`MAX_DEPTH`] relations
//! deep becomes a step of its own, run before the step it came from and
//! read by name in its place, until no step is deeper. A materialized part
//! holds what the step would have computed there, once, so the program
//! computes the same thing; it pays for writing the part, on deep steps
//! only.

use substrait::proto::{Plan, Rel};

use crate::emit::read_step;
use crate::forward::{rel_inputs, rel_inputs_mut, root_of, step_columns, width};
use crate::program::Step;

/// The most relations deep a step may be: 64 levels of protobuf nesting,
/// leaving 36 of datafusion-python's 100 for the plan's frame and the
/// deepest expression.
pub(crate) const MAX_DEPTH: usize = 32;

/// `steps` with every step deeper than [`MAX_DEPTH`] relations split, its
/// parts inserted before it as steps named `{step}_part_{n}`.
pub(crate) fn split_steps(steps: &mut Vec<Step>) {
    let mut out = Vec::with_capacity(steps.len());
    for mut step in steps.drain(..) {
        let mut parts = Vec::new();
        split_plan(&mut step.plan, &step.name, &mut parts);
        out.extend(parts);
        out.push(step);
    }
    *steps = out;
}

/// Split `plan`, the plan of step `name`, appending its parts, in the
/// order they must run, to `parts`.
fn split_plan(plan: &mut Plan, name: &str, parts: &mut Vec<Step>) {
    let Ok((root, _)) = root_of(plan) else {
        return;
    };
    if depth(root) < MAX_DEPTH {
        return; // most steps: nothing to cut, nothing to copy
    }
    let mut root = root.clone();
    let mut cut = Vec::new();
    height(&mut root, name, &mut cut);
    if cut.is_empty() {
        return;
    }
    for (part, rel, width) in cut {
        parts.push(Step {
            name: part,
            plan: Plan {
                relations: vec![substrait::proto::PlanRel {
                    rel_type: Some(substrait::proto::plan_rel::RelType::Root(
                        substrait::proto::RelRoot {
                            input: Some(rel),
                            names: step_columns(width),
                        },
                    )),
                }],
                ..plan.clone()
            },
        });
    }
    for r in plan.relations.iter_mut() {
        if let Some(substrait::proto::plan_rel::RelType::Root(r)) = r.rel_type.as_mut() {
            r.input = Some(root);
            break;
        }
    }
}

/// How many relations deep `rel` is, walked with a loop.
fn depth(rel: &Rel) -> usize {
    let mut deepest = 0;
    let mut stack = vec![(rel, 1)];
    while let Some((r, d)) = stack.pop() {
        deepest = deepest.max(d);
        if let Some(kind) = &r.rel_type {
            stack.extend(rel_inputs(kind).into_iter().map(|i| (i, d + 1)));
        }
    }
    deepest
}

/// `rel`'s height in relations, after replacing each subtree
/// [`MAX_DEPTH`] deep with a read of a part of step `name`, appended to
/// `cut` (bottom-up, so a part only reads parts before it). Recursion is
/// bounded: a subtree is cut before it is [`MAX_DEPTH`] deep, so no call
/// is deeper than the plan, whose depth is what this exists to bound.
fn height(rel: &mut Rel, name: &str, cut: &mut Vec<(String, Rel, usize)>) -> usize {
    let h = 1 + match rel.rel_type.as_mut() {
        Some(kind) => rel_inputs_mut(kind)
            .into_iter()
            .map(|input| height(input, name, cut))
            .max()
            .unwrap_or(0),
        None => 0,
    };
    if h < MAX_DEPTH {
        return h;
    }
    let Ok(w) = width(rel) else {
        return h; // a relation ddx cannot size stays, with all beneath it
    };
    let part = format!("{name}_part_{}", cut.len());
    let sub = std::mem::replace(rel, read_step(&part, step_columns(w)));
    cut.push((part, sub, w));
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{plan, project_emit, read_table};
    use crate::expr::{call, field, lit_f64};
    use crate::functions::{Extensions, Function, Functions};
    use substrait::proto::r#type::{Fp64, Kind, Nullability, Struct};
    use substrait::proto::{NamedStruct, Type};

    fn depth(plan: &Plan) -> usize {
        let mut d = 0;
        let mut cur = root_of(plan).ok().map(|r| r.0);
        while let Some(r) = cur {
            d += 1;
            cur = match &r.rel_type {
                Some(substrait::proto::rel::RelType::Project(p)) => p.input.as_deref(),
                _ => None,
            };
        }
        d
    }

    #[test]
    fn a_chain_deeper_than_the_limit_becomes_parts_each_within_it() {
        let fp = Type {
            kind: Some(Kind::Fp64(Fp64 {
                type_variation_reference: 0,
                nullability: Nullability::Nullable as i32,
            })),
        };
        let mut rel = read_table(
            vec!["t".into()],
            NamedStruct {
                names: vec!["a".into()],
                r#struct: Some(Struct {
                    types: vec![fp],
                    type_variation_reference: 0,
                    nullability: Nullability::Required as i32,
                }),
            },
        );
        let mut ext = Extensions::new(&Functions::default());
        let add = ext.anchor(Function::Add);
        for _ in 0..80 {
            rel = project_emit(
                rel,
                vec![call(add, vec![field(0), lit_f64(1.0)])],
                Some(vec![1]),
            );
        }
        let mut steps = vec![Step {
            name: "__ddx_t_value".into(),
            plan: plan(rel, vec!["a".into()], &ext),
        }];
        split_steps(&mut steps);
        let names: Vec<&str> = steps.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "__ddx_t_value_part_0",
                "__ddx_t_value_part_1",
                "__ddx_t_value"
            ]
        );
        for s in &steps {
            assert!(
                depth(&s.plan) <= MAX_DEPTH,
                "{} is {} deep",
                s.name,
                depth(&s.plan)
            );
            assert_eq!(s.plan.extensions, steps[0].plan.extensions);
        }
        // Each part reads the one before it, and the step the last.
        assert_eq!(
            crate::unbound_reads(&steps[1].plan),
            vec!["__ddx_t_value_part_0"]
        );
        assert_eq!(
            crate::unbound_reads(&steps[2].plan),
            vec!["__ddx_t_value_part_1"]
        );
    }
}
