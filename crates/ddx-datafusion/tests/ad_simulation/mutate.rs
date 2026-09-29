// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Equivalence-preserving rewrites of a Substrait plan.
//!
//! `ddx-ad` reads plans by hand (design.md §4.4): emits, field references,
//! projections that append, joins whose columns are both sides'. It has only
//! ever been fed what DataFusion's producer happens to emit, and another
//! producer (DuckDB's, in M5) will not emit the same shapes. So these rewrite a
//! plan into one that computes the same relation by a different shape, at
//! random nodes, and the caller insists the gradient does not move.
//!
//! None of this is trusted on its own: the caller has DataFusion consume the
//! rewritten plan and compute the loss, and a rewrite that changes the loss is
//! discarded as the harness's mistake, never reported as ddx's.

use ddx_ad::expr::{field, lit_f64, map_fields};
use ddx_ad::forward::width;
use ddx_ad::substrait::proto::aggregate_rel::Measure;
use ddx_ad::substrait::proto::expression::literal::LiteralType;
use ddx_ad::substrait::proto::expression::{Literal, RexType};
use ddx_ad::substrait::proto::join_rel::JoinType;
use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::rel_common::{Emit, EmitKind};
use ddx_ad::substrait::proto::sort_field::{SortDirection, SortKind};
use ddx_ad::substrait::proto::{
    Expression, FilterRel, Plan, ProjectRel, Rel, RelCommon, SortField, SortRel,
};
use ddx_core::test_utils::Rng;

/// Rewrite `plan` at a few random nodes. Returns the plan and the rewrites
/// applied, or `None` when none applied.
pub fn mutate(plan: &Plan, rng: &mut Rng) -> Option<(Plan, Vec<&'static str>)> {
    let mut out = plan.clone();
    let root = out
        .relations
        .iter_mut()
        .find_map(|r| match &mut r.rel_type {
            Some(PlanRelType::Root(root)) => root.input.as_mut(),
            _ => None,
        })?;
    let nodes = count(root);
    let mut targets: Vec<usize> = (0..1 + rng.below(3))
        .map(|_| rng.below(nodes as u64) as usize)
        .collect();
    targets.sort();
    let mut done = Vec::new();
    let mut at = 0;
    *root = walk(root, rng, &targets, &mut at, false, &mut done);
    (!done.is_empty()).then_some((out, done))
}

fn count(rel: &Rel) -> usize {
    1 + inputs(rel).iter().map(|r| count(r)).sum::<usize>()
}

fn inputs(rel: &Rel) -> Vec<&Rel> {
    match &rel.rel_type {
        Some(RelType::Project(p)) => p.input.iter().map(|b| &**b).collect(),
        Some(RelType::Filter(f)) => f.input.iter().map(|b| &**b).collect(),
        Some(RelType::Sort(s)) => s.input.iter().map(|b| &**b).collect(),
        Some(RelType::Fetch(f)) => f.input.iter().map(|b| &**b).collect(),
        Some(RelType::Aggregate(a)) => a.input.iter().map(|b| &**b).collect(),
        Some(RelType::Join(j)) => j.left.iter().chain(&j.right).map(|b| &**b).collect(),
        Some(RelType::Cross(c)) => c.left.iter().chain(&c.right).map(|b| &**b).collect(),
        Some(RelType::Set(s)) => s.inputs.iter().collect(),
        _ => vec![],
    }
}

/// Rebuild `rel` with its inputs walked, then maybe rewrite it.
fn walk(
    rel: &Rel,
    rng: &mut Rng,
    targets: &[usize],
    at: &mut usize,
    under_fetch: bool,
    done: &mut Vec<&'static str>,
) -> Rel {
    let here = *at;
    *at += 1;
    let mut rel = rel.clone();
    let is_fetch = matches!(rel.rel_type, Some(RelType::Fetch(_)));
    // A sort's order is what a fetch above it keeps, so nothing may be
    // inserted between them that reorders rows.
    let is_sort = matches!(rel.rel_type, Some(RelType::Sort(_)));
    let child_under_fetch = is_fetch || (under_fetch && is_sort);
    let mut go = |r: &mut Box<Rel>| **r = walk(r, rng, targets, at, child_under_fetch, done);
    match &mut rel.rel_type {
        Some(RelType::Project(p)) => p.input.iter_mut().for_each(&mut go),
        Some(RelType::Filter(f)) => f.input.iter_mut().for_each(&mut go),
        Some(RelType::Sort(s)) => s.input.iter_mut().for_each(&mut go),
        Some(RelType::Fetch(f)) => f.input.iter_mut().for_each(&mut go),
        Some(RelType::Aggregate(a)) => a.input.iter_mut().for_each(&mut go),
        Some(RelType::Join(j)) => {
            j.left.iter_mut().for_each(&mut go);
            j.right.iter_mut().for_each(&mut go);
        }
        Some(RelType::Cross(c)) => {
            c.left.iter_mut().for_each(&mut go);
            c.right.iter_mut().for_each(&mut go);
        }
        Some(RelType::Set(s)) => {
            for r in &mut s.inputs {
                *r = walk(r, rng, targets, at, false, done);
            }
        }
        _ => {}
    }
    for _ in targets.iter().filter(|&&t| t == here) {
        if let Some((r, kind)) = rewrite(&rel, rng, under_fetch) {
            rel = r;
            done.push(kind);
        }
    }
    rel
}

fn common_emit(emit: Vec<usize>) -> RelCommon {
    RelCommon {
        emit_kind: Some(EmitKind::Emit(Emit {
            output_mapping: emit.into_iter().map(|i| i as i32).collect(),
        })),
        ..Default::default()
    }
}

/// Only the columns `emit` picks from `input`: a projection with no
/// expressions.
fn select(input: Rel, emit: Vec<usize>) -> Rel {
    project(input, vec![], Some(emit))
}

fn project(input: Rel, exprs: Vec<Expression>, emit: Option<Vec<usize>>) -> Rel {
    Rel {
        rel_type: Some(RelType::Project(Box::new(ProjectRel {
            common: emit.map(common_emit),
            input: Some(Box::new(input)),
            expressions: exprs,
            advanced_extension: None,
        }))),
    }
}

fn lit_true() -> Expression {
    Expression {
        rex_type: Some(RexType::Literal(Literal {
            nullable: false,
            type_variation_reference: 0,
            literal_type: Some(LiteralType::Boolean(true)),
        })),
    }
}

/// The emit of `rel`, if it has one.
fn emit_of(rel: &Rel) -> Option<Vec<usize>> {
    let common = match rel.rel_type.as_ref()? {
        RelType::Join(j) => j.common.as_ref(),
        RelType::Aggregate(a) => a.common.as_ref(),
        _ => None,
    };
    match common?.emit_kind.as_ref()? {
        EmitKind::Emit(e) => Some(e.output_mapping.iter().map(|&i| i as usize).collect()),
        _ => None,
    }
}

fn rewrite(rel: &Rel, rng: &mut Rng, under_fetch: bool) -> Option<(Rel, &'static str)> {
    let w = width(rel).ok()?;
    if w == 0 {
        return None;
    }
    let all: Vec<usize> = (0..w).collect();
    // Pick among the rewrites that fit this node, so the ones that need a
    // join or an aggregate are not starved by the nodes that are neither.
    let mut fits: Vec<&str> = vec![
        "identity-project",
        "permute-and-back",
        "filter-true",
        "dropped-column",
    ];
    if !under_fetch {
        fits.push("sort");
    }
    match &rel.rel_type {
        Some(RelType::Join(_)) => fits.extend(["swap-join"; 4]),
        Some(RelType::Aggregate(_)) => fits.extend(["duplicate-measure"; 4]),
        _ => {}
    }
    match fits[rng.below(fits.len() as u64) as usize] {
        "identity-project" => Some((select(rel.clone(), all), "identity-project")),
        "permute-and-back" => {
            let mut perm = all.clone();
            for i in (1..perm.len()).rev() {
                let j = rng.below(i as u64 + 1) as usize;
                perm.swap(i, j);
            }
            let mut inv = vec![0; w];
            for (k, &p) in perm.iter().enumerate() {
                inv[p] = k;
            }
            Some((select(select(rel.clone(), perm), inv), "permute-and-back"))
        }
        "filter-true" => Some((
            Rel {
                rel_type: Some(RelType::Filter(Box::new(FilterRel {
                    common: None,
                    input: Some(Box::new(rel.clone())),
                    condition: Some(Box::new(lit_true())),
                    advanced_extension: None,
                }))),
            },
            "filter-true",
        )),
        "dropped-column" => {
            // Compute a column, then leave it out: a copy of a column, or a
            // constant.
            let e = if rng.below(2) == 0 {
                field(rng.below(w as u64) as usize)
            } else {
                lit_f64(1.5)
            };
            Some((project(rel.clone(), vec![e], Some(all)), "dropped-column"))
        }
        "swap-join" => swap_join(rel),
        "duplicate-measure" => duplicate_measure(rel),
        "sort" if !under_fetch => Some((
            Rel {
                rel_type: Some(RelType::Sort(Box::new(SortRel {
                    common: None,
                    input: Some(Box::new(rel.clone())),
                    sorts: vec![SortField {
                        expr: Some(field(rng.below(w as u64) as usize)),
                        sort_kind: Some(SortKind::Direction(SortDirection::DescNullsLast as i32)),
                    }],
                    advanced_extension: None,
                }))),
            },
            "sort",
        )),
        _ => None,
    }
}

/// An inner join with its sides swapped, then its columns put back in order.
fn swap_join(rel: &Rel) -> Option<(Rel, &'static str)> {
    let Some(RelType::Join(j)) = &rel.rel_type else {
        return None;
    };
    if JoinType::try_from(j.r#type).ok()? != JoinType::Inner {
        return None;
    }
    let (l, r) = (j.left.as_deref()?, j.right.as_deref()?);
    let (wl, wr) = (width(l).ok()?, width(r).ok()?);
    let mut remap = |i: usize| Ok(if i < wl { i + wr } else { i - wl });
    let expression = match &j.expression {
        Some(e) => Some(Box::new(map_fields(e, &mut remap).ok()?)),
        None => None,
    };
    let post_join_filter = match &j.post_join_filter {
        Some(e) => Some(Box::new(map_fields(e, &mut remap).ok()?)),
        None => None,
    };
    let mut swapped = (**j).clone();
    swapped.left = Some(Box::new(r.clone()));
    swapped.right = Some(Box::new(l.clone()));
    swapped.expression = expression;
    swapped.post_join_filter = post_join_filter;
    swapped.common = None;
    // Swapped column k of the original order is at k + wr (left) or k - wl.
    let restore: Vec<usize> = (0..wl + wr)
        .map(|k| if k < wl { k + wr } else { k - wl })
        .collect();
    let emit = match emit_of(rel) {
        Some(m) => m.into_iter().map(|k| restore[k]).collect(),
        None => restore,
    };
    let joined = Rel {
        rel_type: Some(RelType::Join(Box::new(swapped))),
    };
    Some((select(joined, emit), "swap-join"))
}

/// An aggregate with a measure computed twice, the copy then dropped. The
/// copy carries `FILTER (WHERE true)`: DataFusion's consumer names a measure
/// by its expression and refuses two of one name, so an exact copy never
/// reaches ddx.
fn duplicate_measure(rel: &Rel) -> Option<(Rel, &'static str)> {
    let Some(RelType::Aggregate(a)) = &rel.rel_type else {
        return None;
    };
    let first: Measure = a.measures.first()?.clone();
    let mut dup = (**a).clone();
    let direct = width(rel).ok()?;
    let direct = match emit_of(rel) {
        // The direct width, before the emit: groupings then measures.
        Some(_) => {
            let mut plain = (**a).clone();
            plain.common = None;
            width(&Rel {
                rel_type: Some(RelType::Aggregate(Box::new(plain))),
            })
            .ok()?
        }
        None => direct,
    };
    let emit = emit_of(rel).unwrap_or_else(|| (0..direct).collect());
    let mut copy = first;
    copy.filter = Some(lit_true());
    dup.measures.push(copy);
    dup.common = None;
    let agg = Rel {
        rel_type: Some(RelType::Aggregate(Box::new(dup))),
    };
    Some((select(agg, emit), "duplicate-measure"))
}
