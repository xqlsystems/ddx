// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Dropping the columns nothing reads from the plans ddx writes.
//!
//! A region is rebuilt keeping every column (see [`crate::forward`]), and the
//! transposes append cotangents to it, so a backward step carries every
//! intermediate value to the top even where nothing above reads it. An
//! engine's optimizer drops those columns too, but only after consuming the
//! whole plan, and consuming and optimizing a wide plan is most of what a
//! long chain of maps costs. [`prune`] works down from the columns a plan's
//! output needs: a projection keeps only the expressions something reads, a
//! join and a cross join only the columns above them read, an aggregate only
//! the measures read, and a read of an earlier step only the columns it uses,
//! so that step can drop the rest too ([`prune_to`]).
//!
//! A relation it does not know (a window relation, an extension, a set
//! operation other than `UNION ALL`, an aggregate with grouping sets), or an
//! expression whose field references it cannot follow, is kept as it is,
//! with everything beneath it.

use std::collections::BTreeSet;

use substrait::proto::join_rel::JoinType;
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::{Emit, EmitKind};
use substrait::proto::set_rel::SetOp;
use substrait::proto::{Expression, Plan, Rel, RelCommon};

use crate::emit::select;
use crate::expr::{fields_of, map_fields};
use crate::forward::width;

/// For each output column of a relation before pruning, its column after,
/// if it was kept. Every column that was needed is kept.
type Map = Vec<Option<usize>>;

/// Prune every root of `plan`, keeping all of each root's columns.
pub(crate) fn prune_plan(plan: &mut Plan) {
    for r in plan.relations.iter_mut() {
        if let Some(substrait::proto::plan_rel::RelType::Root(root)) = r.rel_type.as_mut() {
            if let Some(input) = root.input.take() {
                let all: Vec<usize> = (0..root.names.len()).collect();
                root.input = Some(prune_to(input, &all));
            }
        }
    }
}

/// Prune `plan`'s root to the output columns named in `used`, keeping their
/// order (at least one, so the step still has rows to count).
/// Whether it pruned anything (if not, the plan is as it was).
pub(crate) fn prune_plan_to_names(plan: &mut Plan, used: &BTreeSet<String>) -> bool {
    let mut pruned = false;
    for r in plan.relations.iter_mut() {
        if let Some(substrait::proto::plan_rel::RelType::Root(root)) = r.rel_type.as_mut() {
            let mut keep: Vec<usize> = (0..root.names.len())
                .filter(|&i| used.contains(&root.names[i]))
                .collect();
            if keep.is_empty() && !root.names.is_empty() {
                keep.push(0);
            }
            if keep.len() == root.names.len() {
                continue;
            }
            if let Some(input) = root.input.take() {
                root.input = Some(prune_to(input, &keep));
                root.names = keep.iter().map(|&i| root.names[i].clone()).collect();
                pruned = true;
            }
        }
    }
    pruned
}

/// `rel` keeping only its output columns `keep`, in that order, with the
/// columns beneath them that nothing else reads dropped.
pub(crate) fn prune_to(rel: Rel, keep: &[usize]) -> Rel {
    let need: BTreeSet<usize> = keep.iter().copied().collect();
    let Ok(w) = width(&rel) else {
        return select_exactly(rel, keep);
    };
    let Ok((pruned, map)) = prune(rel.clone(), &need, w) else {
        return select_exactly(rel, keep);
    };
    let out: Vec<usize> = keep.iter().map(|&c| map[c].expect("needed")).collect();
    let w = Some(map.iter().flatten().count());
    if Some(out.len()) == w && out.iter().enumerate().all(|(i, &c)| i == c) {
        pruned
    } else {
        select(pruned, out)
    }
}

fn select_exactly(rel: Rel, keep: &[usize]) -> Rel {
    let w = width(&rel).ok();
    if Some(keep.len()) == w && keep.iter().enumerate().all(|(i, &c)| i == c) {
        rel
    } else {
        select(rel, keep.to_vec())
    }
}

/// `rel` pruned to the output columns `need` (it may keep more), and where
/// each output column went. An error means a part could not be followed; the
/// caller keeps `rel` as it is.
///
/// `w` is `rel`'s output width, which the caller knows: computing it here
/// would walk the whole subtree at every level.
fn prune(rel: Rel, need: &BTreeSet<usize>, w: usize) -> crate::Result<(Rel, Map)> {
    let Some(mut kind) = rel.rel_type else {
        return Ok((Rel { rel_type: None }, identity(w)));
    };
    let emit = take_emit(&mut kind);
    let direct_need: BTreeSet<usize> = match &emit {
        Some(m) => need.iter().map(|&o| m[o]).collect(),
        None => need.clone(),
    };
    // With no emit, the direct width is the output width.
    let direct_w = if emit.is_none() { Some(w) } else { None };
    let (kind, dmap, narrow) = match prune_direct(kind, &direct_need, direct_w)? {
        Ok(pruned) => pruned,
        Err(mut kind) => {
            // Not a relation pruned through: as it was, emit included.
            set_emit(&mut kind, emit);
            return Ok((
                Rel {
                    rel_type: Some(kind),
                },
                identity(w),
            ));
        }
    };
    let mut kind = kind;
    let outputs: Vec<usize> = match &emit {
        Some(m) => m.clone(),
        None => (0..dmap.len()).collect(),
    };
    let new_direct = dmap.iter().flatten().count();
    if emit.is_some() || narrow {
        // Emit exactly the needed outputs, in their order.
        let kept: Vec<usize> = (0..outputs.len()).filter(|o| need.contains(o)).collect();
        let mapping: Vec<usize> = kept
            .iter()
            .map(|&o| dmap[outputs[o]].expect("a needed column was kept"))
            .collect();
        let mut map = vec![None; outputs.len()];
        for (i, &o) in kept.iter().enumerate() {
            map[o] = Some(i);
        }
        let is_identity =
            mapping.len() == new_direct && mapping.iter().enumerate().all(|(i, &c)| i == c);
        set_emit(&mut kind, (!is_identity).then_some(mapping));
        Ok((
            Rel {
                rel_type: Some(kind),
            },
            map,
        ))
    } else {
        Ok((
            Rel {
                rel_type: Some(kind),
            },
            dmap,
        ))
    }
}

fn identity(w: usize) -> Map {
    (0..w).map(Some).collect()
}

/// Prune a relation's direct output (before any emit) to `need`. Returns the
/// relation, where each direct column went, and whether narrowing its output
/// with an emit is worth it (a projection or a join, whose outputs a parent
/// would otherwise carry); `None` for a relation not pruned through.
/// `direct_w` is the direct width when the caller knows it. A relation not
/// pruned through comes back as `Err`, unchanged.
#[allow(clippy::type_complexity)]
fn prune_direct(
    kind: RelType,
    need: &BTreeSet<usize>,
    direct_w: Option<usize>,
) -> crate::Result<Result<(RelType, Map, bool), RelType>> {
    Ok(Ok(match kind {
        RelType::Project(mut p) => {
            let input = *p.input.take().expect("a projection has an input");
            let w_in = match direct_w {
                Some(w) => w - p.expressions.len(),
                None => width(&input)?,
            };
            let mut child_need: BTreeSet<usize> =
                need.iter().copied().filter(|&c| c < w_in).collect();
            let mut kept = Vec::new();
            for (j, e) in p.expressions.iter().enumerate() {
                if need.contains(&(w_in + j)) {
                    child_need.extend(fields_of(e)?);
                    kept.push(j);
                }
            }
            let (input, cmap) = prune(input, &child_need, w_in)?;
            let exprs = kept
                .iter()
                .map(|&j| remap(&p.expressions[j], &cmap))
                .collect::<crate::Result<Vec<_>>>()?;
            let w_child = cmap.iter().flatten().count();
            let mut dmap: Map = cmap.clone();
            dmap.extend(
                (0..p.expressions.len())
                    .map(|j| kept.iter().position(|&k| k == j).map(|i| w_child + i)),
            );
            p.input = Some(Box::new(input));
            p.expressions = exprs;
            (RelType::Project(p), dmap, true)
        }
        RelType::Filter(mut f) => {
            let input = *f.input.take().expect("a filter has an input");
            let cond = f.condition.take().map(|c| *c);
            let mut child_need = need.clone();
            if let Some(c) = &cond {
                child_need.extend(fields_of(c)?);
            }
            let w_in = match direct_w {
                Some(w) => w,
                None => width(&input)?,
            };
            let (input, cmap) = prune(input, &child_need, w_in)?;
            f.condition = cond.map(|c| remap(&c, &cmap)).transpose()?.map(Box::new);
            f.input = Some(Box::new(input));
            (RelType::Filter(f), cmap, false)
        }
        RelType::Fetch(mut f) => {
            let input = *f.input.take().expect("a fetch has an input");
            let w_in = match direct_w {
                Some(w) => w,
                None => width(&input)?,
            };
            let (input, cmap) = prune(input, need, w_in)?;
            f.input = Some(Box::new(input));
            (RelType::Fetch(f), cmap, false)
        }
        RelType::Sort(mut s) => {
            let input = *s.input.take().expect("a sort has an input");
            let mut child_need = need.clone();
            for k in &s.sorts {
                if let Some(e) = &k.expr {
                    child_need.extend(fields_of(e)?);
                }
            }
            let w_in = match direct_w {
                Some(w) => w,
                None => width(&input)?,
            };
            let (input, cmap) = prune(input, &child_need, w_in)?;
            for k in s.sorts.iter_mut() {
                if let Some(e) = k.expr.as_mut() {
                    *e = remap(e, &cmap)?;
                }
            }
            s.input = Some(Box::new(input));
            (RelType::Sort(s), cmap, false)
        }
        RelType::Join(mut j) => {
            let kind = JoinType::try_from(j.r#type).unwrap_or(JoinType::Unspecified);
            let (keeps_left, keeps_right) = match kind {
                JoinType::Inner
                | JoinType::Outer
                | JoinType::Left
                | JoinType::Right
                | JoinType::LeftSingle
                | JoinType::RightSingle => (true, true),
                JoinType::LeftSemi | JoinType::LeftAnti => (true, false),
                JoinType::RightSemi | JoinType::RightAnti => (false, true),
                _ => return Ok(Err(RelType::Join(j))),
            };
            let left = *j.left.take().expect("a join has a left");
            let right = *j.right.take().expect("a join has a right");
            let (wl, wr) = (width(&left)?, width(&right)?);
            // The condition reads both sides, numbered left then right.
            let mut cond_fields = BTreeSet::new();
            for e in j.expression.iter().chain(j.post_join_filter.iter()) {
                cond_fields.extend(fields_of(e)?);
            }
            let out_offset_right = if keeps_left { wl } else { 0 };
            let mut ln: BTreeSet<usize> = cond_fields.iter().copied().filter(|&c| c < wl).collect();
            let mut rn: BTreeSet<usize> = cond_fields
                .iter()
                .filter(|&&c| c >= wl)
                .map(|&c| c - wl)
                .collect();
            for &c in need {
                if keeps_left && c < wl {
                    ln.insert(c);
                } else if keeps_right {
                    rn.insert(c - out_offset_right);
                }
            }
            let (left, lmap) = prune(left, &ln, wl)?;
            let (right, rmap) = prune(right, &rn, wr)?;
            let wl2 = lmap.iter().flatten().count();
            let both: Map = lmap
                .iter()
                .copied()
                .chain(rmap.iter().map(|c| c.map(|c| c + wl2)))
                .collect();
            j.expression = j
                .expression
                .map(|e| remap(&e, &both))
                .transpose()?
                .map(Box::new);
            j.post_join_filter = j
                .post_join_filter
                .map(|e| remap(&e, &both))
                .transpose()?
                .map(Box::new);
            let dmap: Map = match (keeps_left, keeps_right) {
                (true, true) => both,
                (true, false) => lmap,
                _ => rmap,
            };
            j.left = Some(Box::new(left));
            j.right = Some(Box::new(right));
            (RelType::Join(j), dmap, true)
        }
        RelType::Cross(mut c) => {
            let left = *c.left.take().expect("a cross join has a left");
            let right = *c.right.take().expect("a cross join has a right");
            let (wl, wr) = (width(&left)?, width(&right)?);
            let ln = need.iter().copied().filter(|&x| x < wl).collect();
            let rn = need.iter().filter(|&&x| x >= wl).map(|&x| x - wl).collect();
            let (left, lmap) = prune(left, &ln, wl)?;
            let (right, rmap) = prune(right, &rn, wr)?;
            let wl2 = lmap.iter().flatten().count();
            let dmap: Map = lmap
                .into_iter()
                .chain(rmap.into_iter().map(|c| c.map(|c| c + wl2)))
                .collect();
            c.left = Some(Box::new(left));
            c.right = Some(Box::new(right));
            (RelType::Cross(c), dmap, true)
        }
        RelType::Aggregate(mut a) => {
            #[allow(deprecated)]
            if a.groupings.len() > 1 {
                return Ok(Err(RelType::Aggregate(a)));
            }
            let input = *a.input.take().expect("an aggregate has an input");
            // Older producers list the groupings only per grouping.
            #[allow(deprecated)]
            let ng = if a.grouping_expressions.is_empty() {
                a.groupings
                    .first()
                    .map_or(0, |g| g.grouping_expressions.len())
            } else {
                a.grouping_expressions.len()
            };
            let mut child_need = BTreeSet::new();
            #[allow(deprecated)]
            for e in a.grouping_expressions.iter().chain(
                a.groupings
                    .iter()
                    .flat_map(|g| g.grouping_expressions.iter()),
            ) {
                child_need.extend(fields_of(e)?);
            }
            let kept: Vec<usize> = (0..a.measures.len())
                .filter(|m| need.contains(&(ng + m)))
                .collect();
            for &m in &kept {
                child_need.extend(measure_fields(&a.measures[m])?);
            }
            let w_in = width(&input)?;
            let (input, cmap) = prune(input, &child_need, w_in)?;
            a.grouping_expressions = a
                .grouping_expressions
                .iter()
                .map(|e| remap(e, &cmap))
                .collect::<crate::Result<_>>()?;
            #[allow(deprecated)]
            for g in a.groupings.iter_mut() {
                g.grouping_expressions = g
                    .grouping_expressions
                    .iter()
                    .map(|e| remap(e, &cmap))
                    .collect::<crate::Result<_>>()?;
            }
            let mut measures = Vec::with_capacity(kept.len());
            for &m in &kept {
                measures.push(remap_measure(&a.measures[m], &cmap)?);
            }
            let mut dmap: Map = (0..ng).map(Some).collect();
            dmap.extend(
                (0..a.measures.len()).map(|m| kept.iter().position(|&k| k == m).map(|i| ng + i)),
            );
            a.measures = measures;
            a.input = Some(Box::new(input));
            (RelType::Aggregate(a), dmap, false)
        }
        RelType::Set(mut s) => {
            if SetOp::try_from(s.op) != Ok(SetOp::UnionAll) || s.inputs.is_empty() {
                return Ok(Err(RelType::Set(s)));
            }
            let first = &s.inputs[0];
            let w = width(first)?;
            // Every input keeps exactly the needed columns, so they line up.
            let keep: Vec<usize> = need.iter().copied().collect();
            s.inputs = s.inputs.into_iter().map(|r| prune_to(r, &keep)).collect();
            let mut dmap = vec![None; w];
            for (i, &c) in keep.iter().enumerate() {
                dmap[c] = Some(i);
            }
            (RelType::Set(s), dmap, false)
        }
        RelType::Read(mut r) => {
            let unbound = matches!(&r.read_type, Some(ReadType::NamedTable(_)))
                && r.base_schema.as_ref().is_some_and(|s| s.r#struct.is_none());
            if !unbound || r.filter.is_some() || r.projection.is_some() {
                return Ok(Err(RelType::Read(r)));
            }
            // A read of an earlier step names its columns but not their
            // types; naming only the ones used lets that step drop the rest.
            let schema = r.base_schema.as_mut().expect("checked above");
            let w = schema.names.len();
            let keep: Vec<usize> = (0..w).filter(|c| need.contains(c)).collect();
            schema.names = keep.iter().map(|&c| schema.names[c].clone()).collect();
            let mut dmap = vec![None; w];
            for (i, &c) in keep.iter().enumerate() {
                dmap[c] = Some(i);
            }
            (RelType::Read(r), dmap, false)
        }
        other => return Ok(Err(other)),
    }))
}

fn remap(e: &Expression, map: &Map) -> crate::Result<Expression> {
    map_fields(e, &mut |i| {
        map.get(i).copied().flatten().ok_or_else(|| {
            crate::AdError::Internal(format!("pruning dropped column {i}, which is read"))
        })
    })
}

fn measure_fields(m: &substrait::proto::aggregate_rel::Measure) -> crate::Result<Vec<usize>> {
    use substrait::proto::function_argument::ArgType;
    let mut out = Vec::new();
    if let Some(f) = &m.measure {
        for a in &f.arguments {
            if let Some(ArgType::Value(e)) = &a.arg_type {
                out.extend(fields_of(e)?);
            }
        }
        for s in &f.sorts {
            if let Some(e) = &s.expr {
                out.extend(fields_of(e)?);
            }
        }
    }
    if let Some(e) = &m.filter {
        out.extend(fields_of(e)?);
    }
    Ok(out)
}

fn remap_measure(
    m: &substrait::proto::aggregate_rel::Measure,
    map: &Map,
) -> crate::Result<substrait::proto::aggregate_rel::Measure> {
    use substrait::proto::function_argument::ArgType;
    let mut m = m.clone();
    if let Some(f) = m.measure.as_mut() {
        for a in f.arguments.iter_mut() {
            if let Some(ArgType::Value(e)) = a.arg_type.as_mut() {
                *e = remap(e, map)?;
            }
        }
        for s in f.sorts.iter_mut() {
            if let Some(e) = s.expr.as_mut() {
                *e = remap(e, map)?;
            }
        }
    }
    if let Some(e) = m.filter.as_mut() {
        *e = remap(e, map)?;
    }
    Ok(m)
}

/// Remove a relation's emit, returning its output mapping.
fn take_emit(kind: &mut RelType) -> Option<Vec<usize>> {
    let common = common_mut(kind)?;
    match common.emit_kind.take() {
        Some(EmitKind::Emit(Emit { output_mapping })) => {
            Some(output_mapping.into_iter().map(|i| i as usize).collect())
        }
        other => {
            common.emit_kind = other;
            None
        }
    }
}

fn set_emit(kind: &mut RelType, emit: Option<Vec<usize>>) {
    let Some(mapping) = emit else { return };
    let common = match common_mut(kind) {
        Some(c) => c,
        None => match kind {
            RelType::Project(p) => p.common.get_or_insert_with(RelCommon::default),
            RelType::Filter(f) => f.common.get_or_insert_with(RelCommon::default),
            RelType::Fetch(f) => f.common.get_or_insert_with(RelCommon::default),
            RelType::Sort(s) => s.common.get_or_insert_with(RelCommon::default),
            RelType::Join(j) => j.common.get_or_insert_with(RelCommon::default),
            RelType::Cross(c) => c.common.get_or_insert_with(RelCommon::default),
            RelType::Aggregate(a) => a.common.get_or_insert_with(RelCommon::default),
            RelType::Set(s) => s.common.get_or_insert_with(RelCommon::default),
            RelType::Read(r) => r.common.get_or_insert_with(RelCommon::default),
            _ => return,
        },
    };
    common.emit_kind = Some(EmitKind::Emit(Emit {
        output_mapping: mapping.into_iter().map(|i| i as i32).collect(),
    }));
}

fn common_mut(kind: &mut RelType) -> Option<&mut RelCommon> {
    match kind {
        RelType::Project(p) => p.common.as_mut(),
        RelType::Filter(f) => f.common.as_mut(),
        RelType::Fetch(f) => f.common.as_mut(),
        RelType::Sort(s) => s.common.as_mut(),
        RelType::Join(j) => j.common.as_mut(),
        RelType::Cross(c) => c.common.as_mut(),
        RelType::Aggregate(a) => a.common.as_mut(),
        RelType::Set(s) => s.common.as_mut(),
        RelType::Read(r) => r.common.as_mut(),
        RelType::Window(w) => w.common.as_mut(),
        RelType::ExtensionSingle(e) => e.common.as_mut(),
        RelType::ExtensionMulti(e) => e.common.as_mut(),
        RelType::ExtensionLeaf(e) => e.common.as_mut(),
        _ => None,
    }
}
