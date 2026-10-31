// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A window aggregate over each whole partition, as a `GROUP BY` joined
//! back.
//!
//! `SUM(e) OVER (PARTITION BY i)` is each partition's `SUM`, copied to each
//! of its rows. Reverse mode reads it as exactly that:
//!
//! ```sql
//! SELECT r.*, g.v FROM rows r
//! JOIN (SELECT i, SUM(e) AS v FROM rows GROUP BY i) g ON r.i IS NOT DISTINCT FROM g.i
//! ```
//!
//! Then the aggregate is saved like any other, and every rule for a saved
//! aggregate applies to it. A softmax written with windows, `e / SUM(e)
//! OVER (PARTITION BY i)`, reaches the shape of one written with `GROUP BY`
//! and a join, and gets the same program: the normalizer rule of
//! [`crate::contraction`] holds for both.
//!
//! A window is rewritten when it has a reduce rule ([`crate::reduce`]),
//! reads its whole partition, and isn't `DISTINCT`. The rows beneath it
//! are computed twice, so they must give the same rows both times: no
//! volatile function, `LIMIT` or other window beneath it.

use std::borrow::Cow;

use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::expression::{RexType, WindowFunction};
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::{Emit, EmitKind};
use substrait::proto::{
    AggregateFunction, AggregationPhase, Expression, Plan, PlanRel, ProjectRel, Rel, RelCommon,
};

use crate::emit::{aggregate, join};
use crate::error::Result;
use crate::expr::{call, field};
use crate::forward::{inline_references, referenced, width};
use crate::functions::{Extensions, Function, Functions};
use crate::optics::{Compose, Cosmos, Deep, Exprs, Fold, PlanRels, RelKind, Setter, Tree};
use crate::reduce::{reduce_rule, whole_partition};

/// `plan` with each window aggregate over a whole partition rewritten as a
/// grouped aggregate joined back (see the module docs).
pub(crate) fn windows_as_aggregates(plan: &Plan) -> Result<Cow<'_, Plan>> {
    let functions = Functions::from_plan(plan)?;
    let rewritten = |w: &WindowFunction| -> Result<bool> {
        let rule = reduce_rule(&functions, w.function_reference, w.invocation, &w.arguments)?;
        Ok(rule.is_some()
            && whole_partition(w)
            && w.invocation != AggregationInvocation::Distinct as i32)
    };
    let mut any = false;
    for rel in PlanRels.then(Deep).to_vec(plan) {
        if let Some(RelType::Project(p)) = &rel.rel_type {
            for w in windows(&p.expressions) {
                any |= rewritten(&w)?;
            }
        }
    }
    if !any {
        return Ok(Cow::Borrowed(plan));
    }
    let mut ext = Extensions::new(&functions);
    let relations = plan.relations.clone();
    let mut plan = plan.clone();
    let mut failed = None;
    PlanRels.then(Deep).modify(&mut plan, |rel| {
        let Some(RelType::Project(p)) = &rel.rel_type else {
            return;
        };
        match regrouped(p.as_ref(), &relations, &functions, &mut ext, &rewritten) {
            Ok(Some(new)) => *rel = new,
            Ok(None) => {}
            Err(e) => failed = failed.take().or(Some(e)),
        }
    });
    if let Some(e) = failed {
        return Err(e);
    }
    plan.extensions = ext.declarations();
    plan.extension_urns = ext.urns();
    Ok(Cow::Owned(plan))
}

/// The window functions among `exprs` and inside them, in order.
fn windows(exprs: &[Expression]) -> Vec<WindowFunction> {
    let mut out = Vec::new();
    for e in exprs {
        Cosmos.each(e, |x| {
            if let Some(RexType::WindowFunction(w)) = &x.rex_type {
                out.push(w.clone());
            }
        });
    }
    out
}

/// `p` with its rewritten windows (those `rewritten` accepts) read from an
/// aggregate of its input joined to it, one per set of partition keys, or
/// `None` if it has none, or its input can't be computed twice.
fn regrouped(
    p: &ProjectRel,
    relations: &[PlanRel],
    functions: &Functions,
    ext: &mut Extensions,
    rewritten: &dyn Fn(&WindowFunction) -> Result<bool>,
) -> Result<Option<Rel>> {
    let Some(input) = p.input.as_deref() else {
        return Ok(None);
    };
    // The partition keys of each aggregate, and its windows.
    let mut groups: Vec<(Vec<Expression>, Vec<WindowFunction>)> = Vec::new();
    for w in windows(&p.expressions) {
        if !rewritten(&w)? {
            continue;
        }
        match groups.iter_mut().find(|(keys, _)| *keys == w.partitions) {
            Some((_, ws)) => {
                if !ws.contains(&w) {
                    ws.push(w);
                }
            }
            None => groups.push((w.partitions.clone(), vec![w])),
        }
    }
    if groups.is_empty() || !twice(input, relations, functions, rewritten, &mut Vec::new())? {
        return Ok(None);
    }
    // Its width, with the relations it reads through references written out.
    let Ok(below) = inline_references(input, relations).and_then(|r| width(&r)) else {
        return Ok(None);
    };
    // The input, joined to each aggregate in turn: (window, its column).
    let mut rel = input.clone();
    let mut at: Vec<(WindowFunction, usize)> = Vec::new();
    let mut joined = below;
    for (keys, ws) in groups {
        let n = ws.len();
        let mut agg = aggregate(input.clone(), keys.clone(), Vec::new());
        if let Some(RelType::Aggregate(a)) = &mut agg.rel_type {
            for w in &ws {
                a.measures.push(substrait::proto::aggregate_rel::Measure {
                    measure: Some(AggregateFunction {
                        function_reference: w.function_reference,
                        arguments: w.arguments.clone(),
                        options: w.options.clone(),
                        output_type: w.output_type.clone(),
                        phase: AggregationPhase::InitialToResult as i32,
                        sorts: Vec::new(),
                        invocation: w.invocation,
                        ..Default::default()
                    }),
                    filter: None,
                });
            }
        }
        let mut on = None;
        for (k, key) in keys.iter().enumerate() {
            let same = ext.anchor(Function::IsNotDistinctFrom);
            let here = call(same, vec![key.clone(), field(joined + k)]);
            on = Some(match on {
                Some(before) => call(ext.anchor(Function::And), vec![before, here]),
                None => here,
            });
        }
        rel = match on {
            Some(on) => join(rel, agg, on, substrait::proto::join_rel::JoinType::Inner),
            None => Rel {
                rel_type: Some(RelType::Cross(Box::new(substrait::proto::CrossRel {
                    left: Some(Box::new(rel)),
                    right: Some(Box::new(agg)),
                    ..Default::default()
                }))),
            },
        };
        for (m, w) in ws.into_iter().enumerate() {
            at.push((w, joined + keys.len() + m));
        }
        joined += keys.len() + n;
    }
    let added = joined - below;
    let mut expressions = p.expressions.clone();
    for e in &mut expressions {
        Cosmos.modify(e, |x| {
            if let Some(RexType::WindowFunction(w)) = &x.rex_type {
                if let Some((_, c)) = at.iter().find(|(v, _)| v == w) {
                    *x = field(*c);
                }
            }
        });
    }
    // The projection's own columns move past the aggregates' columns.
    let width = below + p.expressions.len();
    let old: Vec<usize> = match p.common.as_ref().and_then(|c| c.emit_kind.as_ref()) {
        Some(EmitKind::Emit(e)) => e
            .output_mapping
            .iter()
            .map(|&i| usize::try_from(i).unwrap_or(usize::MAX))
            .collect(),
        _ => (0..width).collect(),
    };
    let output_mapping = old
        .into_iter()
        .map(|i| if i < below { i } else { i + added })
        .map(|i| i32::try_from(i).unwrap_or(i32::MAX))
        .collect();
    Ok(Some(Rel {
        rel_type: Some(RelType::Project(Box::new(ProjectRel {
            common: Some(RelCommon {
                emit_kind: Some(EmitKind::Emit(Emit { output_mapping })),
                ..p.common.clone().unwrap_or_default()
            }),
            input: Some(Box::new(rel)),
            expressions,
            advanced_extension: p.advanced_extension.clone(),
        }))),
    }))
}

/// Can `rel` be computed twice and give the same rows? Not if it calls a
/// volatile function, cuts rows with a `LIMIT`, computes a window other
/// than one `rewritten` accepts (a rank may order ties differently each
/// time; a rewritten window becomes a saved aggregate, computed once). A
/// shared relation of `relations` it reads is checked too, each once
/// (`seen`); one that can't be found, or reads itself, can't.
fn twice(
    rel: &Rel,
    relations: &[PlanRel],
    functions: &Functions,
    rewritten: &dyn Fn(&WindowFunction) -> Result<bool>,
    seen: &mut Vec<i32>,
) -> Result<bool> {
    for kind in Tree.then(RelKind).to_vec(rel) {
        match kind {
            RelType::Fetch(_) | RelType::Window(_) => return Ok(false),
            RelType::Reference(x) => {
                if seen.contains(&x.subtree_ordinal) {
                    return Ok(false);
                }
                seen.push(x.subtree_ordinal);
                let Ok(shared) = referenced(relations, x.subtree_ordinal) else {
                    return Ok(false);
                };
                if !twice(shared, relations, functions, rewritten, seen)? {
                    return Ok(false);
                }
                seen.pop();
            }
            _ => {}
        }
        for e in Exprs.to_vec(kind) {
            let mut found = Vec::new();
            Cosmos.each(e, |x| found.push(x.clone()));
            for x in found {
                match &x.rex_type {
                    Some(RexType::ScalarFunction(f))
                        if functions.is_volatile(f.function_reference)? =>
                    {
                        return Ok(false)
                    }
                    Some(RexType::WindowFunction(w)) if !rewritten(w)? => return Ok(false),
                    _ => {}
                }
            }
        }
    }
    Ok(true)
}
