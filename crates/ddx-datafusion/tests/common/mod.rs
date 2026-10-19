// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Shared plumbing for the integration tests.
//!
//! Only the mechanical parts live here — pulling a column out of a result set.
//! Each test file keeps its own fixture setup, because *what* it registers is
//! part of what it is testing: `path_a.rs` deliberately never calls `install`,
//! and `regressions.rs` needs tables with specific column types.

// Each integration test is its own binary and uses a different subset of this
// module, so unused helpers are expected rather than a smell.
#![allow(dead_code)]

pub mod ad;

use datafusion::arrow::array::{Array, Float64Array, RecordBatch};
use datafusion::arrow::datatypes::DataType;
use std::collections::HashMap;

use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::{Plan, PlanRel, ReferenceRel, Rel};
use prost::Message;

/// Column 0 of `batches` as `f64`s.
///
/// Panics if it is not `Float64`: every derivative ddx emits is DOUBLE-typed, so
/// anything else is a failure worth surfacing loudly right here rather than as a
/// confusing comparison further down the test.
pub fn f64_column(batches: &[RecordBatch]) -> Vec<f64> {
    let mut out = Vec::new();
    for b in batches {
        let a = b
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("every derivative is emitted DOUBLE-typed");
        out.extend((0..a.len()).map(|i| a.value(i)));
    }
    out
}

/// The Arrow type of column 0 — used where the *type* is the claim under test,
/// not just the values.
pub fn column_type(batches: &[RecordBatch]) -> DataType {
    batches[0].schema().field(0).data_type().clone()
}

/// The Substrait plan DataFusion produces for `sql`, optimized or not.
pub async fn substrait_of(
    ctx: &datafusion::prelude::SessionContext,
    sql: &str,
    optimized: bool,
) -> ddx_ad::substrait::proto::Plan {
    let df = ctx.sql(sql).await.unwrap();
    let lp = if optimized {
        df.into_optimized_plan().unwrap()
    } else {
        df.into_unoptimized_plan()
    };
    *datafusion_substrait::logical_plan::producer::to_substrait_plan(&lp, &ctx.state()).unwrap()
}

/// The relations of `plan` that carry an emit and are not projections: none
/// in a plan ddx writes. DuckDB's consumer ignores an emit on a join,
/// filter, sort, fetch, cross join or set, and returns the leading columns
/// with no error, so ddx narrows anything but a projection with a
/// projection over it.
pub fn emits_off_projections(plan: &ddx_ad::substrait::proto::Plan) -> Vec<&'static str> {
    use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
    use ddx_ad::substrait::proto::rel::RelType;
    use ddx_ad::substrait::proto::Rel;
    fn has_emit(c: &Option<ddx_ad::substrait::proto::RelCommon>) -> bool {
        use ddx_ad::substrait::proto::rel_common::EmitKind;
        matches!(
            c.as_ref().and_then(|c| c.emit_kind.as_ref()),
            Some(EmitKind::Emit(_))
        )
    }
    let mut out = Vec::new();
    let mut stack: Vec<&Rel> = plan
        .relations
        .iter()
        .filter_map(|r| match &r.rel_type {
            Some(PlanRelType::Root(root)) => root.input.as_ref(),
            Some(PlanRelType::Rel(r)) => Some(r),
            None => None,
        })
        .collect();
    while let Some(r) = stack.pop() {
        let (name, common, inputs): (&'static str, _, Vec<&Rel>) = match &r.rel_type {
            Some(RelType::Project(p)) => ("project", &None, p.input.iter().map(|b| &**b).collect()),
            Some(RelType::Filter(f)) => {
                ("filter", &f.common, f.input.iter().map(|b| &**b).collect())
            }
            Some(RelType::Sort(s)) => ("sort", &s.common, s.input.iter().map(|b| &**b).collect()),
            Some(RelType::Fetch(f)) => ("fetch", &f.common, f.input.iter().map(|b| &**b).collect()),
            Some(RelType::Aggregate(a)) => (
                "aggregate",
                &a.common,
                a.input.iter().map(|b| &**b).collect(),
            ),
            Some(RelType::Join(j)) => (
                "join",
                &j.common,
                j.left.iter().chain(j.right.iter()).map(|b| &**b).collect(),
            ),
            Some(RelType::Cross(c)) => (
                "cross",
                &c.common,
                c.left.iter().chain(c.right.iter()).map(|b| &**b).collect(),
            ),
            Some(RelType::Set(s)) => ("set", &s.common, s.inputs.iter().collect()),
            Some(RelType::Read(rd)) => ("read", &rd.common, Vec::new()),
            _ => ("other", &None, Vec::new()),
        };
        if has_emit(common) {
            out.push(name);
        }
        stack.extend(inputs);
    }
    out
}

/// `plan` with each subtree that two places read written once, at the end
/// of `plan.relations`, and read through a `ReferenceRel` from each: the
/// plan as a DAG, as a producer that shares a CTE read twice writes it. A
/// subtree is shared when distinct subtrees read it, or one reads it twice,
/// so a subtree read only from inside a shared one is not shared again.
pub fn shared(plan: &Plan) -> Plan {
    /// How many times distinct subtrees read each subtree, by encoding.
    fn readers(rel: &Rel, seen: &mut HashMap<Vec<u8>, usize>) -> Vec<u8> {
        let encoding = rel.encode_to_vec();
        if seen.contains_key(&encoding) {
            return encoding;
        }
        seen.insert(encoding.clone(), 0);
        if let Some(kind) = &rel.rel_type {
            for child in inputs(kind) {
                let c = readers(child, seen);
                *seen.get_mut(&c).unwrap() += 1;
            }
        }
        encoding
    }
    fn share(
        rel: &Rel,
        counts: &HashMap<Vec<u8>, usize>,
        ordinals: &mut HashMap<Vec<u8>, i32>,
        relations: &mut Vec<PlanRel>,
    ) -> Rel {
        let encoding = rel.encode_to_vec();
        if counts[&encoding] < 2 {
            return children_shared(rel, counts, ordinals, relations);
        }
        if let Some(&ordinal) = ordinals.get(&encoding) {
            return reference(ordinal);
        }
        let ordinal = relations.len() as i32;
        ordinals.insert(encoding, ordinal);
        relations.push(PlanRel { rel_type: None });
        let body = children_shared(rel, counts, ordinals, relations);
        relations[ordinal as usize].rel_type = Some(PlanRelType::Rel(body));
        reference(ordinal)
    }
    fn children_shared(
        rel: &Rel,
        counts: &HashMap<Vec<u8>, usize>,
        ordinals: &mut HashMap<Vec<u8>, i32>,
        relations: &mut Vec<PlanRel>,
    ) -> Rel {
        let mut out = rel.clone();
        if let Some(kind) = &mut out.rel_type {
            for child in inputs_mut(kind) {
                *child = share(child, counts, ordinals, relations);
            }
        }
        out
    }
    let mut out = plan.clone();
    let at = out
        .relations
        .iter()
        .position(|r| matches!(r.rel_type, Some(PlanRelType::Root(_))))
        .unwrap();
    let Some(PlanRelType::Root(root)) = &out.relations[at].rel_type else {
        unreachable!()
    };
    let input = root.input.clone().unwrap();
    let mut counts = HashMap::new();
    readers(&input, &mut counts);
    let mut relations = std::mem::take(&mut out.relations);
    let input = children_shared(&input, &counts, &mut HashMap::new(), &mut relations);
    let Some(PlanRelType::Root(root)) = &mut relations[at].rel_type else {
        unreachable!()
    };
    root.input = Some(input);
    out.relations = relations;
    out
}

pub fn reference(ordinal: i32) -> Rel {
    Rel {
        rel_type: Some(RelType::Reference(ReferenceRel {
            subtree_ordinal: ordinal,
        })),
    }
}

pub fn inputs(kind: &RelType) -> Vec<&Rel> {
    match kind {
        RelType::Filter(r) => r.input.as_deref().into_iter().collect(),
        RelType::Fetch(r) => r.input.as_deref().into_iter().collect(),
        RelType::Aggregate(r) => r.input.as_deref().into_iter().collect(),
        RelType::Sort(r) => r.input.as_deref().into_iter().collect(),
        RelType::Project(r) => r.input.as_deref().into_iter().collect(),
        RelType::Window(r) => r.input.as_deref().into_iter().collect(),
        RelType::Join(r) => [r.left.as_deref(), r.right.as_deref()]
            .into_iter()
            .flatten()
            .collect(),
        RelType::Cross(r) => [r.left.as_deref(), r.right.as_deref()]
            .into_iter()
            .flatten()
            .collect(),
        RelType::Set(r) => r.inputs.iter().collect(),
        _ => Vec::new(),
    }
}

pub fn inputs_mut(kind: &mut RelType) -> Vec<&mut Rel> {
    match kind {
        RelType::Filter(r) => r.input.as_deref_mut().into_iter().collect(),
        RelType::Fetch(r) => r.input.as_deref_mut().into_iter().collect(),
        RelType::Aggregate(r) => r.input.as_deref_mut().into_iter().collect(),
        RelType::Sort(r) => r.input.as_deref_mut().into_iter().collect(),
        RelType::Project(r) => r.input.as_deref_mut().into_iter().collect(),
        RelType::Window(r) => r.input.as_deref_mut().into_iter().collect(),
        RelType::Join(r) => [r.left.as_deref_mut(), r.right.as_deref_mut()]
            .into_iter()
            .flatten()
            .collect(),
        RelType::Cross(r) => [r.left.as_deref_mut(), r.right.as_deref_mut()]
            .into_iter()
            .flatten()
            .collect(),
        RelType::Set(r) => r.inputs.iter_mut().collect(),
        _ => Vec::new(),
    }
}

/// How many relations `plan` shares through references.
pub fn shared_count(plan: &Plan) -> usize {
    plan.relations
        .iter()
        .filter(|r| matches!(r.rel_type, Some(PlanRelType::Rel(_))))
        .count()
}
