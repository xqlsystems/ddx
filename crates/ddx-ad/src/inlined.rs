// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! How large a plan's columns are, written out as expressions of the tables
//! they read.

use substrait::proto::join_rel::JoinType;
use substrait::proto::plan_rel::RelType as PlanRelType;
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::EmitKind;
use substrait::proto::{Expression, Plan, Rel};

use crate::error::{AdError, Result};
use crate::expr::{as_field, value_args};
use crate::forward::{grouping_expressions, rel_name, width};
use crate::optics::{cata, Compose, Cosmos, Fold, Inputs, RelKind};

/// The size, in expression nodes, of `plan`'s largest column written out
/// down to the tables it reads: each reference to a computed column replaced
/// by that column's expression, through every relation, saturating at
/// `u64::MAX`.
///
/// A Substrait consumer that names each computed column after its
/// expression, as DataFusion's does, gives names about this long. They grow
/// exponentially in a chain of columns that each read the one before
/// twice, a softmax's say, and planning slows with them: so this tells an
/// adapter when to run a step another way (as [`to_sql`](crate::to_sql)'s
/// SQL, whose names are short).
///
/// `None` if `plan` holds a relation it cannot size, such as a reference to
/// a shared one or an extension's: the adapter then runs it as it would
/// any plan, and a size alone never refuses one.
pub fn inlined_size(plan: &Plan) -> Option<u64> {
    sized(plan).ok()
}

fn sized(plan: &Plan) -> Result<u64> {
    let mut largest = 0;
    for r in &plan.relations {
        let rel = match &r.rel_type {
            Some(PlanRelType::Root(root)) => root.input.as_ref(),
            Some(PlanRelType::Rel(rel)) => Some(rel),
            None => None,
        };
        if let Some(rel) = rel {
            let columns = cata(
                rel,
                |rel, found| {
                    RelKind.then(Inputs).each(rel, |i| found.push(i));
                    Ok::<_, AdError>(())
                },
                |rel, inputs: Vec<Vec<u64>>| columns(rel, inputs, &mut largest),
            )?;
            largest = columns.into_iter().fold(largest, u64::max);
        }
    }
    Ok(largest)
}

/// The size of each column of `rel`, given each of its inputs' columns';
/// `largest` raised to the largest it computes, an emit's dropped columns
/// included.
fn columns(rel: &Rel, mut inputs: Vec<Vec<u64>>, largest: &mut u64) -> Result<Vec<u64>> {
    let kind = rel
        .rel_type
        .as_ref()
        .ok_or_else(|| AdError::InvalidPlan("an empty relation".into()))?;
    let mut first = || -> Result<Vec<u64>> {
        (!inputs.is_empty())
            .then(|| inputs.remove(0))
            .ok_or_else(|| AdError::InvalidPlan("a relation with no input".into()))
    };
    let (direct, common) = match kind {
        // Its emit, if any, is in its width already.
        RelType::Read(_) => return Ok(vec![1; width(rel)?]),
        RelType::Filter(f) => (first()?, f.common.as_ref()),
        RelType::Sort(s) => (first()?, s.common.as_ref()),
        RelType::Fetch(f) => (first()?, f.common.as_ref()),
        // Each column as large as its largest input's.
        RelType::Set(s) => {
            let mut out = first()?;
            for input in inputs.drain(..) {
                for (o, n) in out.iter_mut().zip(input) {
                    *o = (*o).max(n);
                }
            }
            (out, s.common.as_ref())
        }
        RelType::Project(p) => {
            let mut out = first()?;
            let computed: Vec<u64> = p.expressions.iter().map(|e| size(e, &out)).collect();
            out.extend(computed);
            (out, p.common.as_ref())
        }
        RelType::Join(j) => {
            let (mut l, r) = (first()?, first()?);
            let out = match JoinType::try_from(j.r#type).unwrap_or(JoinType::Unspecified) {
                JoinType::LeftSemi | JoinType::LeftAnti => l,
                JoinType::RightSemi | JoinType::RightAnti => r,
                JoinType::LeftMark => [l, vec![1]].concat(),
                JoinType::RightMark => [r, vec![1]].concat(),
                _ => {
                    l.extend(r);
                    l
                }
            };
            (out, j.common.as_ref())
        }
        RelType::Cross(c) => ([first()?, first()?].concat(), c.common.as_ref()),
        RelType::Aggregate(a) => {
            let input = first()?;
            let mut out: Vec<u64> = grouping_expressions(a)?
                .into_iter()
                .map(|e| size(e, &input))
                .collect();
            for m in &a.measures {
                let args = match &m.measure {
                    Some(f) => value_args(&f.arguments)?,
                    None => Vec::new(),
                };
                out.push(
                    args.iter()
                        .fold(1, |n, e| n.saturating_add(size(e, &input))),
                );
            }
            (out, a.common.as_ref())
        }
        RelType::Window(w) => {
            let mut out = first()?;
            let over = w
                .partition_expressions
                .iter()
                .chain(w.sorts.iter().filter_map(|s| s.expr.as_ref()))
                .fold(0, |n: u64, e| n.saturating_add(size(e, &out)));
            let mut computed = Vec::new();
            for f in &w.window_functions {
                let args = value_args(&f.arguments)?;
                computed.push(args.iter().fold(over.saturating_add(1), |n, e| {
                    n.saturating_add(size(e, &out))
                }));
            }
            out.extend(computed);
            (out, w.common.as_ref())
        }
        other => {
            return Err(AdError::NotImplemented(format!(
                "the size of a {} relation",
                rel_name(other)
            )))
        }
    };
    *largest = direct.iter().copied().fold(*largest, u64::max);
    match common.and_then(|c| c.emit_kind.as_ref()) {
        Some(EmitKind::Emit(e)) => e
            .output_mapping
            .iter()
            .map(|&i| {
                usize::try_from(i)
                    .ok()
                    .and_then(|i| direct.get(i).copied())
                    .ok_or_else(|| AdError::InvalidPlan(format!("an emit of column {i}")))
            })
            .collect(),
        _ => Ok(direct),
    }
}

/// The size of `e` written out, each of its input's columns `columns` in
/// size: a reference counts as its column, every other node as 1.
fn size(e: &Expression, columns: &[u64]) -> u64 {
    let mut n: u64 = 0;
    Cosmos.each(e, |x| {
        let own = as_field(x)
            .and_then(|f| columns.get(f).copied())
            .unwrap_or(1);
        n = n.saturating_add(own);
    });
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{plan, project_emit, read_step};
    use crate::expr::{call, field, lit_f64};
    use crate::functions::{Extensions, Functions};

    #[test]
    fn a_chain_that_reads_each_column_twice_doubles_at_every_layer() {
        // v -> v * v + 1, ten times over, keeping only the new column.
        let mut rel = read_step("t", vec!["v".into()]);
        for _ in 0..10 {
            let square = call(0, vec![field(0), field(0)]);
            rel = project_emit(
                rel,
                vec![call(1, vec![square, lit_f64(1.0)])],
                Some(vec![1]),
            );
        }
        let p = plan(
            rel,
            vec!["v".into()],
            &Extensions::new(&Functions::default()),
        );
        // Each layer is 2·n + 3 for its input's n: the sum, the product,
        // the literal and two copies of the input.
        let n = (0..10).fold(1, |n, _| 2 * n + 3);
        assert_eq!(inlined_size(&p), Some(n));
    }

    #[test]
    fn a_read_s_emit_picks_its_columns_once() {
        let mut rel = read_step("t", vec!["a".into(), "b".into()]);
        if let Some(RelType::Read(r)) = rel.rel_type.as_mut() {
            r.common = Some(substrait::proto::RelCommon {
                emit_kind: Some(EmitKind::Emit(substrait::proto::rel_common::Emit {
                    output_mapping: vec![1],
                })),
                ..Default::default()
            });
        }
        let p = plan(
            project_emit(rel, vec![call(0, vec![field(0), field(0)])], Some(vec![1])),
            vec!["v".into()],
            &Extensions::new(&Functions::default()),
        );
        assert_eq!(inlined_size(&p), Some(3));
    }

    #[test]
    fn a_column_an_emit_drops_still_counts() {
        let rel = read_step("t", vec!["v".into()]);
        let big = (0..5).fold(field(0), |e, _| call(0, vec![e.clone(), e]));
        let p = plan(
            project_emit(rel, vec![big], Some(vec![0])),
            vec!["v".into()],
            &Extensions::new(&Functions::default()),
        );
        assert_eq!(inlined_size(&p), Some(63));
    }

    #[test]
    fn a_union_s_column_is_as_large_as_its_largest_input_s() {
        let small = read_step("t", vec!["v".into()]);
        let big = (0..3).fold(field(0), |e, _| call(0, vec![e.clone(), e]));
        let big = project_emit(read_step("t", vec!["v".into()]), vec![big], Some(vec![1]));
        let union = Rel {
            rel_type: Some(RelType::Set(substrait::proto::SetRel {
                inputs: vec![small, big],
                op: substrait::proto::set_rel::SetOp::UnionAll as i32,
                ..Default::default()
            })),
        };
        // A column over the union reads its second input's 15 nodes twice.
        let over = project_emit(
            union,
            vec![call(0, vec![field(0), field(0)])],
            Some(vec![1]),
        );
        let p = plan(
            over,
            vec!["v".into()],
            &Extensions::new(&Functions::default()),
        );
        assert_eq!(inlined_size(&p), Some(31));
    }

    #[test]
    fn a_relation_it_cannot_size_gives_no_size() {
        let reference = Rel {
            rel_type: Some(RelType::Reference(substrait::proto::ReferenceRel {
                subtree_ordinal: 0,
            })),
        };
        let p = plan(
            reference,
            vec!["v".into()],
            &Extensions::new(&Functions::default()),
        );
        assert_eq!(inlined_size(&p), None);
    }
}
