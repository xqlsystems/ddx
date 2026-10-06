// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Fusing a projection into the projection beneath it.
//!
//! The plans ddx writes stack projections: an engine's producer writes one
//! per `SELECT`, and ddx adds one to pick columns (an emit is only on a
//! projection, design.md §4.2), to mask a tangent, or to compute a window's
//! argument. Each is a level of nesting, and depth costs: datafusion-python's
//! consumer decodes a plan with protobuf's default limit of 100 nested
//! messages (two per relation), and a consumer recurses once per relation.
//!
//! A projection `P` over a projection `Q` is one projection over `Q`'s input
//! when one of them only picks columns (has no expressions), and the fused one
//! neither repeats work nor deepens an expression or a name:
//!
//! - DataFusion names a computed column by its expression, so two levels of
//!   computed columns keep the boundary between them, which keeps each name
//!   short (a tangent through stacked aggregates grew its names, and its
//!   plan, exponentially without it);
//! - each column `P` passes through bare may be any of `Q`'s, a computed one
//!   at most once, so nothing is computed twice;
//! - an expression of `P` may read only `Q`'s columns that are themselves
//!   bare columns of `Q`'s input, so no expression grows;
//! - no two of the fused projection's expressions are equal;
//! - no window function moves. DataFusion's consumer names a window by its
//!   text, column names included, and refuses two of one name in one
//!   projection: moved onto another input, two windows over two columns of
//!   one name become one name.
//!
//! Computing an expression of `Q` inside one of `P`'s would make the plan
//! shallower only by making that expression deeper, and protobuf counts that
//! nesting too. A projection with anything in its `common` beyond an emit (a
//! hint, an extension) is left as it is.

use substrait::proto::expression::RexType;
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::EmitKind;
use substrait::proto::{Expression, Plan, ProjectRel, Rel, RelCommon};

use crate::emit::project_emit;
use crate::expr::{as_field, contains, field, fields_of, map_fields};
use crate::forward::{rel_inputs_mut, width};

/// Fuse every projection of `plan` into the projections beneath it, where
/// that is free (see the module docs).
pub(crate) fn fuse_plan(plan: &mut Plan) {
    for r in plan.relations.iter_mut() {
        if let Some(substrait::proto::plan_rel::RelType::Root(root)) = r.rel_type.as_mut() {
            if let Some(input) = root.input.as_mut() {
                fuse(input);
            }
        }
    }
}

/// [`fuse_plan`] on one tree. A loop, not recursion: a chain of relations
/// can be hundreds deep.
fn fuse(rel: &mut Rel) {
    let mut stack = vec![rel];
    while let Some(r) = stack.pop() {
        while let Some(fused) = fused(r) {
            *r = fused;
        }
        if let Some(kind) = r.rel_type.as_mut() {
            stack.extend(rel_inputs_mut(kind));
        }
    }
}

/// `rel` as one projection, if it is a projection over a projection that can
/// be fused into it.
fn fused(rel: &Rel) -> Option<Rel> {
    let p = plain_project(rel)?;
    let below = p.input.as_deref()?;
    let q = plain_project(below)?;
    // One of the two only picks columns: each boundary between two levels
    // of computed columns stays, since DataFusion names a computed column
    // by its expression, and a boundary is what keeps a name short.
    if !p.expressions.is_empty() && !q.expressions.is_empty() {
        return None;
    }
    let input = q.input.as_deref()?;
    let w = width(input).ok()?;
    let lower = outputs(q, w)?;
    let upper = outputs(p, lower.len())?;
    // Each column of `Q`'s output, as a bare column of `Q`'s input if it is
    // one.
    let bare: Vec<Option<usize>> = lower.iter().map(as_field).collect();
    let mut passed = vec![0usize; lower.len()];
    let mut out = Vec::with_capacity(upper.len());
    let window = |e: &Expression| matches!(e.rex_type, Some(RexType::WindowFunction(_)));
    for e in &upper {
        if let Some(k) = as_field(e) {
            passed[k] += 1;
            if bare[k].is_none() && passed[k] > 1 {
                return None;
            }
            if contains(&lower[k], &window) {
                return None;
            }
            out.push(lower[k].clone());
        } else {
            if contains(e, &window) {
                return None;
            }
            if fields_of(e).ok()?.iter().any(|&k| bare[k].is_none()) {
                return None;
            }
            out.push(map_fields(e, &mut |k| Ok(bare[k].expect("checked"))).ok()?);
        }
    }
    // Bare columns of the input stay direct; everything else is an
    // expression.
    let mut expressions = Vec::new();
    let mut emit = Vec::with_capacity(out.len());
    for e in out {
        match as_field(&e) {
            Some(c) => emit.push(c),
            None => {
                emit.push(w + expressions.len());
                expressions.push(e);
            }
        }
    }
    // Two equal expressions compute one value twice.
    let mut seen = std::collections::HashSet::new();
    if !expressions
        .iter()
        .all(|e| seen.insert(prost::Message::encode_to_vec(e)))
    {
        return None;
    }
    let identity =
        emit.len() == w + expressions.len() && emit.iter().enumerate().all(|(i, &c)| i == c);
    Some(project_emit(
        input.clone(),
        expressions,
        (!identity).then_some(emit),
    ))
}

/// `rel` as a projection with nothing in its `common` but an emit.
fn plain_project(rel: &Rel) -> Option<&ProjectRel> {
    let Some(RelType::Project(p)) = &rel.rel_type else {
        return None;
    };
    let plain = |c: &RelCommon| {
        c.hint.is_none()
            && c.advanced_extension.is_none()
            && matches!(
                c.emit_kind,
                None | Some(EmitKind::Direct(_)) | Some(EmitKind::Emit(_))
            )
    };
    (p.advanced_extension.is_none() && p.common.as_ref().is_none_or(plain)).then_some(p)
}

/// A projection's output columns, each as an expression over its input
/// (`width` columns wide): an input column as a bare field.
fn outputs(p: &ProjectRel, width: usize) -> Option<Vec<Expression>> {
    let direct: Vec<Expression> = (0..width)
        .map(field)
        .chain(p.expressions.iter().cloned())
        .collect();
    // A subquery's references are not all to this row; it stays where it is.
    let subquery = |e: &Expression| matches!(e.rex_type, Some(RexType::Subquery(_)));
    if direct.iter().any(|e| contains(e, &subquery)) {
        return None;
    }
    match p.common.as_ref().and_then(|c| c.emit_kind.as_ref()) {
        Some(EmitKind::Emit(e)) => e
            .output_mapping
            .iter()
            .map(|&i| usize::try_from(i).ok().and_then(|i| direct.get(i).cloned()))
            .collect(),
        _ => Some(direct),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{plan, read_table, select};
    use crate::expr::{call, lit_f64};
    use crate::functions::{Extensions, Functions};
    use substrait::proto::r#type::{Fp64, Kind, Nullability, Struct};
    use substrait::proto::{NamedStruct, Type};

    fn t() -> Rel {
        let fp = Type {
            kind: Some(Kind::Fp64(Fp64 {
                type_variation_reference: 0,
                nullability: Nullability::Nullable as i32,
            })),
        };
        read_table(
            vec!["t".into()],
            NamedStruct {
                names: vec!["a".into(), "b".into(), "c".into()],
                r#struct: Some(Struct {
                    types: vec![fp.clone(), fp.clone(), fp],
                    type_variation_reference: 0,
                    nullability: Nullability::Required as i32,
                }),
            },
        )
    }

    fn depth(rel: &Rel) -> usize {
        let mut d = 0;
        let mut cur = Some(rel);
        while let Some(r) = cur {
            d += 1;
            cur = match &r.rel_type {
                Some(RelType::Project(p)) => p.input.as_deref(),
                _ => None,
            };
        }
        d
    }

    #[test]
    fn a_select_over_a_projection_is_one_projection() {
        let mut ext = Extensions::new(&Functions::default());
        let add = ext.anchor("add");
        // a + b, then pick (c, a + b).
        let q = project_emit(t(), vec![call(add, vec![field(0), field(1)])], None);
        let p = select(q, vec![2, 3]);
        let mut pl = plan(p, vec!["c".into(), "s".into()], &ext);
        fuse_plan(&mut pl);
        let root = crate::forward::root_of(&pl).unwrap().0.clone();
        assert_eq!(depth(&root), 2, "{root:?}");
        let Some(RelType::Project(p)) = &root.rel_type else {
            panic!()
        };
        assert_eq!(p.expressions.len(), 1);
        assert_eq!(width(&root).unwrap(), 2);
    }

    #[test]
    fn a_computed_column_read_inside_an_expression_is_not_inlined() {
        let mut ext = Extensions::new(&Functions::default());
        let add = ext.anchor("add");
        let mul = ext.anchor("multiply");
        let q = project_emit(t(), vec![call(add, vec![field(0), field(1)])], None);
        // (a + b) * 2 would deepen the expression: left as two projections.
        let p = project_emit(
            q,
            vec![call(mul, vec![field(3), lit_f64(2.0)])],
            Some(vec![4]),
        );
        let mut pl = plan(p, vec!["x".into()], &ext);
        fuse_plan(&mut pl);
        let root = crate::forward::root_of(&pl).unwrap().0.clone();
        assert_eq!(depth(&root), 3);
    }

    #[test]
    fn a_computed_column_passed_through_twice_is_not_computed_twice() {
        let mut ext = Extensions::new(&Functions::default());
        let add = ext.anchor("add");
        let q = project_emit(t(), vec![call(add, vec![field(0), field(1)])], None);
        let p = select(q, vec![3, 3]);
        let mut pl = plan(p, vec!["s".into(), "s2".into()], &ext);
        fuse_plan(&mut pl);
        let root = crate::forward::root_of(&pl).unwrap().0.clone();
        assert_eq!(depth(&root), 3);
    }

    #[test]
    fn an_expression_over_bare_columns_moves_down_and_a_chain_folds() {
        let mut ext = Extensions::new(&Functions::default());
        let mul = ext.anchor("multiply");
        // Three stacked picks and one expression over a picked bare column.
        let r = select(t(), vec![2, 0]);
        let r = select(r, vec![1, 0]);
        let r = project_emit(
            r,
            vec![call(mul, vec![field(1), lit_f64(2.0)])],
            Some(vec![0, 2]),
        );
        let mut pl = plan(r, vec!["a".into(), "c2".into()], &ext);
        fuse_plan(&mut pl);
        let root = crate::forward::root_of(&pl).unwrap().0.clone();
        assert_eq!(depth(&root), 2, "{root:?}");
        let Some(RelType::Project(p)) = &root.rel_type else {
            panic!()
        };
        // c * 2, over the read's column 2.
        assert_eq!(fields_of(&p.expressions[0]).unwrap(), vec![2]);
    }
}
