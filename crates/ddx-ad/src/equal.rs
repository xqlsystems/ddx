// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Columns a region's rows hold equal: what its joins' and filters' `=`
//! and IS NOT DISTINCT FROM say, transitively.

use substrait::proto::expression::RexType;
use substrait::proto::join_rel::JoinType;
use substrait::proto::rel::RelType;
use substrait::proto::{Expression, Rel, RelCommon};

use crate::error::Result;
use crate::expr::{as_field, scalar_args};
use crate::forward::width;
use crate::functions::Functions;

/// Disjoint sets of `0..n`, joined a pair at a time: a union-find.
pub(crate) struct Classes(Vec<usize>);

impl Classes {
    pub(crate) fn new(n: usize) -> Classes {
        Classes((0..n).collect())
    }

    /// The root of `a`'s set, which names it.
    pub(crate) fn root(&mut self, a: usize) -> usize {
        let mut r = a;
        while self.0[r] != r {
            r = self.0[r];
        }
        self.0[a] = r;
        r
    }

    pub(crate) fn join(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.root(a), self.root(b));
        self.0[ra] = rb;
    }

    pub(crate) fn same(&mut self, a: usize, b: usize) -> bool {
        self.root(a) == self.root(b)
    }
}

/// Join in `held` each two columns of `rel` (whose columns start at region
/// column `start`) that every row of it holds equal, by an `=` or IS NOT
/// DISTINCT FROM of an inner join or a filter, beneath relations that only
/// append columns or drop rows. A relation with an emit, whose conditions
/// name columns it may drop or reorder, adds nothing.
pub(crate) fn held_equal(
    functions: &Functions,
    rel: &Rel,
    start: usize,
    held: &mut Classes,
) -> Result<()> {
    let mut conds = Vec::new();
    match &rel.rel_type {
        Some(RelType::Join(j)) if j.r#type == JoinType::Inner as i32 && !emits(&j.common) => {
            let (Some(l), Some(r)) = (j.left.as_deref(), j.right.as_deref()) else {
                return Ok(());
            };
            conds.extend(j.expression.as_deref());
            conds.extend(j.post_join_filter.as_deref());
            held_equal(functions, l, start, held)?;
            held_equal(functions, r, start + width(l)?, held)?;
        }
        Some(RelType::Filter(f)) if !emits(&f.common) => {
            conds.extend(f.condition.as_deref());
            if let Some(input) = f.input.as_deref() {
                held_equal(functions, input, start, held)?;
            }
        }
        Some(RelType::Project(p)) if !emits(&p.common) => {
            if let Some(input) = p.input.as_deref() {
                held_equal(functions, input, start, held)?;
            }
        }
        _ => {}
    }
    for c in conds {
        for conjunct in conjuncts(functions, c)? {
            if let Some((_, a, b)) = equality(functions, &conjunct)? {
                held.join(start + a, start + b);
            }
        }
    }
    Ok(())
}

/// Does a relation with this `common` emit, dropping or reordering columns?
fn emits(common: &Option<RelCommon>) -> bool {
    common.as_ref().is_some_and(|c| c.emit_kind.is_some())
}

/// The conjuncts of `e`, flattening `and`.
pub(crate) fn conjuncts(functions: &Functions, e: &Expression) -> Result<Vec<Expression>> {
    if let Some(RexType::ScalarFunction(f)) = &e.rex_type {
        if functions.name(f.function_reference)? == "and" {
            let mut out = Vec::new();
            for a in scalar_args(f)? {
                out.extend(conjuncts(functions, a)?);
            }
            return Ok(out);
        }
    }
    Ok(vec![e.clone()])
}

/// `(is_not_distinct_from, a, b)` if `e` is `field(a) = field(b)` or
/// `field(a) IS NOT DISTINCT FROM field(b)`.
pub(crate) fn equality(
    functions: &Functions,
    e: &Expression,
) -> Result<Option<(bool, usize, usize)>> {
    let Some(RexType::ScalarFunction(f)) = &e.rex_type else {
        return Ok(None);
    };
    let not_distinct = match functions.name(f.function_reference)? {
        "equal" => false,
        "is_not_distinct_from" => true,
        _ => return Ok(None),
    };
    let [a, b] = scalar_args(f)?[..] else {
        return Ok(None);
    };
    Ok(match (as_field(a), as_field(b)) {
        (Some(a), Some(b)) => Some((not_distinct, a, b)),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{filter, join, plan, read_step};
    use crate::expr::{call, field};
    use crate::forward::root_of;
    use crate::functions::Extensions;
    use substrait::proto::rel_common::{Emit, EmitKind};

    #[test]
    fn a_relation_with_an_emit_holds_nothing_equal() {
        // Its condition names columns its emit may drop or reorder, so read
        // as region columns they may name the wrong ones, or none.
        let mut ext = Extensions::new(&Functions::default());
        let eq = ext.anchor_named("equal");
        let emitting = |mut rel: Rel| {
            let common = match &mut rel.rel_type {
                Some(RelType::Join(j)) => &mut j.common,
                Some(RelType::Filter(f)) => &mut f.common,
                _ => unreachable!(),
            };
            *common = Some(RelCommon {
                emit_kind: Some(EmitKind::Emit(Emit {
                    output_mapping: vec![1],
                })),
                ..Default::default()
            });
            rel
        };
        let a = || read_step("a", vec!["i".into(), "x".into()]);
        let b = || read_step("b", vec!["j".into()]);
        let joined = join(
            a(),
            b(),
            call(eq, vec![field(0), field(2)]),
            JoinType::Inner,
        );
        let filtered = filter(a(), call(eq, vec![field(0), field(1)]));
        for rel in [emitting(joined), emitting(filtered)] {
            let p = plan(rel, vec!["c".into()], &ext);
            let functions = Functions::from_plan(&p).unwrap();
            let (root, _) = root_of(&p).unwrap();
            let mut held = Classes::new(1);
            held_equal(&functions, root, 0, &mut held).unwrap();
        }
    }
}
