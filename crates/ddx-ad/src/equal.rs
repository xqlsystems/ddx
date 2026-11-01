// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Columns a region's rows hold equal: what its joins' and filters' `=`
//! and IS NOT DISTINCT FROM say, transitively.

use substrait::proto::expression::RexType;
use substrait::proto::join_rel::JoinType;
use substrait::proto::rel::RelType;
use substrait::proto::{Expression, Rel};

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
/// append columns or drop rows.
pub(crate) fn held_equal(
    functions: &Functions,
    rel: &Rel,
    start: usize,
    held: &mut Classes,
) -> Result<()> {
    let mut conds = Vec::new();
    match &rel.rel_type {
        Some(RelType::Join(j)) if j.r#type == JoinType::Inner as i32 => {
            let (Some(l), Some(r)) = (j.left.as_deref(), j.right.as_deref()) else {
                return Ok(());
            };
            conds.extend(j.expression.as_deref());
            conds.extend(j.post_join_filter.as_deref());
            held_equal(functions, l, start, held)?;
            held_equal(functions, r, start + width(l)?, held)?;
        }
        Some(RelType::Filter(f)) => {
            conds.extend(f.condition.as_deref());
            if let Some(input) = f.input.as_deref() {
                held_equal(functions, input, start, held)?;
            }
        }
        Some(RelType::Project(p)) if p.common.is_none() => {
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
