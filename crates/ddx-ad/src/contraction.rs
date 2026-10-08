// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The transpose of a contraction, without rebuilding its join.
//!
//! A saved `SUM(e_X · e_Y) … GROUP BY g` over `X JOIN Y ON x.a = y.b`, where
//! `e_X` reads only `X` and `e_Y` only `Y`, is a contraction: a matrix
//! product when `X` and `Y` are matrices. The general reduce rule rebuilds
//! the region `X ⋈ Y` (for a matrix product, `N·D·H` rows), joins it to the
//! cotangent, and sums by each input's dims. For an `X` row `x`, though,
//!
//! ```text
//! Σ over the rows x joined to: c̄ · ∂(e_X · e_Y)/∂x  =  ∂e_X/∂x (x) · Σ_y c̄ · e_Y (y)
//! ```
//!
//! and the sum on the right is a contraction of the cotangent with `Y` alone,
//! as `jax.grad` differentiates a matrix product: `Ā = C̄·Bᵀ`. So `X`'s
//! contribution is
//!
//! ```sql
//! SELECT x.dims, SUM(CASE WHEN e_X IS NULL THEN NULL ELSE t.v END * ∂e_X/∂x)
//! FROM X x JOIN (SELECT y.b, c.g_X, SUM(c.v * e_Y) AS v
//!                FROM Y y JOIN cotangent c ON y.g_Y IS NOT DISTINCT FROM c.g_Y
//!                WHERE y.b IS NOT NULL
//!                GROUP BY y.b, c.g_X) t
//!   ON x.a IS NOT DISTINCT FROM t.b AND x.g_X IS NOT DISTINCT FROM t.g_X
//! GROUP BY x.dims
//! ```
//!
//! which never forms `X ⋈ Y`. (For a key the query joined with `=`, `t`
//! leaves out the NULL keys that `=` never matched, so that every key of the
//! join with `x` can be null-safe, which DataFusion hashes; a key joined with
//! IS NOT DISTINCT FROM keeps them.)
//!
//! **NULLs.** The general rule gives a summed row no cotangent when the
//! aggregate's argument is NULL, since `SUM` skipped it. Here the argument
//! is `e_X · e_Y`, NULL exactly when a factor is. A NULL `e_Y` makes its
//! term `c̄ · e_Y` NULL, which the inner `SUM` skips, as before. A NULL `e_X`
//! makes every row of that `x` NULL, so its contribution is NULL, as before:
//! that is the `CASE`, tested once per `x` instead of once per joined row.
//! Duplicate rows count as before: each copy of a `Y` row adds its term to
//! `t`, and each copy of an `X` row joins `t` and adds to its group.
//!
//! **When it applies.** Exactly one measure has a cotangent, and it is a
//! `SUM`; the region is projections over one inner join, with no filter
//! above it, no post-join filter, no `LIMIT`, no volatile function (a
//! ranking inside a side is checked as the general rule checks it); each side of the join reads one input; the join condition is a
//! conjunction of `=` or `IS NOT DISTINCT FROM` between a column of each
//! side; every grouping key is a column of one side; and the argument,
//! through the projections above the join, is a product of an expression of
//! one side's columns and one of the other's. Otherwise the general rule
//! applies.

use std::collections::BTreeMap;

use substrait::proto::expression::RexType;
use substrait::proto::join_rel::JoinType;
use substrait::proto::rel::RelType;
use substrait::proto::{Expression, Rel};

use crate::emit::{aggregate, filter, join};
use crate::error::Result;
use crate::expr::{as_field, call, field, fields_of, if_then, null_f64, scalar_args};
use crate::forward::{Def, Input, Output, Region, Saved};
use crate::transpose::{Contribution, Transposer};

/// One side of the join: the subtree that computes region columns
/// `start..start + width`, holding one input.
struct Side {
    rel: Rel,
    start: usize,
    width: usize,
    slot: usize,
}

/// A contraction's shape, in each side's own column numbers.
struct Shape {
    sides: [Side; 2],
    /// The join condition: `(left column, right column, is_not_distinct_from)`.
    keys: Vec<(usize, usize, bool)>,
    /// Each grouping key: its side and column.
    groups: Vec<(usize, usize)>,
    /// The factors `e_X` and `e_Y`, over each side's columns.
    factors: [Expression; 2],
}

impl Transposer<'_> {
    /// The transpose of saved aggregate `n` by the contraction rule, if its
    /// shape allows (see the module docs). `measure` is the one measure with
    /// a cotangent, which is column `cot_col` of `cotangent`.
    ///
    /// Returns `false`, having changed nothing, if the shape doesn't allow.
    pub(crate) fn contraction(
        &mut self,
        n: usize,
        measure: usize,
        cotangent: &Rel,
        cot_col: usize,
    ) -> Result<bool> {
        let f = self.f;
        let saved = &f.saved[n];
        let Some(shape) = self.shape(saved, measure)? else {
            return Ok(false);
        };
        // Each side is recomputed, as the general rule recomputes the whole
        // region, so its rankings must be total for the same reason.
        self.check_rankings_are_total(&saved.input)?;
        // Each varied side's contribution, from the other side's factor.
        let mut made = Vec::new();
        for x in 0..2 {
            let side = &shape.sides[x];
            let region = &saved.input;
            if !region.varied[side.start..side.start + side.width]
                .iter()
                .any(|&v| v)
            {
                continue;
            }
            let Some(c) = self.side_contribution(region, &shape, x, cotangent, cot_col)? else {
                return Ok(false);
            };
            made.push((region.slots[side.slot].input, c));
        }
        for (input, c) in made {
            self.contributions.entry(input).or_default().push(c);
        }
        Ok(true)
    }

    /// The contraction's shape, or `None` if the region is not one.
    fn shape(&mut self, saved: &Saved, measure: usize) -> Result<Option<Shape>> {
        let region = &saved.input;
        if region.volatile || !region.cuts.is_empty() || region.slots.len() != 2 {
            return Ok(None);
        }
        let Some(arg) = sum_argument(&self.f.functions, &saved.measures[measure])? else {
            return Ok(None);
        };
        // Walk down the projections to the join.
        let mut rel = &region.rel;
        loop {
            match &rel.rel_type {
                Some(RelType::Project(p)) => match p.input.as_deref() {
                    Some(r) => rel = r,
                    None => return Ok(None),
                },
                Some(RelType::Join(_)) => break,
                _ => return Ok(None),
            }
        }
        let Some(RelType::Join(j)) = &rel.rel_type else {
            return Ok(None);
        };
        if j.r#type != JoinType::Inner as i32 || j.post_join_filter.is_some() {
            return Ok(None);
        }
        let (Some(left), Some(right)) = (j.left.as_deref(), j.right.as_deref()) else {
            return Ok(None);
        };
        let lw = crate::forward::width(left)?;
        let rw = crate::forward::width(right)?;
        // Each side must hold exactly one input, wholly inside it.
        let slot_in = |start: usize, width: usize| -> Option<usize> {
            let mut found = None;
            for (i, s) in region.slots.iter().enumerate() {
                let off = s.offset?;
                if off >= start && off + s.width <= start + width {
                    if found.is_some() {
                        return None;
                    }
                    found = Some(i);
                }
            }
            found
        };
        let (Some(ls), Some(rs)) = (slot_in(0, lw), slot_in(lw, rw)) else {
            return Ok(None);
        };
        if ls == rs {
            return Ok(None);
        }
        let side_of = |c: usize| -> Option<(usize, usize)> {
            if c < lw {
                Some((0, c))
            } else if c < lw + rw {
                Some((1, c - lw))
            } else {
                None
            }
        };
        // Every column above the join is an expression.
        if region.defs[lw + rw..]
            .iter()
            .any(|d| !matches!(d, Def::Expr(_)))
        {
            return Ok(None);
        }
        // The join condition: equalities between the sides.
        let Some(cond) = j.expression.as_deref() else {
            return Ok(None);
        };
        let mut keys = Vec::new();
        for conjunct in self.conjuncts(cond)? {
            let Some((name, a, b)) = self.equality(&conjunct)? else {
                return Ok(None);
            };
            let (Some((sa, ca)), Some((sb, cb))) = (side_of(a), side_of(b)) else {
                return Ok(None);
            };
            match (sa, sb) {
                (0, 1) => keys.push((ca, cb, name)),
                (1, 0) => keys.push((cb, ca, name)),
                _ => return Ok(None),
            }
        }
        if keys.is_empty() {
            return Ok(None);
        }
        // Grouping keys, in the cotangent's order (the saved relation's dims):
        // columns of one side, through the projections.
        let mut groups = Vec::new();
        for d in saved.dims() {
            let Output::Dim(gi) = saved.outputs[d] else {
                return Ok(None);
            };
            let Some(e) = inline_above(region, &saved.groupings[gi], lw + rw) else {
                return Ok(None);
            };
            let Some(c) = as_field(&e) else {
                return Ok(None);
            };
            let Some(sc) = side_of(c) else {
                return Ok(None);
            };
            groups.push(sc);
        }
        // The argument: a product of one factor from each side.
        let Some(arg) = inline_above(region, &arg, lw + rw) else {
            return Ok(None);
        };
        let Some(RexType::ScalarFunction(f)) = &arg.rex_type else {
            return Ok(None);
        };
        if self.f.functions.name(f.function_reference)? != "multiply" {
            return Ok(None);
        }
        let [p, q] = scalar_args(f)?[..] else {
            return Ok(None);
        };
        let side_only = |e: &Expression| -> Result<Option<usize>> {
            let fields = fields_of(e)?;
            let mut side = None;
            for c in fields {
                let Some((s, _)) = side_of(c) else {
                    return Ok(None);
                };
                if side.is_some_and(|t| t != s) {
                    return Ok(None);
                }
                side = Some(s);
            }
            Ok(side)
        };
        let (Some(sp), Some(sq)) = (side_only(p)?, side_only(q)?) else {
            return Ok(None);
        };
        if sp == sq {
            return Ok(None);
        }
        let local =
            |e: &Expression, start: usize| crate::expr::map_fields(e, &mut |c| Ok(c - start));
        let (fl, fr) = if sp == 0 { (p, q) } else { (q, p) };
        let factors = [local(fl, 0)?, local(fr, lw)?];
        Ok(Some(Shape {
            sides: [
                Side {
                    rel: left.clone(),
                    start: 0,
                    width: lw,
                    slot: ls,
                },
                Side {
                    rel: right.clone(),
                    start: lw,
                    width: rw,
                    slot: rs,
                },
            ],
            keys,
            groups,
            factors,
        }))
    }

    /// Side `x`'s contribution: see the module docs. `None` if the gradient
    /// can't be propagated to its input this way.
    fn side_contribution(
        &mut self,
        region: &Region,
        shape: &Shape,
        x: usize,
        cotangent: &Rel,
        cot_col: usize,
    ) -> Result<Option<Contribution>> {
        let y = 1 - x;
        let (xs, ys) = (&shape.sides[x], &shape.sides[y]);
        let same = self.ext.anchor("is_not_distinct_from");
        let is_not_null = self.ext.anchor("is_not_null");
        let and = self.ext.anchor("and");
        let mul = self.ext.anchor("multiply");
        let sum = self.ext.anchor("sum");
        let is_null = self.ext.anchor("is_null");
        let all = |conds: Vec<Expression>| conds.into_iter().reduce(|a, b| call(and, vec![a, b]));

        // The factor of side x, over x's input columns, and its partials.
        let (inputs_x, e_x) = match inline_side(region, xs, &shape.factors[x])? {
            Some(v) => v,
            None => return Ok(None),
        };
        let varied = |c: usize| inputs_x.get(&c).is_some_and(|&rc| region.varied[rc]);
        let partials = self.ew.partials(&e_x, &varied, &mut self.ext)?;
        if partials.is_empty() {
            return Ok(None);
        }

        // t: Y joined to the cotangent on Y's grouping keys, summed by Y's
        // join keys and the cotangent's X-side grouping keys.
        let ndims = shape.groups.len();
        let ywidth = ys.width;
        let on_y: Vec<Expression> = shape
            .groups
            .iter()
            .enumerate()
            .filter(|(_, (s, _))| *s == y)
            .map(|(g, &(_, c))| call(same, vec![field(c), field(ywidth + g)]))
            .collect();
        // With no grouping key on Y's side, Y would meet every cotangent row:
        // a cross join, which the general rule handles instead.
        let Some(on_y) = all(on_y) else {
            return Ok(None);
        };
        let y_keys: Vec<usize> = shape
            .keys
            .iter()
            .map(|&(l, r, _)| if y == 0 { l } else { r })
            .collect();
        // Every key of the join with x below is null-safe, so DataFusion can
        // hash it (it can't hash a join that mixes `=` with IS NOT DISTINCT
        // FROM, and falls back to a nested loop). For a key that was `=`, a
        // NULL never matched, so Y's rows with a NULL there joined nothing:
        // leaving them out of t keeps the meaning.
        let not_null: Vec<Expression> = shape
            .keys
            .iter()
            .zip(&y_keys)
            .filter(|((_, _, not_distinct), _)| !not_distinct)
            .map(|(_, &c)| call(is_not_null, vec![field(c)]))
            .collect();
        let y_rel = match all(not_null) {
            Some(cond) => filter(ys.rel.clone(), cond),
            None => ys.rel.clone(),
        };
        let yc = join(y_rel, cotangent.clone(), on_y, JoinType::Inner);
        let x_groups: Vec<usize> = (0..ndims).filter(|&g| shape.groups[g].0 == x).collect();
        let mut t_keys: Vec<Expression> = y_keys.iter().map(|&c| field(c)).collect();
        t_keys.extend(x_groups.iter().map(|&g| field(ywidth + g)));
        let term = call(
            mul,
            vec![field(ywidth + ndims + cot_col), shape.factors[y].clone()],
        );
        let t = aggregate(yc, t_keys, vec![(sum, vec![term])]);
        let t_value = y_keys.len() + x_groups.len();

        // x joined to t on the join keys and x's grouping keys.
        let xwidth = xs.width;
        let mut on_x = Vec::new();
        for (k, &(l, r, _)) in shape.keys.iter().enumerate() {
            let xc = if x == 0 { l } else { r };
            on_x.push(call(same, vec![field(xc), field(xwidth + k)]));
        }
        for (i, &g) in x_groups.iter().enumerate() {
            let xc = shape.groups[g].1;
            on_x.push(call(
                same,
                vec![field(xc), field(xwidth + y_keys.len() + i)],
            ));
        }
        let xt = join(
            xs.rel.clone(),
            t,
            all(on_x).expect("a contraction has a join key"),
            JoinType::Inner,
        );
        // A NULL factor of x leaves its rows nothing to pass on: the guard
        // the general rule applies to each joined row, once per row of x.
        let guarded = if_then(
            vec![(call(is_null, vec![e_x.clone()]), null_f64())],
            field(xwidth + t_value),
        );
        let slot = &region.slots[xs.slot];
        let offset = slot.offset.expect("a side's input is in the output") - xs.start;
        let dims = match slot.input {
            Input::Table(t) => self.f.tables[t].dims.clone(),
            Input::Saved(s) => self.f.saved[s].dims(),
            Input::Const => return Ok(None),
        };
        let groupings: Vec<Expression> = dims.iter().map(|d| field(offset + d)).collect();
        let mut cols = Vec::new();
        let mut measures = Vec::new();
        for (c, d) in partials {
            // `c` is a column of side x holding input column `c - offset`.
            cols.push(c - offset);
            measures.push((sum, vec![call(mul, vec![guarded.clone(), d])]));
        }
        Ok(Some(Contribution {
            rel: aggregate(xt, groupings, measures),
            cols,
        }))
    }

    /// The conjuncts of `e`, flattening `and`.
    fn conjuncts(&self, e: &Expression) -> Result<Vec<Expression>> {
        if let Some(RexType::ScalarFunction(f)) = &e.rex_type {
            if self.f.functions.name(f.function_reference)? == "and" {
                let mut out = Vec::new();
                for a in scalar_args(f)? {
                    out.extend(self.conjuncts(a)?);
                }
                return Ok(out);
            }
        }
        Ok(vec![e.clone()])
    }

    /// `(is_not_distinct_from, a, b)` if `e` is `field(a) = field(b)` or `field(a) IS
    /// NOT DISTINCT FROM field(b)`.
    fn equality(&self, e: &Expression) -> Result<Option<(bool, usize, usize)>> {
        let Some(RexType::ScalarFunction(f)) = &e.rex_type else {
            return Ok(None);
        };
        let not_distinct = match self.f.functions.name(f.function_reference)? {
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
}

/// `e`, a factor over side `s`'s columns, rewritten over the side's input
/// columns only (through the side's own projections), with a map from
/// each input column it reads (in side numbering) to its region column.
/// `None` if it reads a column that isn't an input or an expression, or
/// one whose gradient ddx refuses.
fn inline_side(
    region: &Region,
    s: &Side,
    e: &Expression,
) -> Result<Option<(BTreeMap<usize, usize>, Expression)>> {
    let mut inputs = BTreeMap::new();
    let Some(out) = inline(e, &mut |c| {
        let rc = s.start + c;
        if region.refusals[rc].is_some() {
            return None;
        }
        match &region.defs[rc] {
            Def::Input { slot, .. } if *slot == s.slot => {
                inputs.insert(c, rc);
                Some(field(c))
            }
            Def::Expr(inner) => {
                let local = crate::expr::map_fields(inner, &mut |f| Ok(f - s.start)).ok()?;
                let (more, x) = inline_side(region, s, &local).ok().flatten()?;
                inputs.extend(more);
                Some(x)
            }
            _ => None,
        }
    })?
    else {
        return Ok(None);
    };
    Ok(Some((inputs, out)))
}

/// The argument of `SUM(argument)`, or `None` for any other aggregate.
fn sum_argument(
    functions: &crate::functions::Functions,
    m: &substrait::proto::AggregateFunction,
) -> Result<Option<Expression>> {
    use substrait::proto::aggregate_function::AggregationInvocation;
    if functions.name(m.function_reference)? != "sum"
        || m.invocation == AggregationInvocation::Distinct as i32
    {
        return Ok(None);
    }
    let args = crate::expr::value_args(&m.arguments)?;
    Ok(match args[..] {
        [a] => Some(a.clone()),
        _ => None,
    })
}

/// `e` with every column at or above `above` (an expression defined above
/// the join) replaced by its definition, recursively. `None` if one isn't an
/// expression, or `e` holds an expression kind `inline` doesn't follow.
fn inline_above(region: &Region, e: &Expression, above: usize) -> Option<Expression> {
    inline(e, &mut |c| {
        if c < above {
            return Some(field(c));
        }
        match &region.defs[c] {
            Def::Expr(inner) => inline_above(region, inner, above),
            _ => None,
        }
    })
    .ok()
    .flatten()
}

/// `e` with each column reference replaced by `f(column)`; `None` if `f`
/// declines one, or `e` holds an expression other than a literal, a column, a
/// scalar function, a `CASE` or a cast.
fn inline(
    e: &Expression,
    f: &mut dyn FnMut(usize) -> Option<Expression>,
) -> Result<Option<Expression>> {
    use substrait::proto::function_argument::ArgType;
    let Some(rex) = &e.rex_type else {
        return Ok(None);
    };
    let mut out = e.clone();
    match (rex, out.rex_type.as_mut().expect("cloned")) {
        (RexType::Literal(_), _) => {}
        (RexType::Selection(_), _) => match as_field(e) {
            Some(c) => match f(c) {
                Some(x) => return Ok(Some(x)),
                None => return Ok(None),
            },
            None => return Ok(None),
        },
        (RexType::ScalarFunction(_), RexType::ScalarFunction(s)) => {
            #[allow(deprecated)]
            if !s.args.is_empty() {
                return Ok(None);
            }
            for a in s.arguments.iter_mut() {
                match a.arg_type.as_mut() {
                    Some(ArgType::Value(v)) => match inline(v, f)? {
                        Some(x) => *v = x,
                        None => return Ok(None),
                    },
                    _ => return Ok(None),
                }
            }
        }
        (RexType::IfThen(_), RexType::IfThen(it)) => {
            for c in it.ifs.iter_mut() {
                for x in [c.r#if.as_mut(), c.then.as_mut()].into_iter().flatten() {
                    match inline(x, f)? {
                        Some(y) => *x = y,
                        None => return Ok(None),
                    }
                }
            }
            if let Some(x) = it.r#else.as_deref_mut() {
                match inline(x, f)? {
                    Some(y) => *x = y,
                    None => return Ok(None),
                }
            }
        }
        (RexType::Cast(_), RexType::Cast(c)) => match c.input.as_deref_mut() {
            Some(x) => match inline(x, f)? {
                Some(y) => *x = y,
                None => return Ok(None),
            },
            None => return Ok(None),
        },
        _ => return Ok(None),
    }
    Ok(Some(out))
}
