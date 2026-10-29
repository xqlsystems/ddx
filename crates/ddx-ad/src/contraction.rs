// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The transpose of a contraction, without rebuilding its join.
//!
//! A saved `SUM(e_X · e_Y) … GROUP BY g` over `X JOIN Y ON x.a = y.b`, where
//! `e_X` reads only `X` and `e_Y` only `Y`, is a contraction: a matrix
//! product when `X` and `Y` are matrices. The general reduce rule rebuilds
//! the region `X ⋈ Y` (for a matrix product, `N·D·H` rows), joins it to the
//! cotangent, and sums by each input's dims. For a row `x` of `X`, though,
//!
//! ```text
//! Σ over the rows x joined to: c̄ · ∂(e_X · e_Y)/∂x  =  ∂e_X/∂x (x) · Σ_y c̄ · e_Y (y)
//! ```
//!
//! and the sum on the right is a contraction of the cotangent with `Y` alone,
//! as `jax.grad` differentiates a matrix product: `Ā = C̄·Bᵀ`. So the
//! cotangent of `e_X` at each row of `X` is
//!
//! ```sql
//! SELECT y.b, c.g_X, SUM(c.v * e_Y) AS v
//! FROM Y y JOIN cotangent c ON y.g_Y IS NOT DISTINCT FROM c.g_Y
//! WHERE y.b IS NOT NULL
//! GROUP BY y.b, c.g_X                                         -- t
//! ```
//!
//! joined to `X` on `x.a IS NOT DISTINCT FROM t.b AND x.g_X IS NOT DISTINCT
//! FROM t.g_X`, and `X`'s own rules (its projections, windows and inputs)
//! carry it from there, as they would carry a cotangent of a column of `X`
//! in the general rule. `X ⋈ Y` is never formed. (For a key the query joined
//! with `=`, `t` leaves out the NULL keys that `=` never matched, so that
//! every key of the join with `X` can be null-safe, which DataFusion hashes;
//! a key joined with IS NOT DISTINCT FROM keeps them.)
//!
//! **Sides of several inputs.** `X` and `Y` need not be single inputs. The
//! inputs of a tree of inner joins can be joined in any order, so the
//! inputs are split by which factors of the product read them, and each side
//! is rebuilt as a join of its own. Attention's output, `SUM(w.w / n.z *
//! v.x) … GROUP BY w.i, v.d` over `weights w JOIN v ON w.j = v.t JOIN norm n
//! ON w.i = n.i`, has the factors `w.w / n.z` and `v.x`, so `X` is `weights
//! ⋈ norm` (one row per score) and `Y` is `v`: the softmax's division is
//! transposed per score, and never per score and feature, as `jax.grad`
//! transposes `p = w / z` and then `p @ v`. An input no factor reads joins
//! the side its join conditions tie it to.
//!
//! **Keys the join makes equal.** Batched attention's output joins `w.s =
//! v.s` and groups by `w.s` (the sequence), so in every joined row `v.s` is
//! the cotangent's `s`, though `v` and the cotangent share no key for it. A
//! join key that the region's equalities, transitively, hold equal to a
//! grouping key of `X` joins `t`'s cotangent on that key too, and `t` is
//! summed by it once. Without it, `t` pairs each row of `Y` with every
//! group of `X`: every two sequences, (sequences × positions)² rows. A side
//! is rebuilt with the equalities its own columns hold this way too: a plan
//! may join the normalizer `n` through `v` (`v.s = n.s AND v.s = w.s`, as
//! DuckDB's does), and without `n.s = w.s` the side `weights ⋈ norm` would
//! join each score to every sequence's normalizer.
//!
//! **A mask.** A conjunct of the join condition that reads only grouping
//! keys, as causal attention's `q.t >= k.t` in `SUM(q.x * k.x) … GROUP BY
//! q.t, k.t` does, selects groups rather than rows: every row of a group
//! passes it or none does. Applied to the cotangent's rows, it leaves the
//! groups the query kept, which is all the transpose needs (and holds even
//! where a cotangent has a row for a group the query dropped).
//!
//! **NULLs.** The general rule gives a summed row no cotangent when the
//! aggregate's argument is NULL, since `SUM` skipped it. Here the argument
//! is `e_X · e_Y`, NULL exactly when a factor is. A NULL `e_Y` makes its
//! term `c̄ · e_Y` NULL, which the inner `SUM` skips, as before. A NULL `e_X`
//! makes every row of that `x` NULL, so its cotangent is NULL, as before:
//! tested once per `x` instead of once per joined row. Duplicate rows count
//! as before: each copy of a `Y` row adds its term to `t`, and each copy of
//! an `X` row joins `t`.
//!
//! **When it applies.** Exactly one measure has a cotangent, and it is a
//! `SUM`; the region is projections over a tree of inner joins (through
//! projections that compute nothing), with no filter above it, no post-join
//! filter, no `LIMIT`, no volatile function and no subquery in a join
//! condition; the argument, through the projections above the joins, is a
//! product whose factors split the join's inputs into two sides; each
//! condition between the sides is `=` or `IS NOT DISTINCT FROM` between a
//! column of each, or a mask, and there is at least one of the first; every
//! grouping key is a column of one side; for each side that is
//! differentiated, the *other* side holds at least one grouping key (so the
//! cotangent joins it on a key, not as a cross join: a matrix-vector product
//! `SUM(a.val * v.val) GROUP BY a.i`, differentiated in `a`, does not
//! qualify); and the argument stays below a size bound once inlined.
//! Otherwise the general rule applies.

use std::cell::Cell;

use substrait::proto::expression::RexType;
use substrait::proto::join_rel::JoinType;
use substrait::proto::rel::RelType;
use substrait::proto::{Expression, Rel};

use crate::elementwise::depends;
use crate::emit::{aggregate, filter, join, project};
use crate::error::Result;
use crate::expr::{as_field, call, field, fields_of, if_then, lit_bool, map_fields, scalar_args};
use crate::forward::{subquery_rounds, width, Def, Output, Region, Saved, Slot};
use crate::functions::Function;
use crate::optics::{subqueries, Fold};
use crate::transpose::Transposer;

/// A subtree of a region's join tree that is not itself an inner join: its
/// relation, and its columns, `start..start + width` of the region.
struct Leaf<'r> {
    rel: &'r Rel,
    start: usize,
    width: usize,
}

/// One side of a contraction, rebuilt as a region of its own.
struct Side {
    region: Region,
    /// Each region column's column here, if it is on this side.
    at: Vec<Option<usize>>,
    /// The side's factor of the product, over its own columns.
    factor: Expression,
}

/// A condition between a contraction's sides, `=` or IS NOT DISTINCT FROM.
struct Key {
    /// Its column on each side.
    cols: [usize; 2],
    not_distinct: bool,
    /// For each side, a grouping key on that side that every row of the
    /// region holds equal to this key (or both NULL), through the join's
    /// equalities, if one does: attention's `a.seq = v.seq`, whose `a.seq`
    /// is grouped by.
    group: [Option<usize>; 2],
}

/// Disjoint sets of `0..n`, joined a pair at a time: a union-find.
struct Classes(Vec<usize>);

impl Classes {
    fn new(n: usize) -> Classes {
        Classes((0..n).collect())
    }

    /// The root of `a`'s set, which names it.
    fn root(&mut self, a: usize) -> usize {
        let mut r = a;
        while self.0[r] != r {
            r = self.0[r];
        }
        self.0[a] = r;
        r
    }

    fn join(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.root(a), self.root(b));
        self.0[ra] = rb;
    }

    fn same(&mut self, a: usize, b: usize) -> bool {
        self.root(a) == self.root(b)
    }
}

/// A contraction's shape, in each side's own column numbers.
struct Shape {
    sides: [Side; 2],
    /// The conditions between the sides.
    keys: Vec<Key>,
    /// Each grouping key: its side and column.
    groups: Vec<(usize, usize)>,
    /// The conditions that read only grouping keys, over the cotangent's
    /// columns.
    masks: Vec<Expression>,
}

impl Transposer<'_> {
    /// The transpose of saved aggregate `n` by the contraction rule, if its
    /// shape allows (see the module docs). `arg` is the argument of its one
    /// measure with a cotangent, a `SUM`, whose cotangent is column `cot_col`
    /// of `cotangent`.
    ///
    /// Returns `false`, having changed nothing, if the shape doesn't allow.
    pub(crate) fn contraction(
        &mut self,
        n: usize,
        arg: &Expression,
        cotangent: &Rel,
        cot_col: usize,
    ) -> Result<bool> {
        let saved = &self.f.saved[n];
        let Some(shape) = self.shape(saved, arg)? else {
            return Ok(false);
        };
        let mut moves = [false; 2];
        for (x, side) in shape.sides.iter().enumerate() {
            let r = &side.region;
            moves[x] = depends(&self.f.functions, &side.factor, &|c| r.varied[c])?;
            // The cotangent meets the other side on its grouping keys; with
            // none there, it would meet every row of it, a cross join, which
            // the general rule handles instead.
            if moves[x] && !shape.groups.iter().any(|&(s, _)| s != x) {
                return Ok(false);
            }
        }
        // Both sides are recomputed, the one that does not move too (it makes
        // t), so their rankings must be total, as the general rule requires of
        // the whole region.
        self.check_rankings_are_total(&saved.input)?;
        for x in (0..2).filter(|&x| moves[x]) {
            self.side(&shape, x, cotangent, cot_col)?;
        }
        self.windows()?;
        Ok(true)
    }

    /// The contraction's shape, or `None` if the region is not one.
    fn shape(&mut self, saved: &Saved, arg: &Expression) -> Result<Option<Shape>> {
        let region = &saved.input;
        if region.volatile
            || !region.cuts.is_empty()
            || region.slots.iter().any(|s| s.offset.is_none())
        {
            return Ok(None);
        }
        // Walk down the projections to the joins.
        let mut rel = &region.rel;
        while let Some(RelType::Project(p)) = &rel.rel_type {
            match p.input.as_deref() {
                Some(r) => rel = r,
                None => return Ok(None),
            }
        }
        let below = width(rel)?;
        // Every column above the joins is an expression.
        if region.defs[below..]
            .iter()
            .any(|d| !matches!(d, Def::Expr(_)))
        {
            return Ok(None);
        }
        let mut leaves = Vec::new();
        let mut conds = Vec::new();
        if !self.flatten(rel, 0, &mut leaves, &mut conds)? || leaves.len() < 2 {
            return Ok(None);
        }
        let leaf_of = |c: usize| {
            leaves
                .iter()
                .position(|l| (l.start..l.start + l.width).contains(&c))
        };
        // Grouping keys, in the cotangent's order (the saved relation's dims):
        // columns beneath the projections.
        let mut group_cols = Vec::new();
        for d in saved.dims() {
            let Output::Dim(gi) = saved.outputs[d] else {
                return Ok(None);
            };
            let Some(e) = inline_above(region, &saved.groupings[gi], below, &budget()) else {
                return Ok(None);
            };
            match as_field(&e) {
                Some(c) if c < below => group_cols.push(c),
                _ => return Ok(None),
            }
        }
        // The argument's factors, and the inputs each reads.
        let Some(arg) = inline_above(region, arg, below, &budget()) else {
            return Ok(None);
        };
        if too_big(&arg) {
            return Ok(None);
        }
        let mut factors = Vec::new();
        self.factors(&arg, &mut factors)?;
        // Split the leaves into the two sides: those a factor reads together
        // are on one side, and a leaf no factor reads goes with those its
        // conditions tie it to.
        let mut side_of = Classes::new(leaves.len());
        let mut read = vec![false; leaves.len()];
        let ties = |sets: &[Vec<usize>], side_of: &mut Classes| {
            for set in sets {
                for w in set.windows(2) {
                    side_of.join(w[0], w[1]);
                }
            }
        };
        let mut factor_leaves = Vec::new();
        for f in &factors {
            let mut ls: Vec<usize> = Vec::new();
            for c in fields_of(f)? {
                let Some(l) = leaf_of(c) else {
                    return Ok(None);
                };
                read[l] = true;
                ls.push(l);
            }
            factor_leaves.push(ls);
        }
        ties(&factor_leaves, &mut side_of);
        let mut cond_leaves = Vec::new();
        for c in &conds {
            if subqueries().any(c, |_| true) {
                return Ok(None);
            }
            let mut ls = Vec::new();
            for f in fields_of(c)? {
                let Some(l) = leaf_of(f) else {
                    return Ok(None);
                };
                ls.push(l);
            }
            ls.sort();
            ls.dedup();
            cond_leaves.push(ls);
        }
        let unread: Vec<Vec<usize>> = cond_leaves
            .iter()
            .filter(|ls| ls.iter().any(|&l| !read[l]))
            .cloned()
            .collect();
        ties(&unread, &mut side_of);
        let roots: Vec<usize> = (0..leaves.len()).map(|l| side_of.root(l)).collect();
        let first = roots[0];
        let Some(other) = roots.iter().copied().find(|&r| r != first) else {
            return Ok(None);
        };
        if roots.iter().any(|&r| r != first && r != other) {
            return Ok(None);
        }
        let side_of_leaf: Vec<usize> = roots.iter().map(|&r| usize::from(r != first)).collect();
        let side_of_col = |c: usize| leaf_of(c).map(|l| side_of_leaf[l]);

        // Columns every row of the region holds equal, or both NULL, through
        // its equalities, transitively: each column's class.
        let mut class = Classes::new(below);
        let mut null_safe = Vec::new();
        for c in &conds {
            if let Some((not_distinct, a, b)) = self.equality(c)? {
                if a < below && b < below {
                    class.join(a, b);
                    if not_distinct {
                        null_safe.push(a);
                    }
                }
            }
        }
        // Whether a class holds an IS NOT DISTINCT FROM, so that its columns
        // may be NULL together.
        let mut not_distinct = vec![false; below];
        for a in null_safe {
            let r = class.root(a);
            not_distinct[r] = true;
        }
        // The conditions: within a side, between them, or a mask.
        let mut within = [Vec::new(), Vec::new()];
        let mut keys = Vec::new();
        let mut masks = Vec::new();
        for (c, ls) in conds.into_iter().zip(cond_leaves) {
            let sides: Vec<usize> = ls.iter().map(|&l| side_of_leaf[l]).collect();
            if sides
                .iter()
                .all(|&s| s == sides.first().copied().unwrap_or(0))
            {
                within[sides.first().copied().unwrap_or(0)].push((c, ls));
                continue;
            }
            if let Some((not_distinct, a, b)) = self.equality(&c)? {
                match (side_of_col(a), side_of_col(b)) {
                    (Some(0), Some(1)) => {
                        keys.push((a, b, not_distinct));
                        continue;
                    }
                    (Some(1), Some(0)) => {
                        keys.push((b, a, not_distinct));
                        continue;
                    }
                    _ => {}
                }
            }
            let mut group_only = true;
            let mask = map_fields(&c, &mut |f| {
                let g = group_cols.iter().position(|&gc| gc == f);
                group_only &= g.is_some();
                Ok(g.unwrap_or(f))
            })?;
            if !group_only {
                return Ok(None);
            }
            masks.push(mask);
        }
        if keys.is_empty() {
            return Ok(None);
        }
        // Within each side, the equalities its classes imply but its own
        // conditions don't: batched attention's `n.s = a.s`, which a plan
        // may state through the other side (`v.s = n.s AND v.s = a.s`, as
        // DuckDB's does). Without them, the side is rebuilt joining every
        // sequence's rows to every other's. Every row of the region holds
        // them, so they drop only rows that join no row of the region. One
        // is `=` where its class is all `=`, so DataFusion hashes it with
        // the side's own `=` keys, and IS NOT DISTINCT FROM otherwise.
        let equal = self.ext.anchor(Function::Equal);
        let same = self.ext.anchor(Function::IsNotDistinctFrom);
        for (s, within) in within.iter_mut().enumerate() {
            let mut local = Classes::new(below);
            for (c, _) in within.iter() {
                if let Some((_, a, b)) = self.equality(c)? {
                    if a < below && b < below {
                        local.join(a, b);
                    }
                }
            }
            for a in (0..below).filter(|&a| side_of_col(a) == Some(s)) {
                let tied = (0..a)
                    .find(|&b| side_of_col(b) == Some(s) && class.same(a, b) && !local.same(a, b));
                let Some(b) = tied else {
                    continue;
                };
                local.join(a, b);
                let f = if not_distinct[class.root(a)] {
                    same
                } else {
                    equal
                };
                let mut ls = vec![leaf_of(b).expect("below"), leaf_of(a).expect("below")];
                ls.sort();
                ls.dedup();
                within.push((call(f, vec![field(b), field(a)]), ls));
            }
        }
        // Each side, rebuilt from its leaves.
        let mut product: [Option<Expression>; 2] = [None, None];
        let mul = self.ext.anchor(Function::Multiply);
        for (f, ls) in factors.into_iter().zip(&factor_leaves) {
            // A factor that reads no input is a constant; it goes with side 0.
            let s = ls.first().map_or(0, |&l| side_of_leaf[l]);
            product[s] = Some(match product[s].take() {
                Some(p) => call(mul, vec![p, f]),
                None => f,
            });
        }
        let and = self.ext.anchor(Function::And);
        let mut sides = Vec::new();
        for (s, within) in within.into_iter().enumerate() {
            let mine: Vec<usize> = (0..leaves.len())
                .filter(|&l| side_of_leaf[l] == s)
                .collect();
            let Some(mut side) = rebuild(region, &leaves, &mine, within, and)? else {
                return Ok(None);
            };
            // A side whose leaves no factor reads has the factor 1: never, as
            // a product's factors read both sides, but kept total.
            let Some(f) = product[s].take() else {
                return Ok(None);
            };
            let at = side.at.clone();
            side.factor = map_fields(&f, &mut |c| Ok(at[c].expect("a factor of this side")))?;
            sides.push(side);
        }
        let [left, right]: [Side; 2] = sides.try_into().ok().expect("two sides");
        let keys = keys
            .into_iter()
            .map(|(a, b, not_distinct)| {
                let mut group = [None, None];
                for (s, g) in group.iter_mut().enumerate() {
                    *g = group_cols
                        .iter()
                        .position(|&gc| side_of_col(gc) == Some(s) && class.same(gc, a));
                }
                Key {
                    cols: [left.at[a].expect("side 0"), right.at[b].expect("side 1")],
                    not_distinct,
                    group,
                }
            })
            .collect();
        let groups = group_cols
            .iter()
            .map(|&c| {
                let s = side_of_col(c).expect("a column beneath the projections");
                (s, [&left, &right][s].at[c].expect("on its side"))
            })
            .collect();
        Ok(Some(Shape {
            sides: [left, right],
            keys,
            groups,
            masks,
        }))
    }

    /// Collect the leaves of the inner-join tree `rel` (whose columns start
    /// at region column `start`), and its join conditions' conjuncts in the
    /// region's numbering. `false` if a join has a post-join filter.
    fn flatten<'r>(
        &self,
        rel: &'r Rel,
        start: usize,
        leaves: &mut Vec<Leaf<'r>>,
        conds: &mut Vec<Expression>,
    ) -> Result<bool> {
        match &rel.rel_type {
            Some(RelType::Join(j)) if j.r#type == JoinType::Inner as i32 => {
                if j.post_join_filter.is_some() {
                    return Ok(false);
                }
                let (Some(l), Some(r)) = (j.left.as_deref(), j.right.as_deref()) else {
                    return Ok(false);
                };
                if let Some(c) = j.expression.as_deref() {
                    for conjunct in self.conjuncts(c)? {
                        conds.push(map_fields(&conjunct, &mut |f| Ok(f + start))?);
                    }
                }
                let lw = width(l)?;
                Ok(self.flatten(l, start, leaves, conds)?
                    && self.flatten(r, start + lw, leaves, conds)?)
            }
            // A projection that computes nothing keeps its input's columns.
            Some(RelType::Project(p)) if p.expressions.is_empty() && p.common.is_none() => {
                match p.input.as_deref() {
                    Some(input) => self.flatten(input, start, leaves, conds),
                    None => Ok(false),
                }
            }
            _ => {
                leaves.push(Leaf {
                    rel,
                    start,
                    width: width(rel)?,
                });
                Ok(true)
            }
        }
    }

    /// The factors of a product `e`: `e` itself unless it multiplies.
    fn factors(&self, e: &Expression, out: &mut Vec<Expression>) -> Result<()> {
        if let Some(RexType::ScalarFunction(f)) = &e.rex_type {
            if self.f.functions.name(f.function_reference)? == "multiply" {
                if let [a, b] = scalar_args(f)?[..] {
                    self.factors(a, out)?;
                    return self.factors(b, out);
                }
            }
        }
        out.push(e.clone());
        Ok(())
    }

    /// The transpose of side `x`: its region, with its factor appended as a
    /// column and joined to `t` (see the module docs), transposed from the
    /// factor's cotangent by the region rules.
    fn side(&mut self, shape: &Shape, x: usize, cotangent: &Rel, cot_col: usize) -> Result<()> {
        let y = 1 - x;
        let (xs, ys) = (&shape.sides[x], &shape.sides[y]);
        let same = self.ext.anchor(Function::IsNotDistinctFrom);
        let and = self.ext.anchor(Function::And);
        let mul = self.ext.anchor(Function::Multiply);
        let sum = self.ext.anchor(Function::Sum);
        let is_null = self.ext.anchor(Function::IsNull);
        let all = |conds: Vec<Expression>| conds.into_iter().reduce(|a, b| call(and, vec![a, b]));

        // t: Y joined to the cotangent on Y's grouping keys, summed by Y's
        // join keys and the cotangent's X-side grouping keys. A join key that
        // the region holds equal to an X-side grouping key joins the
        // cotangent on it too, and is summed by once: without it, t would
        // pair every row of Y with every group of X, as attention's backward
        // would pair every two sequences and heads.
        let ndims = shape.groups.len();
        let ywidth = ys.region.defs.len();
        let y_keys: Vec<usize> = shape.keys.iter().map(|k| k.cols[y]).collect();
        // Each pair once: a plan may repeat a join key (DuckDB's does).
        let mut on_y: Vec<(usize, usize)> = Vec::new();
        let y_groups = shape
            .groups
            .iter()
            .enumerate()
            .filter(|(_, (s, _))| *s == y)
            .map(|(g, &(_, c))| (c, g));
        let key_groups = shape
            .keys
            .iter()
            .filter_map(|k| k.group[x].map(|g| (k.cols[y], g)));
        for pair in y_groups.chain(key_groups) {
            if !on_y.contains(&pair) {
                on_y.push(pair);
            }
        }
        let on_y = all(on_y
            .into_iter()
            .map(|(c, g)| call(same, vec![field(c), field(ywidth + g)]))
            .collect())
        .expect("checked: the other side holds a grouping key");
        // Every key of the join with x below is null-safe, so DataFusion can
        // hash it (it can't hash a join that mixes `=` with IS NOT DISTINCT
        // FROM, and falls back to a nested loop). For a key that was `=`, a
        // NULL never matched, so Y's rows with a NULL there joined nothing:
        // leaving them out of t keeps the meaning.
        let not_null: Vec<Expression> = shape
            .keys
            .iter()
            .zip(&y_keys)
            .filter(|(k, _)| !k.not_distinct)
            // `IS NOT NULL`, from names every engine ddx targets knows.
            .map(|(_, &c)| {
                if_then(
                    vec![(call(is_null, vec![field(c)]), lit_bool(false))],
                    lit_bool(true),
                )
            })
            .collect();
        let y_rel = match all(not_null) {
            Some(cond) => filter(ys.region.rel.clone(), cond),
            None => ys.region.rel.clone(),
        };
        let cotangent = match all(shape.masks.clone()) {
            Some(mask) => filter(cotangent.clone(), mask),
            None => cotangent.clone(),
        };
        let yc = join(y_rel, cotangent, on_y, JoinType::Inner);
        let x_groups: Vec<usize> = (0..ndims).filter(|&g| shape.groups[g].0 == x).collect();
        // t's keys, each column once: a plan may repeat a join key (DuckDB's
        // does), and an aggregate's output has each distinct key once.
        let mut t_cols: Vec<usize> = Vec::new();
        let mut key_of = |c: usize| match t_cols.iter().position(|&k| k == c) {
            Some(i) => i,
            None => {
                t_cols.push(c);
                t_cols.len() - 1
            }
        };
        let y_at: Vec<usize> = shape
            .keys
            .iter()
            .map(|k| match k.group[x] {
                Some(g) => key_of(ywidth + g),
                None => key_of(k.cols[y]),
            })
            .collect();
        let group_at: Vec<usize> = x_groups.iter().map(|&g| key_of(ywidth + g)).collect();
        let term = call(
            mul,
            vec![field(ywidth + ndims + cot_col), ys.factor.clone()],
        );
        let t_value = t_cols.len();
        let t_keys = t_cols.iter().map(|&c| field(c)).collect();
        let mut t = aggregate(yc, t_keys, vec![(sum, vec![term])]);
        // t is most of the work. A side of one input and no window reads it
        // once; any other reads it once per input and once per window, so it
        // is materialized, as a window's summed cotangent is.
        let several = xs.region.slots.len() > 1
            || xs
                .region
                .defs
                .iter()
                .any(|d| matches!(d, Def::Window { .. }));
        if several {
            let name = format!("{}contraction_{}", self.f.namespace, self.steps_made());
            let names: Vec<String> = (0..=t_value).map(|i| format!("c{i}")).collect();
            self.steps.push((name.clone(), t, names.clone()));
            t = crate::emit::read_step(&name, names);
        }

        // X, its factor a column of its own, joined to t on the join keys and
        // X's grouping keys.
        let mut region = xs.region.clone();
        let factor_col = region.defs.len();
        let varied = depends(&self.f.functions, &xs.factor, &|c| region.varied[c])?;
        region
            .rounds
            .push(subquery_rounds(&self.f.functions, &xs.factor)?);
        region.defs.push(Def::Expr(xs.factor.clone()));
        region.varied.push(varied);
        region.refusals.push(None);
        let rel = project(region.rel.clone(), vec![xs.factor.clone()]);
        let t_at = factor_col + 1;
        let mut on_x: Vec<(usize, usize)> = Vec::new();
        let key_pairs = shape.keys.iter().map(|k| k.cols[x]).zip(y_at);
        let group_pairs = x_groups.iter().map(|&g| shape.groups[g].1).zip(group_at);
        for pair in key_pairs.chain(group_pairs) {
            if !on_x.contains(&pair) {
                on_x.push(pair);
            }
        }
        let on_x = on_x
            .into_iter()
            .map(|(xc, at)| call(same, vec![field(xc), field(t_at + at)]))
            .collect();
        let base = join(
            rel,
            t,
            all(on_x).expect("a contraction has a join key"),
            JoinType::Inner,
        );
        // A NULL factor of x leaves its rows nothing to pass on: the guard
        // the general rule applies to each joined row, once per row of x.
        let seed = self.skips_null(factor_col, field(t_at + t_value));
        self.region_once(&region, base, vec![(factor_col, seed)])
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

/// The side made of `mine`, leaves of `region`, joined one at a time, each
/// condition in `within` (with the leaves it reads) at the first join where
/// all its leaves are present (`and` is the anchor of `and`). Each leaf
/// joined is one a condition ties to those before it. `None` if there is
/// none (the side's own conditions leave it in pieces, and rebuilding it
/// would be a cross join the query never formed, `p × z` in `p JOIN v ON
/// p.j = v.t JOIN z ON v.u = z.i`), or if a leaf's columns are not its own.
fn rebuild(
    region: &Region,
    leaves: &[Leaf],
    mine: &[usize],
    within: Vec<(Expression, Vec<usize>)>,
    and: u32,
) -> Result<Option<Side>> {
    let mut at = vec![None; region.defs.len()];
    let mut side: Option<Region> = None;
    let mut placed = Vec::new();
    // The order: each next leaf shares a condition with those placed.
    let mut order = vec![mine[0]];
    let mut rest: Vec<usize> = mine[1..].to_vec();
    while !rest.is_empty() {
        let tied = rest.iter().position(|&l| {
            within.iter().any(|(_, ls)| {
                ls.contains(&l)
                    && ls.iter().any(|x| order.contains(x))
                    && ls.iter().all(|x| *x == l || order.contains(x))
            })
        });
        let Some(i) = tied else {
            return Ok(None);
        };
        order.push(rest.remove(i));
    }
    let mut pending = within;
    for l in order {
        let leaf = &leaves[l];
        let Some(part) = slice(region, leaf)? else {
            return Ok(None);
        };
        let offset = side.as_ref().map_or(0, |s| s.defs.len());
        for c in 0..leaf.width {
            at[leaf.start + c] = Some(offset + c);
        }
        placed.push(l);
        let (ready, rest): (Vec<_>, Vec<_>) = pending
            .into_iter()
            .partition(|(_, ls)| ls.iter().all(|x| placed.contains(x)));
        pending = rest;
        let mut conds = Vec::new();
        for (c, _) in ready {
            conds.push(map_fields(&c, &mut |f| Ok(at[f].expect("placed")))?);
        }
        side = Some(match side {
            None => {
                let mut s = part;
                if let Some(c) = conds.into_iter().reduce(|a, b| call(and, vec![a, b])) {
                    s.rel = filter(s.rel, c);
                }
                s
            }
            Some(mut s) => {
                let left = std::mem::take(&mut s.rel);
                let right = part.rel.clone();
                s.append(part)?;
                let cond = conds
                    .into_iter()
                    .reduce(|a, b| call(and, vec![a, b]))
                    .expect("the order ties each leaf to those before it");
                s.rel = join(left, right, cond, JoinType::Inner);
                s
            }
        });
    }
    Ok(side.map(|region| Side {
        region,
        at,
        factor: lit_bool(true),
    }))
}

/// `leaf`'s columns of `region`, as a region of their own. `None` if one of
/// them reads a column outside the leaf.
fn slice(region: &Region, leaf: &Leaf) -> Result<Option<Region>> {
    let range = leaf.start..leaf.start + leaf.width;
    let local = |c: usize| {
        range
            .contains(&c)
            .then(|| c - leaf.start)
            .ok_or_else(|| crate::error::AdError::Internal("outside the leaf".into()))
    };
    let mut slots: Vec<Slot> = Vec::new();
    let mut slot_at = vec![None; region.slots.len()];
    for (i, s) in region.slots.iter().enumerate() {
        if let Some(o) = s.offset.filter(|o| range.contains(o)) {
            slot_at[i] = Some(slots.len());
            slots.push(Slot {
                offset: Some(o - leaf.start),
                ..s.clone()
            });
        }
    }
    let mut defs = Vec::with_capacity(leaf.width);
    for d in &region.defs[range.clone()] {
        let def = match d {
            Def::Input { slot, col } => match slot_at[*slot] {
                Some(slot) => Def::Input { slot, col: *col },
                None => return Ok(None),
            },
            Def::Expr(e) => match map_fields(e, &mut |c| local(c)) {
                Ok(e) => Def::Expr(e),
                Err(_) => return Ok(None),
            },
            Def::Window {
                keys,
                function,
                beneath,
                offset,
            } => {
                let (Ok(function), Ok(offset)) =
                    (map_fields(function, &mut |c| local(c)), local(*offset))
                else {
                    return Ok(None);
                };
                let keys = match keys {
                    Some(keys) => match keys.iter().map(|&k| local(k)).collect() {
                        Ok(keys) => Some(keys),
                        Err(_) => return Ok(None),
                    },
                    None => None,
                };
                Def::Window {
                    keys,
                    function,
                    beneath: beneath.clone(),
                    offset,
                }
            }
            Def::Const => Def::Const,
            Def::Branch => Def::Branch,
            Def::Union(arms) => match arms
                .iter()
                .map(|&(b, c)| Ok((local(b)?, local(c)?)))
                .collect::<Result<Vec<_>>>()
            {
                Ok(arms) => Def::Union(arms),
                Err(_) => return Ok(None),
            },
        };
        defs.push(def);
    }
    Ok(Some(Region {
        rel: leaf.rel.clone(),
        defs,
        varied: region.varied[range.clone()].to_vec(),
        refusals: region.refusals[range.clone()].to_vec(),
        outputs: (0..leaf.width).collect(),
        slots,
        cuts: Vec::new(),
        volatile: region.volatile,
        rounds: region.rounds[range].to_vec(),
    }))
}

/// Is `e` too large to inline? Inlining copies an expression once per use, so
/// a chain of projections that each read the last twice doubles at every
/// level; past this size the general rule, which computes each column once,
/// handles the region instead.
fn too_big(e: &Expression) -> bool {
    use prost::Message;
    e.encoded_len() > 1 << 16
}

/// How many expression nodes one inlining may build. [`too_big`] measures
/// an expression once it is built, but a chain that doubles at every level
/// would take time and memory exponential in its depth to build first; past
/// this many nodes, inlining gives up and the general rule applies. Every
/// node costs at least two bytes encoded, so an expression [`too_big`]
/// accepts never runs out.
fn budget() -> Cell<usize> {
    Cell::new(1 << 16)
}

/// `e` with every column at or above `above` (an expression defined above
/// the join) replaced by its definition, recursively. `None` if one isn't an
/// expression, or `e` holds an expression kind `inline` doesn't follow.
fn inline_above(
    region: &Region,
    e: &Expression,
    above: usize,
    budget: &Cell<usize>,
) -> Option<Expression> {
    inline(e, budget, &mut |c| {
        if c < above {
            return Some(field(c));
        }
        match &region.defs[c] {
            Def::Expr(inner) => inline_above(region, inner, above, budget),
            _ => None,
        }
    })
    .ok()
    .flatten()
}

/// `e` with each column reference replaced by `f(column)`; `None` if `f`
/// declines one, `e` holds an expression other than a literal, a column, a
/// scalar function, a `CASE` or a cast, or `budget` (a count of nodes, which
/// each node visited spends) runs out.
fn inline(
    e: &Expression,
    budget: &Cell<usize>,
    f: &mut dyn FnMut(usize) -> Option<Expression>,
) -> Result<Option<Expression>> {
    use substrait::proto::function_argument::ArgType;
    let Some(left) = budget.get().checked_sub(1) else {
        return Ok(None);
    };
    budget.set(left);
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
                    Some(ArgType::Value(v)) => match inline(v, budget, f)? {
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
                    match inline(x, budget, f)? {
                        Some(y) => *x = y,
                        None => return Ok(None),
                    }
                }
            }
            if let Some(x) = it.r#else.as_deref_mut() {
                match inline(x, budget, f)? {
                    Some(y) => *x = y,
                    None => return Ok(None),
                }
            }
        }
        (RexType::Cast(_), RexType::Cast(c)) => match c.input.as_deref_mut() {
            Some(x) => match inline(x, budget, f)? {
                Some(y) => *x = y,
                None => return Ok(None),
            },
            None => return Ok(None),
        },
        _ => return Ok(None),
    }
    Ok(Some(out))
}
