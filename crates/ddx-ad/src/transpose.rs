// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! One transpose rule per relational primitive.
//!
//! JAX differentiates a function by giving each primitive a rule and composing
//! them. A query is a composition of relational primitives, and ddx does the
//! same with these:
//!
//! | Primitive | SQL | Transpose |
//! |---|---|---|
//! | **map** | a projected expression `y = f(x₁, x₂, …)` | `x̄ᵢ += ȳ · ∂f/∂xᵢ`, row by row ([`crate::Elementwise`]) |
//! | **select** | `WHERE`, a join condition, a semi-join, a filter on a rank | the cotangent stays on the rows that were kept |
//! | **broadcast** | a join, which pairs each row with every row it matches | sum the cotangent back over the rows each input row was copied to |
//! | **reduce** | a grouped `SUM` | broadcast the group's cotangent to every row that was summed |
//! | | `AVG` | the same, divided by the group's count |
//! | | `MAX`, `MIN` | the group's cotangent to the rows that attain it, shared evenly at a tie (`jax.grad`'s convention for `jnp.max`) |
//!
//! A matrix product, `SUM(a.v * b.v) … GROUP BY` over a join, is not a
//! primitive: it is broadcast, map and reduce, and its transpose is theirs
//! composed, which gives `Ā = Σ_out C̄·B` and `B̄ = Σ_batch A·C̄` (design.md
//! §4.3). A `COUNT` does not change when its argument does, so its transpose is
//! zero.
//!
//! # How the rules are applied
//!
//! A saved aggregate's transpose is the reduce rule: join the recomputed region
//! beneath it to the aggregate's cotangent on the grouping keys, so every row
//! carries the cotangent of the group it was summed into. The region's
//! transpose then walks its columns right to left. A map column passes its
//! cotangent to the columns it reads. Select needs no step of its own, because
//! the recomputed region contains only the rows the forward pass kept. At an
//! input, the broadcast rule sums the cotangent by the input's dims: that is
//! the input's contribution.
//!
//! Each column's cotangent is appended as a new column rather than inlined into
//! the expressions that use it, so a shared subexpression is computed once.

use std::collections::BTreeMap;

use ddx_core::Ddx;
use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::function_argument::ArgType;
use substrait::proto::join_rel::JoinType;
use substrait::proto::rel::RelType;
use substrait::proto::{AggregateFunction, CrossRel, Expression, Rel};

use crate::elementwise::{depends, Elementwise};
use crate::emit::{aggregate, join, project};
use crate::error::{AdError, Result};
use crate::expr::{as_number, call, field, fields_of, if_then, lit_f64, null_f64, window};
use crate::forward::{subquery_rounds, width, Def, Forward, Input, Output, Region};
use crate::functions::Extensions;

/// An input's cotangent from one place that reads it: the input's dims, then
/// the cotangent of each of its columns in `cols`.
pub(crate) struct Contribution {
    pub rel: Rel,
    pub cols: Vec<usize>,
}

/// Applies the rules, collecting each input's contributions.
pub(crate) struct Transposer<'a> {
    pub f: &'a Forward,
    pub ext: Extensions,
    ew: Elementwise<'a>,
    pub contributions: BTreeMap<Input, Vec<Contribution>>,
}

impl<'a> Transposer<'a> {
    pub fn new(f: &'a Forward, ddx: &'a Ddx) -> Self {
        Transposer {
            f,
            ext: Extensions::new(&f.functions),
            ew: Elementwise::new(ddx, &f.functions),
            contributions: BTreeMap::new(),
        }
    }

    /// The transpose of saved aggregate `n`: the reduce rule for each measure
    /// in `cols` (the output columns that have a cotangent), then the
    /// transpose of the region beneath it.
    ///
    /// `cotangent` reads the aggregate's cotangent: its dims, then one column
    /// per entry of `cols`.
    pub fn saved(&mut self, n: usize, cols: &[usize], cotangent: Rel) -> Result<()> {
        let saved = &self.f.saved[n];
        let mut region = saved.input.clone();
        let width = region.defs.len();
        let dims = saved.dims();

        // Each measure's argument becomes a column of the region, so the
        // chain rule can start from it.
        let mut args = Vec::new();
        // (argument column, rule, index into `cols`, output column)
        let mut rules = Vec::new();
        for (i, &col) in cols.iter().enumerate() {
            let Output::Value(m) = saved.outputs[col] else {
                return Err(AdError::NotImplemented(
                    "gradient through a GROUP BY key: the aggregate groups by a value that \
                     depends on a wrt column, and that value is then used in the loss"
                        .into(),
                ));
            };
            let (rule, arg) = match reduce_rule(&self.f.functions, &saved.measures[m])? {
                Reduce::Constant => continue,
                Reduce::Of(rule, arg) => (rule, arg),
            };
            rules.push((width + args.len(), rule, i, col));
            args.push(*arg);
        }
        for e in &args {
            let varied = depends(&self.f.functions, e, &|c| region.varied[c])?;
            region.defs.push(Def::Expr(e.clone()));
            region.rounds.push(subquery_rounds(&self.f.functions, e)?);
            region.varied.push(varied);
            region.refusals.push(None);
        }
        let rows = project(region.rel.clone(), args);
        let rows_width = region.defs.len();
        let keys: Vec<Expression> = dims
            .iter()
            .map(|&d| match saved.outputs[d] {
                Output::Dim(g) => saved.groupings[g].clone(),
                Output::Value(_) => unreachable!("dims() returns grouping columns"),
            })
            .collect();

        // AVG divides by its group's count, and MAX and MIN find the rows
        // that attain the group's extreme and how many do. Each is a window
        // over the recomputed rows themselves, partitioned by the group's
        // keys, never a comparison with the saved aggregate: a recomputation
        // need not be bit-identical to the forward pass (a grouped SUM over
        // several partitions adds in arrival order), and a saved maximum no
        // recomputed row equals would silently send no gradient at all.
        let count = self.ext.anchor("count");
        let sum = self.ext.anchor("sum");
        let isnan = self.ext.anchor("isnan");
        let mut rel = rows;
        let mut next = rows_width;
        let mut stat_at = BTreeMap::new();
        let mut extreme_at = BTreeMap::new();
        let mut windows = Vec::new();
        for &(arg_col, rule, _, _) in &rules {
            let at = next + windows.len();
            match rule {
                Rule::Sum => continue,
                Rule::Mean => {
                    windows.push(window(count, vec![field(arg_col)], keys.clone()));
                    stat_at.insert(arg_col, at);
                }
                Rule::Extreme(name) => {
                    // NaN arguments are left out. DataFusion's grouped MAX
                    // skips a NaN where its ungrouped and window MAX return
                    // it; a MAX that gave a finite value skipped them, so
                    // leaving them out agrees with it, and a NaN row gets no
                    // gradient, as a NULL one gets none.
                    let f = self.ext.anchor(name);
                    let arg = if_then(
                        vec![(call(isnan, vec![field(arg_col)]), null_f64())],
                        field(arg_col),
                    );
                    windows.push(window(f, vec![arg], keys.clone()));
                    extreme_at.insert(arg_col, at);
                }
            }
        }
        if !windows.is_empty() {
            next += windows.len();
            rel = project(rel, windows);
        }
        let attaining: Vec<Expression> = extreme_at
            .iter()
            .map(|(&arg_col, &at)| {
                let attains = if_then(
                    vec![(
                        self.attains(arg_col, at, jitters(&region, arg_col)),
                        lit_f64(1.0),
                    )],
                    lit_f64(0.0),
                );
                window(sum, vec![attains], keys.clone())
            })
            .collect();
        for (k, &arg_col) in extreme_at.keys().enumerate() {
            stat_at.insert(arg_col, next + k);
        }
        if !attaining.is_empty() {
            next += attaining.len();
            rel = project(rel, attaining);
        }

        // Every row joined to the cotangent of the group it went into: the
        // broadcast every reduce rule starts from.
        let cotangent_at = next + dims.len();
        let key_positions: Vec<usize> = (0..dims.len()).collect();
        let base = self.join_on(rel, cotangent, keys, &key_positions, next)?;

        let divide = self.ext.anchor("divide");
        let is_null = self.ext.anchor("is_null");
        let mut seeds = Vec::new();
        for (arg_col, rule, i, _) in rules {
            let cot = field(cotangent_at + i);
            let seed = match rule {
                // Every summed row gets the group's cotangent.
                Rule::Sum => cot,
                // A mean is a sum divided by the group's count.
                Rule::Mean => call(divide, vec![cot, field(stat_at[&arg_col])]),
                // Only the rows equal to the extreme get it, shared evenly
                // among them: jax.grad's convention for jnp.max at a tie. The
                // others get none, NULL rather than 0, so it stays none
                // through a partial that is not finite there (0 · ∞ is NaN:
                // an infinite value in the data that does not attain a MIN).
                Rule::Extreme(_) => if_then(
                    vec![(
                        self.attains(arg_col, extreme_at[&arg_col], jitters(&region, arg_col)),
                        call(divide, vec![cot, field(stat_at[&arg_col])]),
                    )],
                    null_f64(),
                ),
            };
            // An aggregate skips a row whose argument is NULL, so nothing in
            // that row moves the loss: its cotangent is NULL, not the
            // group's, or the row's other inputs (the `p` of `SUM(p + q)`
            // where `q` is NULL) would get gradient through it.
            let seed = if_then(
                vec![(call(is_null, vec![field(arg_col)]), null_f64())],
                seed,
            );
            seeds.push((arg_col, seed));
        }
        self.region(&region, base, seeds)
    }

    /// A column's cotangent: the sum of the terms from the columns that read
    /// it. A NULL term is no contribution (a row an aggregate skipped), so
    /// it is skipped rather than added, as NULL + t would lose t, and the sum
    /// is NULL only if every term is. One flat expression, each term in it a
    /// fixed number of times: a fold that nested the running sum named it
    /// three times a step, and DataFusion's Substrait consumer names a column
    /// by its expression, so ten readers made a 100 MB plan.
    fn null_skipping_sum(&mut self, mut terms: Vec<Expression>) -> Expression {
        if terms.len() == 1 {
            return terms.pop().expect("one term");
        }
        let add = self.ext.anchor("add");
        let and = self.ext.anchor("and");
        let is_null = self.ext.anchor("is_null");
        let none = terms
            .iter()
            .map(|t| call(is_null, vec![t.clone()]))
            .reduce(|a, b| call(and, vec![a, b]))
            .expect("at least two terms");
        let total = terms
            .into_iter()
            .map(|t| if_then(vec![(call(is_null, vec![t.clone()]), lit_f64(0.0))], t))
            .reduce(|a, b| call(add, vec![a, b]))
            .expect("at least two terms");
        if_then(vec![(none, null_f64())], total)
    }

    /// Does the row's argument (column `arg`) attain the group's extreme
    /// (column `extreme`)? Equal to it, or, where it is finite, within a few
    /// ulps of it: the extreme and the arguments are recomputed, and two
    /// groups that tie in exact arithmetic can differ in the last bit (a sum
    /// over several partitions adds in arrival order), which would give the
    /// whole cotangent to whichever rounded higher on that run. Within the
    /// tolerance they share it, as at an exact tie, the same way every run.
    /// Only an argument that can jitter gets the tolerance (see [`jitters`]):
    /// one computed from table values alone is the same every run, and two
    /// of its values a few ulps apart do not tie (MAX(1, 1 + 2 ulps) has
    /// gradient (0, 1), as jax.grad gives).
    fn attains(&mut self, arg: usize, extreme: usize, tolerant: bool) -> Expression {
        let equal = self.ext.anchor("equal");
        if !tolerant {
            return call(equal, vec![field(arg), field(extreme)]);
        }
        let or = self.ext.anchor("or");
        let and = self.ext.anchor("and");
        let lte = self.ext.anchor("lte");
        let abs = self.ext.anchor("abs");
        let subtract = self.ext.anchor("subtract");
        let multiply = self.ext.anchor("multiply");
        let size = call(abs, vec![field(extreme)]);
        let finite = call(lte, vec![size.clone(), lit_f64(f64::MAX)]);
        let gap = call(abs, vec![call(subtract, vec![field(arg), field(extreme)])]);
        let close = call(
            lte,
            vec![gap, call(multiply, vec![lit_f64(8.0 * f64::EPSILON), size])],
        );
        call(
            or,
            vec![
                call(equal, vec![field(arg), field(extreme)]),
                call(and, vec![finite, close]),
            ],
        )
    }

    /// Join `left` (whose columns before `left_width` are a region's) to
    /// `right` on `left_keys[k] = right column right_keys[k]`, null-safely; a
    /// cross join when there are no keys.
    pub fn join_on(
        &mut self,
        left: Rel,
        right: Rel,
        left_keys: Vec<Expression>,
        right_keys: &[usize],
        left_width: usize,
    ) -> Result<Rel> {
        if left_keys.is_empty() {
            return Ok(Rel {
                rel_type: Some(RelType::Cross(Box::new(CrossRel {
                    common: None,
                    left: Some(Box::new(left)),
                    right: Some(Box::new(right)),
                    advanced_extension: None,
                }))),
            });
        }
        // IS NOT DISTINCT FROM, so a NULL group keeps its gradient.
        let same = self.ext.anchor("is_not_distinct_from");
        let and = self.ext.anchor("and");
        let cond = left_keys
            .into_iter()
            .zip(right_keys)
            .map(|(key, &r)| call(same, vec![key, field(left_width + r)]))
            .reduce(|a, b| call(and, vec![a, b]))
            .expect("at least one key");
        Ok(join(left, right, cond, JoinType::Inner))
    }

    /// The transpose of a recomputed region, from `seeds` (a column and its
    /// cotangent, an expression over `base`).
    ///
    /// `base` is the rebuilt region, possibly joined to a cotangent on its
    /// right, so the region's columns keep their numbers in it.
    pub fn region(
        &mut self,
        region: &Region,
        base: Rel,
        seeds: Vec<(usize, Expression)>,
    ) -> Result<()> {
        self.check_rankings_are_total(region)?;
        let mut width = width(&base)?;
        let mut rel = base;
        let mut pending: BTreeMap<usize, Vec<Expression>> = BTreeMap::new();
        for (c, e) in seeds {
            pending.entry(c).or_default().push(e);
        }
        // (slot, input column) → the column holding its cotangent.
        let mut at_inputs: BTreeMap<usize, BTreeMap<usize, usize>> = BTreeMap::new();
        // Cotangent columns are projected in batches: a column joins the
        // current batch unless one of its terms reads a column still in it,
        // which flushes the batch first. The step is then as deep as the
        // chain of columns that read each other, not as the region is wide
        // (one projection per column nested hundreds deep, and a clone of
        // that overflowed a worker thread's stack).
        let mut batch: Vec<Expression> = Vec::new();
        for c in (0..region.defs.len()).rev() {
            let Some(terms) = pending.remove(&c) else {
                continue;
            };
            if let Some(why) = &region.refusals[c] {
                return Err(why.clone());
            }
            if !region.varied[c] {
                continue;
            }
            let mut reads_batch = false;
            for t in &terms {
                reads_batch |= fields_of(t)?.into_iter().any(|f| f >= width);
            }
            if reads_batch {
                width += batch.len();
                rel = project(rel, std::mem::take(&mut batch));
            }
            let here = width + batch.len();
            batch.push(self.null_skipping_sum(terms));
            match &region.defs[c] {
                Def::Expr(e) => {
                    for (read, term) in self.map(e, here, &region.varied)? {
                        pending.entry(read).or_default().push(term);
                    }
                }
                Def::Input { slot, col } => {
                    at_inputs.entry(*slot).or_default().insert(*col, here);
                }
                Def::Window { .. } | Def::Const => {
                    return Err(AdError::Internal(format!(
                        "gradient reached column {c}, a {:?}, with no refusal recorded",
                        region.defs[c]
                    )))
                }
            }
        }
        if !batch.is_empty() {
            rel = project(rel, batch);
        }
        if !pending.is_empty() {
            return Err(AdError::Internal(format!(
                "cotangents for columns {:?} were never propagated",
                pending.keys().collect::<Vec<_>>()
            )));
        }
        for (slot, cols) in at_inputs {
            self.broadcast(region, &rel, slot, cols)?;
        }
        Ok(())
    }

    /// A window function in a region is recomputed in the backward pass, and
    /// a filter on its rank must keep the same rows it kept forward. SQL does
    /// not order ties, so an engine may rank tied rows differently each time;
    /// that only cannot happen when the ranking is total. So every window
    /// must partition or order by each dim of the rows beneath it; otherwise
    /// the program is refused rather than risk sending gradient to rows the
    /// forward pass did not keep.
    fn check_rankings_are_total(&self, region: &Region) -> Result<()> {
        if region.volatile {
            return Err(AdError::NotImplemented(
                "a volatile function (random(), now(), …) in rows that carry gradient: ddx \
                 recomputes them for the backward pass, and the function would not give the \
                 same values again; materialize its result as a table first"
                    .into(),
            ));
        }
        // Constant data is recomputed too, and ddx has no dims for it, so it
        // cannot show a ranking or LIMIT inside it is total.
        if region
            .slots
            .iter()
            .any(|s| s.input == Input::Const && s.ordered)
        {
            return Err(AdError::NotImplemented(
                "a window function or LIMIT over data that reads no wrt table, joined into                  rows that carry gradient: ddx recomputes it for the backward pass and cannot                  show it keeps the same rows (its ties may be ordered differently);                  materialize it as a table first"
                    .into(),
            ));
        }
        for cut in &region.cuts {
            for &(input, offset, one) in &cut.covers {
                let dims = match input {
                    Input::Table(t) => self.f.tables[t].dims.clone(),
                    Input::Saved(n) => self.f.saved[n].dims(),
                    Input::Const if one => continue,
                    Input::Const => {
                        return Err(AdError::NotImplemented(
                            "a LIMIT, or a ranking on a semi-join's right side, over rows                              joined to data ddx has no dims for: ddx cannot show which rows                              it keeps when it recomputes them"
                                .into(),
                        ))
                    }
                };
                if one {
                    continue;
                }
                let total = match &cut.keys {
                    Some(keys) => dims.iter().all(|d| keys.contains(&(offset + d))),
                    None => false,
                };
                if !total {
                    return Err(AdError::NotImplemented(
                        "a LIMIT (or a ranking on a semi-join's right side) whose ORDER BY                          does not include every dim of the rows it cuts. The rows it keeps                          may differ when ddx recomputes it for the backward pass; add the                          remaining dims to its ORDER BY to break ties"
                            .into(),
                    ));
                }
            }
        }
        for (c, def) in region.defs.iter().enumerate() {
            let Def::Window { keys } = def else { continue };
            for s in &region.slots {
                let Some(offset) = s.offset.filter(|&o| o < c) else {
                    continue;
                };
                let dims = match s.input {
                    Input::Table(t) => self.f.tables[t].dims.clone(),
                    Input::Saved(n) => self.f.saved[n].dims(),
                    Input::Const if s.at_most_one_row => continue,
                    Input::Const => {
                        return Err(AdError::NotImplemented(
                            "a window function over rows joined to data ddx has no dims \
                             for: ddx cannot show its ranking is total, and recomputing it \
                             could keep different rows than the forward pass"
                                .into(),
                        ))
                    }
                };
                if dims.iter().any(|d| !keys.contains(&(offset + d))) {
                    return Err(AdError::NotImplemented(
                        "a window function whose PARTITION BY and ORDER BY do not include \
                         every dim of the rows it ranks. Its ties may be ranked differently \
                         when ddx recomputes it for the backward pass; add the remaining dims \
                         to its ORDER BY to break ties"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The map rule: `y = f(x₁, …)` with cotangent in column `cotangent`
    /// gives each varied `xᵢ` the term `ȳ · ∂f/∂xᵢ`.
    fn map(
        &mut self,
        f: &Expression,
        cotangent: usize,
        varied: &[bool],
    ) -> Result<Vec<(usize, Expression)>> {
        let mul = self.ext.anchor("multiply");
        let partials = self.ew.partials(f, &|c| varied[c], &mut self.ext)?;
        Ok(partials
            .into_iter()
            .map(|(x, d)| {
                let term = if as_number(&d) == Some(1.0) {
                    field(cotangent)
                } else {
                    call(mul, vec![field(cotangent), d])
                };
                (x, term)
            })
            .collect())
    }

    /// The broadcast rule, at an input: a join copied each input row to every
    /// row it matched, so the input's cotangent is the sum over those rows,
    /// grouped by the input's dims.
    fn broadcast(
        &mut self,
        region: &Region,
        rel: &Rel,
        slot: usize,
        cols: BTreeMap<usize, usize>,
    ) -> Result<()> {
        let s = &region.slots[slot];
        let offset = s.offset.ok_or_else(|| {
            AdError::Internal("gradient reached an input that is not part of the output".into())
        })?;
        let dims = match s.input {
            Input::Table(t) => self.f.tables[t].dims.clone(),
            Input::Saved(n) => self.f.saved[n].dims(),
            Input::Const => {
                return Err(AdError::Internal(
                    "gradient reached a constant input".into(),
                ))
            }
        };
        let sum = self.ext.anchor("sum");
        let groupings = dims.iter().map(|d| field(offset + d)).collect();
        let measures = cols.values().map(|&at| (sum, vec![field(at)])).collect();
        self.contributions
            .entry(s.input)
            .or_default()
            .push(Contribution {
                rel: aggregate(rel.clone(), groupings, measures),
                cols: cols.keys().copied().collect(),
            });
        Ok(())
    }
}

/// Can column `col` of `region` differ in its last bits from one
/// recomputation to the next? Only if it reads the output of an aggregate
/// that rounds: a saved aggregate, constant data a sum or average computes,
/// or a scalar subquery that does, since a grouped sum over several
/// partitions adds in arrival order. A table's values, whether or not the
/// table is differentiated, a maximum or count of them, and elementwise
/// functions of those, are the same every run (#102: otherwise a table's
/// gradient depended on which other tables were differentiated).
fn jitters(region: &Region, col: usize) -> bool {
    let mut stack = vec![col];
    let mut seen = vec![false; region.defs.len()];
    while let Some(c) = stack.pop() {
        if std::mem::replace(&mut seen[c], true) {
            continue;
        }
        match &region.defs[c] {
            Def::Input { slot, .. } => {
                if !matches!(region.slots[*slot].input, Input::Table(_)) {
                    return true;
                }
            }
            Def::Expr(e) => {
                if region.rounds[c] {
                    return true;
                }
                match fields_of(e) {
                    Ok(fields) => stack.extend(fields),
                    Err(_) => return true,
                }
            }
            Def::Const => {
                let slot = region.slots.iter().find(|s| {
                    s.input == Input::Const && s.offset.is_some_and(|o| o <= c && c < o + s.width)
                });
                if slot.is_none_or(|s| s.rounds) {
                    return true;
                }
            }
            Def::Window { .. } => return true,
        }
    }
    false
}

/// Which reduce rule a measure has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rule {
    /// `SUM`: every row gets the group's cotangent.
    Sum,
    /// `AVG`: every row gets the group's cotangent over the group's count.
    Mean,
    /// `MAX` or `MIN` (the name): the rows attaining it share the group's
    /// cotangent.
    Extreme(&'static str),
}

/// What the reduce rules do with one measure.
enum Reduce {
    /// The cotangent flows into this argument by this rule.
    Of(Rule, Box<Expression>),
    /// Unchanged when its argument changes: no cotangent flows.
    Constant,
}

/// The reduce rule for an aggregate function, or a refusal naming what ddx
/// has rules for.
fn reduce_rule(functions: &crate::Functions, f: &AggregateFunction) -> Result<Reduce> {
    let name = functions.name(f.function_reference)?;
    let args: Vec<&Expression> = f
        .arguments
        .iter()
        .filter_map(|a| match &a.arg_type {
            Some(ArgType::Value(e)) => Some(e),
            _ => None,
        })
        .collect();
    let rule = match name {
        "count" => return Ok(Reduce::Constant),
        "sum" => Rule::Sum,
        "avg" | "mean" => Rule::Mean,
        "max" => Rule::Extreme("max"),
        "min" => Rule::Extreme("min"),
        other => {
            return Err(AdError::NotImplemented(format!(
                "the aggregate `{other}` over a value that carries gradient; ddx has transpose \
                 rules for SUM, AVG, MAX, MIN and COUNT"
            )))
        }
    };
    // DISTINCT changes which rows a sum or mean counts; it makes no difference
    // to a max or min.
    if f.invocation == AggregationInvocation::Distinct as i32 && !matches!(rule, Rule::Extreme(_)) {
        return Err(AdError::NotImplemented(format!(
            "{}(DISTINCT …) over a value that carries gradient",
            name.to_uppercase()
        )));
    }
    match args.as_slice() {
        [arg] => Ok(Reduce::Of(rule, Box::new((*arg).clone()))),
        _ => Err(AdError::InvalidPlan(format!(
            "{} with {} arguments",
            name.to_uppercase(),
            args.len()
        ))),
    }
}
