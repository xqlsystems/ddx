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
//! | **map** | a projected expression `y = f(x₁, x₂, …)` | `x̄ᵢ += ȳ · ∂f/∂xᵢ`, row by row ([`crate::elementwise::Elementwise`]) |
//! | **select** | `WHERE`, a join condition, a semi-join, a filter on a rank | the cotangent stays on the rows that were kept |
//! | **broadcast** | a join, which pairs each row with every row it matches | sum the cotangent back over the rows each input row was copied to |
//! | **concatenate** | `UNION ALL` | each row's cotangent to the input that gave the row |
//! | **reduce** | a grouped `SUM` | broadcast the group's cotangent to every row that was summed |
//! | | `AVG` | the same, divided by the group's count |
//! | | `MAX`, `MIN` | the group's cotangent to the rows that attain it, shared evenly at a tie (`jax.grad`'s convention for `jnp.max`) |
//! | | a window `SUM`, `AVG` | each row's cotangent (over its frame's count, for `AVG`), summed over the frame mirrored |
//! | | a window `MAX`, `MIN` over each whole partition | the grouped rule, from the sum of the partition's cotangents |
//!
//! A matrix product, `SUM(a.v * b.v) … GROUP BY` over a join, is not a
//! primitive: it is broadcast, map and reduce, and its transpose is theirs
//! composed, which gives `Ā = Σ_out C̄·B` and `B̄ = Σ_batch A·C̄`. A `COUNT`
//! does not change when its argument does, so its transpose is zero. Each
//! reduce rule is the transpose of the reduction's linear form,
//! [`crate::reduce`]'s, which forward mode renders as a tangent.
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
use std::rc::Rc;

use ddx_core::Ddx;
use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::expression::window_function::bound::{Following, Kind, Preceding};
use substrait::proto::expression::window_function::Bound;
use substrait::proto::expression::{RexType, WindowFunction};
use substrait::proto::function_argument::ArgType;
use substrait::proto::join_rel::JoinType;
use substrait::proto::rel::RelType;
use substrait::proto::{CrossRel, Expression, FunctionArgument, Rel};

use crate::elementwise::{depends, Elementwise};
use crate::emit::{aggregate, join, project};
use crate::error::{AdError, Result};
use crate::expr::{
    as_number, call, field, fields_of, if_then, lit_f64, map_fields, null_f64, window,
};
use crate::forward::{subquery_rounds, width, Def, Forward, Input, Output, Region};
use crate::functions::{Extensions, Function};
use crate::reduce::{no_rule, reduce_rule, Linear, Reduce};

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
    pub(crate) ew: Elementwise<'a>,
    pub contributions: BTreeMap<Input, Vec<Contribution>>,
    /// Steps the transposes need materialized before the contributions that
    /// read them: a region's cotangents, when several inputs read them.
    /// `(name, relation, column names)`, in the order they must run.
    pub steps: Vec<(String, Rel, Vec<String>)>,
    region_steps: usize,
    /// The windows a region transpose reached, not yet transposed.
    windows: Vec<Gathered>,
}

/// Windows over one set of rows, and their cotangents so far.
struct Gathered {
    /// The rows beneath the windows.
    beneath: Rc<Region>,
    /// Each window function, over `beneath`'s columns, with its cotangents
    /// summed back to those rows: one relation per place that reads it.
    functions: Vec<(Expression, Vec<Rel>)>,
}

impl<'a> Transposer<'a> {
    pub fn new(f: &'a Forward, ddx: &'a Ddx) -> Self {
        Transposer {
            f,
            ext: Extensions::new(&f.functions),
            ew: Elementwise::new(ddx, &f.functions),
            contributions: BTreeMap::new(),
            steps: Vec::new(),
            region_steps: 0,
            windows: Vec::new(),
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
        let width = saved.input.defs.len();
        let dims = saved.dims();

        // Each measure's argument becomes a column of the region, so the
        // chain rule can start from it.
        let mut args = Vec::new();
        // Each measure's reduction, and its index into `cols`.
        let mut reductions = Vec::new();
        for (i, &col) in cols.iter().enumerate() {
            let Output::Value(m) = saved.outputs[col] else {
                return Err(AdError::NotImplemented(
                    "gradient through a GROUP BY key: the aggregate groups by a value that \
                     depends on a wrt column, and that value is then used in the loss (a SELECT \
                     DISTINCT groups by every column it returns, so one over such a column \
                     is the same)"
                        .into(),
                ));
            };
            let f = &saved.measures[m];
            let functions = &self.f.functions;
            let (form, arg) =
                match reduce_rule(functions, f.function_reference, f.invocation, &f.arguments)? {
                    Some(Reduce::Constant) => continue,
                    Some(Reduce::Of(form, arg)) => (form, arg.clone()),
                    None => return Err(no_rule(functions.name(f.function_reference)?)),
                };
            let arg_col = width + args.len();
            reductions.push((Reduction::new(arg_col, form), i));
            args.push(arg);
        }
        // A contraction's transpose needs no rebuilt join (see
        // `crate::contraction`).
        if let ([(r, i)], [arg]) = (&reductions[..], &args[..]) {
            if r.form
                == (Linear {
                    extreme: None,
                    shared: false,
                })
                && self.contraction(n, arg, &cotangent, *i)?
            {
                return Ok(());
            }
        }
        let mut region = saved.input.clone();
        for e in &args {
            let varied = depends(&self.f.functions, e, &|c| region.varied[c])?;
            region.defs.push(Def::Expr(e.clone()));
            region.rounds.push(subquery_rounds(&self.f.functions, e)?);
            region.varied.push(varied);
            region.refusals.push(None);
        }
        let mut rel = project(region.rel.clone(), args);
        let mut next = region.defs.len();
        let keys: Vec<Expression> = dims
            .iter()
            .map(|&d| match saved.outputs[d] {
                Output::Dim(g) => saved.groupings[g].clone(),
                Output::Value(_) => unreachable!("dims() returns grouping columns"),
            })
            .collect();
        for (r, _) in reductions.iter_mut() {
            r.tolerant = jitters(self.f, &region, r.arg);
        }
        let mut counted: Vec<Reduction> = reductions.iter().map(|(r, _)| *r).collect();
        self.counts(&mut rel, &mut next, &mut counted, &keys);

        // Every row joined to the cotangent of the group it went into: the
        // broadcast every reduce rule starts from.
        let cotangent_at = next + dims.len();
        let key_positions: Vec<usize> = (0..dims.len()).collect();
        let base = self.join_on(rel, cotangent, keys, &key_positions, next, JoinType::Inner)?;
        let mut seeds = Vec::new();
        for (r, &(_, i)) in counted.iter().zip(&reductions) {
            seeds.push((r.arg, self.seed(r, field(cotangent_at + i))));
        }
        self.region(&region, base, seeds)
    }

    /// Append to `rel` (`width` columns) the columns each reduction's
    /// [`Self::seed`] reads beside its argument, each a window over the
    /// group's rows (partitioned by `keys`), and record where they are: the
    /// group's extreme, unless the reduction has it already, and the number
    /// of rows that count, when they share the cotangent.
    ///
    /// Each is computed over the recomputed rows themselves, never compared
    /// with a saved aggregate: a recomputation need not be bit-identical to
    /// the forward pass (a grouped SUM over several partitions adds in
    /// arrival order), and a saved maximum no recomputed row equals would
    /// silently send no gradient at all.
    fn counts(
        &mut self,
        rel: &mut Rel,
        width: &mut usize,
        reductions: &mut [Reduction],
        keys: &[Expression],
    ) {
        // NaN arguments are left out of an extreme. DataFusion's grouped MAX
        // skips a NaN where its ungrouped and window MAX return it; a MAX
        // that gave a finite value skipped them, so leaving them out agrees
        // with it, and a NaN row gets no gradient, as a NULL one gets none.
        let isnan = self.ext.anchor(Function::IsNan);
        let mut extremes = Vec::new();
        for r in reductions.iter_mut() {
            let (Some(f), None) = (r.form.extreme, r.extreme) else {
                continue;
            };
            let f = self.ext.anchor(f);
            let arg = if_then(
                vec![(call(isnan, vec![field(r.arg)]), null_f64())],
                field(r.arg),
            );
            r.extreme = Some(*width + extremes.len());
            extremes.push(window(f, vec![arg], keys.to_vec()));
        }
        if !extremes.is_empty() {
            *width += extremes.len();
            *rel = project(std::mem::take(rel), extremes);
        }
        // An AVG divides by its group's count; the rows attaining an extreme
        // share it by how many do.
        let count = self.ext.anchor(Function::Count);
        let sum = self.ext.anchor(Function::Sum);
        let mut totals = Vec::new();
        for r in reductions.iter_mut().filter(|r| r.form.shared) {
            let total = match r.extreme {
                None => window(count, vec![field(r.arg)], keys.to_vec()),
                Some(extreme) => {
                    let attains = if_then(
                        vec![(self.attains(r.arg, extreme, r.tolerant), lit_f64(1.0))],
                        lit_f64(0.0),
                    );
                    window(sum, vec![attains], keys.to_vec())
                }
            };
            r.total = Some(*width + totals.len());
            totals.push(total);
        }
        if !totals.is_empty() {
            *width += totals.len();
            *rel = project(std::mem::take(rel), totals);
        }
    }

    /// The transpose of reduction `r` (whose columns [`Self::counts`]
    /// placed), at one of its rows: the group's cotangent `cotangent` times
    /// the row's `coef` (see [`crate::reduce`]).
    fn seed(&mut self, r: &Reduction, cotangent: Expression) -> Expression {
        let divide = self.ext.anchor(Function::Divide);
        let share = match r.total {
            Some(total) => call(divide, vec![cotangent, field(total)]),
            None => cotangent,
        };
        // Only the rows equal to an extreme get it, shared evenly among
        // them: jax.grad's convention for jnp.max at a tie. The others get
        // none, NULL rather than 0, so it stays none through a partial that
        // is not finite there (0 · ∞ is NaN: an infinite value in the data
        // that does not attain a MIN).
        let seed = match r.extreme {
            None => share,
            Some(extreme) => if_then(
                vec![(self.attains(r.arg, extreme, r.tolerant), share)],
                null_f64(),
            ),
        };
        self.skips_null(r.arg, seed)
    }

    /// `seed`, the cotangent of a reduction's argument (column `arg`), or
    /// NULL where the argument is. A reduction skips a row whose argument is
    /// NULL, so nothing in that row moves the loss: its cotangent is NULL,
    /// not the group's, or the row's other inputs (the `p` of `SUM(p + q)`
    /// where `q` is NULL) would get gradient through it.
    pub(crate) fn skips_null(&mut self, arg: usize, seed: Expression) -> Expression {
        let is_null = self.ext.anchor(Function::IsNull);
        if_then(vec![(call(is_null, vec![field(arg)]), null_f64())], seed)
    }

    /// The dims of the inputs beneath a window, as columns of `beneath`:
    /// they tell its rows apart, so cotangents are summed back to them.
    fn row_keys(&self, beneath: &Region) -> Result<Vec<usize>> {
        let mut keys = Vec::new();
        for (j, slot) in beneath.slots.iter().enumerate() {
            let Some(o) = slot.offset else { continue };
            match slot.input {
                Input::Const if slot.at_most_one_row => continue,
                Input::Const => {
                    return Err(AdError::NotImplemented(
                        "a window function that carries gradient over rows joined to data ddx \
                         has no dims for: ddx cannot tell its rows apart to sum their \
                         cotangents; materialize the data as a table first"
                            .into(),
                    ))
                }
                Input::Table(_) | Input::Saved(_) => {
                    keys.extend(self.slot_dims(beneath, j)?.into_iter().map(|d| o + d))
                }
            }
        }
        // A UNION ALL's inputs can give rows with the same dims.
        keys.extend((0..beneath.defs.len()).filter(|&c| matches!(beneath.defs[c], Def::Branch)));
        Ok(keys)
    }

    /// Gather the cotangent of a region's window column, `function` over the
    /// rows `beneath` it (whose column 0 is the region's column `offset`),
    /// with the cotangent of each of `rel`'s rows in its column `cotangent`.
    ///
    /// A window is a reduction over the rows it read, so its transpose runs
    /// over those rows, each once. Above the window, a filter may have
    /// dropped some, a join copied others, and a sparse cotangent left out
    /// the rest; so `rel`'s cotangents are summed back to the window's rows
    /// by the dims of the inputs beneath it (the broadcast rule). The
    /// transpose itself waits for [`Self::windows`], so that a window read
    /// along several paths is transposed once, from all of them.
    fn gather(
        &mut self,
        beneath: &Rc<Region>,
        offset: usize,
        function: &Expression,
        rel: &Rel,
        cotangent: usize,
    ) -> Result<()> {
        let keys = self.row_keys(beneath)?;
        let sum = self.ext.anchor(Function::Sum);
        let summed = aggregate(
            rel.clone(),
            keys.iter().map(|&k| field(offset + k)).collect(),
            vec![(sum, vec![field(cotangent)])],
        );
        // Materialized, so the transpose beneath reads it rather than
        // carrying a copy of this region's: a plan is a tree, and a copy at
        // each of a stack of windows grew as a power of its depth.
        let name = format!("{}window_{}", self.f.namespace, self.steps_made());
        let names: Vec<String> = (0..=keys.len()).map(|i| format!("c{i}")).collect();
        self.steps.push((name.clone(), summed, names.clone()));
        let summed = crate::emit::read_step(&name, names);
        let function = map_fields(function, &mut |i| Ok(i - offset))?;
        let i = match self
            .windows
            .iter()
            .position(|g| Rc::ptr_eq(&g.beneath, beneath))
        {
            Some(i) => i,
            None => {
                self.windows.push(Gathered {
                    beneath: Rc::clone(beneath),
                    functions: Vec::new(),
                });
                self.windows.len() - 1
            }
        };
        let functions = &mut self.windows[i].functions;
        match functions.iter_mut().find(|(f, _)| *f == function) {
            Some((_, summed_at)) => summed_at.push(summed),
            None => functions.push((function, vec![summed])),
        }
        Ok(())
    }

    /// Transpose every gathered window, each set of rows beneath windows
    /// once. A region that reads a window's rows holds the region beneath
    /// it, so every place that reads a set of rows is a larger region:
    /// taking the largest first, all of a set's cotangents are gathered
    /// before it is transposed.
    pub(crate) fn windows(&mut self) -> Result<()> {
        while let Some(i) =
            (0..self.windows.len()).max_by_key(|&i| self.windows[i].beneath.defs.len())
        {
            let Gathered { beneath, functions } = self.windows.swap_remove(i);
            let keys = self.row_keys(&beneath)?;
            let positions: Vec<usize> = (0..keys.len()).collect();
            let mut rel = beneath.rel.clone();
            let mut width = beneath.defs.len();
            let mut seeds = Vec::new();
            for (function, summed_at) in functions {
                // Each place's sums, joined to the rows beneath the window,
                // recomputed; a row with none gets NULL.
                let mut cotangents = Vec::new();
                for summed in summed_at {
                    rel = self.join_on(
                        rel,
                        summed,
                        keys.iter().map(|&k| field(k)).collect(),
                        &positions,
                        width,
                        JoinType::Left,
                    )?;
                    cotangents.push(field(width + keys.len()));
                    width += keys.len() + 1;
                }
                let cotangent = self.null_skipping_sum(cotangents);
                rel = project(rel, vec![cotangent, function.clone()]);
                let (cotangent, own) = (width, width + 1);
                width += 2;
                seeds.extend(
                    self.window(&beneath, own, &function, cotangent, &mut rel, &mut width)?,
                );
            }
            self.region_once(&beneath, rel, seeds)?;
        }
        Ok(())
    }

    fn window(
        &mut self,
        region: &Region,
        c: usize,
        function: &Expression,
        cotangent: usize,
        rel: &mut Rel,
        width: &mut usize,
    ) -> Result<Vec<(usize, Expression)>> {
        let Some(RexType::WindowFunction(w)) = &function.rex_type else {
            return Err(AdError::Internal(format!(
                "window column {c} holds no window function"
            )));
        };
        let functions = &self.f.functions;
        let Some(Reduce::Of(form, arg)) =
            reduce_rule(functions, w.function_reference, w.invocation, &w.arguments)?
        else {
            return Err(AdError::Internal(format!(
                "gradient reached window column {c}, which has no rule"
            )));
        };
        let arg = arg.clone();
        let sum = self.ext.anchor(Function::Sum);
        let seed = match form.extreme {
            // Row r's value is Σ over the rows s of its frame of coef(r)·ṫ(s),
            // coef(r) 1 for a sum and 1 / (the frame's count) for a mean. The
            // transpose sends ȳ(r)·coef(r) to every row of r's frame, so row
            // s gets the sum over the rows whose frame holds s: the frame
            // mirrored (k PRECEDING ↔ k FOLLOWING), in the same order. Over
            // a whole partition, that is the partition.
            None => {
                let arg_col = *width;
                let mut cols = vec![arg.clone()];
                let mut share = field(cotangent);
                if form.shared {
                    let count = self.ext.anchor(Function::Count);
                    let divide = self.ext.anchor(Function::Divide);
                    cols.push(over(w, count, vec![arg.clone()]));
                    share = call(divide, vec![share, field(arg_col + 1)]);
                }
                *width += cols.len();
                *rel = project(std::mem::take(rel), cols);
                let mut mirrored = over(w, sum, vec![share]);
                if let Some(RexType::WindowFunction(m)) = &mut mirrored.rex_type {
                    let (lower, upper) = (m.lower_bound.take(), m.upper_bound.take());
                    m.lower_bound = upper.map(mirror);
                    m.upper_bound = lower.map(mirror);
                }
                *rel = project(std::mem::take(rel), vec![mirrored]);
                *width += 1;
                self.skips_null(arg_col, field(*width - 1))
            }
            // Only over a whole partition (forward mode refuses a running
            // extreme): the partition's cotangent is the sum of its rows',
            // and the extreme's rule passes it on. The window's own column
            // is the partition's extreme, as forward mode reads it: a
            // partition whose MAX is NaN has no row attaining it.
            Some(_) => {
                let keys = w.partitions.clone();
                let arg_col = *width;
                *rel = project(
                    std::mem::take(rel),
                    vec![
                        arg.clone(),
                        window(sum, vec![field(cotangent)], keys.clone()),
                    ],
                );
                *width += 2;
                let mut tolerant = subquery_rounds(&self.f.functions, &arg)?;
                for f in fields_of(&arg)? {
                    tolerant |= jitters(self.f, region, f);
                }
                let mut r = [Reduction {
                    extreme: Some(c),
                    tolerant,
                    ..Reduction::new(arg_col, form)
                }];
                self.counts(rel, width, &mut r, &keys);
                self.seed(&r[0], field(arg_col + 1))
            }
        };
        *rel = project(std::mem::take(rel), vec![seed]);
        *width += 1;
        self.map(&arg, *width - 1, &region.varied)
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
        let add = self.ext.anchor(Function::Add);
        let and = self.ext.anchor(Function::And);
        let is_null = self.ext.anchor(Function::IsNull);
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
    /// (column `extreme`)? See [`crate::reduce::attains`]. The extreme and
    /// the arguments are recomputed, which is why an argument that can
    /// jitter (see [`jitters`]) gets the tolerance.
    fn attains(&mut self, arg: usize, extreme: usize, tolerant: bool) -> Expression {
        crate::reduce::attains(&mut self.ext, field(arg), field(extreme), tolerant)
    }

    /// Join `left` (whose columns before `left_width` are a region's) to
    /// `right`, by `kind`, on `left_keys[k] = right column right_keys[k]`, null-safely; a
    /// cross join when there are no keys.
    pub fn join_on(
        &mut self,
        left: Rel,
        right: Rel,
        left_keys: Vec<Expression>,
        right_keys: &[usize],
        left_width: usize,
        kind: JoinType,
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
        let same = self.ext.anchor(Function::IsNotDistinctFrom);
        let and = self.ext.anchor(Function::And);
        let cond = left_keys
            .into_iter()
            .zip(right_keys)
            .map(|(key, &r)| call(same, vec![key, field(left_width + r)]))
            .reduce(|a, b| call(and, vec![a, b]))
            .expect("at least one key");
        Ok(join(left, right, cond, kind))
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
        self.region_once(region, base, seeds)?;
        self.windows()
    }

    /// [`Self::region`], leaving the windows it reaches gathered.
    pub(crate) fn region_once(
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
                Def::Window {
                    function,
                    beneath,
                    offset,
                    ..
                } => {
                    // The window reads this column's cotangent.
                    width += batch.len();
                    rel = project(rel, std::mem::take(&mut batch));
                    self.gather(beneath, *offset, function, &rel, here)?;
                }
                Def::Union(arms) => {
                    // Each input's column gets the cotangent in the rows it
                    // gave, and NULL, none, in the others'. An input whose
                    // column reads no wrt table gets none, as a map's
                    // constant operand does.
                    for &(branch, col) in arms.iter().filter(|&&(_, col)| region.varied[col]) {
                        pending
                            .entry(col)
                            .or_default()
                            .push(if_then(vec![(field(branch), field(here))], null_f64()));
                    }
                }
                Def::Const | Def::Branch => {
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
        // Each input's contribution sums its cotangents over the region's
        // rows, so each rebuilds the region. A region that reads one relation
        // twice (a CTE read twice) doubles its inputs at each level, so its
        // backward step grew as the square of the forward query. With three
        // or more inputs, the columns the contributions read, each input's
        // dims and cotangents, are materialized once, and every contribution
        // reads them. With two (a contraction of two varied operands) the
        // region is rebuilt twice instead, which costs at most twice and
        // never writes a contraction's join (N x D x H rows) out.
        if at_inputs.len() > 2 {
            let mut needed = std::collections::BTreeSet::new();
            for (&slot, cols) in &at_inputs {
                let offset = region.slots[slot].offset.unwrap_or(0);
                needed.extend(self.slot_dims(region, slot)?.iter().map(|d| offset + d));
                needed.extend(cols.values().copied());
            }
            let needed: Vec<usize> = needed.into_iter().collect();
            let name = format!("{}region_{}", self.f.namespace, self.steps_made());
            let names: Vec<String> = (0..needed.len()).map(|i| format!("c{i}")).collect();
            self.steps.push((
                name.clone(),
                crate::emit::select(rel, needed.clone()),
                names.clone(),
            ));
            let read = crate::emit::read_step(&name, names);
            let at = |c: usize| {
                needed
                    .binary_search(&c)
                    .expect("a column materialized above")
            };
            for (slot, cols) in at_inputs {
                self.broadcast(region, &read, slot, cols, &at)?;
            }
            return Ok(());
        }
        for (slot, cols) in at_inputs {
            self.broadcast(region, &rel, slot, cols, &|c| c)?;
        }
        Ok(())
    }

    /// How many region steps this program has made so far, counting those
    /// already handed to the program.
    pub(crate) fn steps_made(&mut self) -> usize {
        self.region_steps += 1;
        self.region_steps - 1
    }

    /// The dims of input `slot` of `region`, as columns of the input.
    fn slot_dims(&self, region: &Region, slot: usize) -> Result<Vec<usize>> {
        match region.slots[slot].input {
            Input::Table(t) => Ok(self.f.tables[t].dims.clone()),
            Input::Saved(n) => Ok(self.f.saved[n].dims()),
            Input::Const => Err(AdError::Internal(
                "gradient reached a constant input".into(),
            )),
        }
    }

    /// A window function in a region is recomputed in the backward pass, and
    /// a filter on its rank must keep the same rows it kept forward. SQL does
    /// not order ties, so an engine may rank tied rows differently each time;
    /// that only cannot happen when the ranking is total. So every window
    /// must partition or order by each dim of the rows beneath it; otherwise
    /// the program is refused rather than risk sending gradient to rows the
    /// forward pass did not keep.
    pub(crate) fn check_rankings_are_total(&self, region: &Region) -> Result<()> {
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
                "a window function or LIMIT over data that reads no wrt table, joined into \
                 rows that carry gradient: ddx recomputes it for the backward pass and cannot \
                 show it keeps the same rows (its ties may be ordered differently); \
                 materialize it as a table first"
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
                            "a LIMIT, or a ranking on a semi-join's right side, over rows \
                             joined to data ddx has no dims for: ddx cannot show which rows \
                             it keeps when it recomputes them"
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
                        "a LIMIT (or a ranking on a semi-join's right side) whose ORDER BY \
                         does not include every dim of the rows it cuts. The rows it keeps \
                         may differ when ddx recomputes it for the backward pass; add the \
                         remaining dims to its ORDER BY to break ties"
                            .into(),
                    ));
                }
            }
        }
        for (c, def) in region.defs.iter().enumerate() {
            let Def::Window {
                keys: Some(keys), ..
            } = def
            else {
                continue;
            };
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

    /// The transpose of a map, a projection computing each row from one input
    /// row: `y = f(x₁, …)` with cotangent in column `cotangent`
    /// gives each varied `xᵢ` the term `ȳ · ∂f/∂xᵢ`.
    fn map(
        &mut self,
        f: &Expression,
        cotangent: usize,
        varied: &[bool],
    ) -> Result<Vec<(usize, Expression)>> {
        let mul = self.ext.anchor(Function::Multiply);
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
    /// `at` maps a column of the region (with its cotangents) to its column
    /// in `rel`, which is the region itself or the columns of it a region
    /// step materialized.
    fn broadcast(
        &mut self,
        region: &Region,
        rel: &Rel,
        slot: usize,
        cols: BTreeMap<usize, usize>,
        at: &dyn Fn(usize) -> usize,
    ) -> Result<()> {
        let s = &region.slots[slot];
        let offset = s.offset.ok_or_else(|| {
            AdError::Internal("gradient reached an input that is not part of the output".into())
        })?;
        let dims = self.slot_dims(region, slot)?;
        let sum = self.ext.anchor(Function::Sum);
        let groupings = dims.iter().map(|d| field(at(offset + d))).collect();
        let measures = cols.values().map(|&c| (sum, vec![field(at(c))])).collect();
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

/// Window function `w`'s frame, partitions and ordering, computing `anchor`
/// of `args`.
fn over(w: &WindowFunction, anchor: u32, args: Vec<Expression>) -> Expression {
    Expression {
        rex_type: Some(RexType::WindowFunction(WindowFunction {
            function_reference: anchor,
            arguments: args
                .into_iter()
                .map(|e| FunctionArgument {
                    arg_type: Some(ArgType::Value(e)),
                })
                .collect(),
            options: Vec::new(),
            output_type: None,
            invocation: AggregationInvocation::All as i32,
            ..w.clone()
        })),
    }
}

/// A frame bound seen from the other side: `k PRECEDING` ↔ `k FOLLOWING`.
fn mirror(b: Bound) -> Bound {
    Bound {
        kind: b.kind.map(|k| match k {
            Kind::Preceding(p) => Kind::Following(Following { offset: p.offset }),
            Kind::Following(f) => Kind::Preceding(Preceding { offset: f.offset }),
            other => other,
        }),
    }
}

/// One reduction's transpose: its argument's column and linear form, where
/// [`Transposer::counts`] put the columns its seed reads, and whether its
/// argument can jitter between recomputations (see [`jitters`]).
#[derive(Debug, Clone, Copy)]
struct Reduction {
    arg: usize,
    form: Linear,
    tolerant: bool,
    extreme: Option<usize>,
    total: Option<usize>,
}

impl Reduction {
    fn new(arg: usize, form: Linear) -> Self {
        Reduction {
            arg,
            form,
            tolerant: false,
            extreme: None,
            total: None,
        }
    }
}

/// Can column `col` of `region` differ in its last bits from one
/// recomputation to the next? Only if it reads the output of an aggregate
/// that rounds: a saved `SUM` or `AVG`, constant data a sum or average
/// computes, or a scalar subquery that does, since a grouped sum over several
/// partitions adds in arrival order. A table's values, whether or not the
/// table is differentiated, a maximum or count of them, and elementwise
/// functions of those, are the same every run (#102: otherwise a table's
/// gradient depended on which other tables were differentiated). A saved
/// aggregate is followed into the region it aggregates: its `MAX` of a
/// table's values is exact, its `MAX` of a `SUM` is not.
fn jitters(f: &Forward, region: &Region, col: usize) -> bool {
    let mut stack = vec![col];
    let mut seen = vec![false; region.defs.len()];
    while let Some(c) = stack.pop() {
        if std::mem::replace(&mut seen[c], true) {
            continue;
        }
        match &region.defs[c] {
            Def::Input { slot, col } => match region.slots[*slot].input {
                Input::Table(_) => {}
                Input::Saved(n) => {
                    if saved_jitters(f, n, *col) {
                        return true;
                    }
                }
                Input::Const => return true,
            },
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
            Def::Branch => {}
            Def::Union(arms) => stack.extend(arms.iter().map(|&(_, col)| col)),
        }
    }
    false
}

/// Can output column `col` of saved aggregate `n` differ in its last bits
/// between recomputations? A measure that rounds can; so can a key or a
/// measure computed from columns that can.
fn saved_jitters(f: &Forward, n: usize, col: usize) -> bool {
    let saved = &f.saved[n];
    let over = |e: &Expression| -> bool {
        if subquery_rounds(&f.functions, e).unwrap_or(true) {
            return true;
        }
        match fields_of(e) {
            Ok(fields) => fields.into_iter().any(|c| jitters(f, &saved.input, c)),
            Err(_) => true,
        }
    };
    match saved.outputs.get(col) {
        Some(Output::Dim(g)) => saved.groupings.get(*g).is_none_or(over),
        Some(Output::Value(m)) => {
            let Some(measure) = saved.measures.get(*m) else {
                return true;
            };
            if f.functions
                .rounds(measure.function_reference)
                .unwrap_or(true)
            {
                return true;
            }
            measure.arguments.iter().any(|a| match &a.arg_type {
                Some(substrait::proto::function_argument::ArgType::Value(e)) => over(e),
                _ => false,
            })
        }
        None => true,
    }
}
