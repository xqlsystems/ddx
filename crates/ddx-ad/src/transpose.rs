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
use crate::expr::{as_number, call, field};
use crate::forward::{width, Def, Forward, Input, Output, Region};
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
        let mut seeds = Vec::new();
        for (i, &col) in cols.iter().enumerate() {
            let Output::Value(m) = saved.outputs[col] else {
                return Err(AdError::NotImplemented(
                    "gradient through a GROUP BY key: the aggregate groups by a value that \
                     depends on a wrt column, and that value is then used in the loss"
                        .into(),
                ));
            };
            match reduce_rule(&self.f.functions, &saved.measures[m])? {
                Reduce::Sum(arg) => {
                    seeds.push((width + args.len(), dims.len() + i));
                    args.push(*arg);
                }
                Reduce::Constant => {}
            }
        }
        for e in &args {
            let varied = depends(&self.f.functions, e, &|c| region.varied[c])?;
            region.defs.push(Def::Expr(e.clone()));
            region.varied.push(varied);
            region.refusals.push(None);
        }
        let rows = project(region.rel.clone(), args);

        // The reduce rule's broadcast: every row joined to the cotangent of
        // the group it was summed into.
        let region_width = region.defs.len();
        let keys: Vec<Expression> = dims
            .iter()
            .map(|&d| match saved.outputs[d] {
                Output::Dim(g) => saved.groupings[g].clone(),
                Output::Value(_) => unreachable!("dims() returns grouping columns"),
            })
            .collect();
        let base = self.join_on(rows, cotangent, keys, region_width)?;
        let seeds = seeds
            .into_iter()
            .map(|(arg, cot)| (arg, field(region_width + cot)))
            .collect();
        self.region(&region, base, seeds)
    }

    /// Join `left` (whose columns before `left_width` are a region's) to
    /// `right` on `left_keys[i] = right column i`, null-safely; a cross join
    /// when there are no keys.
    pub fn join_on(
        &mut self,
        left: Rel,
        right: Rel,
        left_keys: Vec<Expression>,
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
            .enumerate()
            .map(|(k, key)| call(same, vec![key, field(left_width + k)]))
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
        let mut width = width(&base)?;
        let mut rel = base;
        let mut pending: BTreeMap<usize, Vec<Expression>> = BTreeMap::new();
        for (c, e) in seeds {
            pending.entry(c).or_default().push(e);
        }
        // (slot, input column) → the column holding its cotangent.
        let mut at_inputs: BTreeMap<usize, BTreeMap<usize, usize>> = BTreeMap::new();
        let add = self.ext.anchor("add");
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
            // This column's cotangent: the sum over the columns that read it.
            let sum = terms
                .into_iter()
                .reduce(|a, b| call(add, vec![a, b]))
                .expect("an entry has at least one term");
            rel = project(rel, vec![sum]);
            let here = width;
            width += 1;
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

/// What the reduce rule does with one measure.
enum Reduce {
    /// A sum: the cotangent is broadcast onto this argument.
    Sum(Box<Expression>),
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
    match name {
        "count" => Ok(Reduce::Constant),
        "sum" => {
            if f.invocation == AggregationInvocation::Distinct as i32 {
                return Err(AdError::NotImplemented(
                    "SUM(DISTINCT …) over a value that carries gradient".into(),
                ));
            }
            match args.as_slice() {
                [arg] => Ok(Reduce::Sum(Box::new((*arg).clone()))),
                _ => Err(AdError::InvalidPlan(format!(
                    "SUM with {} arguments",
                    args.len()
                ))),
            }
        }
        other => Err(AdError::NotImplemented(format!(
            "the aggregate `{other}` over a value that carries gradient; ddx has transpose \
             rules for SUM and COUNT"
        ))),
    }
}
