// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Forward mode: tangents beside values, through the same operators.
//!
//! A relation's **dual** is the relation with one more column for each of
//! its columns that depends on a `wrt` column: the column's tangent, its
//! directional derivative along the caller's tangent of the `wrt` columns.
//! The dual keeps every column where the relation had it and appends the
//! tangents after them, so an expression over the relation's columns reads
//! the same columns of its dual. Each operator carries tangents its own way
//! (design.md §8, M4.5):
//!
//! | Primitive | Tangent |
//! |---|---|
//! | map `y = f(x₁, …)` | `ẏ = Σ ∂f/∂xᵢ · ẋᵢ`, the partials from `ddx-core` |
//! | select (filter, join condition, semi-join, `LIMIT`, sort) | the kept rows keep their tangents |
//! | join, cross join | each side's tangents come along with its values |
//! | `SUM`, `AVG` (aggregate or window) | the same function over the tangents |
//! | `MAX`, `MIN` | the mean tangent of the rows attaining it (`jax.jvp` of `jnp.max`) |
//! | `COUNT` | none |
//! | `UNION ALL` | each input's tangents, zero where an input has none |
//! | `ddx_stop_gradient(x)` | none |
//!
//! A `wrt` table is read joined to its tangent table on its dims; a row the
//! tangent table lacks has tangent 0. An earlier step of a program (forward
//! over reverse) is read from its own dual, already materialized.
//!
//! Nothing is recomputed and nothing is saved: values and tangents are
//! computed together, in one pass, so a `MAX`'s attaining rows are compared
//! with exactly the values the `MAX` saw, with no tolerance.
//!
//! # NULL
//!
//! A tangent is NULL where its value is, and a NULL tangent beside a value
//! that is not NULL means zero: the value does not move with the NULL it was
//! computed past (`coalesce(x, 0)` where `x` is NULL). Aggregates and the
//! final output read tangents through [`Dualizer::masked`], which pins both.
//!
//! # Refusals
//!
//! Some columns that depend on `wrt` have no tangent ddx can give: a rank, a
//! `GROUP BY` key computed from a `wrt` value, an aggregate with no rule.
//! Each carries the refusal, raised only if something reads its tangent, the
//! way reverse mode refuses only where gradient reaches.

use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::aggregate_rel::Measure;
use substrait::proto::expression::window_function::bound::Kind as BoundKind;
use substrait::proto::expression::window_function::Bound;
use substrait::proto::expression::{RexType, WindowFunction};
use substrait::proto::function_argument::ArgType;
use substrait::proto::join_rel::JoinType;
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::set_rel::SetOp;
use substrait::proto::{
    AggregateFunction, AggregateRel, CrossRel, Expression, FetchRel, FilterRel, JoinRel,
    NamedStruct, ProjectRel, ReadRel, Rel, RelCommon, SetRel, SortRel,
};

use crate::elementwise::{depends, Elementwise};
use crate::emit::{self, project_emit, read_step, select};
use crate::error::{AdError, Result};
use crate::expr::{
    as_field, as_number, call, cast, field, fp64, if_then, lit_f64, map_fields, null_f64,
    uncorrelated_scalar,
};
use crate::forward::{
    apply_emit, collect_subqueries, grouping_expressions, input, read_outputs, rel_expressions,
    rel_inputs, rel_name, width,
};
use crate::functions::{Extensions, Functions};
use crate::relation::Table;

/// A column's tangent.
#[derive(Debug, Clone)]
pub(crate) enum Tan {
    /// It does not depend on a `wrt` column: no tangent, as if zero.
    Zero,
    /// Column `i` of the dual holds it.
    Col(usize),
    /// It depends on a `wrt` column, but has no tangent ddx can give.
    Refused(AdError),
}

impl Tan {
    fn varied(&self) -> bool {
        !matches!(self, Tan::Zero)
    }
}

/// A relation's dual: the relation's `width` columns, then the tangents
/// `tans` point to.
#[derive(Debug, Clone)]
pub(crate) struct Dual {
    pub rel: Rel,
    pub width: usize,
    pub tans: Vec<Tan>,
}

/// Where a read of a named table gets its tangents.
#[derive(Debug, Clone)]
pub(crate) enum Source {
    /// A `wrt` table, joined to the table `tangent` on its dims. That table
    /// holds the dims, then one tangent per value, in the table's order.
    Wrt { table: Table, tangent: String },
    /// An earlier step, read from its dual: the table `dual`, whose columns
    /// are `columns`, the step's own first; the step's column `c` has the
    /// tangent `tans[c]`, a [`Tan::Col`] naming a column of the dual.
    Step {
        dual: String,
        columns: Vec<String>,
        tans: Vec<Tan>,
    },
}

/// Where a read of the table with these names (and the schema the read
/// gives it) gets its tangents, or `None` for constant data.
pub(crate) type SourceFn<'a> =
    dyn FnMut(&[String], Option<&NamedStruct>) -> Result<Option<Source>> + 'a;

/// Builds duals of one plan's relations.
pub(crate) struct Dualizer<'a> {
    functions: &'a Functions,
    ew: Elementwise<'a>,
    /// The function declarations of the plan being written.
    pub ext: Extensions,
    source: &'a mut SourceFn<'a>,
}

/// A tangent before its relation's emit: `Col` is a direct column.
type DirectTan = Tan;

impl<'a> Dualizer<'a> {
    pub fn new(
        ddx: &'a ddx_core::Ddx,
        functions: &'a Functions,
        source: &'a mut SourceFn<'a>,
    ) -> Self {
        Dualizer {
            functions,
            ew: Elementwise::new(ddx, functions),
            ext: Extensions::new(functions),
            source,
        }
    }

    /// The dual of `rel`.
    ///
    /// A chain of single-input relations can be hundreds deep (a program's
    /// region step is a projection per batch of columns), deeper than a
    /// 2 MB worker stack allows for recursion. So the chain is walked down
    /// with a loop, and each relation's dual built over its input's on the
    /// way back up; only joins and unions recurse.
    pub fn dual(&mut self, rel: &Rel) -> Result<Dual> {
        let mut chain: Vec<&RelType> = Vec::new();
        let mut cur = rel;
        let mut d = loop {
            if !self.reads_varied(cur)? {
                let width = width(cur)?;
                break Dual {
                    rel: cur.clone(),
                    width,
                    tans: vec![Tan::Zero; width],
                };
            }
            let kind = cur
                .rel_type
                .as_ref()
                .ok_or_else(|| AdError::InvalidPlan("an empty relation".into()))?;
            self.refuse_varied_subqueries(kind)?;
            let below = match kind {
                RelType::Project(p) => &p.input,
                RelType::Filter(f) => &f.input,
                RelType::Sort(s) => &s.input,
                RelType::Fetch(f) => &f.input,
                RelType::Aggregate(a) => &a.input,
                RelType::Read(r) => break self.read(r)?,
                RelType::Join(j) => break self.join(j)?,
                RelType::Cross(c) => break self.cross(c)?,
                RelType::Set(s) => break self.set(s)?,
                other => {
                    return Err(AdError::NotImplemented(format!(
                        "a {} relation on the path from a wrt table to the output",
                        rel_name(other)
                    )))
                }
            };
            chain.push(kind);
            cur = input(below)?;
        };
        while let Some(kind) = chain.pop() {
            d = match kind {
                RelType::Project(p) => self.project(p, d)?,
                RelType::Filter(f) => self.filter(f, d)?,
                RelType::Sort(s) => self.sort(s, d)?,
                RelType::Fetch(f) => self.fetch(f, d)?,
                RelType::Aggregate(a) => self.aggregate(a, d)?,
                _ => unreachable!("only single-input relations are chained"),
            };
        }
        Ok(d)
    }

    /// `tan`, NULL where `value` is NULL and 0 where it is NULL beside a
    /// value that is not (see the module docs).
    pub fn masked(&mut self, value: Expression, tan: Expression) -> Expression {
        let is_null = self.ext.anchor("is_null");
        if_then(
            vec![
                (call(is_null, vec![value]), null_f64()),
                (call(is_null, vec![tan.clone()]), lit_f64(0.0)),
            ],
            tan,
        )
    }

    /// Does anything under `rel` read a table with tangents? Conservatively
    /// true for a subquery that is not an uncorrelated scalar one.
    fn reads_varied(&mut self, rel: &Rel) -> Result<bool> {
        let mut stack = vec![rel];
        while let Some(r) = stack.pop() {
            let Some(kind) = &r.rel_type else { continue };
            if let RelType::Read(read) = kind {
                if let Some(ReadType::NamedTable(t)) = &read.read_type {
                    if (self.source)(&t.names, read.base_schema.as_ref())?.is_some() {
                        return Ok(true);
                    }
                }
                continue;
            }
            for e in rel_expressions(kind) {
                let mut subqueries = Vec::new();
                collect_subqueries(e, &mut subqueries);
                for sq in subqueries {
                    match uncorrelated_scalar(sq) {
                        Some(inner) => stack.push(inner),
                        None => return Ok(true),
                    }
                }
            }
            stack.extend(rel_inputs(kind));
        }
        Ok(false)
    }

    /// A subquery in a relation on the varied path is carried through as a
    /// constant, which is only right if it reads nothing varied.
    fn refuse_varied_subqueries(&mut self, kind: &RelType) -> Result<()> {
        for e in rel_expressions(kind) {
            let mut subqueries = Vec::new();
            collect_subqueries(e, &mut subqueries);
            for sq in subqueries {
                let varied = match uncorrelated_scalar(sq) {
                    Some(inner) => self.reads_varied(inner)?,
                    None => true,
                };
                if varied {
                    return Err(AdError::NotImplemented(
                        "a subquery that is correlated, or reads a wrt table, inside an \
                         expression on the path from a wrt table to the output; write it as a \
                         join"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The tangent of `e`, an expression over the columns of a dual whose
    /// tangents are `tans`: `Σ ∂e/∂x · ẋ`. `None` when `e` depends on no
    /// varied column.
    fn tangent_of(&mut self, e: &Expression, tans: &[Tan]) -> Result<Option<Expression>> {
        let varied = |f: usize| tans.get(f).is_some_and(Tan::varied);
        if !depends(self.functions, e, &varied)? {
            return Ok(None);
        }
        let partials = self.ew.partials(e, &varied, &mut self.ext)?;
        let mul = self.ext.anchor("multiply");
        let add = self.ext.anchor("add");
        let is_null = self.ext.anchor("is_null");
        let mut terms = Vec::new();
        for (x, d) in partials {
            let dx = match &tans[x] {
                Tan::Col(j) => field(*j),
                Tan::Refused(why) => return Err(why.clone()),
                Tan::Zero => continue,
            };
            terms.push(if as_number(&d) == Some(1.0) {
                dx
            } else {
                call(mul, vec![d, dx])
            });
        }
        // A NULL term is a NULL input, which moves nothing (the module docs'
        // NULL rule): it adds 0, or `coalesce(a, 0) + b` with `a` NULL would
        // lose `ḃ`. A lone term is left as it is; a reader masks it.
        // It depends on a wrt column, through conditions only (a CASE test, a
        // comparison): its tangent is 0, as jax.jvp and grad give, not none.
        if terms.is_empty() {
            return Ok(Some(lit_f64(0.0)));
        }
        if terms.len() > 1 {
            terms = terms
                .into_iter()
                .map(|t| if_then(vec![(call(is_null, vec![t.clone()]), lit_f64(0.0))], t))
                .collect();
        }
        Ok(terms.into_iter().reduce(|a, b| call(add, vec![a, b])))
    }

    /// The dual of a relation whose direct output (before its emit) is
    /// `direct` columns wide with tangents `tans`, its emit `common`. The
    /// relation itself is `build(emit)`, given the emit to write.
    fn emitted(
        &self,
        common: Option<&RelCommon>,
        width: usize,
        tans: Vec<DirectTan>,
        build: impl FnOnce(Option<RelCommon>) -> Rel,
        direct_of: &dyn Fn(usize) -> usize,
    ) -> Result<Dual> {
        let outputs = apply_emit(common, (0..width).collect())?;
        let mut emit: Vec<usize> = outputs.iter().map(|&o| direct_of(o)).collect();
        let n = outputs.len();
        let mut out_tans = Vec::with_capacity(n);
        for &o in &outputs {
            out_tans.push(match &tans[o] {
                Tan::Col(j) => {
                    emit.push(*j);
                    Tan::Col(emit.len() - 1)
                }
                other => other.clone(),
            });
        }
        // The relation's direct columns are its own, then one per tangent.
        // An emit that keeps them all, in order, needs no projection.
        let direct = width + tans.iter().filter(|t| matches!(t, Tan::Col(_))).count();
        let identity = emit.len() == direct && emit.iter().enumerate().all(|(i, &c)| i == c);
        Ok(Dual {
            rel: if identity {
                build(None)
            } else {
                with_emit(build(None), emit)
            },
            width: n,
            tans: out_tans,
        })
    }

    fn read(&mut self, r: &ReadRel) -> Result<Dual> {
        let Some(ReadType::NamedTable(t)) = &r.read_type else {
            return Err(AdError::Internal(
                "reads_varied() let through a read that is not a named table".into(),
            ));
        };
        let source = (self.source)(&t.names, r.base_schema.as_ref())?
            .ok_or_else(|| AdError::Internal("a varied read with no source".into()))?;
        let outputs = apply_emit(r.common.as_ref(), read_outputs(r)?)?;
        let filter = r.filter.as_deref().cloned();
        match source {
            Source::Wrt { table, tangent } => {
                let schema = table.schema.clone();
                let w = schema.names.len();
                let (k, v) = (table.dims.len(), table.values.len());
                let columns: Vec<String> = table
                    .dims
                    .iter()
                    .chain(&table.values)
                    .map(|&c| table.columns()[c].clone())
                    .collect();
                let base = emit::read(t.names.clone(), schema, filter, None);
                let same = self.ext.anchor("is_not_distinct_from");
                let and = self.ext.anchor("and");
                let cond = table
                    .dims
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| call(same, vec![field(d), field(w + i)]))
                    .reduce(|a, b| call(and, vec![a, b]))
                    .expect("a wrt table has a dim");
                let joined = emit::join(base, read_step(&tangent, columns), cond, JoinType::Left);
                let mut exprs = Vec::new();
                let mut emit: Vec<usize> = outputs.clone();
                let mut tans = Vec::with_capacity(outputs.len());
                for &o in &outputs {
                    match table.values.iter().position(|&x| x == o) {
                        Some(j) => {
                            let t = cast(field(w + k + j), fp64());
                            exprs.push(self.masked(field(o), t));
                            emit.push(w + k + v + exprs.len() - 1);
                            tans.push(Tan::Col(emit.len() - 1));
                        }
                        None => tans.push(Tan::Zero),
                    }
                }
                Ok(Dual {
                    rel: project_emit(joined, exprs, Some(emit)),
                    width: outputs.len(),
                    tans,
                })
            }
            Source::Step {
                dual,
                columns,
                tans: step_tans,
            } => {
                // A read of a step names the columns it takes (a program's
                // steps read only what they use), so each is found by name,
                // and its tangent read beside it by its dual's name for it.
                let named: Vec<String> = r
                    .base_schema
                    .as_ref()
                    .map(|s| s.names.clone())
                    .unwrap_or_default();
                let step_columns = &columns[..step_tans.len()];
                let mut read_columns = named.clone();
                let mut read_tans = Vec::with_capacity(named.len());
                for n in &named {
                    let c = step_columns.iter().position(|x| x == n).ok_or_else(|| {
                        AdError::Internal(format!(
                            "a read of `{dual}` names a column `{n}` it lacks"
                        ))
                    })?;
                    read_tans.push(match &step_tans[c] {
                        Tan::Col(j) => {
                            read_columns.push(columns[*j].clone());
                            Tan::Col(read_columns.len() - 1)
                        }
                        other => other.clone(),
                    });
                }
                let base = match filter {
                    Some(f) => emit::filter(read_step(&dual, read_columns), f),
                    None => read_step(&dual, read_columns),
                };
                let mut emit: Vec<usize> = outputs.clone();
                let mut tans = Vec::with_capacity(outputs.len());
                for &o in &outputs {
                    match read_tans.get(o) {
                        Some(Tan::Col(j)) => {
                            emit.push(*j);
                            tans.push(Tan::Col(emit.len() - 1));
                        }
                        Some(Tan::Refused(why)) => tans.push(Tan::Refused(why.clone())),
                        _ => tans.push(Tan::Zero),
                    }
                }
                Ok(Dual {
                    rel: select(base, emit),
                    width: outputs.len(),
                    tans,
                })
            }
        }
    }

    fn project(&mut self, p: &ProjectRel, d: Dual) -> Result<Dual> {
        let w_in = d.width;
        let big = d.rel_width();
        let m = p.expressions.len();
        let mut tangent_exprs = Vec::new();
        // Tangents of MAX/MIN windows: they read the window's own column, so
        // they go in a second projection over this one.
        let mut deferred: Vec<(usize, Expression)> = Vec::new();
        let mut expr_tans: Vec<DirectTan> = Vec::with_capacity(m);
        for (i, e) in p.expressions.iter().enumerate() {
            let tan = if let Some(c) = as_field(e) {
                // A bare reference is its column, tangent and all.
                match d.tans.get(c) {
                    Some(Tan::Col(j)) => Tan::Col(*j),
                    Some(other) => other.clone(),
                    None => Tan::Zero,
                }
            } else if let Some(RexType::WindowFunction(wf)) = &e.rex_type {
                match self.window(wf, &d.tans, big + i)? {
                    WindowTan::Zero => Tan::Zero,
                    WindowTan::Refused(why) => Tan::Refused(why),
                    WindowTan::Now(t) => {
                        tangent_exprs.push(t);
                        Tan::Col(big + m + tangent_exprs.len() - 1)
                    }
                    WindowTan::Later(t) => {
                        deferred.push((i, t));
                        Tan::Zero // filled in below
                    }
                }
            } else {
                match self.tangent_of(e, &d.tans) {
                    Ok(Some(t)) => {
                        tangent_exprs.push(t);
                        Tan::Col(big + m + tangent_exprs.len() - 1)
                    }
                    Ok(None) => Tan::Zero,
                    Err(why) => Tan::Refused(why),
                }
            };
            expr_tans.push(tan);
        }
        let q = tangent_exprs.len();
        let mut exprs = p.expressions.clone();
        exprs.extend(tangent_exprs);
        let mut rel = project_emit(d.rel, exprs, None);
        if !deferred.is_empty() {
            let base = big + m + q;
            let mut later = Vec::new();
            for (k, (i, t)) in deferred.into_iter().enumerate() {
                expr_tans[i] = Tan::Col(base + k);
                later.push(t);
            }
            rel = project_emit(rel, later, None);
        }
        // Direct columns: the input's dual, the expressions, their tangents.
        let mut tans: Vec<DirectTan> = d.tans.clone();
        tans.extend(expr_tans);
        let direct_of = |o: usize| if o < w_in { o } else { big + (o - w_in) };
        let common = p.common.as_ref();
        self.emitted(common, w_in + m, tans, |_| rel, &direct_of)
    }

    /// A window function's tangent, in a projection over a dual whose
    /// tangents are `tans`; `at` is the window's own column.
    fn window(&mut self, wf: &WindowFunction, tans: &[Tan], at: usize) -> Result<WindowTan> {
        let varied = |f: usize| tans.get(f).is_some_and(Tan::varied);
        let mut reads_varied = false;
        let args = value_args(&wf.arguments);
        for e in args
            .iter()
            .copied()
            .chain(&wf.partitions)
            .chain(wf.sorts.iter().filter_map(|s| s.expr.as_ref()))
        {
            reads_varied |= depends(self.functions, e, &varied)?;
        }
        if !reads_varied {
            return Ok(WindowTan::Zero);
        }
        let name = self.functions.name(wf.function_reference)?.to_string();
        let arg_varied = match args.as_slice() {
            [a] => depends(self.functions, a, &varied)?,
            _ => false,
        };
        let refuse = |what: &str| -> Result<WindowTan> {
            Ok(WindowTan::Refused(AdError::NotImplemented(format!(
                "a window function ({what}) whose result depends on a wrt column is used as a \
                 value that carries a tangent"
            ))))
        };
        match name.as_str() {
            "count" => Ok(WindowTan::Zero),
            "sum" | "avg" if !arg_varied => Ok(WindowTan::Zero),
            "sum" | "avg" => {
                if wf.invocation == AggregationInvocation::Distinct as i32 {
                    return refuse(&format!("{name}(DISTINCT …)"));
                }
                let arg = args[0].clone();
                let Some(t) = self.tangent_of(&arg, tans)? else {
                    return Ok(WindowTan::Zero);
                };
                let mut tw = wf.clone();
                tw.arguments = vec![value_arg(self.masked(arg, t))];
                tw.output_type = None;
                Ok(WindowTan::Now(Expression {
                    rex_type: Some(RexType::WindowFunction(tw)),
                }))
            }
            "max" | "min" if !arg_varied => Ok(WindowTan::Zero),
            "max" | "min" => {
                let whole = wf.sorts.is_empty()
                    && unbounded(wf.lower_bound.as_ref())
                    && unbounded(wf.upper_bound.as_ref());
                if !whole {
                    return refuse(&format!("a running {name} over an ordered frame"));
                }
                let arg = args[0].clone();
                let Some(t) = self.tangent_of(&arg, tans)? else {
                    return Ok(WindowTan::Zero);
                };
                // The rows attaining the extreme (the window's own column),
                // and the mean of their tangents, over the same partition.
                let equal = self.ext.anchor("equal");
                let avg = self.ext.anchor("avg");
                let masked = self.masked(arg.clone(), t);
                let attaining = if_then(
                    vec![(call(equal, vec![arg, field(at)]), masked)],
                    null_f64(),
                );
                Ok(WindowTan::Later(crate::expr::window(
                    avg,
                    vec![attaining],
                    wf.partitions.clone(),
                )))
            }
            other => refuse(other),
        }
    }

    fn filter(&mut self, f: &FilterRel, d: Dual) -> Result<Dual> {
        let (w, tans) = (d.width, d.tans.clone());
        let condition = f.condition.clone();
        self.emitted(
            f.common.as_ref(),
            w,
            tans,
            |common| Rel {
                rel_type: Some(RelType::Filter(Box::new(FilterRel {
                    common,
                    input: Some(Box::new(d.rel)),
                    condition,
                    advanced_extension: None,
                }))),
            },
            &|o| o,
        )
    }

    fn sort(&mut self, s: &SortRel, d: Dual) -> Result<Dual> {
        let (w, tans) = (d.width, d.tans.clone());
        let sorts = s.sorts.clone();
        self.emitted(
            s.common.as_ref(),
            w,
            tans,
            |common| Rel {
                rel_type: Some(RelType::Sort(Box::new(SortRel {
                    common,
                    input: Some(Box::new(d.rel)),
                    sorts,
                    advanced_extension: None,
                }))),
            },
            &|o| o,
        )
    }

    fn fetch(&mut self, f: &FetchRel, d: Dual) -> Result<Dual> {
        let (w, tans) = (d.width, d.tans.clone());
        self.emitted(
            f.common.as_ref(),
            w,
            tans,
            |common| Rel {
                rel_type: Some(RelType::Fetch(Box::new(FetchRel {
                    common,
                    input: Some(Box::new(d.rel)),
                    ..f.clone()
                }))),
            },
            &|o| o,
        )
    }

    fn join(&mut self, j: &JoinRel) -> Result<Dual> {
        let kind = JoinType::try_from(j.r#type).unwrap_or(JoinType::Unspecified);
        let l = self.dual(input(&j.left)?)?;
        let r = self.dual(input(&j.right)?)?;
        let (wl, wr) = (l.width, r.width);
        let big_l = l.rel_width();
        // The join's expressions read the left's columns, then the right's.
        let shift = |e: &Option<Box<Expression>>| -> Result<Option<Box<Expression>>> {
            e.as_deref()
                .map(|e| map_fields(e, &mut |f| Ok(if f < wl { f } else { f - wl + big_l })))
                .transpose()
                .map(|e| e.map(Box::new))
        };
        let expression = shift(&j.expression)?;
        let post_join_filter = shift(&j.post_join_filter)?;
        let right_tans = |tans: &[Tan]| -> Vec<Tan> {
            tans.iter()
                .map(|t| match t {
                    Tan::Col(c) => Tan::Col(big_l + c),
                    other => other.clone(),
                })
                .collect()
        };
        type Direct = Box<dyn Fn(usize) -> usize>;
        let (width, tans, direct_of): (usize, Vec<Tan>, Direct) = match kind {
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Outer => {
                let mut tans = l.tans.clone();
                tans.extend(right_tans(&r.tans));
                (
                    wl + wr,
                    tans,
                    Box::new(move |o| if o < wl { o } else { big_l + (o - wl) }),
                )
            }
            JoinType::LeftSemi | JoinType::LeftAnti => (wl, l.tans.clone(), Box::new(|o| o)),
            JoinType::RightSemi | JoinType::RightAnti => (wr, r.tans.clone(), Box::new(|o| o)),
            other => {
                return Err(AdError::NotImplemented(format!(
                    "a {} join on the path from a wrt table to the output",
                    other.as_str_name()
                )))
            }
        };
        let (left, right) = (l.rel, r.rel);
        let r#type = j.r#type;
        self.emitted(
            j.common.as_ref(),
            width,
            tans,
            |common| Rel {
                rel_type: Some(RelType::Join(Box::new(JoinRel {
                    common,
                    left: Some(Box::new(left)),
                    right: Some(Box::new(right)),
                    expression,
                    post_join_filter,
                    r#type,
                    advanced_extension: None,
                }))),
            },
            &*direct_of,
        )
    }

    fn cross(&mut self, c: &CrossRel) -> Result<Dual> {
        let l = self.dual(input(&c.left)?)?;
        let r = self.dual(input(&c.right)?)?;
        let (wl, wr) = (l.width, r.width);
        let big_l = l.rel_width();
        let mut tans = l.tans.clone();
        tans.extend(r.tans.iter().map(|t| match t {
            Tan::Col(c) => Tan::Col(big_l + c),
            other => other.clone(),
        }));
        let (left, right) = (l.rel, r.rel);
        self.emitted(
            c.common.as_ref(),
            wl + wr,
            tans,
            |common| Rel {
                rel_type: Some(RelType::Cross(Box::new(CrossRel {
                    common,
                    left: Some(Box::new(left)),
                    right: Some(Box::new(right)),
                    advanced_extension: None,
                }))),
            },
            &|o| if o < wl { o } else { big_l + (o - wl) },
        )
    }

    fn aggregate(&mut self, a: &AggregateRel, d: Dual) -> Result<Dual> {
        let groupings: Vec<Expression> = grouping_expressions(a)?.into_iter().cloned().collect();
        let varied = |f: usize| d.tans.get(f).is_some_and(Tan::varied);
        let g = groupings.len();
        let mut tans: Vec<DirectTan> = Vec::new();
        for k in &groupings {
            tans.push(if depends(self.functions, k, &varied)? {
                Tan::Refused(AdError::NotImplemented(
                    "a tangent through a GROUP BY key: the aggregate groups by a value that \
                     depends on a wrt column, and that value is then used as a number"
                        .into(),
                ))
            } else {
                Tan::Zero
            });
        }
        // Windows the MAX and MIN rules read: each the extreme of a measure's
        // argument over its group, appended to the input.
        let mut windows = Vec::new();
        let mut tangent_measures = Vec::new();
        let base = d.rel_width();
        let m = a.measures.len();
        for measure in &a.measures {
            let tan = self.measure(measure, &groupings, &d.tans, base, &mut windows);
            tans.push(match tan {
                Ok(Some(tm)) => {
                    tangent_measures.push(tm);
                    Tan::Col(g + m + tangent_measures.len() - 1)
                }
                Ok(None) => Tan::Zero,
                Err(why) => Tan::Refused(why),
            });
        }
        let mut rel_input = if windows.is_empty() {
            d.rel
        } else {
            project_emit(d.rel, windows.clone(), None)
        };
        // Each tangent measure's argument as a column beneath it, so the
        // measure reads a column, not the expression. DataFusion names an
        // aggregate's output after its expression, and a tangent reads its
        // input's tangent more than once (`masked`): written into the
        // measure, it doubled at every aggregate above, 2ⁿ for n layers.
        let at = base + windows.len();
        let mut args = Vec::new();
        for tm in tangent_measures.iter_mut() {
            if let Some(f) = tm.measure.as_mut() {
                for arg in f.arguments.iter_mut() {
                    if let Some(ArgType::Value(e)) = arg.arg_type.as_mut() {
                        args.push(std::mem::replace(e, field(at + args.len())));
                    }
                }
            }
        }
        if !args.is_empty() {
            rel_input = project_emit(rel_input, args, None);
        }
        let mut measures = a.measures.clone();
        measures.extend(tangent_measures);
        let template = a.clone();
        self.emitted(
            a.common.as_ref(),
            g + m,
            tans,
            |common| Rel {
                rel_type: Some(RelType::Aggregate(Box::new(AggregateRel {
                    common,
                    input: Some(Box::new(rel_input)),
                    measures,
                    advanced_extension: None,
                    ..template
                }))),
            },
            &|o| o,
        )
    }

    /// A measure's tangent measure, or `None` when it has none. `windows`
    /// collects the windows it reads, appended to the input after `base`
    /// columns.
    fn measure(
        &mut self,
        measure: &Measure,
        groupings: &[Expression],
        tans: &[Tan],
        base: usize,
        windows: &mut Vec<Expression>,
    ) -> Result<Option<Measure>> {
        let f = measure
            .measure
            .as_ref()
            .ok_or_else(|| AdError::InvalidPlan("an aggregate measure with no function".into()))?;
        let varied = |c: usize| tans.get(c).is_some_and(Tan::varied);
        let args = value_args(&f.arguments);
        let mut arg_varied = false;
        for e in &args {
            arg_varied |= depends(self.functions, e, &varied)?;
        }
        if !arg_varied {
            return Ok(None);
        }
        let name = self.functions.name(f.function_reference)?.to_string();
        if name == "count" {
            // A count of varied values moves with none of them: tangent 0,
            // as grad gives it gradient 0.
            let sum = self.ext.anchor("sum");
            return Ok(Some(Measure {
                measure: Some(AggregateFunction {
                    function_reference: sum,
                    arguments: vec![value_arg(lit_f64(0.0))],
                    invocation: AggregationInvocation::All as i32,
                    phase: f.phase,
                    ..Default::default()
                }),
                filter: None,
            }));
        }
        if f.invocation == AggregationInvocation::Distinct as i32 {
            return Err(AdError::NotImplemented(format!(
                "{name}(DISTINCT …) over values that depend on a wrt column"
            )));
        }
        if measure.filter.is_some() {
            return Err(AdError::NotImplemented(
                "an aggregate with a FILTER clause over values that depend on a wrt column".into(),
            ));
        }
        if !f.sorts.is_empty() {
            return Err(AdError::NotImplemented(format!(
                "an ordered aggregate ({name}) over values that depend on a wrt column"
            )));
        }
        let [arg] = args.as_slice() else {
            return Err(AdError::NotImplemented(format!(
                "no forward-mode rule for the aggregate `{name}` with {} arguments",
                args.len()
            )));
        };
        let arg = (*arg).clone();
        let Some(t) = self.tangent_of(&arg, tans)? else {
            return Ok(None);
        };
        let tangent = |function_reference: u32, arg: Expression| Measure {
            measure: Some(AggregateFunction {
                function_reference,
                arguments: vec![value_arg(arg)],
                invocation: AggregationInvocation::All as i32,
                phase: f.phase,
                ..Default::default()
            }),
            filter: None,
        };
        match name.as_str() {
            // Linear: the same function over the tangents.
            "sum" | "avg" => {
                let masked = self.masked(arg, t);
                Ok(Some(tangent(f.function_reference, masked)))
            }
            // The mean tangent of the rows attaining the group's extreme,
            // found by a window over the group, never by comparing with the
            // aggregate. NaN is left out of the window, as DataFusion's
            // grouped MAX leaves it out.
            "max" | "min" => {
                let isnan = self.ext.anchor("isnan");
                let extreme = self.ext.anchor(&name);
                let equal = self.ext.anchor("equal");
                let avg = self.ext.anchor("avg");
                let skip_nan = if_then(
                    vec![(call(isnan, vec![arg.clone()]), null_f64())],
                    arg.clone(),
                );
                windows.push(crate::expr::window(
                    extreme,
                    vec![skip_nan],
                    groupings.to_vec(),
                ));
                let at = base + windows.len() - 1;
                let masked = self.masked(arg.clone(), t);
                let attaining = if_then(
                    vec![(call(equal, vec![arg, field(at)]), masked)],
                    null_f64(),
                );
                Ok(Some(tangent(avg, attaining)))
            }
            other => Err(AdError::NotImplemented(format!(
                "no forward-mode rule for the aggregate `{other}`"
            ))),
        }
    }

    fn set(&mut self, s: &SetRel) -> Result<Dual> {
        if SetOp::try_from(s.op) != Ok(SetOp::UnionAll) {
            return Err(AdError::NotImplemented(
                "a set operation other than UNION ALL on the path from a wrt table to the \
                 output"
                    .into(),
            ));
        }
        let duals = s
            .inputs
            .iter()
            .map(|r| self.dual(r))
            .collect::<Result<Vec<_>>>()?;
        let w = duals
            .first()
            .ok_or_else(|| AdError::InvalidPlan("a set operation with no inputs".into()))?
            .width;
        // A column has a tangent if any input gives it one.
        let mut out: Vec<Tan> = vec![Tan::Zero; w];
        for d in &duals {
            for (c, t) in d.tans.iter().enumerate() {
                match (&out[c], t) {
                    (Tan::Refused(_), _) | (_, Tan::Zero) => {}
                    (_, Tan::Refused(why)) => out[c] = Tan::Refused(why.clone()),
                    (_, Tan::Col(_)) => out[c] = Tan::Col(0),
                }
            }
        }
        let carried: Vec<usize> = (0..w).filter(|&c| matches!(out[c], Tan::Col(_))).collect();
        let inputs = duals
            .into_iter()
            .map(|d| {
                let big = d.rel_width();
                let mut emit: Vec<usize> = (0..w).collect();
                let mut zeros = Vec::new();
                for &c in &carried {
                    match d.tans[c] {
                        Tan::Col(j) => emit.push(j),
                        _ => {
                            zeros.push(lit_f64(0.0));
                            emit.push(big + zeros.len() - 1);
                        }
                    }
                }
                project_emit(d.rel, zeros, Some(emit))
            })
            .collect();
        for (k, &c) in carried.iter().enumerate() {
            out[c] = Tan::Col(w + k);
        }
        let set = Rel {
            rel_type: Some(RelType::Set(SetRel {
                common: None,
                inputs,
                op: s.op,
                advanced_extension: None,
            })),
        };
        self.emitted(s.common.as_ref(), w, out, |_| set, &|o| o)
    }
}

impl Dual {
    /// Its number of columns, values and tangents.
    pub fn rel_width(&self) -> usize {
        self.width
            + self
                .tans
                .iter()
                .filter(|t| matches!(t, Tan::Col(_)))
                .count()
    }
}

/// A window function's tangent.
enum WindowTan {
    Zero,
    Refused(AdError),
    /// An expression in the same projection.
    Now(Expression),
    /// An expression in a projection over it, which can read the window.
    Later(Expression),
}

/// `rel` with only the columns `emit` picks: as its own emit if it is a
/// projection without one, else through a projection over it. An emit
/// only on a projection: DuckDB's consumer ignores one on anything else
/// (design.md §4.2).
fn with_emit(rel: Rel, emit: Vec<usize>) -> Rel {
    match rel.rel_type {
        Some(RelType::Project(p)) if p.common.is_none() => {
            let ProjectRel {
                input, expressions, ..
            } = *p;
            let input = input.map(|b| *b).unwrap_or_default();
            project_emit(input, expressions, Some(emit))
        }
        other => select(Rel { rel_type: other }, emit),
    }
}

fn unbounded(b: Option<&Bound>) -> bool {
    b.is_none_or(|b| matches!(b.kind, None | Some(BoundKind::Unbounded(_))))
}

fn value_args(arguments: &[substrait::proto::FunctionArgument]) -> Vec<&Expression> {
    arguments
        .iter()
        .filter_map(|a| match &a.arg_type {
            Some(ArgType::Value(e)) => Some(e),
            _ => None,
        })
        .collect()
}

fn value_arg(e: Expression) -> substrait::proto::FunctionArgument {
    substrait::proto::FunctionArgument {
        arg_type: Some(ArgType::Value(e)),
    }
}
