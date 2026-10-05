// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Reading the forward pass: what is saved, what is recomputed, and which
//! columns carry gradient.
//!
//! # Saved aggregates and recomputed regions
//!
//! Reverse mode needs values from the forward pass, and like any AD system
//! ddx chooses which to save and which to recompute (JAX's
//! `jax.checkpoint` policies make the same choice). ddx saves the output of
//! every aggregate that depends on a `wrt` table: a [`Saved`] relation,
//! materialized once as a forward step. That keeps each saved relation no
//! larger than a layer's output. Everything between two saved aggregates is
//! row-local work (projections, filters, joins, window rankings), a
//! [`Region`], and is recomputed inside each backward step instead of saved.
//! A contraction's join, `N × D × H` rows for nn.py's first layer, is never
//! written out.
//!
//! # A recomputed region
//!
//! A region is rebuilt with the same relations in the same order, so it
//! produces the same rows, with two differences. Its inputs are read from the
//! saved relations or in full from the tables. And no projection drops a
//! column: each appends its expressions to everything beneath it. So every
//! intermediate value is a column of the rebuilt relation, described by a
//! [`Def`]: an input column, or an expression over columns to its left. The
//! transposes walk those columns right to left, which is a reverse topological
//! order because a projection can only read columns to its left.
//!
//! # Activity
//!
//! A column is **varied** when it depends on a `wrt` column, other than
//! through `ddx_stop_gradient`. Only varied columns can receive gradient.
//! Anything that reads no `wrt` table is constant, and is copied into the
//! rebuilt region unread, whatever it contains. Some varied columns have no
//! derivative (a window rank, the `NULL` side of an outer join); each carries
//! a refusal that is raised only if gradient actually reaches it.

use std::collections::{BTreeSet, HashMap};

use prost::Message;

use substrait::proto::aggregate_rel::Grouping;
use substrait::proto::expression::{window_function, RexType, WindowFunction};
use substrait::proto::function_argument::ArgType;
use substrait::proto::join_rel::JoinType;
use substrait::proto::plan_rel::RelType as PlanRelType;
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::EmitKind;
use substrait::proto::{
    AggregateFunction, AggregateRel, CrossRel, Expression, FetchRel, JoinRel, NamedStruct, Plan,
    PlanRel, ProjectRel, ReadRel, Rel, RelCommon, SortRel,
};

use crate::elementwise::depends;
use crate::emit;
use crate::error::{AdError, Result};
use crate::expr::{
    as_field, children, contains, field, fields_of, if_then, lit_bool, map_fields,
    uncorrelated_scalar,
};
use crate::functions::Functions;
use crate::relation::{find_column, table_matches, ColumnRef, Table};

/// What an input of a region is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Input {
    /// A `wrt` table, by index into [`Forward::tables`].
    Table(usize),
    /// A saved aggregate, by index into [`Forward::saved`].
    Saved(usize),
    /// A subtree that reads no `wrt` table, copied in as it was.
    Const,
}

/// Where an input's columns sit in its region.
#[derive(Debug, Clone)]
pub struct Slot {
    /// The input.
    pub input: Input,
    /// Its first column in the rebuilt region, or `None` when its columns are
    /// not part of the output (the right side of a semi-join).
    pub offset: Option<usize>,
    /// Its number of columns.
    pub width: usize,
    /// Whether the plan guarantees it has at most one row: a saved aggregate
    /// with no grouping, or constant data the plan shows is one row. `grad`
    /// needs this of every input its loss reads.
    pub at_most_one_row: bool,
    /// Whether it is constant data computed with a window function or a
    /// `LIMIT`, which a recomputation need not repeat row for row.
    pub ordered: bool,
    /// Whether it is constant data computed by an aggregate or window
    /// function that rounds (see [`Functions::rounds`]): its values can
    /// differ in their last bits between recomputations (a sum over several
    /// partitions adds in arrival order). A table's values cannot, nor a
    /// maximum's.
    pub rounds: bool,
}

/// Rows cut by an ordering: a `LIMIT` (after an `ORDER BY` on `keys`, or with
/// none), or a window function a semi-join's right side computed. The
/// recomputed region keeps the same rows only if the ordering is total over
/// the inputs it covers. Self-contained: `keys` and the offsets in `covers`
/// are in one numbering, so it outlives the columns it was recorded over.
#[derive(Debug, Clone)]
pub struct Cut {
    /// The columns it orders by, or `None` for a `LIMIT` with no `ORDER BY`.
    pub keys: Option<Vec<usize>>,
    /// Each input it cuts: the input, its first column, and whether it has at
    /// most one row.
    pub covers: Vec<(Input, usize, bool)>,
}

/// How a column of a rebuilt region is computed.
#[derive(Debug, Clone)]
pub enum Def {
    /// Column `col` of the input in slot `slot`.
    Input {
        /// The slot.
        slot: usize,
        /// The input's column.
        col: usize,
    },
    /// A scalar expression over earlier columns: the map primitive.
    Expr(Expression),
    /// A window function: a value computed across rows (a rank), with no
    /// derivative. `keys` are the columns it partitions and orders by, or
    /// `None` if its value is the same however the rows of a partition are
    /// ordered (an exact aggregate over each whole partition: `MAX(x) OVER
    /// (PARTITION BY g)`), so it needs no ties broken.
    Window {
        /// The columns it partitions and orders by.
        keys: Option<Vec<usize>>,
    },
    /// A column of a constant input.
    Const,
}

/// A rebuilt, recomputed region of row-local work.
#[derive(Debug, Clone)]
pub struct Region {
    /// The rebuilt relation. See the module docs.
    pub rel: Rel,
    /// How each of its columns is computed.
    pub defs: Vec<Def>,
    /// Whether each column is varied.
    pub varied: Vec<bool>,
    /// A column whose gradient ddx must refuse to propagate, and why. Checked
    /// only if gradient actually reaches it.
    pub refusals: Vec<Option<AdError>>,
    /// For each output column of the original relation, its column here.
    pub outputs: Vec<usize>,
    /// Its inputs.
    pub slots: Vec<Slot>,
    /// Where its rows were cut by an ordering (see [`Cut`]).
    pub cuts: Vec<Cut>,
    /// Whether it calls a volatile function (`random()`, `now()`), which a
    /// recomputation would not repeat.
    pub volatile: bool,
    /// For each column, whether its own expression reads a scalar subquery
    /// computed by a function that rounds (see [`Slot::rounds`]).
    pub rounds: Vec<bool>,
}

impl Region {
    fn width(&self) -> usize {
        self.defs.len()
    }

    /// Append the columns of `other` after this region's, renumbering them.
    fn append(&mut self, other: Region) -> Result<usize> {
        let shift = self.width();
        let slot_shift = self.slots.len();
        for d in other.defs {
            self.defs.push(match d {
                Def::Input { slot, col } => Def::Input {
                    slot: slot + slot_shift,
                    col,
                },
                Def::Expr(e) => Def::Expr(map_fields(&e, &mut |i| Ok(i + shift))?),
                Def::Window { keys } => Def::Window {
                    keys: keys.map(|keys| keys.into_iter().map(|k| k + shift).collect()),
                },
                Def::Const => Def::Const,
            });
        }
        self.varied.extend(other.varied);
        self.rounds.extend(other.rounds);
        self.refusals.extend(other.refusals);
        self.slots.extend(other.slots.into_iter().map(|s| Slot {
            offset: s.offset.map(|o| o + shift),
            ..s
        }));
        self.cuts
            .extend(other.cuts.into_iter().map(|c| c.shifted(shift)));
        self.volatile |= other.volatile;
        Ok(shift)
    }

    fn push(&mut self, def: Def, varied: bool) -> usize {
        self.defs.push(def);
        self.rounds.push(false);
        self.varied.push(varied);
        self.refusals.push(None);
        self.defs.len() - 1
    }

    /// A cut by `keys` over every input whose columns come before `before`.
    fn cut(&self, keys: Option<Vec<usize>>, before: usize) -> Cut {
        Cut {
            keys,
            covers: self
                .slots
                .iter()
                .filter_map(|s| {
                    s.offset
                        .filter(|&o| o < before)
                        .map(|o| (s.input, o, s.at_most_one_row))
                })
                .collect(),
        }
    }

    /// Its windows and cuts, as cuts that need none of its columns: for a
    /// semi-join's right side, whose columns are dropped.
    fn detached_cuts(&self) -> Vec<Cut> {
        let mut cuts = self.cuts.clone();
        for (c, d) in self.defs.iter().enumerate() {
            if let Def::Window { keys: Some(keys) } = d {
                cuts.push(self.cut(Some(keys.clone()), c));
            }
        }
        cuts
    }

    fn refuse(&mut self, col: usize, why: AdError) {
        if self.refusals[col].is_none() {
            self.refusals[col] = Some(why);
        }
    }
}

impl Cut {
    fn shifted(self, shift: usize) -> Cut {
        Cut {
            keys: self
                .keys
                .map(|k| k.into_iter().map(|c| c + shift).collect()),
            covers: self
                .covers
                .into_iter()
                .map(|(i, o, one)| (i, o + shift, one))
                .collect(),
        }
    }
}

/// What an output column of a saved aggregate is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// A dim: its `i`th grouping key.
    Dim(usize),
    /// A value: its `i`th measure.
    Value(usize),
}

/// An aggregate that depends on a `wrt` table, saved by the forward pass.
#[derive(Debug, Clone)]
pub struct Saved {
    /// What it aggregates.
    pub input: Region,
    /// Its grouping keys, over `input`'s columns.
    pub groupings: Vec<Expression>,
    /// Its measures, with arguments over `input`'s columns.
    pub measures: Vec<AggregateFunction>,
    /// What each output column is.
    pub outputs: Vec<Output>,
    /// Whether each output column is varied.
    pub varied: Vec<bool>,
    /// The relation that computes it, reading its own inputs as saved.
    pub rel: Rel,
}

impl Saved {
    /// The output columns that are dims: its grouping keys.
    pub fn dims(&self) -> Vec<usize> {
        (0..self.outputs.len())
            .filter(|&c| matches!(self.outputs[c], Output::Dim(_)))
            .collect()
    }
}

/// A forward query, read.
#[derive(Debug, Clone)]
pub struct Forward {
    /// The plan's function table.
    pub functions: Functions,
    /// The `wrt` tables.
    pub tables: Vec<Table>,
    /// The saved aggregates, each after every one it reads.
    pub saved: Vec<Saved>,
    /// The region above the last saved aggregate: the query's output.
    pub output: Region,
    /// The output column names.
    pub output_names: Vec<String>,
    /// The prefix of every table name the program materializes.
    pub namespace: String,
}

/// The name of saved aggregate `n`'s table, in a program's `namespace`.
pub fn saved_name(namespace: &str, n: usize) -> String {
    format!("{namespace}saved_{n}")
}

/// A fresh prefix for one program's table names: `__ddx_{id}_`.
///
/// Every table a program materializes lives under it, so two programs on one
/// engine never read or replace each other's tables, and none takes a name a
/// user is likely to have. The id is unique within the process (a counter)
/// and unlikely to repeat across processes (the clock).
pub fn new_namespace() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let clock = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("__ddx_{:x}_{n:x}_", clock & 0xffff_ffff)
}

/// The column names of a saved relation or its cotangent, `width` wide.
pub fn step_columns(width: usize) -> Vec<String> {
    (0..width).map(|i| format!("c{i}")).collect()
}

impl Forward {
    /// Read `plan`, with `wrt` naming the table columns that are values.
    pub fn new(plan: &Plan, wrt: &[ColumnRef]) -> Result<Forward> {
        Forward::in_namespace(plan, wrt, new_namespace())
    }

    /// [`Forward::new`], naming every table the program writes under
    /// `namespace` rather than a fresh one.
    pub fn in_namespace(plan: &Plan, wrt: &[ColumnRef], namespace: String) -> Result<Forward> {
        let functions = Functions::from_plan(plan)?;
        let (root, output_names) = root_of(plan)?;
        let root = &inline_references(root, &plan.relations)?;
        if wrt.is_empty() {
            return Err(AdError::UnknownWrt(
                "no wrt columns were given; name at least one table column".into(),
            ));
        }
        let mut b = Builder {
            functions: &functions,
            namespace: namespace.clone(),
            wrt,
            tables: Vec::new(),
            saved: Vec::new(),
            by_encoding: HashMap::new(),
            seen: BTreeSet::new(),
            reads: HashMap::new(),
        };
        let output = b.lower(root)?;
        b.check_every_wrt_was_read()?;
        let (tables, saved) = (b.tables, b.saved);
        Ok(Forward {
            functions,
            tables,
            saved,
            output,
            output_names,
            namespace,
        })
    }
}

pub(crate) fn root_of(plan: &Plan) -> Result<(&Rel, Vec<String>)> {
    let mut roots = plan.relations.iter().filter_map(|r| match &r.rel_type {
        Some(PlanRelType::Root(root)) => Some(root),
        _ => None,
    });
    let root = roots
        .next()
        .ok_or_else(|| AdError::InvalidPlan("the plan has no root relation".into()))?;
    if roots.next().is_some() {
        return Err(AdError::NotImplemented(
            "a plan with more than one root relation".into(),
        ));
    }
    let rel = root
        .input
        .as_ref()
        .ok_or_else(|| AdError::InvalidPlan("the root relation is empty".into()))?;
    Ok((rel, root.names.clone()))
}

/// `root` with every `ReferenceRel` replaced by a copy of the relation it
/// names. A producer shares a subtree read more than once this way (DuckDB, a
/// CTE read twice), where DataFusion writes the tree out; ddx reads trees, and
/// saves each distinct aggregate once whichever way it was written. A cycle
/// or a missing ordinal is an invalid plan, and a reference inside a
/// subquery expression is refused rather than read as reading nothing.
pub(crate) fn inline_references(root: &Rel, relations: &[PlanRel]) -> Result<Rel> {
    fn inline(rel: &mut Rel, relations: &[PlanRel], path: &mut Vec<i32>) -> Result<()> {
        if let Some(RelType::Reference(r)) = &rel.rel_type {
            let ordinal = r.subtree_ordinal;
            if path.contains(&ordinal) {
                return Err(AdError::InvalidPlan(format!(
                    "relation {ordinal} refers to itself through ReferenceRel"
                )));
            }
            let target = usize::try_from(ordinal)
                .ok()
                .and_then(|i| relations.get(i))
                .and_then(|r| match &r.rel_type {
                    Some(PlanRelType::Rel(r)) => Some(r.clone()),
                    Some(PlanRelType::Root(root)) => root.input.clone(),
                    None => None,
                })
                .ok_or_else(|| {
                    AdError::InvalidPlan(format!(
                        "a ReferenceRel names relation {ordinal}, which the plan does not have"
                    ))
                })?;
            *rel = target;
            path.push(ordinal);
            inline(rel, relations, path)?;
            path.pop();
            return Ok(());
        }
        if let Some(kind) = rel.rel_type.as_mut() {
            for input in rel_inputs_mut(kind) {
                inline(input, relations, path)?;
            }
        }
        Ok(())
    }
    let mut out = root.clone();
    inline(&mut out, relations, &mut Vec::new())?;
    if mentions_reference(&out) {
        return Err(AdError::NotImplemented(
            "a ReferenceRel inside a subquery expression".into(),
        ));
    }
    Ok(out)
}

/// Does `rel` still hold a `ReferenceRel` anywhere, subqueries included?
fn mentions_reference(rel: &Rel) -> bool {
    use substrait::proto::expression::subquery::SubqueryType;
    fn subquery_rels<'e>(e: &'e Expression, out: &mut Vec<&'e Rel>) {
        if let Some(RexType::Subquery(sq)) = &e.rex_type {
            match &sq.subquery_type {
                Some(SubqueryType::Scalar(s)) => out.extend(s.input.as_deref()),
                Some(SubqueryType::InPredicate(s)) => out.extend(s.haystack.as_deref()),
                Some(SubqueryType::SetPredicate(s)) => out.extend(s.tuples.as_deref()),
                Some(SubqueryType::SetComparison(s)) => out.extend(s.right.as_deref()),
                None => {}
            }
        }
        for c in children(e) {
            subquery_rels(c, out);
        }
    }
    let mut stack = vec![rel];
    while let Some(r) = stack.pop() {
        let Some(kind) = &r.rel_type else { continue };
        if matches!(kind, RelType::Reference(_)) {
            return true;
        }
        for e in rel_expressions(kind) {
            subquery_rels(e, &mut stack);
        }
        stack.extend(rel_inputs(kind));
    }
    false
}

struct Builder<'a> {
    functions: &'a Functions,
    namespace: String,
    wrt: &'a [ColumnRef],
    tables: Vec<Table>,
    saved: Vec<Saved>,
    /// Each saved aggregate, by the encoding of the aggregate it was read from.
    by_encoding: HashMap<Vec<u8>, usize>,
    /// Every table name the plan reads, for the error when a `wrt` names none.
    seen: BTreeSet<String>,
    /// [`Builder::reads_wrt`]'s answers, by relation: the plan is borrowed for
    /// the whole read, so its relations stay put.
    reads: HashMap<*const Rel, bool>,
}

impl Builder<'_> {
    /// Lower `rel`. A chain of single-input relations (projections,
    /// filters, sorts, limits) can be hundreds deep, a `jvp` program's step
    /// most of all, deeper than a 2 MB worker stack allows for recursion; so
    /// the chain is walked down with a loop, and each relation lowered over
    /// its input's region on the way back up. Only joins and aggregates
    /// recurse.
    fn lower(&mut self, rel: &Rel) -> Result<Region> {
        let mut chain = Vec::new();
        let mut cur = rel;
        let mut s = loop {
            if !self.reads_wrt(cur)? {
                break self.constant(cur)?;
            }
            let below = match &cur.rel_type {
                Some(RelType::Project(p)) => &p.input,
                Some(RelType::Filter(f)) => &f.input,
                Some(RelType::Sort(so)) => &so.input,
                Some(RelType::Fetch(fe)) => &fe.input,
                _ => break self.lower_one(cur, None)?,
            };
            chain.push(cur);
            cur = input(below)?;
        };
        while let Some(r) = chain.pop() {
            s = self.lower_one(r, Some(s))?;
        }
        Ok(s)
    }

    /// Lower `rel`, which reads a `wrt` table, over `below`, its input's
    /// region, if it is a single-input relation [`Builder::lower`] walked
    /// down to.
    fn lower_one(&mut self, rel: &Rel, below: Option<Region>) -> Result<Region> {
        let kind = rel
            .rel_type
            .as_ref()
            .ok_or_else(|| AdError::InvalidPlan("an empty relation".into()))?;
        let mut below = below;
        let mut lowered = |this: &mut Self, r: &Option<Box<Rel>>| match below.take() {
            Some(s) => Ok(s),
            None => this.lower(input(r)?),
        };
        // A subquery in this relation's expressions is carried through as a
        // constant (see `uncorrelated_scalar`). That is only right if it reads
        // no wrt table; otherwise the gradient through it would be dropped.
        for e in rel_expressions(kind) {
            let mut subqueries = Vec::new();
            collect_subqueries(e, &mut subqueries);
            for sq in subqueries {
                let reads = match uncorrelated_scalar(sq) {
                    Some(inner) => self.reads_wrt(inner)?,
                    None => true,
                };
                if reads {
                    return Err(AdError::NotImplemented(
                        "a subquery that is correlated, or reads a wrt table, inside an \
                         expression on the path from a wrt table to the output; write it as a \
                         join"
                            .into(),
                    ));
                }
            }
        }
        let volatile = !matches!(kind, RelType::Aggregate(_))
            && rel_expressions(kind)
                .into_iter()
                .any(|e| self.calls_volatile(e));
        let (mut s, direct, common) = match kind {
            RelType::Read(r) => {
                let s = self.read(r)?;
                let direct = read_outputs(r)?;
                (s, direct, r.common.as_ref())
            }
            RelType::Project(p) => {
                let below = lowered(self, &p.input)?;
                let (s, direct) = self.project(below, p)?;
                (s, direct, p.common.as_ref())
            }
            RelType::Filter(f) => {
                let mut s = lowered(self, &f.input)?;
                let cond = remap(f.condition.as_deref(), &s.outputs)?;
                s.rel = match cond {
                    Some(c) => emit::filter(s.rel, c),
                    None => s.rel,
                };
                let direct = s.outputs.clone();
                (s, direct, f.common.as_ref())
            }
            RelType::Sort(so) => {
                let mut s = lowered(self, &so.input)?;
                let mut sorts = so.sorts.clone();
                for sf in sorts.iter_mut() {
                    if let Some(e) = sf.expr.as_mut() {
                        *e = map_fields(e, &mut |i| lookup(&s.outputs, i))?;
                    }
                }
                s.rel = Rel {
                    rel_type: Some(RelType::Sort(Box::new(SortRel {
                        common: None,
                        input: Some(Box::new(s.rel)),
                        sorts,
                        advanced_extension: None,
                    }))),
                };
                let direct = s.outputs.clone();
                (s, direct, so.common.as_ref())
            }
            RelType::Fetch(fe) => {
                let mut s = lowered(self, &fe.input)?;
                let below = input(&fe.input)?;
                // The rows a LIMIT keeps are the first in its input's order:
                // an ORDER BY's keys, or none.
                let keys = match &below.rel_type {
                    Some(RelType::Sort(so)) => {
                        let mut keys = Vec::new();
                        for e in so.sorts.iter().filter_map(|sf| sf.expr.as_ref()) {
                            if let Some(f) = order_key(e) {
                                keys.push(lookup(&s.outputs, f)?);
                            }
                        }
                        Some(keys)
                    }
                    _ => None,
                };
                let cut = s.cut(keys, s.width());
                s.cuts.push(cut);
                s.rel = Rel {
                    rel_type: Some(RelType::Fetch(Box::new(FetchRel {
                        common: None,
                        input: Some(Box::new(s.rel)),
                        ..(**fe).clone()
                    }))),
                };
                let direct = s.outputs.clone();
                (s, direct, fe.common.as_ref())
            }
            RelType::Join(j) => {
                let (s, direct) = self.join(j)?;
                (s, direct, j.common.as_ref())
            }
            RelType::Cross(c) => {
                let (s, direct) = self.cross(c)?;
                (s, direct, c.common.as_ref())
            }
            RelType::Aggregate(a) => {
                let n = self.read_saved(a)?;
                let width = self.saved[n].outputs.len();
                let mut s = empty(emit::read_step(
                    &saved_name(&self.namespace, n),
                    step_columns(width),
                ));
                s.slots.push(Slot {
                    input: Input::Saved(n),
                    offset: Some(0),
                    width,
                    at_most_one_row: self.saved[n].groupings.is_empty(),
                    ordered: false,
                    rounds: false,
                });
                for c in 0..width {
                    let v = self.saved[n].varied[c];
                    s.push(Def::Input { slot: 0, col: c }, v);
                }
                (s, (0..width).collect(), None)
            }
            other => {
                return Err(AdError::NotImplemented(format!(
                    "a {} relation on the path from a wrt table to the output",
                    rel_name(other)
                )))
            }
        };
        s.outputs = apply_emit(common, direct)?;
        s.volatile |= volatile;
        Ok(s)
    }

    /// A subtree that reads no `wrt` table: copied as it is.
    fn constant(&mut self, rel: &Rel) -> Result<Region> {
        let width = width(rel)?;
        let mut s = empty(rel.clone());
        s.slots.push(Slot {
            input: Input::Const,
            offset: Some(0),
            width,
            at_most_one_row: at_most_one_row(rel),
            ordered: is_ordered(rel),
            rounds: rounds_under(self.functions, rel)?,
        });
        for _ in 0..width {
            s.push(Def::Const, false);
        }
        s.outputs = (0..width).collect();
        s.volatile = self.volatile_under(rel);
        Ok(s)
    }

    fn read(&mut self, r: &ReadRel) -> Result<Region> {
        let Some(ReadType::NamedTable(t)) = &r.read_type else {
            return Err(AdError::Internal(
                "reads_wrt() let through a read that is not a named table".into(),
            ));
        };
        let schema = r.base_schema.clone().ok_or_else(|| {
            AdError::InvalidPlan(format!("the read of `{}` has no schema", t.names.join(".")))
        })?;
        let table = self.table(&t.names, &schema)?;
        let filter = r.filter.as_deref().cloned();
        let mut s = empty(emit::read(t.names.clone(), schema.clone(), filter, None));
        let width = schema.names.len();
        s.slots.push(Slot {
            input: Input::Table(table),
            offset: Some(0),
            width,
            at_most_one_row: false,
            ordered: false,
            rounds: false,
        });
        for c in 0..width {
            let v = self.tables[table].values.contains(&c);
            s.push(Def::Input { slot: 0, col: c }, v);
        }
        Ok(s)
    }

    /// Register a `wrt` table the first time it is read.
    fn table(&mut self, names: &[String], schema: &NamedStruct) -> Result<usize> {
        if let Some(i) = self.tables.iter().position(|t| t.names == names) {
            if self.tables[i].schema != *schema {
                return Err(AdError::InvalidPlan(format!(
                    "table `{}` is read with two different schemas",
                    names.join(".")
                )));
            }
            return Ok(i);
        }
        let table = wrt_table(names, schema, self.wrt)?;
        self.tables.push(table);
        Ok(self.tables.len() - 1)
    }

    /// A projection over `s`, its input's region.
    fn project(&mut self, mut s: Region, p: &ProjectRel) -> Result<(Region, Vec<usize>)> {
        let inputs = s.outputs.clone();
        let mut exprs = Vec::with_capacity(p.expressions.len());
        let mut direct = inputs.clone();
        for e in &p.expressions {
            let e = map_fields(e, &mut |i| lookup(&inputs, i))?;
            // A bare column reference is the column it references, not a copy
            // of it, so one value has one column.
            if let Some(c) = as_field(&e) {
                direct.push(c);
                continue;
            }
            let col = if let Some(RexType::WindowFunction(w)) = &e.rex_type {
                let keys = (!self.whole_partition_exact(w)?).then(|| {
                    w.partitions
                        .iter()
                        .chain(w.sorts.iter().filter_map(|s| s.expr.as_ref()))
                        .filter_map(order_key)
                        .collect()
                });
                let varied = fields_of(&e)?.into_iter().any(|f| s.varied[f]);
                let col = s.push(Def::Window { keys }, varied);
                s.refuse(
                    col,
                    AdError::NotImplemented(
                        "a window function's result is used as a value that carries gradient; \
                         a rank or row number has no derivative"
                            .into(),
                    ),
                );
                col
            } else if contains(&e, &|x| {
                matches!(x.rex_type, Some(RexType::WindowFunction(_)))
            }) {
                return Err(AdError::NotImplemented(
                    "a window function nested inside a larger expression".into(),
                ));
            } else {
                let varied = depends(self.functions, &e, &|f| s.varied[f])?;
                let col = s.push(Def::Expr(e.clone()), varied);
                s.rounds[col] = subquery_rounds(self.functions, &e)?;
                col
            };
            direct.push(col);
            exprs.push(e);
        }
        let windows: Vec<usize> = (s.width() - exprs.len()..s.width())
            .filter(|&c| matches!(s.defs[c], Def::Window { .. }))
            .collect();
        s.rel = emit::project(s.rel, exprs);
        if !windows.is_empty() {
            let width = s.width();
            s.rel = rename_in_place(s.rel, width, &windows);
        }
        Ok((s, direct))
    }

    /// Is `w` an exact aggregate (`MAX`, `MIN`, `COUNT`) over each whole
    /// partition, with no `ORDER BY` and no frame narrower than it? Then its
    /// value does not depend on how a partition's rows are ordered, and a
    /// recomputation gives the same one.
    fn whole_partition_exact(&self, w: &WindowFunction) -> Result<bool> {
        use substrait::proto::expression::window_function::bound::Kind;
        let unbounded = |b: &Option<window_function::Bound>| match b {
            None => true,
            Some(b) => matches!(b.kind, None | Some(Kind::Unbounded(_))),
        };
        Ok(w.sorts.is_empty()
            && unbounded(&w.lower_bound)
            && unbounded(&w.upper_bound)
            && !self.functions.rounds(w.function_reference)?)
    }

    fn join(&mut self, j: &JoinRel) -> Result<(Region, Vec<usize>)> {
        let kind = JoinType::try_from(j.r#type).unwrap_or(JoinType::Unspecified);
        let mut s = self.lower(input(&j.left)?)?;
        let right = self.lower(input(&j.right)?)?;
        let left_outputs = s.outputs.clone();
        let right_outputs = right.outputs.clone();
        let left_width = s.width();
        let right_width = right.width();
        let right_varied: Vec<usize> = (0..right_width).filter(|&c| right.varied[c]).collect();
        let left_varied: Vec<usize> = (0..left_width).filter(|&c| s.varied[c]).collect();
        let right_rel = right.rel.clone();

        // The join's own expressions read left's outputs, then right's.
        let joined: Vec<usize> = left_outputs
            .iter()
            .copied()
            .chain(right_outputs.iter().map(|c| c + left_width))
            .collect();
        let cond = remap(j.expression.as_deref(), &joined)?;
        let post = remap(j.post_join_filter.as_deref(), &joined)?;

        let (direct, semi) = match kind {
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Outer => {
                (joined.clone(), false)
            }
            JoinType::LeftSemi | JoinType::LeftAnti => (left_outputs.clone(), true),
            other => {
                return Err(AdError::NotImplemented(format!(
                    "a {} join on the path from a wrt table to the output",
                    other.as_str_name()
                )))
            }
        };
        if semi {
            // Only the left side's columns come out. The right side is still
            // read, to decide which rows do, so its orderings still count.
            s.cuts.extend(right.detached_cuts());
            s.volatile |= right.volatile;
            s.slots.extend(
                right
                    .slots
                    .into_iter()
                    .map(|sl| Slot { offset: None, ..sl }),
            );
        } else {
            s.append(right)?;
        }
        let outer = |cols: &[usize], shift: usize, s: &mut Region| {
            for &c in cols {
                s.refuse(
                    c + shift,
                    AdError::NotImplemented(
                        "gradient through the NULL-extended side of an outer join".into(),
                    ),
                );
            }
        };
        match kind {
            JoinType::Left => outer(&right_varied, left_width, &mut s),
            JoinType::Right => outer(&left_varied, 0, &mut s),
            JoinType::Outer => {
                outer(&left_varied, 0, &mut s);
                outer(&right_varied, left_width, &mut s);
            }
            _ => {}
        }
        s.rel = Rel {
            rel_type: Some(RelType::Join(Box::new(JoinRel {
                common: None,
                left: Some(Box::new(s.rel)),
                right: Some(Box::new(right_rel)),
                expression: cond.map(Box::new),
                post_join_filter: post.map(Box::new),
                r#type: j.r#type,
                advanced_extension: None,
            }))),
        };
        Ok((s, direct))
    }

    fn cross(&mut self, c: &CrossRel) -> Result<(Region, Vec<usize>)> {
        let mut s = self.lower(input(&c.left)?)?;
        let right = self.lower(input(&c.right)?)?;
        let right_rel = right.rel.clone();
        let right_outputs = right.outputs.clone();
        let shift = s.append(right)?;
        let direct = s
            .outputs
            .iter()
            .copied()
            .chain(right_outputs.iter().map(|o| o + shift))
            .collect();
        s.rel = Rel {
            rel_type: Some(RelType::Cross(Box::new(CrossRel {
                common: None,
                left: Some(Box::new(s.rel)),
                right: Some(Box::new(right_rel)),
                advanced_extension: None,
            }))),
        };
        Ok((s, direct))
    }

    /// The saved relation for aggregate `a`, read on first sight.
    ///
    /// A query that reads a CTE twice gets the CTE's plan twice: Substrait
    /// plans are trees, and producers inline. Two identical aggregates are one
    /// saved relation, computed and stored once, and both readers' cotangents
    /// are added into one (design.md §4.4's fan-in). Treating them as two
    /// would also be correct, by linearity, but would do the work twice. This
    /// is the other half of the save-or-recompute policy the module docs
    /// describe.
    fn read_saved(&mut self, a: &AggregateRel) -> Result<usize> {
        let encoding = a.encode_to_vec();
        if let Some(&n) = self.by_encoding.get(&encoding) {
            return Ok(n);
        }
        let n = self.read_aggregate(a)?;
        self.by_encoding.insert(encoding, n);
        Ok(n)
    }

    fn read_aggregate(&mut self, a: &AggregateRel) -> Result<usize> {
        let mut input = self.lower(input(&a.input)?)?;
        let outputs = &input.outputs;
        let groupings = grouping_expressions(a)?
            .into_iter()
            .map(|e| map_fields(e, &mut |i| lookup(outputs, i)))
            .collect::<Result<Vec<_>>>()?;
        let mut measures = Vec::with_capacity(a.measures.len());
        for m in &a.measures {
            let f = m.measure.as_ref().ok_or_else(|| {
                AdError::InvalidPlan("an aggregate measure with no function".into())
            })?;
            let mut f = f.clone();
            for arg in f.arguments.iter_mut() {
                if let Some(ArgType::Value(e)) = arg.arg_type.as_mut() {
                    *e = map_fields(e, &mut |i| lookup(outputs, i))?;
                }
            }
            #[allow(deprecated)]
            for e in f.args.iter_mut() {
                *e = map_fields(e, &mut |i| lookup(outputs, i))?;
            }
            for sf in f.sorts.iter_mut() {
                if let Some(e) = sf.expr.as_mut() {
                    *e = map_fields(e, &mut |i| lookup(outputs, i))?;
                }
            }
            if m.filter.is_some() {
                return Err(AdError::NotImplemented(
                    "an aggregate with a FILTER clause over a wrt table".into(),
                ));
            }
            measures.push(f);
        }
        let varied_input = |f: usize| input.varied[f];
        let mut direct = Vec::new();
        let mut direct_varied = Vec::new();
        for (i, g) in groupings.iter().enumerate() {
            direct.push(Output::Dim(i));
            direct_varied.push(depends(self.functions, g, &varied_input)?);
        }
        for (i, m) in measures.iter().enumerate() {
            direct.push(Output::Value(i));
            let mut v = false;
            for arg in &m.arguments {
                if let Some(ArgType::Value(e)) = &arg.arg_type {
                    v |= depends(self.functions, e, &varied_input)?;
                }
            }
            direct_varied.push(v);
        }
        let picked = apply_emit(a.common.as_ref(), (0..direct.len()).collect())?;
        let rel = Rel {
            rel_type: Some(RelType::Aggregate(Box::new(AggregateRel {
                common: a.common.clone(),
                input: Some(Box::new(input.rel.clone())),
                #[allow(deprecated)]
                groupings: vec![Grouping {
                    grouping_expressions: groupings.clone(),
                    expression_references: (0..groupings.len() as u32).collect(),
                }],
                grouping_expressions: groupings.clone(),
                measures: a
                    .measures
                    .iter()
                    .zip(&measures)
                    .map(|(m, f)| substrait::proto::aggregate_rel::Measure {
                        measure: Some(f.clone()),
                        filter: m.filter.clone(),
                    })
                    .collect(),
                advanced_extension: None,
            }))),
        };
        // The region beneath is recomputed with the keys and the measures'
        // arguments, so a volatile one makes it unrepeatable too.
        let args = measures
            .iter()
            .flat_map(|m| &m.arguments)
            .filter_map(|a| match &a.arg_type {
                Some(ArgType::Value(e)) => Some(e),
                _ => None,
            });
        input.volatile |= groupings.iter().chain(args).any(|e| self.calls_volatile(e));
        self.saved.push(Saved {
            input,
            groupings,
            measures,
            outputs: picked.iter().map(|&i| direct[i]).collect(),
            varied: picked.iter().map(|&i| direct_varied[i]).collect(),
            rel,
        });
        Ok(self.saved.len() - 1)
    }

    /// Does `e` call a volatile function?
    fn calls_volatile(&self, e: &Expression) -> bool {
        let direct = contains(e, &|x| match &x.rex_type {
            Some(RexType::ScalarFunction(f)) => self
                .functions
                .is_volatile(f.function_reference)
                .unwrap_or(false),
            _ => false,
        });
        let mut subqueries = Vec::new();
        collect_subqueries(e, &mut subqueries);
        direct
            || subqueries.into_iter().any(|sq| {
                // A subquery ddx cannot read is refused elsewhere.
                uncorrelated_scalar(sq).is_some_and(|r| self.volatile_under(r))
            })
    }

    /// Does anything in `rel` call a volatile function?
    fn volatile_under(&self, rel: &Rel) -> bool {
        let Some(kind) = rel.rel_type.as_ref() else {
            return false;
        };
        rel_expressions(kind)
            .into_iter()
            .any(|e| self.calls_volatile(e))
            || rel_inputs(kind).into_iter().any(|r| self.volatile_under(r))
    }

    /// Does anything under `rel` read a `wrt` table? Records every table name
    /// read along the way. Conservatively true for a subquery inside an
    /// expression, which lowering then refuses rather than treats as constant.
    fn reads_wrt(&mut self, rel: &Rel) -> Result<bool> {
        // Each relation's answer is remembered, so the plan is walked once in
        // all: `lower` asks at every level, and walking the subtree each time
        // made reading a plan quadratic in its depth.
        let key = rel as *const Rel;
        if let Some(&found) = self.reads.get(&key) {
            return Ok(found);
        }
        // Post-order: a relation's answer once its inputs' are known.
        let mut stack: Vec<(&Rel, bool)> = vec![(rel, false)];
        while let Some((r, ready)) = stack.pop() {
            let k = r as *const Rel;
            if self.reads.contains_key(&k) {
                continue;
            }
            let Some(kind) = &r.rel_type else {
                self.reads.insert(k, false);
                continue;
            };
            if let RelType::Read(read) = kind {
                let mut found = false;
                if let Some(ReadType::NamedTable(t)) = &read.read_type {
                    self.seen.insert(t.names.join("."));
                    found = self.wrt.iter().any(|w| table_matches(&w.table, &t.names));
                }
                self.reads.insert(k, found);
                continue;
            }
            // A subquery inside an expression: an uncorrelated scalar one is
            // read like any other input; anything else is conservatively
            // assumed to read a wrt table, so lowering refuses it rather than
            // treating it as constant.
            let mut below: Vec<&Rel> = rel_inputs(kind);
            let mut correlated = false;
            for e in rel_expressions(kind) {
                let mut subqueries = Vec::new();
                collect_subqueries(e, &mut subqueries);
                for sq in subqueries {
                    match uncorrelated_scalar(sq) {
                        Some(inner) => below.push(inner),
                        None => correlated = true,
                    }
                }
            }
            if ready {
                let found = correlated
                    || below
                        .iter()
                        .any(|b| self.reads.get(&(*b as *const Rel)) == Some(&true));
                self.reads.insert(k, found);
            } else {
                stack.push((r, true));
                stack.extend(below.into_iter().map(|b| (b, false)));
            }
        }
        Ok(self.reads[&key])
    }

    fn check_every_wrt_was_read(&self) -> Result<()> {
        check_every_wrt_was_read(self.wrt, &self.seen, &self.tables)
    }
}

/// The column an ordering or partitioning expression orders by, if it is a
/// bare column: only then does it tell the rows of each value apart. An
/// expression over columns can tie rows that differ in all of them
/// (`ORDER BY i % 2` ties every even `i`), so it counts as no key, and a
/// ranking or `LIMIT` whose order is total only through one is refused as
/// one that does not break ties.
fn order_key(e: &Expression) -> Option<usize> {
    as_field(e)
}

/// Refuse a `wrt` entry that names no table the plan reads where gradient
/// can flow (`tables`), or names more than one of the tables it reads at
/// all (`seen`).
pub(crate) fn check_every_wrt_was_read(
    wrt: &[ColumnRef],
    seen: &BTreeSet<String>,
    tables: &[Table],
) -> Result<()> {
    for w in wrt {
        // A bare name that matches tables in two schemas would take both
        // as wrt tables; which rows get a gradient must not be a guess.
        let matching: Vec<&String> = seen
            .iter()
            .filter(|t| {
                table_matches(
                    &w.table,
                    &t.split('.').map(str::to_string).collect::<Vec<_>>(),
                )
            })
            .collect();
        if matching.len() > 1 {
            return Err(AdError::UnknownWrt(format!(
                "`{}` names more than one table the query reads: {matching:?}; \
                 qualify it",
                w.table
            )));
        }
        let read = tables
            .iter()
            .any(|t| table_matches(&w.table, &t.names));
        if !read {
            // A producer may drop a schema qualifier (Ibis writes `s.t`
            // as `t`), and then only the bare name can match.
            let hint = match w.table.rsplit_once('.') {
                Some((_, last)) if seen.iter().any(|t| t.eq_ignore_ascii_case(last)) => {
                    format!("; the plan names its tables without a schema, so name it `{last}`")
                }
                _ => String::new(),
            };
            return Err(AdError::UnknownWrt(format!(
                "the query does not read a table `{}` (or reads it only where no gradient \
                 can reach); it reads {:?}{hint}",
                w.table, seen
            )));
        }
    }
    Ok(())
}

/// The `wrt` table `names` (with `schema`): its values are the columns `wrt`
/// names, its dims the rest. Refused unless every value is floating point and
/// some column is a dim.
pub(crate) fn wrt_table(names: &[String], schema: &NamedStruct, wrt: &[ColumnRef]) -> Result<Table> {
    if schema.r#struct.as_ref().map(|s| s.types.len()) != Some(schema.names.len()) {
        return Err(AdError::NotImplemented(format!(
            "table `{}` has nested columns",
            names.join(".")
        )));
    }
    let mut values = Vec::new();
    for w in wrt.iter().filter(|w| table_matches(&w.table, names)) {
        let col = find_column(&schema.names, &w.column).ok_or_else(|| {
            AdError::UnknownWrt(format!(
                "table `{}` has no column `{}`; its columns are {:?}",
                names.join("."),
                w.column,
                schema.names
            ))
        })?;
        if !values.contains(&col) {
            values.push(col);
        }
    }
    values.sort_unstable();
    // A gradient is taken only with respect to floating-point values, as
    // jax.grad requires inexact inputs. An integer column is piecewise
    // constant to the engine (`val / 2` truncates), while ddx would
    // differentiate real arithmetic: a silently wrong gradient.
    let types = schema
        .r#struct
        .as_ref()
        .map(|t| t.types.as_slice())
        .unwrap_or(&[]);
    for &v in &values {
        let float = matches!(
            types.get(v).and_then(|t| t.kind.as_ref()),
            Some(
                substrait::proto::r#type::Kind::Fp32(_)
                    | substrait::proto::r#type::Kind::Fp64(_)
            )
        );
        if !float {
            return Err(AdError::InvalidWrt(format!(
                "column `{}` of table `{}` is not a floating-point column; a gradient is \
                 taken only with respect to REAL or DOUBLE values. Store it as DOUBLE",
                schema.names[v],
                names.join(".")
            )));
        }
    }
    let dims: Vec<usize> = (0..schema.names.len())
        .filter(|c| !values.contains(c))
        .collect();
    // A cotangent is keyed by its table's dims. With none, nothing tells
    // one row's gradient from another's, and every row would get the sum.
    if dims.is_empty() {
        return Err(AdError::InvalidWrt(format!(
            "every column of table `{}` is a wrt column, so no column identifies its \
             rows and their gradients cannot be told apart. Add a dim column (a row \
             index or coordinate) to the table",
            names.join(".")
        )));
    }
    Ok(Table {
        names: names.to_vec(),
        schema: schema.clone(),
        dims,
        values,
    })
}

/// `rel` (`width` columns wide) with each column in `cols` replaced, in its
/// place, by `CASE WHEN true THEN c ELSE c END`: the same values under a
/// different name. A region is rebuilt keeping every column, so a window
/// function computed over rows that already carry an identical window
/// function's column would give the engine two columns of one name, which
/// DataFusion refuses (a rank filter over a rank filter). Renaming each window
/// column as soon as it is computed means no later one can collide with it.
fn rename_in_place(rel: Rel, width: usize, cols: &[usize]) -> Rel {
    let exprs = cols
        .iter()
        .map(|&c| if_then(vec![(lit_bool(true), field(c))], field(c)))
        .collect();
    let emit = (0..width)
        .map(|c| match cols.iter().position(|&w| w == c) {
            Some(k) => width + k,
            None => c,
        })
        .collect();
    emit::project_emit(rel, exprs, Some(emit))
}

/// Whether `rel` computes a window function or cuts rows with a `LIMIT`
/// anywhere in it: then which rows it gives can depend on how the engine
/// orders ties, and a recomputation need not give the same ones.
fn is_ordered(rel: &Rel) -> bool {
    let Some(kind) = rel.rel_type.as_ref() else {
        return false;
    };
    if matches!(kind, RelType::Fetch(_) | RelType::Window(_)) {
        return true;
    }
    let window = |e: &Expression| {
        contains(e, &|x| {
            matches!(x.rex_type, Some(RexType::WindowFunction(_)))
        })
    };
    rel_expressions(kind).into_iter().any(window) || rel_inputs(kind).into_iter().any(is_ordered)
}

fn empty(rel: Rel) -> Region {
    Region {
        rel,
        defs: Vec::new(),
        varied: Vec::new(),
        refusals: Vec::new(),
        outputs: Vec::new(),
        slots: Vec::new(),
        cuts: Vec::new(),
        volatile: false,
        rounds: Vec::new(),
    }
}

pub(crate) fn input(r: &Option<Box<Rel>>) -> Result<&Rel> {
    r.as_deref()
        .ok_or_else(|| AdError::InvalidPlan("a relation with no input".into()))
}

fn lookup(outputs: &[usize], i: usize) -> Result<usize> {
    outputs.get(i).copied().ok_or_else(|| {
        AdError::InvalidPlan(format!(
            "field {i} is referenced, but the input has only {} columns",
            outputs.len()
        ))
    })
}

fn remap(e: Option<&Expression>, outputs: &[usize]) -> Result<Option<Expression>> {
    e.map(|e| map_fields(e, &mut |i| lookup(outputs, i)))
        .transpose()
}

pub(crate) fn apply_emit(common: Option<&RelCommon>, direct: Vec<usize>) -> Result<Vec<usize>> {
    match common.and_then(|c| c.emit_kind.as_ref()) {
        Some(EmitKind::Emit(e)) => e
            .output_mapping
            .iter()
            .map(|&i| {
                usize::try_from(i)
                    .ok()
                    .and_then(|i| direct.get(i).copied())
                    .ok_or_else(|| AdError::InvalidPlan(format!("emit refers to column {i}")))
            })
            .collect(),
        _ => Ok(direct),
    }
}

/// A read's output columns, before any emit: its projection mask, or every
/// column.
pub(crate) fn read_outputs(r: &ReadRel) -> Result<Vec<usize>> {
    let all = r.base_schema.as_ref().map_or(0, |s| s.names.len());
    match r.projection.as_ref().and_then(|m| m.select.as_ref()) {
        Some(sel) => sel
            .struct_items
            .iter()
            .map(|it| {
                if it.child.is_some() {
                    return Err(AdError::NotImplemented(
                        "a read that selects inside a nested column".into(),
                    ));
                }
                usize::try_from(it.field)
                    .ok()
                    .filter(|&f| f < all)
                    .ok_or_else(|| {
                        AdError::InvalidPlan(format!("a read selects field {}", it.field))
                    })
            })
            .collect(),
        None => Ok((0..all).collect()),
    }
}

/// Does the plan guarantee `rel` has at most one row? True for an aggregate
/// with no grouping, a one-row `VALUES`, and anything that only filters,
/// projects, sorts, limits or cross-joins such relations. False when unsure.
fn at_most_one_row(rel: &Rel) -> bool {
    let Some(kind) = &rel.rel_type else {
        return false;
    };
    let below = |r: &Option<Box<Rel>>| r.as_deref().is_some_and(at_most_one_row);
    match kind {
        RelType::Aggregate(a) => grouping_expressions(a).is_ok_and(|g| g.is_empty()),
        RelType::Project(p) => below(&p.input),
        RelType::Filter(f) => below(&f.input),
        RelType::Sort(s) => below(&s.input),
        RelType::Fetch(f) => below(&f.input),
        RelType::Cross(c) => below(&c.left) && below(&c.right),
        RelType::Read(r) => match &r.read_type {
            Some(ReadType::VirtualTable(v)) => {
                // Older producers fill the deprecated `values`.
                #[allow(deprecated)]
                let rows = v.values.len() + v.expressions.len();
                rows <= 1
            }
            _ => false,
        },
        _ => false,
    }
}

/// The grouping keys of a single-grouping-set aggregate.
pub(crate) fn grouping_expressions(a: &AggregateRel) -> Result<Vec<&Expression>> {
    if a.groupings.len() > 1 {
        return Err(AdError::NotImplemented(
            "GROUPING SETS, ROLLUP or CUBE over a wrt table".into(),
        ));
    }
    let Some(g) = a.groupings.first() else {
        return Ok(Vec::new());
    };
    if !a.grouping_expressions.is_empty() || !g.expression_references.is_empty() {
        return g
            .expression_references
            .iter()
            .map(|&r| {
                a.grouping_expressions.get(r as usize).ok_or_else(|| {
                    AdError::InvalidPlan(format!("a grouping refers to expression {r}"))
                })
            })
            .collect();
    }
    #[allow(deprecated)]
    Ok(g.grouping_expressions.iter().collect())
}

/// The number of output columns of `rel`.
pub fn width(rel: &Rel) -> Result<usize> {
    let kind = rel
        .rel_type
        .as_ref()
        .ok_or_else(|| AdError::InvalidPlan("an empty relation".into()))?;
    let (direct, common) =
        match kind {
            RelType::Read(r) => {
                let n = match &r.projection.as_ref().and_then(|m| m.select.as_ref()) {
                    Some(sel) => sel.struct_items.len(),
                    None => r.base_schema.as_ref().map_or(0, |s| {
                        s.r#struct.as_ref().map_or(s.names.len(), |t| t.types.len())
                    }),
                };
                (n, r.common.as_ref())
            }
            RelType::Project(p) => (
                width(input(&p.input)?)? + p.expressions.len(),
                p.common.as_ref(),
            ),
            RelType::Filter(f) => (width(input(&f.input)?)?, f.common.as_ref()),
            RelType::Sort(s) => (width(input(&s.input)?)?, s.common.as_ref()),
            RelType::Fetch(f) => (width(input(&f.input)?)?, f.common.as_ref()),
            RelType::Join(j) => {
                let (l, r) = (width(input(&j.left)?)?, width(input(&j.right)?)?);
                let n = match JoinType::try_from(j.r#type).unwrap_or(JoinType::Unspecified) {
                    JoinType::LeftSemi | JoinType::LeftAnti => l,
                    JoinType::RightSemi | JoinType::RightAnti => r,
                    JoinType::LeftMark => l + 1,
                    JoinType::RightMark => r + 1,
                    _ => l + r,
                };
                (n, j.common.as_ref())
            }
            RelType::Cross(c) => (
                width(input(&c.left)?)? + width(input(&c.right)?)?,
                c.common.as_ref(),
            ),
            RelType::Aggregate(a) => {
                if a.groupings.len() > 1 {
                    return Err(AdError::NotImplemented(
                        "GROUPING SETS, ROLLUP or CUBE in a differentiated query".into(),
                    ));
                }
                (
                    grouping_expressions(a)?.len() + a.measures.len(),
                    a.common.as_ref(),
                )
            }
            RelType::Set(s) => (
                width(s.inputs.first().ok_or_else(|| {
                    AdError::InvalidPlan("a set operation with no inputs".into())
                })?)?,
                s.common.as_ref(),
            ),
            other => {
                return Err(AdError::NotImplemented(format!(
                    "a {} relation in a differentiated query",
                    rel_name(other)
                )))
            }
        };
    match common.and_then(|c| c.emit_kind.as_ref()) {
        Some(EmitKind::Emit(e)) => Ok(e.output_mapping.len()),
        _ => Ok(direct),
    }
}

/// Does anything under `rel`, a subquery's included, call an aggregate or
/// window function that rounds (see [`Functions::rounds`])?
pub(crate) fn rounds_under(functions: &Functions, rel: &Rel) -> Result<bool> {
    let Some(kind) = rel.rel_type.as_ref() else {
        return Ok(false);
    };
    let calls = |anchor: u32| functions.rounds(anchor);
    match kind {
        RelType::Aggregate(a) => {
            for m in &a.measures {
                if let Some(f) = &m.measure {
                    if calls(f.function_reference)? {
                        return Ok(true);
                    }
                }
            }
        }
        RelType::Window(w) => {
            for f in &w.window_functions {
                if calls(f.function_reference)? {
                    return Ok(true);
                }
            }
        }
        _ => {}
    }
    for e in rel_expressions(kind) {
        let mut windows = Vec::new();
        collect_window_functions(e, &mut windows);
        for f in windows {
            if calls(f)? {
                return Ok(true);
            }
        }
        if subquery_rounds(functions, e)? {
            return Ok(true);
        }
    }
    for r in rel_inputs(kind) {
        if rounds_under(functions, r)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Does `e` read a subquery computed by a function that rounds? A
/// subquery that is not an uncorrelated scalar one is assumed to.
pub(crate) fn subquery_rounds(functions: &Functions, e: &Expression) -> Result<bool> {
    let mut subqueries = Vec::new();
    collect_subqueries(e, &mut subqueries);
    for sq in subqueries {
        match uncorrelated_scalar(sq) {
            Some(inner) if !rounds_under(functions, inner)? => {}
            _ => return Ok(true),
        }
    }
    Ok(false)
}

/// The function of every window function inside `e`; not inside subqueries.
fn collect_window_functions(e: &Expression, out: &mut Vec<u32>) {
    match &e.rex_type {
        Some(RexType::WindowFunction(w)) => out.push(w.function_reference),
        Some(RexType::Subquery(_)) => return,
        _ => {}
    }
    for c in children(e) {
        collect_window_functions(c, out);
    }
}

/// Every subquery inside `e`, outermost first; not inside the subqueries.
pub(crate) fn collect_subqueries<'e>(
    e: &'e Expression,
    out: &mut Vec<&'e substrait::proto::expression::Subquery>,
) {
    if let Some(RexType::Subquery(sq)) = &e.rex_type {
        out.push(sq);
        return;
    }
    for c in children(e) {
        collect_subqueries(c, out);
    }
}

fn one(r: &Option<Box<Rel>>) -> Vec<&Rel> {
    r.as_deref().into_iter().collect()
}

pub(crate) fn rel_inputs(kind: &RelType) -> Vec<&Rel> {
    match kind {
        RelType::Filter(r) => one(&r.input),
        RelType::Fetch(r) => one(&r.input),
        RelType::Aggregate(r) => one(&r.input),
        RelType::Sort(r) => one(&r.input),
        RelType::Project(r) => one(&r.input),
        RelType::Join(r) => one(&r.left).into_iter().chain(one(&r.right)).collect(),
        RelType::Cross(r) => one(&r.left).into_iter().chain(one(&r.right)).collect(),
        RelType::Set(r) => r.inputs.iter().collect(),
        RelType::ExtensionSingle(r) => one(&r.input),
        RelType::ExtensionMulti(r) => r.inputs.iter().collect(),
        _ => Vec::new(),
    }
}

pub(crate) fn rel_inputs_mut(kind: &mut RelType) -> Vec<&mut Rel> {
    fn one(r: &mut Option<Box<Rel>>) -> Vec<&mut Rel> {
        r.as_deref_mut().into_iter().collect()
    }
    match kind {
        RelType::Filter(r) => one(&mut r.input),
        RelType::Fetch(r) => one(&mut r.input),
        RelType::Aggregate(r) => one(&mut r.input),
        RelType::Sort(r) => one(&mut r.input),
        RelType::Project(r) => one(&mut r.input),
        RelType::Join(r) => {
            let (l, rt) = (&mut r.left, &mut r.right);
            one(l).into_iter().chain(one(rt)).collect()
        }
        RelType::Cross(r) => {
            let (l, rt) = (&mut r.left, &mut r.right);
            one(l).into_iter().chain(one(rt)).collect()
        }
        RelType::Set(r) => r.inputs.iter_mut().collect(),
        RelType::ExtensionSingle(r) => one(&mut r.input),
        RelType::ExtensionMulti(r) => r.inputs.iter_mut().collect(),
        _ => Vec::new(),
    }
}

pub(crate) fn rel_expressions(kind: &RelType) -> Vec<&Expression> {
    match kind {
        RelType::Filter(r) => r.condition.as_deref().into_iter().collect(),
        RelType::Project(r) => r.expressions.iter().collect(),
        RelType::Join(r) => r
            .expression
            .as_deref()
            .into_iter()
            .chain(r.post_join_filter.as_deref())
            .collect(),
        RelType::Sort(r) => r.sorts.iter().filter_map(|s| s.expr.as_ref()).collect(),
        RelType::Aggregate(r) => {
            let mut out: Vec<&Expression> = r.grouping_expressions.iter().collect();
            for m in &r.measures {
                if let Some(f) = &m.measure {
                    for a in &f.arguments {
                        if let Some(ArgType::Value(e)) = &a.arg_type {
                            out.push(e);
                        }
                    }
                }
            }
            out.extend(r.measures.iter().filter_map(|m| m.filter.as_ref()));
            out
        }
        // A read's filters, and a virtual table's rows (DataFusion writes a
        // projection over one empty row, `SELECT random()`, as one).
        RelType::Read(r) => {
            let mut out: Vec<&Expression> = r
                .filter
                .as_deref()
                .into_iter()
                .chain(r.best_effort_filter.as_deref())
                .collect();
            if let Some(ReadType::VirtualTable(v)) = &r.read_type {
                out.extend(v.expressions.iter().flat_map(|row| &row.fields));
            }
            out
        }
        _ => Vec::new(),
    }
}

pub(crate) fn rel_name(kind: &RelType) -> &'static str {
    match kind {
        RelType::Read(_) => "read",
        RelType::Filter(_) => "filter",
        RelType::Fetch(_) => "LIMIT/OFFSET",
        RelType::Aggregate(_) => "aggregate",
        RelType::Sort(_) => "sort",
        RelType::Join(_) => "join",
        RelType::Project(_) => "projection",
        RelType::Set(_) => "set operation (UNION, INTERSECT, EXCEPT)",
        RelType::ExtensionSingle(_) | RelType::ExtensionMulti(_) | RelType::ExtensionLeaf(_) => {
            "extension"
        }
        RelType::Cross(_) => "cross join",
        RelType::Reference(_) => "reference",
        RelType::Write(_) => "write",
        RelType::Ddl(_) => "DDL",
        RelType::Update(_) => "update",
        RelType::HashJoin(_) | RelType::MergeJoin(_) | RelType::NestedLoopJoin(_) => {
            "physical join"
        }
        RelType::Window(_) => "window",
        RelType::Exchange(_) => "exchange",
        RelType::Expand(_) => "expand",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_namespace_separates_its_clock_from_its_counter() {
        // Without a separator, clock 0xabc1 with counter 2 and clock 0xabc
        // with counter 0x12 both read `__ddx_abc12_`.
        let ns = new_namespace();
        let parts: Vec<&str> = ns
            .strip_prefix("__ddx_")
            .and_then(|r| r.strip_suffix('_'))
            .unwrap()
            .split('_')
            .collect();
        assert_eq!(parts.len(), 2, "{ns}");
        assert!(parts
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_hexdigit())));
        assert_ne!(ns, new_namespace());
    }
}
