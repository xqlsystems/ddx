// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Activity analysis: which columns carry gradient.
//!
//! # The three roles a column can play
//!
//! In the XQL data model a relation is an N-dimensional array in tidy form: one
//! row per coordinate tuple, with the coordinates and the data in columns. So a
//! column is either a **dim** (a coordinate identifying the row) or a **value**
//! (design.md §1 calls these dimensions and variables; §4.3 says *dim* and
//! *val*, the spelling this crate uses). Differentiation adds a third role,
//! because not every value is differentiated: a **constant** is a value that
//! gradient doesn't flow through, such as an input-data column in a query whose
//! parameters are the weights.
//!
//! This module answers *one* of those questions: which columns are values that
//! carry gradient. It deliberately does **not** classify dims, and "not active"
//! must never be read as "is a dim" — an input-data column is neither. Dim-ness
//! is **structural**: which columns identify a row is stated by the plan, in the
//! join conditions and grouping keys of whichever node consumes the relation, so
//! the transpose rules read it from there rather than from an intrinsic property
//! guessed at here (design.md §4.4).
//!
//! # Active = varied ∧ useful
//!
//! The user names the wrt columns and ddx derives the rest, rather than asking
//! for a naming convention. Following the AD literature's standard activity
//! analysis, a column is **active** when it is both:
//!
//! - **varied**: its value depends on a wrt column, outside any
//!   `ddx_stop_gradient`, and
//! - **useful**: it reaches an output column through *value* positions — not
//!   only through a join condition, filter, grouping key, sort key or `CASE`
//!   condition, in each of which the result is piecewise constant.
//!
//! Only active columns get a cotangent. A dim computed arithmetically from other
//! dims (`height * width_count + width AS inp`) is inactive because no wrt
//! column reaches it; an input-data column is inactive for the same reason.
//!
//! The two halves are asymmetric on purpose. *Varied* over-approximates — it
//! counts control-position reads, so a column can be varied whose true
//! derivative is zero — because a false positive costs at worst a typed error
//! from a rule that can't handle the column, while a false negative silently
//! drops a gradient. *Useful* follows the definition above exactly, so that a
//! value which genuinely only orders or filters rows needs no transpose rule and
//! no tag.
//!
//! # What is differentiated
//!
//! [`Activity::analyze`] seeds **every** output column of the plan's root, so a
//! root with two active columns is the gradient of their *sum*. That is the
//! contract until `vjp_query` exists to take the output columns explicitly
//! (design.md §4.4); [`Activity::seeded`] reports what was seeded so a caller
//! never has to guess.

use std::fmt;

use substrait::proto::rel::RelType;

use crate::columns::{Col, ColumnDef, Columns, Field};
use crate::error::{AdError, Result};
use crate::expr::{agg_refs, refs, window_rel_refs, Reads};
use crate::index::{NodeId, NodeKind, PlanIndex, TableRef};
use crate::markers::Functions;
use crate::names::AggKind;

/// A column of a named table: the unit a gradient is taken with respect to.
///
/// The plan-level twin of `ddx_core::ColRef`, the same concept one layer up (a
/// column reference, qualified by where it comes from). Not called `Param`: what
/// you differentiate with respect to needn't be a *parameter* — a sensitivity to
/// an input column is an ordinary request — and the AD literature's word for the
/// role is *independent variable*.
///
/// The column name is matched exactly against the table's schema as the plan
/// records it (DataFusion lower-cases unquoted identifiers, so `val`, not `VAL`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColumnRef {
    pub table: TableRef,
    pub column: String,
}

impl ColumnRef {
    /// The column `column` of `table`.
    pub fn new(table: TableRef, column: impl Into<String>) -> Self {
        ColumnRef {
            table,
            column: column.into(),
        }
    }
}

impl fmt::Display for ColumnRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.table, self.column)
    }
}

/// Which columns of which nodes carry gradient.
#[derive(Debug, Clone)]
pub struct Activity {
    varied: Vec<Vec<bool>>,
    useful: Vec<Vec<bool>>,
    seeded: Vec<Col>,
}

impl Activity {
    /// Analyse a plan with respect to `wrt`, seeding every output column of the
    /// root (see the module docs).
    pub fn analyze(
        index: &PlanIndex<'_>,
        columns: &Columns<'_>,
        fns: &Functions,
        wrt: &[ColumnRef],
    ) -> Result<Self> {
        check_wrt(index, columns, wrt)?;
        let varied = varied(index, columns, fns, wrt)?;
        let useful = useful(index, columns, fns)?;
        Ok(Activity {
            varied,
            useful,
            seeded: columns.cols(index.root()).collect(),
        })
    }

    /// Does column `col` of `node` depend on a wrt column?
    ///
    /// `col` must be a column of `node` in the analysed plan — one handed out by
    /// [`Columns::cols`].
    pub fn is_varied(&self, node: NodeId, col: Col) -> bool {
        self.varied[node.index()][col.index()]
    }

    /// Does column `col` of `node` reach an output column through value
    /// positions? Independent of the wrt columns: this is a property of the plan.
    pub fn is_useful(&self, node: NodeId, col: Col) -> bool {
        self.useful[node.index()][col.index()]
    }

    /// Does column `col` of `node` carry gradient?
    pub fn is_active(&self, node: NodeId, col: Col) -> bool {
        self.is_varied(node, col) && self.is_useful(node, col)
    }

    /// The columns of `node` that carry gradient.
    pub fn active(&self, node: NodeId) -> Vec<Col> {
        (0..self.varied[node.index()].len())
            .map(Col::new)
            .filter(|&c| self.is_active(node, c))
            .collect()
    }

    /// The root columns the backward pass is seeded at: every root column, so
    /// several active ones mean the gradient of their sum (module docs).
    pub fn seeded(&self) -> &[Col] {
        &self.seeded
    }

    /// Every wrt column must actually carry gradient somewhere, or its gradient
    /// is zero — which is worth an error rather than a column of zeros. Checked
    /// per column, so asking for one live and one dead column is refused just as
    /// loudly as asking for the dead one alone.
    ///
    /// Runs after the structural refusals in `Analysis::new`, so a plan shape ddx
    /// can't differentiate is reported as such rather than as an unreachable
    /// column.
    pub(crate) fn check_reachable(
        &self,
        index: &PlanIndex<'_>,
        columns: &Columns<'_>,
        wrt: &[ColumnRef],
    ) -> Result<()> {
        let dead: Vec<String> = wrt
            .iter()
            .filter(|p| !self.reaches_output(index, columns, p))
            .map(|p| format!("`{p}`"))
            .collect();
        if dead.is_empty() {
            return Ok(());
        }
        Err(AdError::InvalidWrt(format!(
            "no gradient can reach {}: the column never reaches the query's output as a value. \
             It may be used only as a dim (a join or grouping key), only inside a filter, or be \
             cut off by a ddx_stop_gradient",
            dead.join(", ")
        )))
    }

    fn reaches_output(
        &self,
        index: &PlanIndex<'_>,
        columns: &Columns<'_>,
        wrt: &ColumnRef,
    ) -> bool {
        index.nodes().iter().any(|n| {
            matches!(&n.kind, NodeKind::Read { table: Some(t) } if *t == wrt.table)
                && columns.cols(n.id).any(|c| {
                    matches!(columns.of(n.id)[c.index()],
                             ColumnDef::Source { name, .. } if name == wrt.column)
                        && self.is_active(n.id, c)
                })
        })
    }
}

/// Every wrt column must name a table the plan reads and a column that table
/// exposes. A typo would otherwise be a gradient of zero.
fn check_wrt(index: &PlanIndex<'_>, columns: &Columns<'_>, wrt: &[ColumnRef]) -> Result<()> {
    if wrt.is_empty() {
        return Err(AdError::InvalidWrt(
            "no columns to differentiate with respect to".into(),
        ));
    }
    for (i, p) in wrt.iter().enumerate() {
        if let Some(dup) = wrt[..i].iter().find(|q| *q == p) {
            return Err(AdError::InvalidWrt(format!(
                "`{dup}` is listed twice; each wrt column must appear once"
            )));
        }
    }
    let sources = index.sources();
    for p in wrt {
        let Some(reads) = sources.get(&p.table) else {
            let read: Vec<String> = sources.keys().map(|t| t.to_string()).collect();
            return Err(AdError::InvalidWrt(format!(
                "`{p}`: the query never reads table `{}` (it reads: {})",
                p.table,
                read.join(", ")
            )));
        };
        // A read may project its table down to a subset of columns, so "in the
        // schema" and "actually read" are different questions with different
        // remedies: a typo, versus a column the query doesn't select.
        let in_schema = reads
            .iter()
            .any(|&r| base_names(index, r).contains(&p.column));
        if !in_schema {
            return Err(AdError::InvalidWrt(format!(
                "`{p}`: table `{}` has no column `{}` (its columns: {})",
                p.table,
                p.column,
                base_names(index, reads[0]).join(", ")
            )));
        }
        let is_read = reads.iter().any(|&r| {
            columns
                .of(r)
                .iter()
                .any(|d| matches!(d, ColumnDef::Source { name, .. } if *name == p.column))
        });
        if !is_read {
            return Err(AdError::InvalidWrt(format!(
                "`{p}`: the query reads table `{}` but not its `{}` column — the plan projects \
                 that column away, so no gradient could reach it",
                p.table, p.column
            )));
        }
    }
    Ok(())
}

fn base_names<'a>(index: &PlanIndex<'a>, read: NodeId) -> &'a [String] {
    match index.node(read).rel.rel_type.as_ref() {
        Some(RelType::Read(r)) => r.base_schema.as_ref().map_or(&[], |s| &s.names),
        _ => &[],
    }
}

/// A node's window relation, if it is one: the partition and sort expressions
/// every window function in it shares.
fn window_frame<'a>(
    index: &PlanIndex<'a>,
    id: NodeId,
) -> (
    &'a [substrait::proto::Expression],
    &'a [substrait::proto::SortField],
) {
    match index.node(id).rel.rel_type.as_ref() {
        Some(RelType::Window(w)) => (&w.partition_expressions, &w.sorts),
        _ => (&[], &[]),
    }
}

fn varied(
    index: &PlanIndex<'_>,
    columns: &Columns<'_>,
    fns: &Functions,
    wrt: &[ColumnRef],
) -> Result<Vec<Vec<bool>>> {
    let mut varied: Vec<Vec<bool>> = vec![Vec::new(); index.nodes().len()];
    for id in index.forward_order() {
        let node = index.node(id);
        let (partitions, sorts) = window_frame(index, id);
        let reads = |fields: Vec<Field>, varied: &[Vec<bool>]| -> Result<bool> {
            for f in fields {
                let (n, c) = columns.resolve_input(index, id, f)?;
                if varied[n.index()][c.index()] {
                    return Ok(true);
                }
            }
            Ok(false)
        };
        let mut out = Vec::with_capacity(columns.of(id).len());
        for def in columns.of(id) {
            out.push(match *def {
                ColumnDef::Source { name, .. } => match &node.kind {
                    NodeKind::Read { table: Some(t) } => {
                        wrt.iter().any(|p| p.table == *t && p.column == name)
                    }
                    _ => false,
                },
                ColumnDef::Input(field) => {
                    let (n, c) = columns.resolve_input(index, id, field)?;
                    varied[n.index()][c.index()]
                }
                ColumnDef::Computed(e) | ColumnDef::GroupKey(e) => {
                    reads(refs(e, fns, Reads::Varied)?, &varied)?
                }
                ColumnDef::Measure(m) => {
                    let f = m.measure.as_ref().ok_or_else(|| {
                        AdError::InvalidPlan("an aggregate measure has no function".into())
                    })?;
                    // A row count is a count of rows: its value doesn't depend
                    // on the values counted, so its derivative is zero and it
                    // needs no rule and no tag.
                    if AggKind::from_name(&fns.base_name(f.function_reference)?)
                        == Some(AggKind::Count)
                    {
                        false
                    } else {
                        reads(agg_refs(f, fns, Reads::Varied)?, &varied)?
                    }
                }
                ColumnDef::Window(w) => reads(
                    window_rel_refs(w, partitions, sorts, fns, Reads::Varied)?,
                    &varied,
                )?,
            });
        }
        varied[id.index()] = out;
    }
    Ok(varied)
}

/// Which columns reach an output column through value positions. A property of
/// the plan alone — the wrt columns don't enter into it.
fn useful(index: &PlanIndex<'_>, columns: &Columns<'_>, fns: &Functions) -> Result<Vec<Vec<bool>>> {
    let mut useful: Vec<Vec<bool>> = index
        .nodes()
        .iter()
        .map(|n| vec![false; columns.of(n.id).len()])
        .collect();
    let root = index.root();
    useful[root.index()].iter_mut().for_each(|u| *u = true);

    for id in index.backward_order() {
        let (partitions, sorts) = window_frame(index, id);
        let mut reached = Vec::new();
        for (c, def) in columns.of(id).iter().enumerate() {
            if !useful[id.index()][c] {
                continue;
            }
            match *def {
                // A grouping key is a dim: it identifies the output row rather
                // than contributing a value to it, so nothing flows through it.
                ColumnDef::Source { .. } | ColumnDef::GroupKey(_) => {}
                ColumnDef::Input(field) => reached.push(field),
                ColumnDef::Computed(e) => reached.extend(refs(e, fns, Reads::Value)?),
                ColumnDef::Measure(m) => {
                    if let Some(f) = &m.measure {
                        reached.extend(agg_refs(f, fns, Reads::Value)?);
                    }
                }
                ColumnDef::Window(w) => {
                    reached.extend(window_rel_refs(w, partitions, sorts, fns, Reads::Value)?)
                }
            }
        }
        for field in reached {
            let (n, c) = columns.resolve_input(index, id, field)?;
            useful[n.index()][c.index()] = true;
        }
    }
    Ok(useful)
}
