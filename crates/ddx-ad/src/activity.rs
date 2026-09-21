// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Activity analysis: which columns carry gradient.
//!
//! # The three roles of a column
//!
//! In the XQL data model a relation is an N-dimensional array in tidy form: one
//! row for each coordinate tuple, with the coordinates and the data in columns.
//! A column is therefore a dim, which is a coordinate that identifies the row,
//! or a value. design.md §1 calls these dimensions and variables, and §4.3 uses
//! the shorter words dim and val, which this crate also uses.
//!
//! Differentiation adds a third role, because ddx does not differentiate every
//! value. A constant is a value that gradient does not flow through, such as an
//! input-data column in a query whose parameters are the weights.
//!
//! This module answers one of those questions: which columns are values that
//! carry gradient. It does not classify dims. Read "not active" as "carries no
//! gradient", and never as "is a dim", because an input-data column is neither.
//!
//! Dim-ness is structural. The plan states which columns identify a row, in the
//! join conditions and the grouping keys of the node that consumes the relation.
//! The transpose rules read dim-ness from the plan, and not from a property that
//! this module infers (design.md §4.4).
//!
//! # Active means varied and useful
//!
//! The user names the wrt columns and ddx derives the rest, so the user needs no
//! naming convention. This is the standard activity analysis of the AD
//! literature. A column is active when both of these hold:
//!
//! - The column is varied. Its value depends on a wrt column, outside any
//!   `ddx_stop_gradient` call.
//! - The column is useful. It reaches an output column through value positions.
//!   A column that reaches the output only through a control position is not
//!   useful. The control positions are a join condition, a filter, a grouping
//!   key, a sort key and a `CASE` condition. In each of them the result is
//!   piecewise constant.
//!
//! Only an active column gets a cotangent. A dim that arithmetic computes from
//! other dims, such as `height * width_count + width AS inp`, is inactive,
//! because no wrt column reaches it. An input-data column is inactive for the
//! same reason.
//!
//! The two halves are asymmetric on purpose. Varied over-approximates: it counts
//! reads in control positions, so a column can be varied when its true
//! derivative is zero. A false positive costs at most a typed error from a rule
//! that cannot handle the column. A false negative drops a gradient in
//! silence. Useful follows the definition above exactly, so a value that only
//! orders or filters rows needs no transpose rule and no tag.
//!
//! # What ddx differentiates
//!
//! [`Activity::analyze`] seeds every output column of the root of the plan. A
//! root with two active columns therefore describes the gradient of the sum of
//! those columns. This is the contract until `vjp_query` exists and takes the
//! output columns as an argument (design.md §4.4). [`Activity::seeded`] reports
//! the columns that ddx seeded, so a caller never has to guess.

use std::fmt;

use substrait::proto::rel::RelType;

use crate::columns::{Col, ColumnDef, Columns, Field};
use crate::error::{AdError, Result};
use crate::expr::{agg_refs, refs, window_rel_refs, Reads};
use crate::index::{NodeId, NodeKind, PlanIndex, TableRef};
use crate::markers::Functions;
use crate::names::AggKind;

/// A column of a named table, which is the unit that a gradient is taken with
/// respect to.
///
/// This type is the plan-level twin of `ddx_core::ColRef`, the same concept one
/// layer up: a column reference, qualified by its origin. The name is not
/// `Param`, because what you differentiate with respect to need not be a
/// parameter. A sensitivity to an input column is an ordinary request, and the
/// AD literature calls the role an independent variable.
///
/// ddx matches the column name exactly against the schema of the table as the
/// plan records it. DataFusion writes an unquoted identifier in lower case, so
/// the name to give is `val` and not `VAL`.
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
    /// Analyse a plan with respect to `wrt`. ddx seeds every output column of
    /// the root, as the module documentation describes.
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

    /// Does the column `col` of `node` depend on a wrt column?
    ///
    /// The `col` argument must be a column of `node` in the analysed plan, which
    /// means a column that [`Columns::cols`] handed out.
    pub fn is_varied(&self, node: NodeId, col: Col) -> bool {
        self.varied[node.index()][col.index()]
    }

    /// Does the column `col` of `node` reach an output column through value
    /// positions? The answer does not depend on the wrt columns, because it is a
    /// property of the plan.
    pub fn is_useful(&self, node: NodeId, col: Col) -> bool {
        self.useful[node.index()][col.index()]
    }

    /// Does the column `col` of `node` carry gradient?
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

    /// The root columns that the backward pass is seeded at. ddx seeds every
    /// root column, so two or more active columns describe the gradient of their
    /// sum. The module documentation gives the reason.
    pub fn seeded(&self) -> &[Col] {
        &self.seeded
    }

    /// Every wrt column must carry gradient somewhere. A column that carries
    /// none has a gradient of zero, and ddx reports an error instead of a column
    /// of zeros.
    ///
    /// ddx checks each wrt column on its own. A request for one live column and
    /// one dead column is refused as loudly as a request for the dead column
    /// alone.
    ///
    /// This check runs after the structural refusals in `Analysis::new`. A plan
    /// shape that ddx cannot differentiate is then reported as such, and not as
    /// a column that gradient cannot reach.
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

/// Every wrt column must name a table that the plan reads, and a column that
/// the table exposes. Without this check, a mistyped name gives a gradient of
/// zero.
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
                "`{p}`: the query reads table `{}` but not its `{}` column. The plan \
                 projects that column away, so no gradient can reach it",
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

/// The partition expressions and the sort expressions of a node, if the node is
/// a window relation. Every window function in that relation shares them.
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

/// Which columns reach an output column through value positions. This is a
/// property of the plan alone, and the wrt columns do not enter into it.
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
