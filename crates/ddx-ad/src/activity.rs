// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Activity analysis: which columns carry gradient.
//!
//! # The three roles a column can play
//!
//! In the XQL data model a relation is an N-dimensional array in tidy form, so
//! its columns are either **dims** (the coordinates identifying a row) or
//! **values** (design.md §1 calls these dimensions and variables; §4.3 and
//! `nn.py` say *dim* and *val*, the spelling this crate uses). Differentiation
//! adds a third role, because not every value is differentiated: a **constant**
//! is a value gradient doesn't flow through — `nn.py`'s `images` is data, not a
//! parameter.
//!
//! This module answers *one* of those questions: which columns are values that
//! carry gradient. It deliberately does **not** classify dims, and "not active"
//! must never be read as "is a dim" — `images` is neither. Dim-ness is
//! **structural**: which columns identify a row is stated by the plan, in the
//! join conditions and grouping keys of whichever node consumes the relation, so
//! the transpose rules read it from there rather than from an intrinsic property
//! guessed at here (design.md §4.4).
//!
//! # Active = varied ∧ useful
//!
//! The user names the wrt columns — `weight.val`, `bias.val` — and ddx derives the
//! rest rather than asking for a naming convention. Following the AD literature's
//! standard activity analysis, a column is **active** when it is both:
//!
//! - **varied**: its value depends on a wrt column (outside any
//!   `ddx_stop_gradient`), and
//! - **useful**: it reaches the query's output through a value position — not
//!   only through a join condition, filter, grouping key or sort.
//!
//! Only active columns get a cotangent. `nn.py`'s `height * 28 + width AS inp` is
//! computed but inactive, and its `images` column is data.
//!
//! The analysis errs towards "varied". Calling a column active when its true
//! derivative is zero costs, at worst, a typed error from a rule that can't
//! handle it; calling an active column inactive would drop its gradient
//! silently.

use std::fmt;

use substrait::proto::rel::RelType;

use crate::columns::{Col, ColumnDef, Columns, Field};
use crate::error::{AdError, Result};
use crate::expr::{args_refs, differentiable_refs};
use crate::index::{NodeId, NodeKind, PlanIndex, TableRef};
use crate::markers::Functions;

/// A column of a named table: the unit a gradient is taken with respect to.
///
/// The plan-level twin of `ddx_core::ColRef`, the same concept one layer up (a
/// column reference, qualified by where it comes from). Not called `Param`: what
/// you differentiate with respect to needn't be a *parameter* —
/// `attention_ad_spike.py` differentiates with respect to `X`, the input — and
/// the AD literature's word for the role is *independent variable*.
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
}

impl Activity {
    /// Analyse a plan with respect to `wrt`, seeding every varied column of the
    /// root as an output to differentiate.
    pub fn analyze(
        index: &PlanIndex<'_>,
        columns: &Columns<'_>,
        fns: &Functions,
        wrt: &[ColumnRef],
    ) -> Result<Self> {
        check_wrt(index, wrt)?;
        let varied = varied(index, columns, fns, wrt)?;
        let useful = useful(index, columns, fns, &varied)?;
        Ok(Activity { varied, useful })
    }

    /// Does column `col` of `node` depend on a wrt column?
    pub fn is_varied(&self, node: NodeId, col: Col) -> bool {
        self.varied[node.index()][col.index()]
    }

    /// Does column `col` of `node` reach the output through a value position?
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
}

/// Every wrt column must name a table the plan reads and a column that table
/// has. A typo would otherwise be a gradient of zero.
fn check_wrt(index: &PlanIndex<'_>, wrt: &[ColumnRef]) -> Result<()> {
    if wrt.is_empty() {
        return Err(AdError::InvalidWrt(
            "no columns to differentiate with respect to".into(),
        ));
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
        let names = base_names(index, reads[0]);
        if !names.contains(&p.column) {
            return Err(AdError::InvalidWrt(format!(
                "`{p}`: table `{}` has no column `{}` (its columns: {})",
                p.table,
                p.column,
                names.join(", ")
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

fn varied(
    index: &PlanIndex<'_>,
    columns: &Columns<'_>,
    fns: &Functions,
    wrt: &[ColumnRef],
) -> Result<Vec<Vec<bool>>> {
    let mut varied: Vec<Vec<bool>> = vec![Vec::new(); index.nodes().len()];
    for id in index.forward_order() {
        let node = index.node(id);
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
                    reads(differentiable_refs(e, fns)?, &varied)?
                }
                ColumnDef::Measure(m) => {
                    let args = m.measure.as_ref().map_or(&[][..], |f| &f.arguments);
                    reads(args_refs(args, fns)?, &varied)?
                }
                ColumnDef::Window(w) => reads(args_refs(&w.arguments, fns)?, &varied)?,
            });
        }
        varied[id.index()] = out;
    }
    Ok(varied)
}

fn useful(
    index: &PlanIndex<'_>,
    columns: &Columns<'_>,
    fns: &Functions,
    varied: &[Vec<bool>],
) -> Result<Vec<Vec<bool>>> {
    let mut useful: Vec<Vec<bool>> = varied.iter().map(|v| vec![false; v.len()]).collect();
    let root = index.root();
    useful[root.index()].clone_from(&varied[root.index()]);
    if !useful[root.index()].contains(&true) {
        return Err(AdError::InvalidWrt(
            "the query's output doesn't depend on any of the wrt columns, so every gradient \
             would be zero"
                .into(),
        ));
    }

    for id in index.backward_order() {
        let mut reached = Vec::new();
        for (c, def) in columns.of(id).iter().enumerate() {
            if !useful[id.index()][c] {
                continue;
            }
            match *def {
                ColumnDef::Source { .. } => {}
                ColumnDef::Input(field) => reached.push(field),
                ColumnDef::Computed(e) => reached.extend(differentiable_refs(e, fns)?),
                ColumnDef::Measure(m) => {
                    let args = m.measure.as_ref().map_or(&[][..], |f| &f.arguments);
                    reached.extend(args_refs(args, fns)?);
                }
                ColumnDef::Window(w) => reached.extend(args_refs(&w.arguments, fns)?),
                ColumnDef::GroupKey(_) if varied[id.index()][c] => {
                    return Err(AdError::NotImplemented(format!(
                        "relation {id} groups by {}, which depends on a wrt column; a grouping \
                         key is a dim, and gradient can't flow through one",
                        Col::new(c)
                    )))
                }
                ColumnDef::GroupKey(_) => {}
            }
        }
        for field in reached {
            let (n, c) = columns.resolve_input(index, id, field)?;
            useful[n.index()][c.index()] = true;
        }
    }
    Ok(useful)
}
