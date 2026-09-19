// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Activity analysis: which columns carry gradient.
//!
//! The user names the parameters — `weight.val`, `bias.val` — and ddx derives
//! the rest rather than asking for a naming convention. A column is **active**
//! when it is both:
//!
//! - **varied**: its value depends on a parameter (outside any
//!   `ddx_stop_gradient`), and
//! - **useful**: it reaches the query's output through a value position — not
//!   only through a join condition, filter, grouping key or sort.
//!
//! Only active columns get a cotangent. Everything else is either a key the
//! backward pass joins on, or a constant: `nn.py`'s `height * 28 + width AS inp`
//! is computed but inactive, and its `images` column is data.
//!
//! The analysis errs towards "varied". Calling a column active when its true
//! derivative is zero costs, at worst, a typed error from a rule that can't
//! handle it; calling an active column inactive would drop its gradient
//! silently.

use std::fmt;

use substrait::proto::rel::RelType;

use crate::columns::{ColumnDef, Columns};
use crate::error::{AdError, Result};
use crate::expr::{args_refs, differentiable_refs};
use crate::index::{NodeId, NodeKind, PlanIndex, RelRef};
use crate::markers::Functions;

/// A parameter to differentiate with respect to: one column of a named table.
///
/// The column name is matched exactly against the table's schema as the plan
/// records it (DataFusion lower-cases unquoted identifiers, so `val`, not `VAL`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Param {
    pub table: RelRef,
    pub column: String,
}

impl Param {
    /// The column `column` of `table`.
    pub fn new(table: RelRef, column: impl Into<String>) -> Self {
        Param {
            table,
            column: column.into(),
        }
    }
}

impl fmt::Display for Param {
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
        wrt: &[Param],
    ) -> Result<Self> {
        check_wrt(index, wrt)?;
        let varied = varied(index, columns, fns, wrt)?;
        let useful = useful(index, columns, fns, &varied)?;
        Ok(Activity { varied, useful })
    }

    /// Does column `col` of `node` depend on a parameter?
    pub fn is_varied(&self, node: NodeId, col: usize) -> bool {
        self.varied[node.0][col]
    }

    /// Does column `col` of `node` reach the output through a value position?
    pub fn is_useful(&self, node: NodeId, col: usize) -> bool {
        self.useful[node.0][col]
    }

    /// Does column `col` of `node` carry gradient?
    pub fn is_active(&self, node: NodeId, col: usize) -> bool {
        self.is_varied(node, col) && self.is_useful(node, col)
    }

    /// The columns of `node` that carry gradient.
    pub fn active(&self, node: NodeId) -> Vec<usize> {
        (0..self.varied[node.0].len())
            .filter(|&c| self.is_active(node, c))
            .collect()
    }
}

/// Every parameter must name a table the plan reads and a column that table
/// has. A typo would otherwise be a gradient of zero.
fn check_wrt(index: &PlanIndex<'_>, wrt: &[Param]) -> Result<()> {
    if wrt.is_empty() {
        return Err(AdError::InvalidWrt(
            "no parameters to differentiate with respect to".into(),
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
    wrt: &[Param],
) -> Result<Vec<Vec<bool>>> {
    let mut varied: Vec<Vec<bool>> = vec![Vec::new(); index.nodes().len()];
    for id in index.forward_order() {
        let node = index.node(id);
        let reads = |refs: Vec<usize>, varied: &[Vec<bool>]| -> Result<bool> {
            for pos in refs {
                let (n, c) = columns.resolve_input(index, id, pos)?;
                if varied[n.0][c] {
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
                ColumnDef::Input(pos) => {
                    let (n, c) = columns.resolve_input(index, id, pos)?;
                    varied[n.0][c]
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
        varied[id.0] = out;
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
    useful[root.0].clone_from(&varied[root.0]);
    if !useful[root.0].contains(&true) {
        return Err(AdError::InvalidWrt(
            "the query's output doesn't depend on any of the parameters, so every gradient \
             would be zero"
                .into(),
        ));
    }

    for id in index.backward_order() {
        let mut reached = Vec::new();
        for (c, def) in columns.of(id).iter().enumerate() {
            if !useful[id.0][c] {
                continue;
            }
            match *def {
                ColumnDef::Source { .. } => {}
                ColumnDef::Input(pos) => reached.push(pos),
                ColumnDef::Computed(e) => reached.extend(differentiable_refs(e, fns)?),
                ColumnDef::Measure(m) => {
                    let args = m.measure.as_ref().map_or(&[][..], |f| &f.arguments);
                    reached.extend(args_refs(args, fns)?);
                }
                ColumnDef::Window(w) => reached.extend(args_refs(&w.arguments, fns)?),
                ColumnDef::GroupKey(_) if varied[id.0][c] => {
                    return Err(AdError::NotImplemented(format!(
                        "relation {id} groups by output column {c}, which depends on a \
                         parameter; a grouping key is a coordinate, and gradient can't flow \
                         through one"
                    )))
                }
                ColumnDef::GroupKey(_) => {}
            }
        }
        for pos in reached {
            let (n, c) = columns.resolve_input(index, id, pos)?;
            useful[n.0][c] = true;
        }
    }
    Ok(useful)
}
