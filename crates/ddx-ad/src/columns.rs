// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Column tracing: what each output column of each node *is*.
//!
//! Substrait refers to columns by position, and a position means something
//! different at every node. A real producer layers these freely — DataFusion
//! puts a `Project` with an `emit` remapping between a contraction's `Join` and
//! its `Aggregate`, and pushes a column mask into every `Read`. The rules can't
//! read "the contracted dim off the join condition" (design.md §4.3) until every
//! position is resolved back to where it came from, which is what this does.
//!
//! Positions inside a node's expressions index its **input row**: the input's
//! columns for a single-input relation, left's then right's for a join. That is a
//! different numbering from the node's own output columns, so the two have
//! separate types — [`Field`] and [`Col`] — rather than both being `usize`.

use std::fmt;

use substrait::proto::aggregate_rel::Measure;
use substrait::proto::consistent_partition_window_rel::WindowRelFunction;
use substrait::proto::expression::mask_expression::StructSelect;
use substrait::proto::join_rel::JoinType;
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::EmitKind;
use substrait::proto::{Expression, ReadRel, Rel, RelCommon};

use crate::error::{AdError, Result};
use crate::index::{NodeId, PlanIndex};

/// An **output column** of a node: an index into what that node produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Col(usize);

/// A position in a node's **input row**: an index into its input's columns, or —
/// for a join — into left's followed by right's. This is what a Substrait field
/// reference inside the node's expressions names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Field(usize);

impl Col {
    /// The column at index `i`.
    pub const fn new(i: usize) -> Self {
        Col(i)
    }

    /// The index.
    pub const fn index(self) -> usize {
        self.0
    }
}

impl Field {
    /// The input-row position at index `i`.
    pub const fn new(i: usize) -> Self {
        Field(i)
    }

    /// The index.
    pub const fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for Col {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "column {}", self.0)
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "input field {}", self.0)
    }
}

/// Where one output column of a node comes from.
#[derive(Debug, Clone, Copy)]
pub enum ColumnDef<'a> {
    /// A column of a `Read`: field `field` of its base schema, named `name`.
    Source { field: usize, name: &'a str },
    /// The node's input-row column at this position, passed through unchanged.
    Input(Field),
    /// A `Project` expression over the input row.
    Computed(&'a Expression),
    /// An `Aggregate` grouping key: an expression over the input row.
    GroupKey(&'a Expression),
    /// An `Aggregate` measure over the input row.
    Measure(&'a Measure),
    /// A window function of a consistent-partition window.
    Window(&'a WindowRelFunction),
}

/// The output columns of every node of a [`PlanIndex`].
#[derive(Debug, Clone)]
pub struct Columns<'a> {
    per_node: Vec<Vec<ColumnDef<'a>>>,
}

impl<'a> Columns<'a> {
    /// Trace every node's output columns.
    pub fn build(index: &PlanIndex<'a>) -> Result<Self> {
        let mut per_node: Vec<Vec<ColumnDef<'a>>> = vec![Vec::new(); index.nodes().len()];
        for id in index.forward_order() {
            let node = index.node(id);
            let input_width: usize = node.inputs.iter().map(|i| per_node[i.index()].len()).sum();
            let defs = direct_columns(node.rel, input_width, &node.inputs, &per_node)?;
            per_node[id.index()] = apply_emit(common(node.rel), defs, id)?;
        }
        Ok(Columns { per_node })
    }

    /// The output columns of `node`, indexed by [`Col`].
    pub fn of(&self, node: NodeId) -> &[ColumnDef<'a>] {
        &self.per_node[node.index()]
    }

    /// Where output column `col` of `node` comes from.
    pub fn def(&self, node: NodeId, col: Col) -> Result<ColumnDef<'a>> {
        self.of(node)
            .get(col.index())
            .copied()
            .ok_or_else(|| AdError::Internal(format!("relation {node} has no {col}")))
    }

    /// Every output column of `node`.
    pub fn cols(&self, node: NodeId) -> impl Iterator<Item = Col> {
        (0..self.of(node).len()).map(Col::new)
    }

    /// How many columns `node`'s input row has.
    pub fn input_width(&self, index: &PlanIndex<'_>, node: NodeId) -> usize {
        index
            .node(node)
            .inputs
            .iter()
            .map(|i| self.per_node[i.index()].len())
            .sum()
    }

    /// Resolve a position in `node`'s input row to the input node, and the output
    /// column *of that input*, which the position names.
    pub fn resolve_input(
        &self,
        index: &PlanIndex<'_>,
        node: NodeId,
        field: Field,
    ) -> Result<(NodeId, Col)> {
        let mut rest = field.index();
        for &input in &index.node(node).inputs {
            let width = self.per_node[input.index()].len();
            if rest < width {
                return Ok((input, Col::new(rest)));
            }
            rest -= width;
        }
        Err(AdError::InvalidPlan(format!(
            "relation {node} refers to {field}, but its input row has only {} fields",
            field.index() - rest
        )))
    }
}

fn common(rel: &Rel) -> Option<&RelCommon> {
    match rel.rel_type.as_ref()? {
        RelType::Read(r) => r.common.as_ref(),
        RelType::Filter(r) => r.common.as_ref(),
        RelType::Project(r) => r.common.as_ref(),
        RelType::Aggregate(r) => r.common.as_ref(),
        RelType::Window(r) => r.common.as_ref(),
        RelType::Sort(r) => r.common.as_ref(),
        RelType::Fetch(r) => r.common.as_ref(),
        RelType::Join(r) => r.common.as_ref(),
        _ => None,
    }
}

/// A relation's columns before its `emit` is applied.
fn direct_columns<'a>(
    rel: &'a Rel,
    input_width: usize,
    inputs: &[NodeId],
    per_node: &[Vec<ColumnDef<'a>>],
) -> Result<Vec<ColumnDef<'a>>> {
    let passthrough = |n: usize| (0..n).map(|i| ColumnDef::Input(Field::new(i)));
    Ok(match rel.rel_type.as_ref() {
        Some(RelType::Read(r)) => read_columns(r)?,
        Some(RelType::Filter(_) | RelType::Sort(_) | RelType::Fetch(_)) => {
            passthrough(input_width).collect()
        }
        Some(RelType::Project(p)) => passthrough(input_width)
            .chain(p.expressions.iter().map(ColumnDef::Computed))
            .collect(),
        Some(RelType::Window(w)) => passthrough(input_width)
            .chain(w.window_functions.iter().map(ColumnDef::Window))
            .collect(),
        Some(RelType::Join(j)) => {
            let left = per_node[inputs[0].index()].len();
            match j.r#type() {
                JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Outer => {
                    passthrough(input_width).collect()
                }
                JoinType::LeftSemi | JoinType::LeftAnti => passthrough(left).collect(),
                JoinType::RightSemi | JoinType::RightAnti => (left..input_width)
                    .map(|i| ColumnDef::Input(Field::new(i)))
                    .collect(),
                JoinType::Unspecified => {
                    return Err(AdError::InvalidPlan("a join has no join type".into()))
                }
                other => {
                    return Err(AdError::NotImplemented(format!(
                        "join type `{}`",
                        other.as_str_name()
                    )))
                }
            }
        }
        Some(RelType::Aggregate(a)) => {
            #[allow(deprecated)] // DataFusion still emits the deprecated form too.
            let keys: Vec<&Expression> = match a.groupings.as_slice() {
                [] => Vec::new(),
                [g] if !g.expression_references.is_empty() => g
                    .expression_references
                    .iter()
                    .map(|&r| {
                        a.grouping_expressions.get(r as usize).ok_or_else(|| {
                            AdError::InvalidPlan(format!(
                                "grouping refers to grouping expression {r}, which doesn't exist"
                            ))
                        })
                    })
                    .collect::<Result<_>>()?,
                [g] => g.grouping_expressions.iter().collect(),
                _ => {
                    return Err(AdError::NotImplemented(
                        "grouping sets (an aggregate with more than one grouping)".into(),
                    ))
                }
            };
            keys.into_iter()
                .map(ColumnDef::GroupKey)
                .chain(a.measures.iter().map(ColumnDef::Measure))
                .collect()
        }
        // `PlanIndex::build` already refused every other relation, so this is a
        // disagreement between the two modules rather than a plan ddx can't read.
        _ => {
            return Err(AdError::Internal(
                "a relation the index accepted has no column tracing".into(),
            ))
        }
    })
}

fn read_columns(r: &ReadRel) -> Result<Vec<ColumnDef<'_>>> {
    let schema = r
        .base_schema
        .as_ref()
        .ok_or_else(|| AdError::InvalidPlan("a read has no base schema".into()))?;
    // `NamedStruct.names` lists nested field names depth-first, so it only lines
    // up one-to-one with the top-level types when nothing is nested.
    if let Some(s) = &schema.r#struct {
        if s.types.len() != schema.names.len() {
            return Err(AdError::NotImplemented(
                "a read of a table with nested (struct-typed) columns".into(),
            ));
        }
    }
    let names = &schema.names;
    let source = |field: usize| ColumnDef::Source {
        field,
        name: names[field].as_str(),
    };
    let Some(StructSelect { struct_items }) = r.projection.as_ref().and_then(|m| m.select.as_ref())
    else {
        return Ok((0..names.len()).map(source).collect());
    };
    struct_items
        .iter()
        .map(|item| {
            if item.child.is_some() {
                return Err(AdError::NotImplemented(
                    "a read projection that selects inside a nested column".into(),
                ));
            }
            let field = usize::try_from(item.field)
                .ok()
                .filter(|&f| f < names.len())
                .ok_or_else(|| {
                    AdError::InvalidPlan(format!(
                        "a read projects field {}, but its schema has {} fields",
                        item.field,
                        names.len()
                    ))
                })?;
            Ok(source(field))
        })
        .collect()
}

fn apply_emit<'a>(
    common: Option<&RelCommon>,
    defs: Vec<ColumnDef<'a>>,
    id: NodeId,
) -> Result<Vec<ColumnDef<'a>>> {
    let Some(EmitKind::Emit(emit)) = common.and_then(|c| c.emit_kind.as_ref()) else {
        return Ok(defs);
    };
    emit.output_mapping
        .iter()
        .map(|&m| {
            usize::try_from(m)
                .ok()
                .and_then(|m| defs.get(m).copied())
                .ok_or_else(|| {
                    AdError::InvalidPlan(format!(
                        "relation {id} emits field {m}, but it has only {} fields",
                        defs.len()
                    ))
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_plans::*;

    /// The node at pre-order position `i`.
    fn node(i: usize) -> NodeId {
        NodeId::new(i)
    }

    fn kinds(cols: &Columns<'_>, n: usize) -> Vec<String> {
        cols.of(node(n))
            .iter()
            .map(|d| match d {
                ColumnDef::Source { name, .. } => format!("src:{name}"),
                ColumnDef::Input(p) => format!("in:{}", p.index()),
                ColumnDef::Computed(_) => "expr".into(),
                ColumnDef::GroupKey(_) => "key".into(),
                ColumnDef::Measure(_) => "measure".into(),
                ColumnDef::Window(_) => "window".into(),
            })
            .collect()
    }

    #[test]
    fn a_read_mask_selects_and_reorders_base_fields() {
        let p = plan(masked_read("t", &["a", "b", "c"], &[2, 0]));
        let idx = PlanIndex::build(&p).unwrap();
        let cols = Columns::build(&idx).unwrap();
        assert_eq!(kinds(&cols, 0), ["src:c", "src:a"]);
    }

    /// DataFusion's shape: a project over a join, emitting a subset reordered.
    #[test]
    fn project_emit_over_a_join_resolves_back_through_both_sides() {
        let j = join(read("a", &["i", "j", "val"]), read("b", &["j", "k", "val"]));
        let pr = emit(project(j, vec![field(0)]), &[6, 2, 4]);
        let p = plan(pr);
        let idx = PlanIndex::build(&p).unwrap();
        let cols = Columns::build(&idx).unwrap();
        assert_eq!(kinds(&cols, 0), ["expr", "in:2", "in:4"]);
        // Input position 4 is right's second column, `k`.
        assert_eq!(
            cols.resolve_input(&idx, node(0), Field::new(4)).unwrap(),
            (node(1), Col::new(4))
        );
        assert_eq!(
            cols.resolve_input(&idx, node(1), Field::new(4)).unwrap(),
            (node(3), Col::new(1))
        );
        assert!(matches!(
            cols.resolve_input(&idx, node(1), Field::new(6)),
            Err(AdError::InvalidPlan(_))
        ));
    }

    #[test]
    fn an_aggregate_is_its_keys_then_its_measures() {
        let agg = aggregate(read("t", &["i", "v"]), &[field(0)], &[(0, vec![field(1)])]);
        let p = plan(agg);
        let idx = PlanIndex::build(&p).unwrap();
        let cols = Columns::build(&idx).unwrap();
        assert_eq!(kinds(&cols, 0), ["key", "measure"]);
    }

    #[test]
    fn a_semi_join_keeps_only_one_side() {
        let p = plan(join_typed(
            read("a", &["x", "y"]),
            read("b", &["z"]),
            JoinType::LeftSemi,
        ));
        let idx = PlanIndex::build(&p).unwrap();
        let cols = Columns::build(&idx).unwrap();
        assert_eq!(kinds(&cols, 0), ["in:0", "in:1"]);

        let p = plan(join_typed(
            read("a", &["x", "y"]),
            read("b", &["z"]),
            JoinType::RightAnti,
        ));
        let idx = PlanIndex::build(&p).unwrap();
        assert_eq!(kinds(&Columns::build(&idx).unwrap(), 0), ["in:2"]);
    }

    #[test]
    fn refuses_what_it_cannot_trace() {
        let bad_emit = plan(emit(read("t", &["x"]), &[1]));
        let idx = PlanIndex::build(&bad_emit).unwrap();
        assert!(matches!(Columns::build(&idx), Err(AdError::InvalidPlan(_))));

        let mark = plan(join_typed(
            read("a", &["x"]),
            read("b", &["y"]),
            JoinType::LeftMark,
        ));
        let idx = PlanIndex::build(&mark).unwrap();
        assert!(matches!(
            Columns::build(&idx),
            Err(AdError::NotImplemented(_))
        ));

        let bad_mask = plan(masked_read("t", &["a"], &[3]));
        let idx = PlanIndex::build(&bad_mask).unwrap();
        assert!(matches!(Columns::build(&idx), Err(AdError::InvalidPlan(_))));
    }
}
