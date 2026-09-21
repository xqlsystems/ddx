// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The annotated-node index: a flat view of a Substrait plan for the backward
//! walk (design.md §4.4).
//!
//! The index is an implementation detail of the walker and not a second IR. It
//! stands in the same relation to a Substrait plan as `ColRef` in `ddx-core`
//! stands to `sqlparser::ast::Expr`. Each node borrows its `Rel` from the plan,
//! and the plan remains the source of truth.
//!
//! The index adds only what the walk needs to schedule itself: which node
//! consumes which, and which named tables feed the plan.

use std::collections::BTreeMap;
use std::fmt;

use substrait::proto::plan_rel::RelType as PlanRelType;
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::{Plan, Rel};

use crate::error::{AdError, Result};

/// A reference to a named table that the plan reads. A gradient is taken with
/// respect to a column of such a table (see [`crate::ColumnRef`]).
///
/// The name is `TableRef` and not `RelRef`, because in Substrait a relation is
/// any node of the plan, including the [`Node::rel`] field below. Only a named
/// table can be differentiated with respect to. design.md §4.4 called this type
/// `RelRef`, and the document now follows the code.
///
/// A Substrait plan is a tree, so a table that the plan reads twice appears as
/// two `Read` nodes. `TableRef` joins the two reads back together, which is
/// where fan-in appears. A parameter table that a model reads once per layer
/// contributes one gradient term per read, and ddx sums the terms
/// (design.md §4.4).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableRef(Vec<String>);

impl TableRef {
    /// A reference to a table by its (possibly qualified) name parts.
    pub fn new<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        TableRef(names.into_iter().map(Into::into).collect())
    }

    /// The name parts, outermost qualifier first.
    pub fn names(&self) -> &[String] {
        &self.0
    }
}

impl fmt::Display for TableRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.join("."))
    }
}

/// The position of a node in the index. ddx assigns the ids in pre-order, so the
/// id of a node is always smaller than the id of each of its inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub(crate) usize);

impl NodeId {
    /// The node at pre-order position `i`.
    ///
    /// This function is for tests. A `NodeId` indexes one analysed plan, so every
    /// caller, the rules of ddx included, takes a `NodeId` from
    /// [`PlanIndex::nodes`] or from [`Node::inputs`]. A `NodeId` built by hand
    /// can name a node that does not exist.
    #[cfg(test)]
    pub(crate) const fn new(i: usize) -> Self {
        NodeId(i)
    }

    /// The id as an index into the slice that [`PlanIndex::nodes`] returns.
    pub const fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// What a relation is, in the terms that the backward walk needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    /// A leaf read. The `table` field is `Some` only for a named table. A named
    /// table is the only kind of read that a gradient can be taken with respect
    /// to.
    Read {
        table: Option<TableRef>,
    },
    Filter,
    Project,
    Join,
    Aggregate,
    /// A `ConsistentPartitionWindowRel`, which is a window as its own relation.
    ///
    /// The forward idiom for Route does not always produce this node. DataFusion
    /// 54 emits `ROW_NUMBER() OVER (…)` as a window-function expression inside a
    /// `Project`, which was checked against its producer. The subject of the
    /// Route rule is therefore that expression and not this node kind. This
    /// variant exists because another producer can use the relation form.
    Window,
    Sort,
    Fetch,
}

/// One relation of the plan.
#[derive(Debug, Clone)]
pub struct Node<'a> {
    pub id: NodeId,
    pub kind: NodeKind,
    /// The relation itself, borrowed from the plan.
    pub rel: &'a Rel,
    /// The relations that feed this one, in the order that Substrait lists
    /// them.
    pub inputs: Vec<NodeId>,
    /// The relation that consumes this one. It is `None` for the root of the
    /// plan.
    pub consumer: Option<NodeId>,
}

/// The annotated-node index over one Substrait plan.
#[derive(Debug, Clone)]
pub struct PlanIndex<'a> {
    nodes: Vec<Node<'a>>,
    root_names: &'a [String],
}

impl<'a> PlanIndex<'a> {
    /// Index `plan`. The plan must have exactly one root relation.
    ///
    /// A relation with no [`NodeKind`] produces an [`AdError::NotImplemented`]
    /// that names the relation. The walk refuses a plan that it cannot see in
    /// full, and never passes over part of one.
    pub fn build(plan: &'a Plan) -> Result<Self> {
        let [plan_rel] = plan.relations.as_slice() else {
            return Err(AdError::InvalidPlan(format!(
                "expected exactly one root relation, found {}",
                plan.relations.len()
            )));
        };
        let (root, root_names) = match plan_rel.rel_type.as_ref() {
            Some(PlanRelType::Root(root)) => (root.input.as_ref(), root.names.as_slice()),
            Some(PlanRelType::Rel(rel)) => (Some(rel), &[][..]),
            None => (None, &[][..]),
        };
        let root =
            root.ok_or_else(|| AdError::InvalidPlan("the root relation has no input".into()))?;

        let mut nodes = Vec::new();
        visit(root, None, 0, &mut nodes)?;
        Ok(PlanIndex { nodes, root_names })
    }

    /// Every node in pre-order, indexed by [`NodeId`].
    pub fn nodes(&self) -> &[Node<'a>] {
        &self.nodes
    }

    /// The node with this id.
    pub fn node(&self, id: NodeId) -> &Node<'a> {
        &self.nodes[id.0]
    }

    /// The root of the plan, which is the relation that nothing consumes.
    pub fn root(&self) -> NodeId {
        NodeId(0)
    }

    /// The names of the output columns of the root, when the plan carries them.
    /// A `RelRoot` carries the names. A bare `Rel` root does not.
    pub fn root_names(&self) -> &'a [String] {
        self.root_names
    }

    /// Consumers before inputs, which is the order in which the backward pass
    /// visits nodes. The cotangent of a node is complete when the walk reaches
    /// the node (design.md §4.4).
    ///
    /// Pre-order is enough here because the plan is a tree. [`Self::sources`]
    /// describes where fan-in appears.
    pub fn backward_order(&self) -> impl DoubleEndedIterator<Item = NodeId> + '_ {
        // Pre-order puts every consumer before its inputs.
        self.nodes.iter().map(|n| n.id)
    }

    /// Inputs before consumers, which is the order in which ddx computes the
    /// forward facts. Those facts are the shape of each column, and the
    /// dependence of a column on a wrt column.
    pub fn forward_order(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.backward_order().rev()
    }

    /// The `Read` nodes of each named table. Two or more entries mean that the
    /// table feeds several consumers, so its gradient is a sum (design.md §4.4).
    ///
    /// All fan-in appears here. A Substrait plan is a tree, so every node has
    /// exactly one consumer and no two consumers share a relation. One relation
    /// feeds several consumers only when the plan reads it more than once.
    ///
    /// Substrait itself does not guarantee the tree property. It holds because
    /// [`Self::build`] refuses a plan with several roots, and refuses any
    /// relation that references another one (`ReferenceRel`). If a later change
    /// relaxes either refusal to support shared CTE subtrees,
    /// [`Self::backward_order`] stops being a valid schedule.
    ///
    /// design.md §4.4 states the general rule for a DAG: process a node only
    /// after every consumer of that node has contributed. On a tree the rule
    /// reduces to a sum across the reads of one table. That is what a parameter
    /// table read once per layer needs, and what an input read by several
    /// projections needs.
    pub fn sources(&self) -> BTreeMap<&TableRef, Vec<NodeId>> {
        let mut out: BTreeMap<&TableRef, Vec<NodeId>> = BTreeMap::new();
        for n in &self.nodes {
            if let NodeKind::Read { table: Some(t) } = &n.kind {
                out.entry(t).or_default().push(n.id);
            }
        }
        out
    }
}

/// How deep a plan can nest. This is a guard on the recursion below, and ddx
/// refuses a deeper plan rather than risk the stack.
///
/// The bound is generous and it is not derived from anything. Decoding gives a
/// smaller bound at no cost, because prost stops at 100 nested messages and a
/// Substrait relation is several messages deep. A plan built in memory never
/// passes through the decoder, and that is the case this guard exists for.
pub(crate) const MAX_DEPTH: usize = 128;

fn visit<'a>(
    rel: &'a Rel,
    consumer: Option<NodeId>,
    depth: usize,
    nodes: &mut Vec<Node<'a>>,
) -> Result<NodeId> {
    if depth > MAX_DEPTH {
        return Err(AdError::InvalidPlan(format!(
            "plan nests deeper than {MAX_DEPTH} relations"
        )));
    }
    let (kind, inputs): (NodeKind, Vec<Option<&'a Rel>>) = match rel.rel_type.as_ref() {
        Some(RelType::Read(r)) => {
            let table = match r.read_type.as_ref() {
                Some(ReadType::NamedTable(t)) => Some(TableRef::new(t.names.iter().cloned())),
                _ => None,
            };
            (NodeKind::Read { table }, vec![])
        }
        Some(RelType::Filter(r)) => (NodeKind::Filter, vec![r.input.as_deref()]),
        Some(RelType::Project(r)) => (NodeKind::Project, vec![r.input.as_deref()]),
        Some(RelType::Aggregate(r)) => (NodeKind::Aggregate, vec![r.input.as_deref()]),
        Some(RelType::Window(r)) => (NodeKind::Window, vec![r.input.as_deref()]),
        Some(RelType::Sort(r)) => (NodeKind::Sort, vec![r.input.as_deref()]),
        Some(RelType::Fetch(r)) => (NodeKind::Fetch, vec![r.input.as_deref()]),
        Some(RelType::Join(r)) => (NodeKind::Join, vec![r.left.as_deref(), r.right.as_deref()]),
        Some(other) => {
            return Err(AdError::NotImplemented(format!(
                "relation `{}` has no place in a differentiated plan",
                rel_name(other)
            )));
        }
        None => return Err(AdError::InvalidPlan("a relation has no type".into())),
    };

    let id = NodeId(nodes.len());
    nodes.push(Node {
        id,
        kind,
        rel,
        inputs: Vec::new(),
        consumer,
    });
    let mut ids = Vec::with_capacity(inputs.len());
    for input in inputs {
        let input = input.ok_or_else(|| {
            AdError::InvalidPlan(format!("relation {id} is missing a required input"))
        })?;
        ids.push(visit(input, Some(id), depth + 1, nodes)?);
    }
    nodes[id.0].inputs = ids;
    Ok(id)
}

/// The variant name of a relation, taken from its `Debug` form. The list of
/// names cannot drift from the `substrait` version that the crate pins.
fn rel_name(t: &RelType) -> String {
    let dbg = format!("{t:?}");
    match dbg.split_once('(') {
        Some((name, _)) => name.to_string(),
        None => dbg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_plans::*;
    use substrait::proto::{plan_rel, PlanRel, SetRel};

    /// `SELECT … FROM a JOIN b GROUP BY …`, which is the shape of a contraction.
    #[test]
    fn indexes_a_join_aggregate() {
        let p = plan(aggregate(
            join(read("a", &["x"]), read("b", &["y"])),
            &[],
            &[],
        ));
        let idx = PlanIndex::build(&p).unwrap();
        let kinds: Vec<_> = idx.nodes().iter().map(|n| n.kind.clone()).collect();
        assert_eq!(
            kinds,
            [
                NodeKind::Aggregate,
                NodeKind::Join,
                NodeKind::Read {
                    table: Some(TableRef::new(["a"]))
                },
                NodeKind::Read {
                    table: Some(TableRef::new(["b"]))
                },
            ]
        );
        assert_eq!(idx.root(), NodeId(0));
        assert_eq!(idx.node(NodeId(1)).inputs, [NodeId(2), NodeId(3)]);
        assert_eq!(idx.node(NodeId(0)).consumer, None);
        assert_eq!(idx.node(NodeId(2)).consumer, Some(NodeId(1)));
    }

    /// The property that the backward walk depends on. The walk reaches a node
    /// only after it reaches the node that consumes it.
    #[test]
    fn backward_order_visits_consumers_first() {
        let p = plan(aggregate(
            join(
                join(read("a", &["x"]), read("b", &["x"])),
                aggregate(read("c", &["x"]), &[], &[]),
            ),
            &[],
            &[],
        ));
        let idx = PlanIndex::build(&p).unwrap();
        let order: Vec<_> = idx.backward_order().collect();
        assert_eq!(order.len(), idx.nodes().len());
        let pos = |id| order.iter().position(|&o| o == id).unwrap();
        for n in idx.nodes() {
            if let Some(c) = n.consumer {
                assert!(pos(c) < pos(n.id), "{c} must precede its input {}", n.id);
            }
        }
        let fwd: Vec<_> = idx.forward_order().collect();
        assert_eq!(fwd, order.into_iter().rev().collect::<Vec<_>>());
    }

    /// One table with several reads, as an input that feeds several projections
    /// produces.
    #[test]
    fn a_table_read_twice_is_fan_in() {
        let p = plan(join(
            join(read("x", &["v"]), read("wq", &["v"])),
            join(read("x", &["v"]), read("wk", &["v"])),
        ));
        let idx = PlanIndex::build(&p).unwrap();
        let sources = idx.sources();
        assert_eq!(sources[&TableRef::new(["x"])].len(), 2);
        assert_eq!(sources[&TableRef::new(["wq"])].len(), 1);
        assert_eq!(sources.len(), 3);
    }

    #[test]
    fn a_bare_rel_root_is_accepted() {
        let p = Plan {
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Rel(read("t", &["x"]))),
            }],
            ..Default::default()
        };
        let idx = PlanIndex::build(&p).unwrap();
        assert_eq!(idx.nodes().len(), 1);
        assert!(idx.root_names().is_empty());
    }

    #[test]
    fn refuses_a_relation_it_cannot_see_through() {
        let set = Rel {
            rel_type: Some(RelType::Set(SetRel::default())),
        };
        let err = PlanIndex::build(&plan(aggregate(set, &[], &[]))).unwrap_err();
        assert_eq!(
            err,
            AdError::NotImplemented("relation `Set` has no place in a differentiated plan".into())
        );
    }

    #[test]
    fn refuses_a_plan_nested_too_deeply() {
        let mut r = read("t", &["x"]);
        for _ in 0..=MAX_DEPTH {
            r = filter(r, lit(true));
        }
        assert!(matches!(
            PlanIndex::build(&plan(r)),
            Err(AdError::InvalidPlan(_))
        ));
    }

    #[test]
    fn refuses_a_missing_input() {
        for t in [
            RelType::Filter(Box::default()),
            RelType::Project(Box::default()),
        ] {
            let p = plan(Rel { rel_type: Some(t) });
            let err = PlanIndex::build(&p).unwrap_err();
            assert!(matches!(err, AdError::InvalidPlan(_)), "{err}");
        }
    }

    #[test]
    fn refuses_zero_or_many_roots() {
        assert!(matches!(
            PlanIndex::build(&Plan::default()),
            Err(AdError::InvalidPlan(_))
        ));
        let mut p = plan(read("t", &["x"]));
        p.relations.push(p.relations[0].clone());
        assert!(matches!(PlanIndex::build(&p), Err(AdError::InvalidPlan(_))));
    }

    #[test]
    fn a_non_table_read_is_a_leaf_without_a_name() {
        let leaf = Rel {
            rel_type: Some(RelType::Read(Box::default())),
        };
        let p = plan(leaf);
        let idx = PlanIndex::build(&p).unwrap();
        assert_eq!(idx.nodes()[0].kind, NodeKind::Read { table: None });
        assert!(idx.sources().is_empty());
    }
}
