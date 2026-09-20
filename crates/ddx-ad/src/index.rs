// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The annotated-node index: a flat view of a Substrait plan for the backward
//! walk (design.md §4.4).
//!
//! This is an implementation detail of the walker, not a competing IR — the same
//! relationship `ddx-core`'s `ColRef` has to `sqlparser::ast::Expr`. Each node
//! borrows its `Rel` from the plan, which stays the source of truth; the index
//! only adds what the walk needs to schedule itself: which node consumes which,
//! and which named tables feed the plan.

use std::collections::BTreeMap;
use std::fmt;

use substrait::proto::plan_rel::RelType as PlanRelType;
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::{Plan, Rel};

use crate::error::{AdError, Result};

/// A reference to a named table the plan reads. The unit a gradient is taken
/// with respect to is a column of one of these (see [`crate::ColumnRef`]).
///
/// Named `TableRef`, not `RelRef`: in Substrait — and in [`Node::rel`] right
/// below — a *relation* is any node of the plan, while only a named table can be
/// differentiated with respect to. (design.md §4.4 called this `RelRef`; the
/// doc follows the code.)
///
/// Substrait plans are trees, so a table read twice appears as two `Read` nodes;
/// `TableRef` is what joins them back together. That is where fan-in shows up —
/// `nn.py`'s `weight` is read once per layer (design.md §4.4).
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

/// A node's position in the index. Ids are assigned in pre-order, so a node's id
/// is always smaller than any of its inputs'.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub(crate) usize);

impl NodeId {
    /// The node at pre-order position `i`.
    pub const fn new(i: usize) -> Self {
        NodeId(i)
    }

    /// The id as an index into [`PlanIndex::nodes`].
    pub const fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// What a relation is, as far as the backward walk cares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    /// A leaf read. `table` is `Some` only for a named table — the only kind a
    /// gradient can be taken with respect to.
    Read {
        table: Option<TableRef>,
    },
    Filter,
    Project,
    Join,
    Aggregate,
    /// A `ConsistentPartitionWindowRel` — a window as its own *relation*.
    ///
    /// Route's forward idiom does **not** land here on every engine: DataFusion
    /// 54 emits `ROW_NUMBER() OVER (…)` as a window-function *expression inside
    /// a `Project`* (verified against its producer), so the Route rule's subject
    /// is that expression, not this node kind. This variant exists because a
    /// producer may legitimately use the relation form instead.
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
    /// The relations feeding this one, in the order Substrait lists them.
    pub inputs: Vec<NodeId>,
    /// The relation consuming this one; `None` for the plan's root.
    pub consumer: Option<NodeId>,
}

/// The annotated-node index over one Substrait plan.
#[derive(Debug, Clone)]
pub struct PlanIndex<'a> {
    nodes: Vec<Node<'a>>,
    root_names: &'a [String],
}

impl<'a> PlanIndex<'a> {
    /// Index `plan`, which must have exactly one root relation.
    ///
    /// A relation with no [`NodeKind`] is an [`AdError::NotImplemented`] naming
    /// it: the walk refuses a plan it can't fully see rather than skip part of it.
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

    /// Every node, indexed by [`NodeId`], in pre-order.
    pub fn nodes(&self) -> &[Node<'a>] {
        &self.nodes
    }

    /// The node with this id.
    pub fn node(&self, id: NodeId) -> &Node<'a> {
        &self.nodes[id.0]
    }

    /// The plan's root: the relation nothing consumes.
    pub fn root(&self) -> NodeId {
        NodeId(0)
    }

    /// The root's output column names, when the plan carries them (a `RelRoot`
    /// does; a bare `Rel` root doesn't).
    pub fn root_names(&self) -> &'a [String] {
        self.root_names
    }

    /// Consumers before inputs: the order the backward pass visits nodes, so a
    /// node's cotangent is complete by the time it is reached (design.md §4.4).
    ///
    /// Pre-order suffices because the plan is a tree — see [`Self::sources`] for
    /// where fan-in actually appears.
    pub fn backward_order(&self) -> impl DoubleEndedIterator<Item = NodeId> + '_ {
        // Pre-order puts every consumer before its inputs.
        self.nodes.iter().map(|n| n.id)
    }

    /// Inputs before consumers: the order forward facts (column shapes,
    /// dependence on a parameter) are computed in.
    pub fn forward_order(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.backward_order().rev()
    }

    /// The `Read` nodes of each named table. More than one entry means the table
    /// feeds several consumers, so its gradient is a sum (design.md §4.4).
    ///
    /// This is where **all** fan-in lives. A Substrait plan is a tree, so every
    /// node has exactly one consumer and no relation is shared between two of
    /// them; the only way one relation feeds several is by being read more than
    /// once. design.md §4.4 describes the general DAG discipline ("process a
    /// node only once every consumer has contributed"); on a tree that reduces
    /// to summing across a table's reads, which is what attention's `X` and
    /// `nn.py`'s per-layer `weight` actually need.
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

/// How deep a plan may nest. Plans arrive as protobuf, whose own decoder stops
/// at 100 levels, so a plan deeper than this can only have been built in
/// memory; refusing it keeps the recursion below from overflowing the stack.
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

/// The variant name of a relation, read off its `Debug` form so the list can't
/// drift from the `substrait` version the crate is pinned to.
fn rel_name(t: &RelType) -> String {
    let dbg = format!("{t:?}");
    dbg.split('(').next().unwrap_or(&dbg).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_plans::*;
    use substrait::proto::{plan_rel, PlanRel, SetRel};

    /// `SELECT … FROM a JOIN b GROUP BY …` — the shape of a contraction.
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

    /// The property the backward walk depends on: a node is reached only after
    /// the node consuming it.
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

    /// Attention's `X` feeding several projections: one table, several reads.
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
