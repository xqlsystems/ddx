// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Optics: where the parts of a plan are, said once.
//!
//! An optic names some parts of a value: a *lens* one part that is always
//! there (a field), a *prism* one that may be (one case of an enum), a
//! *traversal* zero or more (a relation's inputs). One optic both reads the
//! parts ([`Fold`]) and changes them in place ([`Setter`]), and optics
//! compose with [`Compose::then`]: `Deep.then(RelKind).then(AsJoin)` is every
//! join of a plan, subqueries included. This is the idea of Haskell's
//! `lens` library, and of DataFusion's `TreeNode` for its own plans.
//!
//! Each one-level optic below is written as one table ([`optic!`]), from
//! which its shared walk, its in-place walk and its worklist walk
//! ([`Plate`]) are all derived, so the three cannot disagree. (Rust can't
//! write one function for both `&T` and `&mut T`; this crate kept a `_mut`
//! twin of each walk, and the twins drifted apart: a window relation's
//! input, an aggregate's arguments.)
//!
//! [`Tree`], [`Deep`] and [`Cosmos`] are the walks over every node of a
//! tree (Haskell's `cosmos`): every relation following inputs, every
//! relation following inputs and the relations of subqueries, and every
//! expression inside an expression. Each is a loop, not recursion, since a
//! program's relations chain hundreds deep (only a subquery's relations,
//! which nest shallowly, are walked recursively), and each visits nodes in
//! pre-order, left to right, the order recursion would.
//!
//! The optics take generic closures, so a composed walk compiles to direct
//! loops, and none allocates per node.

use std::ops::ControlFlow;

use substrait::proto::aggregate_rel::{Grouping, Measure};
use substrait::proto::expression::if_then::IfClause;
use substrait::proto::expression::multi_or_list::Record;
use substrait::proto::expression::nested::map::KeyValue;
use substrait::proto::expression::nested::NestedType;
use substrait::proto::expression::subquery::SubqueryType;
use substrait::proto::expression::switch_expression::IfValue;
use substrait::proto::expression::{FieldReference, RexType, Subquery, WindowFunction};
use substrait::proto::function_argument::ArgType;
use substrait::proto::plan_rel::RelType as PlanRelType;
use substrait::proto::read_rel::{ReadType, VirtualTable};
use substrait::proto::rel::RelType;
use substrait::proto::{
    AggregateFunction, Expression, FunctionArgument, JoinRel, Plan, PlanRel, ReadRel, Rel,
    SortField,
};

/// Reads the parts an `S` holds, each borrowed for as long as `S` is (so a
/// walk can keep them), stopping when `f` breaks.
pub(crate) trait Fold<S: ?Sized + 'static, A: ?Sized + 'static> {
    fn try_each<'s, B, F: FnMut(&'s A) -> ControlFlow<B>>(
        &self,
        s: &'s S,
        f: &mut F,
    ) -> ControlFlow<B>;

    /// `f` on each part.
    fn each<'s, F: FnMut(&'s A)>(&self, s: &'s S, mut f: F) {
        let _ = self.try_each::<(), _>(s, &mut |a| {
            f(a);
            ControlFlow::Continue(())
        });
    }

    /// Each part, in order.
    fn to_vec<'s>(&self, s: &'s S) -> Vec<&'s A> {
        let mut out = Vec::new();
        self.each(s, |a| out.push(a));
        out
    }

    /// Whether any part satisfies `p`; stops at the first that does.
    fn any(&self, s: &S, mut p: impl FnMut(&A) -> bool) -> bool {
        self.try_each(s, &mut |a| {
            if p(a) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .is_break()
    }
}

/// Changes each part an `S` holds in place, stopping at the first error.
pub(crate) trait Setter<S: ?Sized + 'static, A: ?Sized + 'static> {
    fn try_modify<E, F: FnMut(&mut A) -> Result<(), E>>(
        &self,
        s: &mut S,
        f: &mut F,
    ) -> Result<(), E>;

    /// `f` on each part.
    fn modify(&self, s: &mut S, mut f: impl FnMut(&mut A)) {
        let _ = self.try_modify::<std::convert::Infallible, _>(s, &mut |a| {
            f(a);
            Ok(())
        });
    }
}

/// A one-level optic that also lends its parts mutably for as long as `S`
/// is borrowed, all at once: what a worklist walk needs to push them.
pub(crate) trait Plate<S: ?Sized + 'static, A: ?Sized + 'static>:
    Fold<S, A> + Setter<S, A>
{
    fn each_mut<'s, F: FnMut(&'s mut A)>(&self, s: &'s mut S, f: &mut F);
}

/// `a.then(b)`: the parts `b` names inside each part `a` names. `M` is the
/// type between them.
pub(crate) struct Then<T, U, M: ?Sized>(T, U, std::marker::PhantomData<fn(&M)>);

pub(crate) trait Compose: Sized {
    fn then<U, M: ?Sized>(self, next: U) -> Then<Self, U, M> {
        Then(self, next, std::marker::PhantomData)
    }
}
impl<T> Compose for T {}

impl<S, M, A, T, U> Fold<S, A> for Then<T, U, M>
where
    S: ?Sized + 'static,
    M: ?Sized + 'static,
    A: ?Sized + 'static,
    T: Fold<S, M>,
    U: Fold<M, A>,
{
    fn try_each<'s, B, F: FnMut(&'s A) -> ControlFlow<B>>(
        &self,
        s: &'s S,
        f: &mut F,
    ) -> ControlFlow<B> {
        self.0.try_each(s, &mut |m| self.1.try_each(m, f))
    }
}

impl<S, M, A, T, U> Setter<S, A> for Then<T, U, M>
where
    S: ?Sized + 'static,
    M: ?Sized + 'static,
    A: ?Sized + 'static,
    T: Setter<S, M>,
    U: Setter<M, A>,
{
    fn try_modify<E, F: FnMut(&mut A) -> Result<(), E>>(
        &self,
        s: &mut S,
        f: &mut F,
    ) -> Result<(), E> {
        self.0.try_modify(s, &mut |m| self.1.try_modify(m, f))
    }
}

impl<S, M, A, T, U> Plate<S, A> for Then<T, U, M>
where
    S: ?Sized + 'static,
    M: ?Sized + 'static,
    A: ?Sized + 'static,
    T: Plate<S, M>,
    U: Plate<M, A>,
{
    fn each_mut<'s, F: FnMut(&'s mut A)>(&self, s: &'s mut S, f: &mut F) {
        self.0.each_mut(s, &mut |m| self.1.each_mut(m, f))
    }
}

/// A one-level optic from one table: each arm a pattern on `S` and the
/// parts it holds, each as `optic => place`. Its [`Fold`], [`Setter`] and
/// [`Plate`] are all generated from the table. A pattern binds by
/// reference in all three (`&S` or `&mut S`), so one pattern serves each.
macro_rules! optic {
    ($(#[$m:meta])* $name:ident : $S:ty => $A:ty {
        $($pat:pat => [$($o:expr => $place:expr),* $(,)?]),* $(,)?
    }) => {
        $(#[$m])*
        #[derive(Clone, Copy)]
        pub(crate) struct $name;

        impl Fold<$S, $A> for $name {
            #[allow(unused_variables, unreachable_patterns, deprecated)]
            fn try_each<'s, B, F: FnMut(&'s $A) -> ControlFlow<B>>(
                &self,
                s: &'s $S,
                f: &mut F,
            ) -> ControlFlow<B> {
                match s {
                    $($pat => { $(Fold::try_each(&$o, &$place, f)?;)* })*
                    _ => {}
                }
                ControlFlow::Continue(())
            }
        }

        impl Setter<$S, $A> for $name {
            #[allow(unused_variables, unreachable_patterns, deprecated)]
            fn try_modify<E, F: FnMut(&mut $A) -> Result<(), E>>(
                &self,
                s: &mut $S,
                f: &mut F,
            ) -> Result<(), E> {
                match s {
                    $($pat => { $(Setter::try_modify(&$o, &mut $place, f)?;)* })*
                    _ => {}
                }
                Ok(())
            }
        }

        impl Plate<$S, $A> for $name {
            #[allow(unused_variables, unreachable_patterns, deprecated)]
            fn each_mut<'s, F: FnMut(&'s mut $A)>(&self, s: &'s mut $S, f: &mut F) {
                match s {
                    $($pat => { $(Plate::each_mut(&$o, &mut $place, f);)* })*
                    _ => {}
                }
            }
        }
    };
}

/// The leaf optics, one per shape a protobuf field has: the value itself,
/// an `Option`, an `Option<Box>` and a `Vec`. Each is written three times,
/// once per access, as it is the one place a shape is spelled out.
macro_rules! leaf {
    ($(#[$m:meta])* $name:ident, $S:ty, |$s:ident, $f:ident|
        shared $shared:expr, unique $unique:expr, lend $lend:expr) => {
        $(#[$m])*
        #[derive(Clone, Copy)]
        pub(crate) struct $name;

        impl<T: 'static> Fold<$S, T> for $name {
            fn try_each<'s, B, F: FnMut(&'s T) -> ControlFlow<B>>(
                &self,
                $s: &'s $S,
                $f: &mut F,
            ) -> ControlFlow<B> {
                $shared
            }
        }

        impl<T: 'static> Setter<$S, T> for $name {
            fn try_modify<E, F: FnMut(&mut T) -> Result<(), E>>(
                &self,
                $s: &mut $S,
                $f: &mut F,
            ) -> Result<(), E> {
                $unique
            }
        }

        impl<T: 'static> Plate<$S, T> for $name {
            fn each_mut<'s, F: FnMut(&'s mut T)>(&self, $s: &'s mut $S, $f: &mut F) {
                $lend
            }
        }
    };
}

leaf!(
    /// The value itself.
    Id, T, |s, f|
    shared f(s), unique f(s), lend f(s)
);
leaf!(
    /// The value of an `Option`, if any.
    Opt, Option<T>, |s, f|
    shared match s { Some(t) => f(t), None => ControlFlow::Continue(()) },
    unique match s { Some(t) => f(t), None => Ok(()) },
    lend if let Some(t) = s { f(t) }
);
leaf!(
    /// The value of an `Option<Box<_>>`, if any.
    OptBox, Option<Box<T>>, |s, f|
    shared match s { Some(t) => f(t), None => ControlFlow::Continue(()) },
    unique match s { Some(t) => f(t), None => Ok(()) },
    lend if let Some(t) = s { f(t) }
);
leaf!(
    /// Each value of a `Vec`, in order.
    Each, Vec<T>, |s, f|
    shared { for t in s { f(t)?; } ControlFlow::Continue(()) },
    unique { for t in s { f(t)?; } Ok(()) },
    lend for t in s { f(t) }
);

// ---------------------------------------------------------------- relations

optic! {
    /// The relations of a plan: its root's, then each it shares.
    PlanRels: Plan => Rel {
        Plan { relations, .. } => [Each.then(PlanRelRel) => *relations],
    }
}

optic! {
    PlanRelRel: PlanRel => Rel {
        PlanRel { rel_type: Some(PlanRelType::Root(root)) } => [Opt => root.input],
        PlanRel { rel_type: Some(PlanRelType::Rel(r)) } => [Id => *r],
    }
}

optic! {
    /// A relation's kind: none for an empty relation.
    RelKind: Rel => RelType { Rel { rel_type } => [Opt => *rel_type] }
}

optic! {
    /// The inputs of a relation, in order. The one list of them, a window
    /// relation's included.
    Inputs: RelType => Rel {
        RelType::Filter(r) => [OptBox => r.input],
        RelType::Fetch(r) => [OptBox => r.input],
        RelType::Aggregate(r) => [OptBox => r.input],
        RelType::Sort(r) => [OptBox => r.input],
        RelType::Project(r) => [OptBox => r.input],
        RelType::Window(r) => [OptBox => r.input],
        RelType::ExtensionSingle(r) => [OptBox => r.input],
        RelType::Join(r) => [OptBox => r.left, OptBox => r.right],
        RelType::Cross(r) => [OptBox => r.left, OptBox => r.right],
        RelType::Set(r) => [Each => r.inputs],
        RelType::ExtensionMulti(r) => [Each => r.inputs],
    }
}

optic! {
    /// The expressions of a relation, outermost each: a filter's condition,
    /// a projection's expressions, a join's condition and post-join filter,
    /// a sort's keys, an aggregate's groupings (each grouping's deprecated
    /// copy too) and its measures' arguments,
    /// sort keys and filters, and a read's filters and a virtual table's rows
    /// (DataFusion writes a projection over one empty row, `SELECT
    /// random()`, as one).
    Exprs: RelType => Expression {
        RelType::Filter(r) => [OptBox => r.condition],
        RelType::Project(r) => [Each => r.expressions],
        RelType::Join(r) => [OptBox => r.expression, OptBox => r.post_join_filter],
        RelType::Sort(r) => [Each.then(SortExpr) => r.sorts],
        RelType::Aggregate(r) => [
            Each => r.grouping_expressions,
            Each.then(GroupingExprs) => r.groupings,
            Each.then(MeasureExprs) => r.measures,
        ],
        RelType::Read(r) => [
            OptBox => r.filter,
            OptBox => r.best_effort_filter,
            Opt.then(VirtualRows) => r.read_type,
        ],
    }
}

optic! {
    VirtualRows: ReadType => Expression {
        ReadType::VirtualTable(v) => [Id.then(VirtualFields) => *v],
    }
}

optic! {
    VirtualFields: VirtualTable => Expression {
        VirtualTable { expressions, .. } => [Each.then(StructFields) => *expressions],
    }
}

optic! {
    StructFields: substrait::proto::expression::nested::Struct => Expression {
        substrait::proto::expression::nested::Struct { fields, .. } => [Each => *fields],
    }
}

optic! {
    /// A grouping's own copy of its expressions: deprecated, but older
    /// producers fill only these, and ddx writes them beside the top-level
    /// list, so a rewrite must reach both.
    GroupingExprs: Grouping => Expression {
        Grouping { grouping_expressions, .. } => [Each => *grouping_expressions],
    }
}

optic! {
    SortExpr: SortField => Expression { SortField { expr, .. } => [Opt => *expr] }
}

optic! {
    /// A measure's arguments and sort keys, then its filter.
    MeasureExprs: Measure => Expression {
        Measure { measure, filter } => [
            Opt.then(AggregateExprs) => *measure,
            Opt => *filter,
        ],
    }
}

optic! {
    AggregateExprs: AggregateFunction => Expression {
        AggregateFunction { arguments, sorts, .. } => [
            Each.then(Value) => *arguments,
            Each.then(SortExpr) => *sorts,
        ],
    }
}

optic! {
    /// A join (a prism on a relation's kind).
    AsJoin: RelType => JoinRel { RelType::Join(j) => [Id => **j] }
}

optic! {
    /// A read (a prism on a relation's kind).
    AsRead: RelType => ReadRel { RelType::Read(r) => [Id => **r] }
}

// -------------------------------------------------------------- expressions

optic! {
    /// An expression's kind: none for an empty expression.
    Rex: Expression => RexType { Expression { rex_type } => [Opt => *rex_type] }
}

optic! {
    /// A function argument that is a value.
    Value: FunctionArgument => Expression {
        FunctionArgument { arg_type: Some(ArgType::Value(e)) } => [Id => *e],
    }
}

optic! {
    /// The expressions directly inside an expression of this kind: the
    /// arguments of a function (the deprecated `args` too, which older
    /// producers fill), a window's partitions and sort keys, each branch of
    /// a `CASE`, an `IN` list's value and options, a cast's input, a
    /// struct, list or map literal's values, and a subquery's needles or
    /// left side, which the row computes. Not a subquery's relation.
    SubExprs: RexType => Expression {
        RexType::ScalarFunction(s) => [Each.then(Value) => s.arguments, Each => s.args],
        RexType::WindowFunction(w) => [
            Each.then(Value) => w.arguments,
            Each => w.args,
            Each => w.partitions,
            Each.then(SortExpr) => w.sorts,
        ],
        RexType::IfThen(it) => [Each.then(IfBranch) => it.ifs, OptBox => it.r#else],
        RexType::SwitchExpression(sw) => [
            OptBox => sw.r#match,
            Each.then(SwitchBranch) => sw.ifs,
            OptBox => sw.r#else,
        ],
        RexType::SingularOrList(l) => [OptBox => l.value, Each => l.options],
        RexType::MultiOrList(l) => [Each => l.value, Each.then(RecordFields) => l.options],
        RexType::Cast(c) => [OptBox => c.input],
        RexType::Nested(n) => [Opt.then(NestedValues) => n.nested_type],
        RexType::Subquery(sq) => [Opt.then(SubqueryOuter) => sq.subquery_type],
    }
}

optic! {
    IfBranch: IfClause => Expression { IfClause { r#if, then } => [Opt => *r#if, Opt => *then] }
}

optic! {
    SwitchBranch: IfValue => Expression { IfValue { then, .. } => [Opt => *then] }
}

optic! {
    RecordFields: Record => Expression { Record { fields } => [Each => *fields] }
}

optic! {
    NestedValues: NestedType => Expression {
        NestedType::Struct(s) => [Each => s.fields],
        NestedType::List(l) => [Each => l.values],
        NestedType::Map(m) => [Each.then(MapEntry) => m.key_values],
    }
}

optic! {
    MapEntry: KeyValue => Expression { KeyValue { key, value } => [Opt => *key, Opt => *value] }
}

optic! {
    /// A subquery's expressions of the row it sits in.
    SubqueryOuter: SubqueryType => Expression {
        SubqueryType::InPredicate(s) => [Each => s.needles],
        SubqueryType::SetComparison(s) => [OptBox => s.left],
    }
}

optic! {
    /// The relation of an expression of this kind, if it is a subquery.
    SubqueryRel: RexType => Rel {
        RexType::Subquery(sq) => [Opt.then(SubqueryInner) => sq.subquery_type],
    }
}

optic! {
    SubqueryInner: SubqueryType => Rel {
        SubqueryType::Scalar(s) => [OptBox => s.input],
        SubqueryType::InPredicate(s) => [OptBox => s.haystack],
        SubqueryType::SetPredicate(s) => [OptBox => s.tuples],
        SubqueryType::SetComparison(s) => [OptBox => s.right],
    }
}

optic! {
    /// A subquery (a prism on an expression's kind).
    AsSubquery: RexType => Subquery { RexType::Subquery(sq) => [Id => **sq] }
}

/// Every subquery in an expression, outermost first; not inside them.
pub(crate) fn subqueries() -> impl Fold<Expression, Subquery> {
    Cosmos.then(Rex).then(AsSubquery)
}

optic! {
    /// A field reference (a prism on an expression's kind).
    AsField: RexType => FieldReference { RexType::Selection(r) => [Id => **r] }
}

optic! {
    /// A window function call (a prism on an expression's kind).
    AsWindow: RexType => WindowFunction { RexType::WindowFunction(w) => [Id => *w] }
}

// ------------------------------------------------------------------- walks

/// Every expression in an expression, itself included, in pre-order; not
/// inside a subquery's relation. As a [`Setter`], each expression is
/// changed before the walk enters it, so it walks what `f` left.
#[derive(Clone, Copy)]
pub(crate) struct Cosmos;

impl Fold<Expression, Expression> for Cosmos {
    fn try_each<'s, B, F: FnMut(&'s Expression) -> ControlFlow<B>>(
        &self,
        s: &'s Expression,
        f: &mut F,
    ) -> ControlFlow<B> {
        let mut stack = vec![s];
        while let Some(e) = stack.pop() {
            f(e)?;
            let from = stack.len();
            Rex.then(SubExprs).each(e, |c| stack.push(c));
            stack[from..].reverse();
        }
        ControlFlow::Continue(())
    }
}

impl Setter<Expression, Expression> for Cosmos {
    fn try_modify<E, F: FnMut(&mut Expression) -> Result<(), E>>(
        &self,
        s: &mut Expression,
        f: &mut F,
    ) -> Result<(), E> {
        let mut stack = vec![s];
        while let Some(e) = stack.pop() {
            f(e)?;
            let from = stack.len();
            Rex.then(SubExprs).each_mut(e, &mut |c| stack.push(c));
            stack[from..].reverse();
        }
        Ok(())
    }
}

/// Every relation of a tree, itself included, in pre-order, following
/// inputs only: not into subqueries. As a [`Setter`], each relation is
/// changed before the walk enters it.
#[derive(Clone, Copy)]
pub(crate) struct Tree;

impl Fold<Rel, Rel> for Tree {
    fn try_each<'s, B, F: FnMut(&'s Rel) -> ControlFlow<B>>(
        &self,
        s: &'s Rel,
        f: &mut F,
    ) -> ControlFlow<B> {
        let mut stack = vec![s];
        while let Some(r) = stack.pop() {
            f(r)?;
            let from = stack.len();
            RelKind.then(Inputs).each(r, |c| stack.push(c));
            stack[from..].reverse();
        }
        ControlFlow::Continue(())
    }
}

impl Setter<Rel, Rel> for Tree {
    fn try_modify<E, F: FnMut(&mut Rel) -> Result<(), E>>(
        &self,
        s: &mut Rel,
        f: &mut F,
    ) -> Result<(), E> {
        let mut stack = vec![s];
        while let Some(r) = stack.pop() {
            f(r)?;
            let from = stack.len();
            RelKind.then(Inputs).each_mut(r, &mut |c| stack.push(c));
            stack[from..].reverse();
        }
        Ok(())
    }
}

/// Every relation of a tree, itself included, in pre-order, following
/// inputs and the relations of the subqueries in each relation's
/// expressions (a relation's subqueries before its inputs). As a
/// [`Setter`], each relation is changed before the walk enters it.
#[derive(Clone, Copy)]
pub(crate) struct Deep;

/// The relations of the subqueries anywhere in an expression, outermost
/// first; not inside them.
pub(crate) fn subquery_rels() -> impl Fold<Expression, Rel> + Setter<Expression, Rel> {
    Cosmos.then(Rex).then(SubqueryRel)
}

impl Fold<Rel, Rel> for Deep {
    fn try_each<'s, B, F: FnMut(&'s Rel) -> ControlFlow<B>>(
        &self,
        s: &'s Rel,
        f: &mut F,
    ) -> ControlFlow<B> {
        let mut stack = vec![s];
        while let Some(r) = stack.pop() {
            f(r)?;
            let from = stack.len();
            RelKind
                .then(Exprs)
                .then(subquery_rels())
                .each(r, |c| stack.push(c));
            RelKind.then(Inputs).each(r, |c| stack.push(c));
            stack[from..].reverse();
        }
        ControlFlow::Continue(())
    }
}

impl Setter<Rel, Rel> for Deep {
    fn try_modify<E, F: FnMut(&mut Rel) -> Result<(), E>>(
        &self,
        s: &mut Rel,
        f: &mut F,
    ) -> Result<(), E> {
        let mut stack = vec![s];
        while let Some(r) = stack.pop() {
            f(r)?;
            // A subquery's relations are walked now, recursively (subqueries
            // nest shallowly); the inputs later, off the stack.
            RelKind
                .then(Exprs)
                .then(subquery_rels())
                .try_modify(r, &mut |sub| self.try_modify(sub, f))?;
            let from = stack.len();
            RelKind.then(Inputs).each_mut(r, &mut |c| stack.push(c));
            stack[from..].reverse();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{filter, join, project, read_step};
    use crate::expr::{call, field, lit_bool};
    use substrait::proto::expression::nested::{self, NestedType};
    use substrait::proto::expression::subquery::{InPredicate, Scalar};
    use substrait::proto::expression::Nested;
    use substrait::proto::join_rel::JoinType;
    use substrait::proto::ConsistentPartitionWindowRel;

    fn read(name: &str) -> Rel {
        read_step(name, vec!["c".into()])
    }

    /// The table each read in `rels` names, in order.
    fn names<'a>(rels: impl IntoIterator<Item = &'a Rel>) -> Vec<String> {
        rels.into_iter()
            .filter_map(|r| match &r.rel_type {
                Some(RelType::Read(r)) => match &r.read_type {
                    Some(ReadType::NamedTable(t)) => Some(t.names.join(".")),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    fn scalar(rel: Rel) -> Expression {
        Expression {
            rex_type: Some(RexType::Subquery(Box::new(Subquery {
                subquery_type: Some(SubqueryType::Scalar(Box::new(Scalar {
                    input: Some(Box::new(rel)),
                }))),
            }))),
        }
    }

    #[test]
    fn inputs_lists_a_window_relation_s_input() {
        let window = RelType::Window(Box::new(ConsistentPartitionWindowRel {
            input: Some(Box::new(read("t"))),
            ..Default::default()
        }));
        assert_eq!(names(Inputs.to_vec(&window)), ["t"]);
    }

    #[test]
    fn a_join_s_inputs_come_left_then_right() {
        let j = join(read("l"), read("r"), lit_bool(true), JoinType::Inner);
        assert_eq!(names(RelKind.then(Inputs).to_vec(&j)), ["l", "r"]);
    }

    #[test]
    fn sub_exprs_walk_into_struct_list_and_map_literals() {
        let lit = |fields: Vec<Expression>| Expression {
            rex_type: Some(RexType::Nested(Nested {
                nested_type: Some(NestedType::Struct(nested::Struct { fields })),
                ..Default::default()
            })),
        };
        let e = lit(vec![field(0), lit(vec![field(1)])]);
        let fields: Vec<_> = Cosmos.then(Rex).then(AsField).to_vec(&e);
        assert_eq!(fields.len(), 2);
    }

    #[test]
    fn cosmos_visits_in_pre_order_left_to_right() {
        // f(g(0, 1), 2): the call, then g, its fields, then 2.
        let e = call(0, vec![call(1, vec![field(0), field(1)]), field(2)]);
        let order: Vec<_> = Cosmos
            .to_vec(&e)
            .into_iter()
            .map(|x| match &x.rex_type {
                Some(RexType::ScalarFunction(f)) => format!("f{}", f.function_reference),
                _ => crate::expr::as_field(x).map(|i| i.to_string()).unwrap(),
            })
            .collect();
        assert_eq!(order, ["f0", "f1", "0", "1", "2"]);
    }

    #[test]
    fn a_subquery_s_needles_are_the_row_s_but_its_relation_is_not() {
        let e = Expression {
            rex_type: Some(RexType::Subquery(Box::new(Subquery {
                subquery_type: Some(SubqueryType::InPredicate(Box::new(InPredicate {
                    needles: vec![field(3)],
                    haystack: Some(Box::new(filter(read("h"), field(9)))),
                }))),
            }))),
        };
        let fields: Vec<_> = Cosmos.then(Rex).then(AsField).to_vec(&e);
        assert_eq!(fields.len(), 1, "the needle, not the haystack's filter");
        assert_eq!(names(Tree.to_vec(subquery_rels().to_vec(&e)[0])), ["h"]);
    }

    #[test]
    fn deep_walks_into_subqueries_and_tree_does_not() {
        // A projection over `t` whose expression reads `s` in a subquery.
        let rel = project(read("t"), vec![scalar(read("s"))]);
        assert_eq!(names(Tree.to_vec(&rel)), ["t"]);
        assert_eq!(names(Deep.to_vec(&rel)), ["s", "t"]);
    }

    #[test]
    fn deep_changes_relations_inside_subqueries() {
        let mut rel = project(read("t"), vec![scalar(read("s"))]);
        Deep.then(RelKind).then(AsRead).modify(&mut rel, |r| {
            if let Some(ReadType::NamedTable(t)) = &mut r.read_type {
                t.names = vec![format!("{}2", t.names[0])];
            }
        });
        assert_eq!(names(Deep.to_vec(&rel)), ["s2", "t2"]);
    }

    #[test]
    fn exprs_reaches_a_grouping_s_deprecated_copy_of_its_expressions() {
        // ddx writes each grouping expression twice: top-level, and in the
        // grouping (deprecated). A rewrite through `Exprs` changes both.
        let mut kind = RelType::Aggregate(Box::new(substrait::proto::AggregateRel {
            grouping_expressions: vec![call(7, vec![field(0)])],
            #[allow(deprecated)]
            groupings: vec![Grouping {
                grouping_expressions: vec![call(7, vec![field(0)])],
                expression_references: vec![0],
            }],
            ..Default::default()
        }));
        Exprs.then(Cosmos).then(Rex).modify(&mut kind, |rex| {
            if let RexType::ScalarFunction(f) = rex {
                f.function_reference = 9;
            }
        });
        let anchors: Vec<u32> = Exprs
            .then(Cosmos)
            .then(Rex)
            .to_vec(&kind)
            .into_iter()
            .filter_map(|rex| match rex {
                RexType::ScalarFunction(f) => Some(f.function_reference),
                _ => None,
            })
            .collect();
        assert_eq!(anchors, [9, 9]);
    }

    #[test]
    fn any_stops_at_the_first_hit() {
        let rel = join(read("a"), read("b"), lit_bool(true), JoinType::Inner);
        let mut seen = 0;
        assert!(Tree.then(RelKind).then(AsRead).any(&rel, |_| {
            seen += 1;
            true
        }));
        assert_eq!(seen, 1);
    }

    #[test]
    fn try_modify_stops_at_the_first_error() {
        let mut rel = join(read("a"), read("b"), lit_bool(true), JoinType::Inner);
        let mut seen = 0;
        let out: std::result::Result<(), &str> =
            Tree.then(RelKind)
                .then(AsRead)
                .try_modify(&mut rel, &mut |_| {
                    seen += 1;
                    Err("stop")
                });
        assert_eq!((out, seen), (Err("stop"), 1));
    }

    #[test]
    fn exprs_lists_a_projection_s_expressions_and_tree_reaches_them() {
        let rel = project(filter(read("t"), field(0)), vec![field(1), field(2)]);
        let fields = Tree
            .then(RelKind)
            .then(Exprs)
            .then(Cosmos)
            .then(Rex)
            .then(AsField);
        assert_eq!(fields.to_vec(&rel).len(), 3);
    }
}
