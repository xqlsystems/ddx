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
//! Each one-level optic below is written as one table (`optic!`), from
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
//!
//! # Reading this file
//!
//! - A **whole** is the value an optic looks inside (a `Rel`, a `Plan`);
//!   its **parts** are what the optic finds there (the relation's inputs).
//!   The traits are generic over both: `Fold<Whole, Part>`.
//! - [`Fold`] reads the parts, [`Setter`] changes them, and [`Plate`] lends
//!   them all mutably at once, which a walk needs to keep them on a stack.
//!   The first two are named as in Haskell's `lens` library, and `Plate`
//!   after its `Plated` class, which lends a node's children the same way.
//! - Most optics here look one level down. `optic!` writes one from a
//!   table of cases, and `leaf!` the four that look inside a field's
//!   shape (`Option`, `Vec`, ...). [`Compose::then`] joins two optics into
//!   one that looks two levels down.
//! - [`Tree`], [`Deep`] and [`Cosmos`] are written out by hand, because
//!   they look at every level: each is a loop over a stack.

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

/// Reads the parts a `Whole` holds, in order, stopping early if `visit`
/// returns `ControlFlow::Break`. Each part is lent for as long as the
/// whole is borrowed (`'whole`), so a walk can keep parts on a stack.
///
/// [`ControlFlow`] is the standard library's "keep going or stop" value:
/// `Continue(())` moves on to the next part, `Break(value)` ends the walk
/// and hands `value` back to the caller.
pub(crate) trait Fold<Whole: ?Sized + 'static, Part: ?Sized + 'static> {
    fn try_each<'whole, Stop, Visit: FnMut(&'whole Part) -> ControlFlow<Stop>>(
        &self,
        whole: &'whole Whole,
        visit: &mut Visit,
    ) -> ControlFlow<Stop>;

    /// Call `visit` on each part.
    fn each<'whole, Visit: FnMut(&'whole Part)>(&self, whole: &'whole Whole, mut visit: Visit) {
        let _ = self.try_each::<(), _>(whole, &mut |part| {
            visit(part);
            ControlFlow::Continue(())
        });
    }

    /// Each part, in order.
    fn to_vec<'whole>(&self, whole: &'whole Whole) -> Vec<&'whole Part> {
        let mut parts = Vec::new();
        self.each(whole, |part| parts.push(part));
        parts
    }

    /// Whether any part satisfies `matches`; stops at the first that does.
    fn any(&self, whole: &Whole, mut matches: impl FnMut(&Part) -> bool) -> bool {
        self.try_each(whole, &mut |part| {
            if matches(part) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .is_break()
    }
}

/// Changes each part a `Whole` holds in place, in order, stopping at the
/// first error `change` returns.
pub(crate) trait Setter<Whole: ?Sized + 'static, Part: ?Sized + 'static> {
    fn try_modify<Error, Change: FnMut(&mut Part) -> Result<(), Error>>(
        &self,
        whole: &mut Whole,
        change: &mut Change,
    ) -> Result<(), Error>;

    /// Call `change` on each part.
    fn modify(&self, whole: &mut Whole, mut change: impl FnMut(&mut Part)) {
        let _ = self.try_modify::<std::convert::Infallible, _>(whole, &mut |part| {
            change(part);
            Ok(())
        });
    }
}

/// A one-level optic that can also lend all its parts mutably at once, for
/// as long as the whole is borrowed. A walk over a tree needs this to push
/// a node's children onto its stack and change them later; [`Setter`]
/// lends each part only during its own call.
pub(crate) trait Plate<Whole: ?Sized + 'static, Part: ?Sized + 'static>:
    Fold<Whole, Part> + Setter<Whole, Part>
{
    fn each_mut<'whole, Visit: FnMut(&'whole mut Part)>(
        &self,
        whole: &'whole mut Whole,
        visit: &mut Visit,
    );
}

/// `first.then(second)`: for each part `first` finds, the parts `second`
/// finds inside it. `Middle` is the type of `first`'s parts, which
/// `second` looks inside; Rust can't infer it from `first` and `second`
/// alone, so it is a parameter of its own.
pub(crate) struct Then<First, Second, Middle: ?Sized> {
    first: First,
    second: Second,
    middle: std::marker::PhantomData<fn(&Middle)>,
}

pub(crate) trait Compose: Sized {
    fn then<Second, Middle: ?Sized>(self, second: Second) -> Then<Self, Second, Middle> {
        Then {
            first: self,
            second,
            middle: std::marker::PhantomData,
        }
    }
}
impl<Optic> Compose for Optic {}

impl<Whole, Middle, Part, First, Second> Fold<Whole, Part> for Then<First, Second, Middle>
where
    Whole: ?Sized + 'static,
    Middle: ?Sized + 'static,
    Part: ?Sized + 'static,
    First: Fold<Whole, Middle>,
    Second: Fold<Middle, Part>,
{
    fn try_each<'whole, Stop, Visit: FnMut(&'whole Part) -> ControlFlow<Stop>>(
        &self,
        whole: &'whole Whole,
        visit: &mut Visit,
    ) -> ControlFlow<Stop> {
        self.first
            .try_each(whole, &mut |middle| self.second.try_each(middle, visit))
    }
}

impl<Whole, Middle, Part, First, Second> Setter<Whole, Part> for Then<First, Second, Middle>
where
    Whole: ?Sized + 'static,
    Middle: ?Sized + 'static,
    Part: ?Sized + 'static,
    First: Setter<Whole, Middle>,
    Second: Setter<Middle, Part>,
{
    fn try_modify<Error, Change: FnMut(&mut Part) -> Result<(), Error>>(
        &self,
        whole: &mut Whole,
        change: &mut Change,
    ) -> Result<(), Error> {
        self.first
            .try_modify(whole, &mut |middle| self.second.try_modify(middle, change))
    }
}

impl<Whole, Middle, Part, First, Second> Plate<Whole, Part> for Then<First, Second, Middle>
where
    Whole: ?Sized + 'static,
    Middle: ?Sized + 'static,
    Part: ?Sized + 'static,
    First: Plate<Whole, Middle>,
    Second: Plate<Middle, Part>,
{
    fn each_mut<'whole, Visit: FnMut(&'whole mut Part)>(
        &self,
        whole: &'whole mut Whole,
        visit: &mut Visit,
    ) {
        self.first
            .each_mut(whole, &mut |middle| self.second.each_mut(middle, visit))
    }
}

/// A one-level optic from one table. Each row of the table is a pattern
/// on the whole, then the parts that case holds, each written
/// `optic => place`: the field `place`, looked inside with `optic` (a leaf
/// optic below, or a composed one). The macro writes the optic's [`Fold`],
/// [`Setter`] and [`Plate`] from the one table. Rust's patterns bind by
/// reference when matched against a reference, so the same pattern works
/// on `&Whole` in the first and `&mut Whole` in the other two.
macro_rules! optic {
    ($(#[$attr:meta])* $name:ident : $Whole:ty => $Part:ty {
        $($pattern:pat => [$($optic:expr => $place:expr),* $(,)?]),* $(,)?
    }) => {
        $(#[$attr])*
        #[derive(Clone, Copy)]
        pub(crate) struct $name;

        impl Fold<$Whole, $Part> for $name {
            #[allow(unused_variables, unreachable_patterns, deprecated)]
            fn try_each<'whole, Stop, Visit: FnMut(&'whole $Part) -> ControlFlow<Stop>>(
                &self,
                whole: &'whole $Whole,
                visit: &mut Visit,
            ) -> ControlFlow<Stop> {
                match whole {
                    $($pattern => { $(Fold::try_each(&$optic, &$place, visit)?;)* })*
                    _ => {}
                }
                ControlFlow::Continue(())
            }
        }

        impl Setter<$Whole, $Part> for $name {
            #[allow(unused_variables, unreachable_patterns, deprecated)]
            fn try_modify<Error, Change: FnMut(&mut $Part) -> Result<(), Error>>(
                &self,
                whole: &mut $Whole,
                change: &mut Change,
            ) -> Result<(), Error> {
                match whole {
                    $($pattern => { $(Setter::try_modify(&$optic, &mut $place, change)?;)* })*
                    _ => {}
                }
                Ok(())
            }
        }

        impl Plate<$Whole, $Part> for $name {
            #[allow(unused_variables, unreachable_patterns, deprecated)]
            fn each_mut<'whole, Visit: FnMut(&'whole mut $Part)>(
                &self,
                whole: &'whole mut $Whole,
                visit: &mut Visit,
            ) {
                match whole {
                    $($pattern => { $(Plate::each_mut(&$optic, &mut $place, visit);)* })*
                    _ => {}
                }
            }
        }
    };
}

/// The leaf optics, one per shape a protobuf field has: the value itself,
/// an `Option`, an `Option<Box>` and a `Vec`. Each says how to reach the
/// values inside its shape three times, once per kind of access: `shared`
/// for [`Fold`], `unique` for [`Setter`] and `lend` for [`Plate`].
macro_rules! leaf {
    ($(#[$attr:meta])* $name:ident, $Whole:ty, |$whole:ident, $visit:ident|
        shared $shared:expr, unique $unique:expr, lend $lend:expr) => {
        $(#[$attr])*
        #[derive(Clone, Copy)]
        pub(crate) struct $name;

        impl<Item: 'static> Fold<$Whole, Item> for $name {
            fn try_each<'whole, Stop, Visit: FnMut(&'whole Item) -> ControlFlow<Stop>>(
                &self,
                $whole: &'whole $Whole,
                $visit: &mut Visit,
            ) -> ControlFlow<Stop> {
                $shared
            }
        }

        impl<Item: 'static> Setter<$Whole, Item> for $name {
            fn try_modify<Error, Change: FnMut(&mut Item) -> Result<(), Error>>(
                &self,
                $whole: &mut $Whole,
                $visit: &mut Change,
            ) -> Result<(), Error> {
                $unique
            }
        }

        impl<Item: 'static> Plate<$Whole, Item> for $name {
            fn each_mut<'whole, Visit: FnMut(&'whole mut Item)>(
                &self,
                $whole: &'whole mut $Whole,
                $visit: &mut Visit,
            ) {
                $lend
            }
        }
    };
}

leaf!(
    /// The value itself.
    Id, Item, |item, visit|
    shared visit(item), unique visit(item), lend visit(item)
);
leaf!(
    /// The value of an `Option`, if any.
    Opt, Option<Item>, |option, visit|
    shared match option { Some(item) => visit(item), None => ControlFlow::Continue(()) },
    unique match option { Some(item) => visit(item), None => Ok(()) },
    lend if let Some(item) = option { visit(item) }
);
leaf!(
    /// The value of an `Option<Box<_>>`, if any.
    OptBox, Option<Box<Item>>, |option, visit|
    shared match option { Some(item) => visit(item), None => ControlFlow::Continue(()) },
    unique match option { Some(item) => visit(item), None => Ok(()) },
    lend if let Some(item) = option { visit(item) }
);
leaf!(
    /// Each value of a `Vec`, in order.
    Each, Vec<Item>, |items, visit|
    shared { for item in items { visit(item)?; } ControlFlow::Continue(()) },
    unique { for item in items { visit(item)?; } Ok(()) },
    lend for item in items { visit(item) }
);

// ---------------------------------------------------------------- relations

optic! {
    /// The relations of a plan: its root's, then each it shares.
    PlanRels: Plan => Rel {
        Plan { relations, .. } => [Each.then(PlanRelRel) => *relations],
    }
}

optic! {
    /// The relation of one of a plan's entries: the root's input, or a
    /// relation the plan shares.
    PlanRelRel: PlanRel => Rel {
        PlanRel { rel_type: Some(PlanRelType::Root(root)) } => [Opt => root.input],
        PlanRel { rel_type: Some(PlanRelType::Rel(rel)) } => [Id => *rel],
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
        RelType::Filter(filter) => [OptBox => filter.input],
        RelType::Fetch(fetch) => [OptBox => fetch.input],
        RelType::Aggregate(aggregate) => [OptBox => aggregate.input],
        RelType::Sort(sort) => [OptBox => sort.input],
        RelType::Project(project) => [OptBox => project.input],
        RelType::Window(window) => [OptBox => window.input],
        RelType::ExtensionSingle(extension) => [OptBox => extension.input],
        RelType::Join(join) => [OptBox => join.left, OptBox => join.right],
        RelType::Cross(cross) => [OptBox => cross.left, OptBox => cross.right],
        RelType::Set(set) => [Each => set.inputs],
        RelType::ExtensionMulti(extension) => [Each => extension.inputs],
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
        RelType::Filter(filter) => [OptBox => filter.condition],
        RelType::Project(project) => [Each => project.expressions],
        RelType::Join(join) => [OptBox => join.expression, OptBox => join.post_join_filter],
        RelType::Sort(sort) => [Each.then(SortExpr) => sort.sorts],
        RelType::Aggregate(aggregate) => [
            Each => aggregate.grouping_expressions,
            Each.then(GroupingExprs) => aggregate.groupings,
            Each.then(MeasureExprs) => aggregate.measures,
        ],
        RelType::Read(read) => [
            OptBox => read.filter,
            OptBox => read.best_effort_filter,
            Opt.then(VirtualRows) => read.read_type,
        ],
    }
}

optic! {
    /// The values of a virtual table's rows, if the read is of one.
    VirtualRows: ReadType => Expression {
        ReadType::VirtualTable(table) => [Id.then(VirtualFields) => *table],
    }
}

optic! {
    /// The values of a virtual table's rows, row by row.
    VirtualFields: VirtualTable => Expression {
        VirtualTable { expressions, .. } => [Each.then(StructFields) => *expressions],
    }
}

optic! {
    /// The values of a struct literal: one virtual table row's.
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
    /// The expression a sort key sorts by.
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
    /// An aggregate function's value arguments, then its sort keys.
    AggregateExprs: AggregateFunction => Expression {
        AggregateFunction { arguments, sorts, .. } => [
            Each.then(Value) => *arguments,
            Each.then(SortExpr) => *sorts,
        ],
    }
}

optic! {
    /// A join (a prism on a relation's kind).
    AsJoin: RelType => JoinRel { RelType::Join(join) => [Id => **join] }
}

optic! {
    /// A read (a prism on a relation's kind).
    AsRead: RelType => ReadRel { RelType::Read(read) => [Id => **read] }
}

// -------------------------------------------------------------- expressions

optic! {
    /// An expression's kind: none for an empty expression.
    Rex: Expression => RexType { Expression { rex_type } => [Opt => *rex_type] }
}

optic! {
    /// A function argument that is a value.
    Value: FunctionArgument => Expression {
        FunctionArgument { arg_type: Some(ArgType::Value(value)) } => [Id => *value],
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
        RexType::ScalarFunction(function) => [
            Each.then(Value) => function.arguments,
            Each => function.args,
        ],
        RexType::WindowFunction(window) => [
            Each.then(Value) => window.arguments,
            Each => window.args,
            Each => window.partitions,
            Each.then(SortExpr) => window.sorts,
        ],
        RexType::IfThen(case) => [Each.then(IfBranch) => case.ifs, OptBox => case.r#else],
        RexType::SwitchExpression(switch) => [
            OptBox => switch.r#match,
            Each.then(SwitchBranch) => switch.ifs,
            OptBox => switch.r#else,
        ],
        RexType::SingularOrList(list) => [OptBox => list.value, Each => list.options],
        RexType::MultiOrList(list) => [Each => list.value, Each.then(RecordFields) => list.options],
        RexType::Cast(cast) => [OptBox => cast.input],
        RexType::Nested(literal) => [Opt.then(NestedValues) => literal.nested_type],
        RexType::Subquery(subquery) => [Opt.then(SubqueryOuter) => subquery.subquery_type],
    }
}

optic! {
    /// A `CASE WHEN` branch's condition, then its result.
    IfBranch: IfClause => Expression { IfClause { r#if, then } => [Opt => *r#if, Opt => *then] }
}

optic! {
    /// A `CASE x WHEN` branch's result. (The value it matches is a
    /// literal, not an expression.)
    SwitchBranch: IfValue => Expression { IfValue { then, .. } => [Opt => *then] }
}

optic! {
    /// The values of one option of a multi-column `IN` list.
    RecordFields: Record => Expression { Record { fields } => [Each => *fields] }
}

optic! {
    /// The values of a struct, list or map literal; a map's keys too.
    NestedValues: NestedType => Expression {
        NestedType::Struct(structure) => [Each => structure.fields],
        NestedType::List(list) => [Each => list.values],
        NestedType::Map(map) => [Each.then(MapEntry) => map.key_values],
    }
}

optic! {
    /// A map literal entry's key, then its value.
    MapEntry: KeyValue => Expression { KeyValue { key, value } => [Opt => *key, Opt => *value] }
}

optic! {
    /// A subquery's expressions of the row it sits in.
    SubqueryOuter: SubqueryType => Expression {
        SubqueryType::InPredicate(in_list) => [Each => in_list.needles],
        SubqueryType::SetComparison(comparison) => [OptBox => comparison.left],
    }
}

optic! {
    /// The relation of an expression of this kind, if it is a subquery.
    SubqueryRel: RexType => Rel {
        RexType::Subquery(subquery) => [Opt.then(SubqueryInner) => subquery.subquery_type],
    }
}

optic! {
    /// The relation a subquery runs.
    SubqueryInner: SubqueryType => Rel {
        SubqueryType::Scalar(scalar) => [OptBox => scalar.input],
        SubqueryType::InPredicate(in_list) => [OptBox => in_list.haystack],
        SubqueryType::SetPredicate(exists) => [OptBox => exists.tuples],
        SubqueryType::SetComparison(comparison) => [OptBox => comparison.right],
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

// Each walk below keeps a stack of the nodes still to visit. It pops a
// node, visits it, and pushes its children; the children are pushed in
// order and then reversed in place, so the first child is popped next and
// the walk visits nodes in pre-order, left to right.

impl Fold<Expression, Expression> for Cosmos {
    fn try_each<'whole, Stop, Visit: FnMut(&'whole Expression) -> ControlFlow<Stop>>(
        &self,
        root: &'whole Expression,
        visit: &mut Visit,
    ) -> ControlFlow<Stop> {
        let mut to_visit = vec![root];
        while let Some(expr) = to_visit.pop() {
            visit(expr)?;
            let first_child = to_visit.len();
            Rex.then(SubExprs).each(expr, |child| to_visit.push(child));
            to_visit[first_child..].reverse();
        }
        ControlFlow::Continue(())
    }
}

impl Setter<Expression, Expression> for Cosmos {
    fn try_modify<Error, Change: FnMut(&mut Expression) -> Result<(), Error>>(
        &self,
        root: &mut Expression,
        change: &mut Change,
    ) -> Result<(), Error> {
        let mut to_visit = vec![root];
        while let Some(expr) = to_visit.pop() {
            change(expr)?;
            let first_child = to_visit.len();
            Rex.then(SubExprs)
                .each_mut(expr, &mut |child| to_visit.push(child));
            to_visit[first_child..].reverse();
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
    fn try_each<'whole, Stop, Visit: FnMut(&'whole Rel) -> ControlFlow<Stop>>(
        &self,
        root: &'whole Rel,
        visit: &mut Visit,
    ) -> ControlFlow<Stop> {
        let mut to_visit = vec![root];
        while let Some(rel) = to_visit.pop() {
            visit(rel)?;
            let first_child = to_visit.len();
            RelKind.then(Inputs).each(rel, |input| to_visit.push(input));
            to_visit[first_child..].reverse();
        }
        ControlFlow::Continue(())
    }
}

impl Setter<Rel, Rel> for Tree {
    fn try_modify<Error, Change: FnMut(&mut Rel) -> Result<(), Error>>(
        &self,
        root: &mut Rel,
        change: &mut Change,
    ) -> Result<(), Error> {
        let mut to_visit = vec![root];
        while let Some(rel) = to_visit.pop() {
            change(rel)?;
            let first_child = to_visit.len();
            RelKind
                .then(Inputs)
                .each_mut(rel, &mut |input| to_visit.push(input));
            to_visit[first_child..].reverse();
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
    fn try_each<'whole, Stop, Visit: FnMut(&'whole Rel) -> ControlFlow<Stop>>(
        &self,
        root: &'whole Rel,
        visit: &mut Visit,
    ) -> ControlFlow<Stop> {
        let mut to_visit = vec![root];
        while let Some(rel) = to_visit.pop() {
            visit(rel)?;
            let first_child = to_visit.len();
            RelKind
                .then(Exprs)
                .then(subquery_rels())
                .each(rel, |subquery| to_visit.push(subquery));
            RelKind.then(Inputs).each(rel, |input| to_visit.push(input));
            to_visit[first_child..].reverse();
        }
        ControlFlow::Continue(())
    }
}

impl Setter<Rel, Rel> for Deep {
    fn try_modify<Error, Change: FnMut(&mut Rel) -> Result<(), Error>>(
        &self,
        root: &mut Rel,
        change: &mut Change,
    ) -> Result<(), Error> {
        let mut to_visit = vec![root];
        while let Some(rel) = to_visit.pop() {
            change(rel)?;
            // A subquery's relations are walked now, recursively (subqueries
            // nest shallowly); the inputs later, off the stack. They can't
            // be pushed for later as the Fold does: `Cosmos`, which finds
            // them, keeps its own stack, so it has no [`Plate`] to lend
            // them past its own call.
            RelKind
                .then(Exprs)
                .then(subquery_rels())
                .try_modify(rel, &mut |subquery| self.try_modify(subquery, change))?;
            let first_child = to_visit.len();
            RelKind
                .then(Inputs)
                .each_mut(rel, &mut |input| to_visit.push(input));
            to_visit[first_child..].reverse();
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
