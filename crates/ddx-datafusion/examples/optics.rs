//! Prototype: optics (lenses, prisms, traversals) over Substrait plans, as
//! a way to define each walk of ddx-ad's trees once. Not used by the crates
//! yet; see https://github.com/xqlsystems/ddx/issues/176.
//!
//! An optic names where some parts of a tree are: a lens one part that is
//! always there, a prism one that may be (one case of an enum), a traversal
//! zero or more. Optics compose with `.then`, so a deep path is built from
//! one-level ones, and one optic both reads (`Fold`) and writes (`Setter`)
//! what it names. `Tree`, `Deep` and `Cosmos` are Haskell lens's `cosmos`
//! (every node of a self-similar tree) over relations and expressions.
//!
//! ```text
//! nix develop -c cargo run -p ddx-datafusion --example optics
//! ```

use std::marker::PhantomData;

use datafusion_substrait::substrait::proto;
use proto::expression::field_reference::ReferenceType;
use proto::expression::nested::NestedType;
use proto::expression::subquery::SubqueryType;
use proto::expression::{FieldReference, RexType};
use proto::function_argument::ArgType;
use proto::join_rel::JoinType;
use proto::read_rel::ReadType;
use proto::rel::RelType;
use proto::{AggregateFunction, Expression, FunctionArgument, JoinRel, Rel};

// ---------------------------------------------------------------- optics

/// Read-only optic: the zero or more `A`s an `S` holds, borrowed for as long
/// as `S` is (so a walk can keep them on a stack).
pub trait Fold<S: ?Sized, A: ?Sized> {
    fn each<'s>(&self, s: &'s S, f: &mut dyn FnMut(&'s A));

    fn to_vec<'s>(&self, s: &'s S) -> Vec<&'s A> {
        let mut out = Vec::new();
        self.each(s, &mut |a| out.push(a));
        out
    }
    fn any(&self, s: &S, mut p: impl FnMut(&A) -> bool) -> bool {
        let mut hit = false;
        self.each(s, &mut |a| hit = hit || p(a));
        hit
    }
}

/// Write-only optic: change each `A` in an `S` in place.
pub trait Setter<S: ?Sized, A: ?Sized> {
    fn try_modify<E>(&self, s: &mut S, f: &mut dyn FnMut(&mut A) -> Result<(), E>)
        -> Result<(), E>;

    fn modify(&self, s: &mut S, mut f: impl FnMut(&mut A)) {
        let _ = self.try_modify::<std::convert::Infallible>(s, &mut |a| {
            f(a);
            Ok(())
        });
    }
}

/// A one-level optic that can also hand out its foci mutably for as long
/// as `S` is borrowed: what a worklist walk needs to push them.
pub trait Plate<S: ?Sized, A: ?Sized>: Fold<S, A> + Setter<S, A> {
    fn each_mut<'s>(&self, s: &'s mut S, f: &mut dyn FnMut(&'s mut A));
}

/// `a.then(b)`: `b`'s foci inside each of `a`'s. `A` is the type between.
pub struct Then<T, U, A: ?Sized>(T, U, PhantomData<fn(&A)>);

pub trait Compose: Sized {
    fn then<U, A: ?Sized>(self, next: U) -> Then<Self, U, A> {
        Then(self, next, PhantomData)
    }
}
impl<T> Compose for T {}

impl<S: ?Sized, A: ?Sized + 'static, B: ?Sized, T: Fold<S, A>, U: Fold<A, B>> Fold<S, B>
    for Then<T, U, A>
{
    fn each<'s>(&self, s: &'s S, f: &mut dyn FnMut(&'s B)) {
        self.0.each(s, &mut |a| self.1.each(a, &mut *f))
    }
}
impl<S: ?Sized, A: ?Sized, B: ?Sized, T: Setter<S, A>, U: Setter<A, B>> Setter<S, B>
    for Then<T, U, A>
{
    fn try_modify<E>(
        &self,
        s: &mut S,
        f: &mut dyn FnMut(&mut B) -> Result<(), E>,
    ) -> Result<(), E> {
        self.0.try_modify(s, &mut |a| self.1.try_modify(a, &mut *f))
    }
}
impl<S: ?Sized, A: ?Sized + 'static, B: ?Sized, T: Plate<S, A>, U: Plate<A, B>> Plate<S, B>
    for Then<T, U, A>
{
    fn each_mut<'s>(&self, s: &'s mut S, f: &mut dyn FnMut(&'s mut B)) {
        self.0.each_mut(s, &mut |a| self.1.each_mut(a, &mut *f))
    }
}

/// Defines a one-level optic from ONE table of foci: its `Fold`, `Setter`
/// and `Plate` are all generated from it, so the shared and the mutable
/// walks cannot disagree. Each arm is a pattern on `S` and the foci it
/// holds, each as `optic @ place`.
macro_rules! optic {
    ($(#[$m:meta])* $vis:vis $name:ident : $S:ty => $A:ty {
        $($pat:pat => [$($o:expr => $place:expr),* $(,)?]),* $(,)?
    }) => {
        $(#[$m])* #[derive(Clone, Copy)] $vis struct $name;
        impl Fold<$S, $A> for $name {
            #[allow(unused_variables, unreachable_patterns, deprecated)]
            fn each<'s>(&self, s: &'s $S, f: &mut dyn FnMut(&'s $A)) {
                match s { $($pat => { $(Fold::each(&$o, &$place, &mut *f);)* })* _ => {} }
            }
        }
        impl Setter<$S, $A> for $name {
            #[allow(unused_variables, unreachable_patterns, deprecated)]
            fn try_modify<E>(&self, s: &mut $S, f: &mut dyn FnMut(&mut $A) -> Result<(), E>)
                -> Result<(), E> {
                match s { $($pat => { $(Setter::try_modify(&$o, &mut $place, &mut *f)?;)* })* _ => {} }
                Ok(())
            }
        }
        impl Plate<$S, $A> for $name {
            #[allow(unused_variables, unreachable_patterns, deprecated)]
            fn each_mut<'s>(&self, s: &'s mut $S, f: &mut dyn FnMut(&'s mut $A)) {
                match s { $($pat => { $(Plate::each_mut(&$o, &mut $place, &mut *f);)* })* _ => {} }
            }
        }
    };
}

/// The leaf optics, one per container shape a protobuf field has.
#[derive(Clone, Copy)]
pub struct Id;
#[derive(Clone, Copy)]
pub struct Opt;
#[derive(Clone, Copy)]
pub struct OptBox;
#[derive(Clone, Copy)]
pub struct Each;

macro_rules! leaf {
    ($name:ident, $S:ty, |$s:ident, $f:ident| $shared:expr, $unique:expr) => {
        impl<T> Fold<$S, T> for $name {
            fn each<'s>(&self, $s: &'s $S, $f: &mut dyn FnMut(&'s T)) {
                $shared
            }
        }
        impl<T> Setter<$S, T> for $name {
            fn try_modify<E>(
                &self,
                $s: &mut $S,
                g: &mut dyn FnMut(&mut T) -> Result<(), E>,
            ) -> Result<(), E> {
                let mut err = Ok(());
                let $f = &mut |t: &mut T| {
                    if err.is_ok() {
                        err = g(t)
                    }
                };
                $unique;
                err
            }
        }
        impl<T> Plate<$S, T> for $name {
            fn each_mut<'s>(&self, $s: &'s mut $S, $f: &mut dyn FnMut(&'s mut T)) {
                $unique
            }
        }
    };
}
leaf!(Id, T, |s, f| f(s), f(s));
leaf!(
    Opt,
    Option<T>,
    |s, f| s.iter().for_each(f),
    s.iter_mut().for_each(f)
);
leaf!(
    OptBox,
    Option<Box<T>>,
    |s, f| s.iter().for_each(|b| f(b)),
    s.iter_mut().for_each(|b| f(b))
);
leaf!(
    Each,
    Vec<T>,
    |s, f| s.iter().for_each(f),
    s.iter_mut().for_each(f)
);

// ------------------------------------------------- optics on Substrait plans

optic! {
    /// A relation's kind (a prism: none for an empty relation).
    pub Kind: Rel => RelType { Rel { rel_type } => [Opt => *rel_type] }
}
optic! {
    /// An expression's kind.
    pub Rex: Expression => RexType { Expression { rex_type } => [Opt => *rex_type] }
}
optic! {
    /// A join (a prism on a relation's kind).
    pub AsJoin: RelType => JoinRel { RelType::Join(j) => [Id => **j] }
}
optic! {
    /// A field reference (a prism on an expression's kind).
    pub AsField: RexType => FieldReference { RexType::Selection(r) => [Id => **r] }
}
optic! {
    /// The direct inputs of a relation. The ONE list: `rel_inputs`,
    /// `rel_inputs_mut`, `input_slots` and `take_input_slots` are four today,
    /// and only two of them list a window relation's input.
    pub Inputs: RelType => Rel {
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
    /// A function's value arguments.
    pub Values: FunctionArgument => Expression {
        FunctionArgument { arg_type: Some(ArgType::Value(e)) } => [Id => *e],
    }
}
optic! {
    /// The expressions a relation holds, measure arguments included.
    pub Exprs: RelType => Expression {
        RelType::Filter(r) => [OptBox => r.condition],
        RelType::Project(r) => [Each => r.expressions],
        RelType::Join(r) => [OptBox => r.expression, OptBox => r.post_join_filter],
        RelType::Sort(r) => [Each.then(SortExpr) => r.sorts],
        RelType::Aggregate(r) => [
            Each => r.grouping_expressions,
            Each.then(MeasureParts) => r.measures,
        ],
        RelType::Read(r) => [OptBox => r.filter, OptBox => r.best_effort_filter],
    }
}
optic! {
    pub SortExpr: proto::SortField => Expression { proto::SortField { expr, .. } => [Opt => *expr] }
}
optic! {
    /// A measure's arguments and its filter.
    pub MeasureParts: proto::aggregate_rel::Measure => Expression {
        proto::aggregate_rel::Measure { measure, filter } => [
            Opt.then(FnArgs) => *measure,
            Opt => *filter,
        ],
    }
}
optic! {
    pub FnArgs: AggregateFunction => Expression {
        AggregateFunction { arguments, .. } => [Each.then(Values) => *arguments],
    }
}
optic! {
    /// The direct subexpressions of an expression. The ONE list: today
    /// `children`, `rex_children_mut`, `walk_fields`, `redeclare_expr`,
    /// `redeclare_functions` and `tangent::substitute` each keep their own,
    /// and they disagree on `Nested` and on window partitions.
    pub SubExprs: RexType => Expression {
        RexType::ScalarFunction(s) => [Each.then(Values) => s.arguments, Each => s.args],
        RexType::WindowFunction(w) => [
            Each.then(Values) => w.arguments, Each => w.args,
            Each => w.partitions, Each.then(SortExpr) => w.sorts,
        ],
        RexType::IfThen(it) => [Each.then(IfClause) => it.ifs, OptBox => it.r#else],
        RexType::SwitchExpression(sw) => [
            OptBox => sw.r#match, Each.then(SwitchThen) => sw.ifs, OptBox => sw.r#else,
        ],
        RexType::SingularOrList(l) => [OptBox => l.value, Each => l.options],
        RexType::MultiOrList(l) => [Each => l.value, Each.then(Record) => l.options],
        RexType::Cast(c) => [OptBox => c.input],
        RexType::Nested(n) => [Opt.then(NestedParts) => n.nested_type],
        // A subquery's needles and left side are this row's expressions.
        RexType::Subquery(sq) => [Opt.then(SubqueryOuter) => sq.subquery_type],
    }
}
optic! {
    pub IfClause: proto::expression::if_then::IfClause => Expression {
        proto::expression::if_then::IfClause { r#if, then } => [Opt => *r#if, Opt => *then],
    }
}
optic! {
    pub SwitchThen: proto::expression::switch_expression::IfValue => Expression {
        proto::expression::switch_expression::IfValue { then, .. } => [Opt => *then],
    }
}
optic! {
    pub Record: proto::expression::multi_or_list::Record => Expression {
        proto::expression::multi_or_list::Record { fields } => [Each => *fields],
    }
}
optic! {
    pub NestedParts: NestedType => Expression {
        NestedType::Struct(s) => [Each => s.fields],
        NestedType::List(l) => [Each => l.values],
        NestedType::Map(m) => [Each.then(KeyValue) => m.key_values],
    }
}
optic! {
    pub KeyValue: proto::expression::nested::map::KeyValue => Expression {
        proto::expression::nested::map::KeyValue { key, value } => [Opt => *key, Opt => *value],
    }
}
optic! {
    pub SubqueryOuter: SubqueryType => Expression {
        SubqueryType::InPredicate(s) => [Each => s.needles],
        SubqueryType::SetComparison(s) => [OptBox => s.left],
    }
}
optic! {
    /// The relations of an expression's own subquery, if it is one.
    pub SubqueryRels: RexType => Rel {
        RexType::Subquery(sq) => [Opt.then(SubqueryInner) => sq.subquery_type],
    }
}
optic! {
    pub SubqueryInner: SubqueryType => Rel {
        SubqueryType::Scalar(s) => [OptBox => s.input],
        SubqueryType::InPredicate(s) => [OptBox => s.haystack],
        SubqueryType::SetPredicate(s) => [OptBox => s.tuples],
        SubqueryType::SetComparison(s) => [OptBox => s.right],
    }
}

// ------------------------------------------- closures: Plated's `cosmos`

/// Every expression in an expression, itself included, outermost first;
/// not inside subqueries' relations. A loop, not recursion.
#[derive(Clone, Copy)]
pub struct Cosmos;
impl Fold<Expression, Expression> for Cosmos {
    fn each<'s>(&self, s: &'s Expression, f: &mut dyn FnMut(&'s Expression)) {
        let mut stack = vec![s];
        while let Some(e) = stack.pop() {
            f(e);
            Rex.then(SubExprs).each(e, &mut |c| stack.push(c));
        }
    }
}
impl Setter<Expression, Expression> for Cosmos {
    /// Top-down: `f` sees a node before its (possibly new) children.
    fn try_modify<E>(
        &self,
        s: &mut Expression,
        f: &mut dyn FnMut(&mut Expression) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut stack = vec![s];
        while let Some(e) = stack.pop() {
            f(e)?;
            Rex.then(SubExprs).each_mut(e, &mut |c| stack.push(c));
        }
        Ok(())
    }
}

/// Every relation of a tree, itself included: `Tree` follows inputs only,
/// `Deep` also the relations of subqueries in its expressions. Which walk a
/// site means is now said by name rather than by which helpers it calls.
#[derive(Clone, Copy)]
pub struct Tree;
#[derive(Clone, Copy)]
pub struct Deep;

/// The relations of the subqueries anywhere in an expression.
fn subquery_rels() -> impl Fold<Expression, Rel> + Setter<Expression, Rel> {
    Cosmos.then(Rex).then(SubqueryRels)
}

impl Fold<Rel, Rel> for Tree {
    fn each<'s>(&self, s: &'s Rel, f: &mut dyn FnMut(&'s Rel)) {
        let mut stack = vec![s];
        while let Some(r) = stack.pop() {
            f(r);
            Kind.then(Inputs).each(r, &mut |c| stack.push(c));
        }
    }
}
impl Setter<Rel, Rel> for Tree {
    fn try_modify<E>(
        &self,
        s: &mut Rel,
        f: &mut dyn FnMut(&mut Rel) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut stack = vec![s];
        while let Some(r) = stack.pop() {
            f(r)?;
            Kind.then(Inputs).each_mut(r, &mut |c| stack.push(c));
        }
        Ok(())
    }
}
impl Fold<Rel, Rel> for Deep {
    fn each<'s>(&self, s: &'s Rel, f: &mut dyn FnMut(&'s Rel)) {
        let mut stack = vec![s];
        while let Some(r) = stack.pop() {
            f(r);
            Kind.then(Exprs)
                .then(subquery_rels())
                .each(r, &mut |c| stack.push(c));
            Kind.then(Inputs).each(r, &mut |c| stack.push(c));
        }
    }
}
impl Setter<Rel, Rel> for Deep {
    fn try_modify<E>(
        &self,
        s: &mut Rel,
        f: &mut dyn FnMut(&mut Rel) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut stack = vec![s];
        while let Some(r) = stack.pop() {
            f(r)?;
            // A subquery's relations are walked now (subqueries nest
            // shallowly), its inputs later, off the stack.
            Kind.then(Exprs)
                .then(subquery_rels())
                .try_modify(r, &mut |sub| self.try_modify(sub, f))?;
            Kind.then(Inputs).each_mut(r, &mut |c| stack.push(c));
        }
        Ok(())
    }
}

// --------------------------------------------- call sites, rewritten

/// `emit::condition_joins` (#174): 15 lines plus two 25-line helpers today.
fn condition_joins(root: &mut Rel) {
    Deep.then(Kind).then(AsJoin).modify(root, |j| {
        let typed = !matches!(j.r#type(), JoinType::Inner | JoinType::Unspecified);
        if typed && j.expression.is_none() {
            j.expression = Some(Box::new(lit_true()));
        }
    })
}

/// `forward::mentions_reference`.
fn mentions_reference(rel: &Rel) -> bool {
    Deep.then(Kind)
        .any(rel, |k| matches!(k, RelType::Reference(_)))
}

/// `expr::fields_of` over `walk_fields`' 110 lines: every field a
/// relation's expressions read.
fn fields_of(e: &Expression) -> Vec<&FieldReference> {
    Cosmos.then(Rex).then(AsField).to_vec(e)
}

/// `expr::map_fields`: shift every direct field reference.
fn shift_fields(e: &mut Expression, by: i32) {
    Cosmos.then(Rex).then(AsField).modify(e, |r| {
        if let Some(ReferenceType::DirectReference(seg)) = &mut r.reference_type {
            if let Some(proto::expression::reference_segment::ReferenceType::StructField(s)) =
                &mut seg.reference_type
            {
                s.field += by;
            }
        }
    })
}

/// `share::reads_placeholder` / `compose::substitute`'s read lookups.
fn table_reads(rel: &Rel) -> Vec<String> {
    let mut out = Vec::new();
    Deep.then(Kind).each(rel, &mut |k| {
        if let RelType::Read(r) = k {
            if let Some(ReadType::NamedTable(t)) = &r.read_type {
                out.push(t.names.join("."));
            }
        }
    });
    out
}

fn lit_true() -> Expression {
    Expression {
        rex_type: Some(RexType::Literal(proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::Boolean(true)),
            ..Default::default()
        })),
    }
}

// --------------------------------------------- check against DataFusion

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    use datafusion::prelude::*;
    let ctx = SessionContext::new();
    ctx.sql("CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, -2.0)")
        .await?
        .collect()
        .await?;
    ctx.sql("CREATE TABLE b (i BIGINT, val DOUBLE) AS VALUES (0, 0.25), (1, 3.0)")
        .await?
        .collect()
        .await?;
    for sql in [
        // #171: an uncorrelated EXISTS, a semi join with no condition.
        "SELECT i, val FROM w WHERE EXISTS (SELECT 1 FROM b WHERE b.val > 0)",
        "SELECT i FROM w WHERE i NOT IN (SELECT i FROM b WHERE val > 100)",
        "SELECT SUM(w.val * b.val) FROM w JOIN b ON w.i = b.i",
        "SELECT i, (SELECT MAX(val) FROM b) + val AS v FROM w",
    ] {
        let df = ctx.sql(sql).await?;
        let plan = df.into_optimized_plan()?;
        let mut sp =
            datafusion_substrait::logical_plan::producer::to_substrait_plan(&plan, &ctx.state())?;
        let Some(proto::plan_rel::RelType::Root(root)) = sp.relations[0].rel_type.as_mut() else {
            panic!()
        };
        let rel = root.input.as_mut().unwrap();
        println!("\n{sql}");
        println!(
            "  relations: tree {}, deep {}",
            Tree.to_vec(rel).len(),
            Deep.to_vec(rel).len()
        );
        println!(
            "  joins without a condition: {}",
            Deep.then(Kind)
                .then(AsJoin)
                .to_vec(rel)
                .iter()
                .filter(|j| j.expression.is_none())
                .count()
        );
        let fields: usize = Deep
            .then(Kind)
            .then(Exprs)
            .to_vec(rel)
            .into_iter()
            .map(|e| fields_of(e).len())
            .sum();
        println!(
            "  field references: {fields}; tables read: {:?}; mentions a reference: {}",
            table_reads(rel),
            mentions_reference(rel)
        );
        let consumed =
            datafusion_substrait::logical_plan::consumer::from_substrait_plan(&ctx.state(), &sp)
                .await;
        println!(
            "  consumed as produced: {}",
            consumed
                .as_ref()
                .map(|_| "ok".to_string())
                .unwrap_or_else(|e| e.to_string().lines().next().unwrap().to_string())
        );
        let Some(proto::plan_rel::RelType::Root(root)) = sp.relations[0].rel_type.as_mut() else {
            panic!()
        };
        condition_joins(root.input.as_mut().unwrap());
        let lp =
            datafusion_substrait::logical_plan::consumer::from_substrait_plan(&ctx.state(), &sp)
                .await?;
        let rows = ctx.execute_logical_plan(lp).await?.collect().await?;
        println!(
            "  after condition_joins: consumed and ran, {} rows",
            rows.iter().map(|b| b.num_rows()).sum::<usize>()
        );
    }
    // A setter round trip: shift then unshift every field, unchanged.
    let e = Expression {
        rex_type: Some(RexType::Selection(Box::new(FieldReference {
            reference_type: Some(ReferenceType::DirectReference(
                proto::expression::ReferenceSegment {
                    reference_type: Some(
                        proto::expression::reference_segment::ReferenceType::StructField(Box::new(
                            proto::expression::reference_segment::StructField {
                                field: 3,
                                child: None,
                            },
                        )),
                    ),
                },
            )),
            root_type: None,
        }))),
    };
    let mut x = e.clone();
    shift_fields(&mut x, 5);
    assert_ne!(x, e);
    shift_fields(&mut x, -5);
    assert_eq!(x, e);
    println!("\nsetter round trip ok");
    Ok(())
}
