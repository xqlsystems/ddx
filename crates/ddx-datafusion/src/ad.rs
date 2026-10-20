// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Query-level AD on DataFusion.
//!
//! The simplest way in is SQL itself. [`sql`] runs a statement in which,
//! in a `SELECT` over a table, `grad(f, column)` is the gradient at each row
//! of the number a CTE computes (its objective: a loss, a likelihood, an
//! energy, …), so a training step is one expression:
//!
//! ```sql
//! WITH loss AS (SELECT SUM(val * val) AS l FROM w)
//! SELECT i, val - 0.1 * grad(loss, val) AS val FROM w
//! ```
//!
//! `vjp(f, column, cotangent)` is the same for any CTE `f`, pulled back along
//! `cotangent`, a relation shaped like `f`'s output; and `jvp(e,
//! table.column, tangent)`, a column of any `SELECT`, is the tangent of its
//! expression `e` along `tangent`, a relation shaped like the table:
//!
//! ```sql
//! WITH v AS (SELECT i, 1.0 AS val FROM w)
//! SELECT SUM(val * val) AS l, jvp(SUM(val * val), w.val, v) AS dl FROM w
//! ```
//!
//! Underneath, [`grad`] turns a query whose result is one number into a
//! [`BackwardProgram`], and [`run`] runs it and returns what it computed as
//! DataFrames ([`Outputs`]): the number, and each `wrt` table's gradient,
//! shaped like the table. [`vjp`] does the same for a query with any output,
//! pulling back a cotangent [`run`] takes as the program's input
//! ([`BackwardProgram::inputs`]). [`jvp`] pushes a tangent forward instead
//! ([`ForwardProgram`]); both kinds of program share one vocabulary of input and
//! output tables ([`InputTable`], [`OutputTable`]). Each of the three takes SQL,
//! a plan, or a program ([`Differentiable`]), so they compose: `jvp` of a
//! `grad` program gives Hessian-vector products.
//!
//! While a program runs, every table it writes is named with a prefix unique
//! to that program, `__ddx_{id}_`, so two programs on one context never read
//! each other's tables, and a user's table is never replaced unless its name
//! starts with `__ddx_`, which is reserved. None of them stays on the context
//! once [`run`] returns.
//!
//! ```
//! # use datafusion::prelude::SessionContext;
//! # #[tokio::main]
//! # async fn main() -> datafusion::error::Result<()> {
//! use ddx_datafusion::ad::{self, ColumnRef};
//!
//! let ctx = SessionContext::new();
//! ctx.sql("CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)")
//!     .await?
//!     .collect()
//!     .await?;
//!
//! // In SQL: d/dval of Σ val² is 2·val.
//! let df = ad::sql(
//!     &ctx,
//!     "WITH loss AS (SELECT SUM(val * val) AS l FROM w) \
//!      SELECT i, grad(loss, val) AS val FROM w ORDER BY i",
//! )
//! .await?;
//! # let _ = df.collect().await?;
//!
//! // The same, as a program.
//! let loss = "SELECT SUM(val * val) AS loss FROM w";
//! let program = ad::grad(&ctx, loss, &[ColumnRef::new("w", "val")]).await?;
//! let out = ad::run(&ctx, &program, &[]).await?;
//! out.gradients[0].clone().show().await?; // i, val: 2·val
//!
//! // And Hessian-vector products: jvp of that program, along v = 1.
//! let hvp = ad::jvp(&ctx, &program, &[ColumnRef::new("w", "val")]).await?;
//! let v = ctx.sql("SELECT i, 1.0 AS val FROM w").await?;
//! let out = ad::run(&ctx, &hvp, &[v]).await?;
//! out.gradients[0].clone().show().await?; // i, val, val_tangent: H·v = 2
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use datafusion::common::TableReference;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::prelude::{ident, DataFrame, SessionContext};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;
use ddx_ad::substrait::proto::Plan;
use ddx_ad::Programs;

pub use ddx_ad::{
    AdError, BackwardProgram, Check, ColumnRef, ForwardProgram, InputTable, Of, OutputTable,
    Program, Step, Tangent,
};

use crate::engine::{execute, from_run_error, register, release, to_df_err, DataFusion};

/// The gradient of `of` with respect to the `wrt` columns (see
/// [`ddx_ad::grad`]): of SQL text or a plan, the number the query computes (a
/// loss, a likelihood, an energy, …), which must be one row and one column;
/// of a program, its one output table, which must be too.
///
/// SQL is planned and optimized by `ctx`, and a plan converted to Substrait,
/// before [`ddx_ad::grad`] differentiates it. A refusal arrives as
/// [`DataFusionError::External`] boxing an [`AdError`].
pub async fn grad<D: Differentiable + ?Sized>(
    ctx: &SessionContext,
    of: &D,
    wrt: &[ColumnRef],
) -> Result<BackwardProgram> {
    let of = of.subject(ctx).await?;
    ddx_ad::grad(of.get(), wrt).map_err(to_df_err)
}

/// The vector-Jacobian product of `of` with respect to the `wrt` columns
/// (see [`ddx_ad::vjp`]). Before the program runs, register the cotangent as
/// its input table ([`BackwardProgram::inputs`]), with the columns it lists.
pub async fn vjp<D: Differentiable + ?Sized>(
    ctx: &SessionContext,
    of: &D,
    wrt: &[ColumnRef],
) -> Result<BackwardProgram> {
    let of = of.subject(ctx).await?;
    ddx_ad::vjp(of.get(), wrt).map_err(to_df_err)
}

/// The Jacobian-vector product of `of` with respect to the `wrt` columns
/// (see [`ddx_ad::jvp`]): of SQL text or a plan, a program computing the
/// query's output and, beside it, its tangent; of a program, its steps
/// rewritten, so that of a [`grad`] program each gradient's tangent is the
/// Hessian-vector product `H·v`. There is no `hvp`: it is
/// `jvp(ctx, &grad(ctx, sql, wrt).await?, wrt)`, and one `grad` program
/// serves every direction. Before running the result, register each `wrt`
/// table's tangent as the input table [`ForwardProgram::inputs`] gives for
/// it, with the columns it lists.
pub async fn jvp<D: Differentiable + ?Sized>(
    ctx: &SessionContext,
    of: &D,
    wrt: &[ColumnRef],
) -> Result<ForwardProgram> {
    let of = of.subject(ctx).await?;
    ddx_ad::jvp(of.get(), wrt).map_err(to_df_err)
}

type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

mod sealed {
    /// What a [`Differentiable`](super::Differentiable) is to ddx-ad.
    pub trait Sealed {
        fn subject<'a>(
            &'a self,
            ctx: &'a super::SessionContext,
        ) -> super::BoxFuture<'a, super::Result<Subject<'a>>>;
    }

    /// What ddx-ad differentiates: a plan converted here, or a program as it is.
    pub enum Subject<'a> {
        Plan(Box<super::Plan>),
        Program(&'a (dyn ddx_ad::Differentiable + Sync)),
    }

    impl Subject<'_> {
        pub(super) fn get(&self) -> &dyn ddx_ad::Differentiable {
            match self {
                Subject::Plan(p) => &**p,
                Subject::Program(p) => *p,
            }
        }
    }
}

/// What [`grad`], [`vjp`] and [`jvp`] differentiate: SQL text (a `str` or a
/// `String`), a DataFusion
/// plan (from the DataFrame API, say), or a program of either kind, so they
/// compose: [`jvp`] of a [`grad`] program is forward over reverse, `H·v`;
/// [`vjp`] of a [`jvp`] program is reverse over forward (see
/// [`ddx_ad::Differentiable`]).
///
/// Sealed: these are the things ddx differentiates here.
pub trait Differentiable: sealed::Sealed + Sync {}

impl sealed::Sealed for str {
    fn subject<'a>(
        &'a self,
        ctx: &'a SessionContext,
    ) -> BoxFuture<'a, Result<sealed::Subject<'a>>> {
        Box::pin(async move {
            let lp = ctx.sql(self).await?.into_optimized_plan()?;
            Ok(sealed::Subject::Plan(substrait_of(ctx, &lp).await?))
        })
    }
}
impl Differentiable for str {}

impl sealed::Sealed for String {
    fn subject<'a>(
        &'a self,
        ctx: &'a SessionContext,
    ) -> BoxFuture<'a, Result<sealed::Subject<'a>>> {
        self.as_str().subject(ctx)
    }
}
impl Differentiable for String {}

impl sealed::Sealed for LogicalPlan {
    fn subject<'a>(
        &'a self,
        ctx: &'a SessionContext,
    ) -> BoxFuture<'a, Result<sealed::Subject<'a>>> {
        Box::pin(async move { Ok(sealed::Subject::Plan(substrait_of(ctx, self).await?)) })
    }
}
impl Differentiable for LogicalPlan {}

impl sealed::Sealed for BackwardProgram {
    fn subject<'a>(
        &'a self,
        _ctx: &'a SessionContext,
    ) -> BoxFuture<'a, Result<sealed::Subject<'a>>> {
        Box::pin(async move { Ok(sealed::Subject::Program(self)) })
    }
}
impl Differentiable for BackwardProgram {}

impl sealed::Sealed for ForwardProgram {
    fn subject<'a>(
        &'a self,
        _ctx: &'a SessionContext,
    ) -> BoxFuture<'a, Result<sealed::Subject<'a>>> {
        Box::pin(async move { Ok(sealed::Subject::Program(self)) })
    }
}
impl Differentiable for ForwardProgram {}

/// `plan` as Substrait, refused if the conversion would lose its meaning.
pub(crate) async fn substrait_of(ctx: &SessionContext, plan: &LogicalPlan) -> Result<Box<Plan>> {
    refuse_what_substrait_loses(plan)?;
    refuse_replaced_tables(ctx, plan).await?;
    to_substrait_plan(plan, &ctx.state())
}

/// The first table `plan` reads that is no longer the one registered under
/// its name, if any. A view keeps the tables it was made over: after one is
/// registered again, reading the view reads the old one, while a Substrait
/// plan names the table and its consumer reads the new one (ddx issue
/// #172). A catalog that makes a new table each time it is asked cannot be
/// told apart from one whose table was replaced, so a table is taken as
/// replaced only if asking twice gives the same table, and not the one
/// read.
pub(crate) async fn replaced_table(
    ctx: &SessionContext,
    plan: &LogicalPlan,
) -> Result<Option<TableReference>> {
    use datafusion::common::tree_node::TreeNodeRecursion;
    use datafusion::datasource::source_as_provider;
    let mut scans = Vec::new();
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            scans.push((scan.table_name.clone(), Arc::clone(&scan.source)));
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    let same = |a: &Arc<dyn datafusion::catalog::TableProvider>,
                b: &Arc<dyn datafusion::catalog::TableProvider>| {
        std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
    };
    for (name, source) in scans {
        // A source that is not a registered table's: nothing to compare.
        let Ok(read) = source_as_provider(&source) else {
            continue;
        };
        let Ok(now) = ctx.table_provider(name.clone()).await else {
            return Ok(Some(name));
        };
        if same(&now, &read) {
            continue;
        }
        let again = ctx.table_provider(name.clone()).await?;
        if same(&now, &again) {
            return Ok(Some(name));
        }
    }
    Ok(None)
}

/// Refuse a plan that reads a table no longer registered under its name
/// (see [`replaced_table`]): ddx would differentiate the new one.
pub(crate) async fn refuse_replaced_tables(ctx: &SessionContext, plan: &LogicalPlan) -> Result<()> {
    match replaced_table(ctx, plan).await? {
        Some(name) => Err(to_df_err(AdError::NotImplemented(format!(
            "the query reads a view made over a table `{name}` that has since been \
             registered again: DataFusion's view still reads the old table, while ddx \
             would read the new one (ddx issue #172). Create the view again after \
             registering what it reads"
        )))),
        None => Ok(()),
    }
}

/// Refuse a plan whose meaning DataFusion's Substrait producer does not
/// keep, so ddx would differentiate a different query, silently.
///
/// One case, upstream bug #104: `x NOT IN (subquery)` keeps no row when the
/// subquery returns a NULL. DataFusion plans it as a null-aware anti-join
/// (as an `IN` subquery expression before optimization), and its producer
/// writes a plain anti-join, which keeps those rows. ddx-ad cannot tell from
/// the Substrait plan, which reads as `NOT EXISTS`; only this plan can. As
/// DataFusion does, a `NOT IN` whose two sides cannot be NULL is let through.
fn refuse_what_substrait_loses(plan: &LogicalPlan) -> Result<()> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::logical_expr::ExprSchemable;
    let mut found = false;
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::Join(j) = node {
            found |= j.null_aware;
        }
        let schema = node.inputs().first().map_or(node.schema(), |i| i.schema());
        node.apply_expressions(|e| {
            e.apply(|x| {
                if let Expr::InSubquery(sq) = x {
                    if sq.negated {
                        let needle = sq.expr.nullable(schema).unwrap_or(true);
                        let haystack = sq
                            .subquery
                            .subquery
                            .schema()
                            .fields()
                            .first()
                            .is_none_or(|f| f.is_nullable());
                        found |= needle || haystack;
                    }
                }
                Ok(TreeNodeRecursion::Continue)
            })
        })?;
        Ok(if found {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    })?;
    if found {
        return Err(to_df_err(AdError::NotImplemented(
            "`NOT IN` over a subquery whose values may be NULL: DataFusion's Substrait \
             producer writes it as a plain anti-join, which keeps the rows a NULL in the \
             subquery should exclude (ddx issue #104). Write `NOT EXISTS`, or filter the \
             NULLs out of the subquery"
                .into(),
        )));
    }
    Ok(())
}

/// Run the SQL statement `sql`, in which, as columns of a `SELECT` (see
/// [`ddx_ad::rewrite`], which this runs on DataFusion):
///
/// - `grad(f, column)`, in a `SELECT` over the column's table, is the
///   gradient at each row of the number the CTE `f` computes;
/// - `vjp(f, column, cotangent)` is the same for any `f`, pulled back along
///   `cotangent`, a CTE or table shaped like `f`'s output;
/// - `jvp(e, table.column, …, tangent)`, a column of the statement's
///   `SELECT` or a CTE's, is the tangent of its expression `e` at each row,
///   along `tangent`, a CTE or table shaped like `table`, or a number, the
///   same at every row: `jvp(sin(x), t.x, 1)` is `cos(x)`.
///
/// Calls nest: `f`, `e`'s `SELECT`, or the cotangent or tangent may read
/// another call, and the outer call differentiates through it, so `jvp` of
/// a column holding `grad(loss, val)` along `v` is the Hessian-vector
/// product `H·v`.
///
/// Each program runs first, once, however many calls use it (one per
/// objective for `grad`, per objective and cotangent for `vjp`, per
/// `SELECT` and tangent for `jvp`), and each call is replaced by a read of
/// what it needs before the statement is planned. A statement with no such
/// call is planned as it is. The programs' tables and the cotangents and
/// tangents are dropped once the statement is planned; the returned
/// DataFrame keeps what it reads.
///
/// Each program is built once per context and run again by later calls,
/// as a training loop makes them ([`ddx_ad::Programs`]): it depends on the
/// schemas of the tables its query reads and on the views' definitions,
/// not on the tables' values, and is built again when one of those
/// changes. One caveat: DataFusion's optimizer folds `now()` into a literal,
/// so a program whose query reads it keeps the time of the call that built
/// it.
///
/// An objective defined in a `WITH RECURSIVE` clause is refused, so a
/// training loop cannot yet be written as one recursive statement: each
/// iteration would need its own program. This is ddx's own limit, separate
/// from DataFusion 54's, which plans a recursive CTE wrongly unless the outer
/// query reads every one of its columns.
pub async fn sql(ctx: &SessionContext, sql: &str) -> Result<DataFrame> {
    let mut frames = sql_all(ctx, &[sql]).await?;
    Ok(frames.pop().expect("one statement in, one frame out"))
}

/// [`sql`] for several statements at once. Statements whose calls need the
/// same program share one run of it, so a training step can update each
/// parameter table with its own statement and pay for the backward pass
/// once.
pub async fn sql_all(ctx: &SessionContext, statements: &[&str]) -> Result<Vec<DataFrame>> {
    // Which programs to run, how each is built and run, and how each
    // statement reads them are ddx_ad::rewrite's; DataFusion plans and runs.
    let mut engine = DataFusion(ctx.clone());
    let programs = programs(ctx);
    let out = ddx_ad::rewrite(&mut engine, &programs, statements, &GenericDialect {})
        .await
        .map_err(from_run_error)?;
    let frames = plan_each(ctx, &out.statements).await;
    // A planned DataFrame holds the tables it reads, so they can leave the
    // catalog now.
    out.release(&mut engine, &programs).await?;
    frames
}

/// The programs [`sql_all`] keeps on `ctx` (and every context that shares
/// its state), put there on first use.
fn programs(ctx: &SessionContext) -> Arc<Programs> {
    let state = ctx.state_ref();
    if let Some(p) = state.read().config().get_extension::<Programs>() {
        return p;
    }
    let mut state = state.write();
    // Another call may have put them there since the read.
    if let Some(p) = state.config().get_extension::<Programs>() {
        return p;
    }
    let p = Arc::new(Programs::new());
    state.config_mut().set_extension(Arc::clone(&p));
    p
}

/// Each of `statements`, planned.
async fn plan_each(ctx: &SessionContext, statements: &[String]) -> Result<Vec<DataFrame>> {
    let mut frames = Vec::with_capacity(statements.len());
    for sql in statements {
        frames.push(ctx.sql(sql).await?);
    }
    Ok(frames)
}

/// What a program computed, as DataFrames that hold their rows: nothing it
/// wrote stays on the context.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Outputs {
    /// The query's output: its columns, then, for a `jvp` program, each
    /// column's tangent as `{column}_tangent`, as `jvp(…)` in SQL returns
    /// them ([`OutputTable::query`]).
    pub value: DataFrame,
    /// One per `wrt` table, in the program's order
    /// ([`Program::gradients`]): the table's dims, then each value's
    /// gradient under the value's name, and for `jvp` of a program with
    /// gradients each one's tangent, named as in `value`.
    pub gradients: Vec<DataFrame>,
}

/// Run `program` on `inputs`, one per [`Program::inputs`] in order (a
/// `vjp`'s cotangent, a `jvp`'s tangents; none for `grad`), each with at
/// least the columns its table names, and return what it computed.
///
/// First its checks: that each `wrt` table's dims identify its rows, that
/// each input's keys do. A row back from one is [`AdError::InvalidWrt`],
/// and nothing runs. Then its steps, each materialized in turn. Nothing it
/// writes stays on `ctx`, whether it succeeds or fails.
///
/// A program depends on the tables' names and schemas, not their values, so
/// build it once and run it on every training step. One program's runs must
/// not overlap: they write the same tables while they run. Build a program
/// per concurrent caller instead.
///
/// ```no_run
/// # use datafusion::prelude::SessionContext;
/// # use ddx_datafusion::ad::{self, ColumnRef};
/// # async fn hvp(ctx: &SessionContext) -> datafusion::error::Result<()> {
/// let wrt = [ColumnRef::new("w", "val")];
/// let grad = ad::grad(ctx, "SELECT SUM(val * val * val) AS l FROM w", &wrt).await?;
/// let hvp = ad::jvp(ctx, &grad, &wrt).await?;
/// let v = ctx.sql("SELECT i, 1.0 AS val FROM w").await?;
/// let out = ad::run(ctx, &hvp, &[v]).await?;
/// out.gradients[0].clone().show().await?; // i, val, val_tangent: ∇l and H·v
/// # Ok(())
/// # }
/// ```
pub async fn run<P: Program + ?Sized>(
    ctx: &SessionContext,
    program: &P,
    inputs: &[DataFrame],
) -> Result<Outputs> {
    let tables = program.inputs();
    if inputs.len() != tables.len() {
        let names: Vec<&str> = tables.iter().map(|t| t.name.as_str()).collect();
        return Err(DataFusionError::Plan(format!(
            "the program takes {} input tables ({}), and was given {}",
            tables.len(),
            names.join(", "),
            inputs.len()
        )));
    }
    let result = async {
        for (table, df) in tables.iter().zip(inputs) {
            let columns: Vec<Expr> = table.columns.iter().map(ident).collect();
            register(ctx, &table.name, df.clone().select(columns)?).await?;
        }
        execute(ctx, program).await?;
        let mut gradients = Vec::with_capacity(program.gradients().len());
        for g in program.gradients() {
            gradients.push(ctx.sql(&g.query()).await?);
        }
        Ok(Outputs {
            value: ctx.sql(&program.value().query()).await?,
            gradients,
        })
    }
    .await;
    // A planned DataFrame holds the tables it reads, so they can leave the
    // catalog now.
    release(ctx, program)?;
    for table in tables {
        ctx.deregister_table(table.name.as_str())?;
    }
    result
}
