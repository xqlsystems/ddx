// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Query-level reverse-mode AD (ddx v2) on DataFusion.
//!
//! The simplest way in is SQL itself. [`sql`] runs a statement in which
//! `grad(f, table.column)` is the gradient of the number a CTE computes (its
//! objective: a loss, a likelihood, an energy, …), as a relation shaped like
//! the table:
//!
//! ```sql
//! WITH loss AS (SELECT SUM(val * val) AS l FROM w)
//! SELECT w.i, w.val - 0.1 * g.val AS val
//! FROM w JOIN grad(loss, w.val) g ON w.i = g.i
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
//!      SELECT i, val FROM grad(loss, w.val) ORDER BY i",
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

use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::prelude::{ident, DataFrame, SessionContext};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;
use ddx_ad::substrait::proto::Plan;

pub use ddx_ad::{
    AdError, BackwardProgram, Check, ColumnRef, ForwardProgram, InputTable, Of, OutputTable,
    Program, Step, Tangent, Verified,
};

use crate::engine::{execute, register, release, to_df_err};
use ddx_ad::Statements;

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
            Ok(sealed::Subject::Plan(substrait_of(ctx, &lp)?))
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
        Box::pin(async move { Ok(sealed::Subject::Plan(substrait_of(ctx, self)?)) })
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
fn substrait_of(ctx: &SessionContext, plan: &LogicalPlan) -> Result<Box<Plan>> {
    refuse_what_substrait_loses(plan)?;
    to_substrait_plan(plan, &ctx.state())
}

/// DataFusion's plan of `SELECT * FROM table WHERE predicate`, for
/// [`ddx_ad::Options::restrict`]; `None` if it does not plan, and then every
/// row of the gradient is computed, which is always right.
async fn restriction_plan(ctx: &SessionContext, table: &str, predicate: &str) -> Option<Plan> {
    let sql = format!("SELECT * FROM {table} WHERE {predicate}");
    let lp = ctx.sql(&sql).await.ok()?.into_unoptimized_plan();
    to_substrait_plan(&lp, &ctx.state()).ok().map(|p| *p)
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

/// Run the SQL statement `sql`, in which `grad(f, table.column, …)` in a
/// `FROM` clause is the gradient of the number the CTE `f` computes (see
/// [`ddx_ad::sql`]): a relation shaped like `table`, its dims and the named
/// columns' gradients.
///
/// Each objective's [`grad`] program runs first, once, however many calls use it,
/// and each call is replaced by a query over the gradient it needs before the
/// statement is planned. A statement with no such call is planned as it is.
/// The programs' tables are dropped once the statement is planned; the
/// returned DataFrame keeps what it reads.
///
/// An objective defined in a `WITH RECURSIVE` clause is refused, so a training loop
/// cannot yet be written as one recursive statement (design.md §5). This is
/// ddx's own limit, separate from DataFusion 54's recursive-CTE planning bug
/// (design.md §3.6).
pub async fn sql(ctx: &SessionContext, sql: &str) -> Result<DataFrame> {
    let mut frames = sql_all(ctx, &[sql]).await?;
    Ok(frames.pop().expect("one statement in, one frame out"))
}

/// [`sql`] for several statements at once. Statements whose `grad` calls
/// differentiate the same objective share one run of its program, so a
/// training step can update each parameter table with its own statement and
/// pay for the backward pass once.
pub async fn sql_all(ctx: &SessionContext, statements: &[&str]) -> Result<Vec<DataFrame>> {
    // Which programs to run, and how each statement reads their gradients,
    // are ddx_ad::Statements'; running them and planning the result are
    // DataFusion's.
    let planned = Statements::plan(statements, &GenericDialect {}).map_err(to_df_err)?;
    let mut ran: Vec<BackwardProgram> = Vec::with_capacity(planned.jobs().len());
    let result = async {
        for job in planned.jobs() {
            // Only the gradient rows the statements read, where they say.
            let mut options = ddx_ad::Options::new();
            for (table, predicate) in &job.restrict {
                if let Some(select) = restriction_plan(ctx, table, predicate).await {
                    options = options.restrict(table.clone(), select);
                }
            }
            let lp = ctx.sql(&job.query).await?.into_optimized_plan()?;
            let plan = substrait_of(ctx, &lp)?;
            let program = ddx_ad::grad_with(&*plan, &job.wrt, &options).map_err(to_df_err)?;
            ran.push(program);
            execute(ctx, ran.last().expect("just pushed"), &mut Verified::new()).await?;
        }
        let rewritten = planned
            .rewrite(&ran.iter().collect::<Vec<_>>())
            .map_err(to_df_err)?;
        let mut frames = Vec::with_capacity(rewritten.len());
        for sql in &rewritten {
            frames.push(ctx.sql(sql).await?);
        }
        Ok(frames)
    }
    .await;
    // A planned DataFrame holds the tables it reads, so the programs' tables
    // can leave the catalog now.
    for program in &ran {
        release(ctx, program)?;
    }
    result
}

/// What a program computed, as DataFrames that hold their rows: nothing it
/// wrote stays on the context.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Outputs {
    /// The query's output: its columns, then, for a `jvp` program, each
    /// column's tangent as `{column}_tangent` ([`OutputTable::query`]).
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
/// per concurrent caller instead. When many runs read `wrt` tables whose
/// keys do not change, see [`run_verified`].
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
    run_verified(ctx, program, inputs, &mut Verified::new()).await
}

/// [`run`], skipping each check of a `wrt` table whose fact `verified`
/// holds, and recording every check's fact in it once the run has
/// succeeded (see [`Verified`]). The inputs are new on every run, so their
/// checks always run.
///
/// A check is a grouped scan of its table on every run; on a large `wrt`
/// table, the checks are most of a `jvp`'s run. Keep one [`Verified`] across
/// the runs that read the same tables, and [`Verified::forget`] a table
/// whenever its keys may change:
///
/// ```no_run
/// # use datafusion::prelude::SessionContext;
/// # use ddx_datafusion::ad::{self, ColumnRef, Verified};
/// # async fn cg(ctx: &SessionContext) -> datafusion::error::Result<()> {
/// let wrt = [ColumnRef::new("w", "val")];
/// let grad = ad::grad(ctx, "SELECT SUM(val * val) AS l FROM w", &wrt).await?;
/// let hvp = ad::jvp(ctx, &grad, &wrt).await?;
/// let mut verified = Verified::new();
/// for k in 0..10 {
///     // A new direction each iteration; w's keys are checked on the first
///     // run only.
///     let v = ctx.sql(&format!("SELECT i, {k}.0 AS val FROM w")).await?;
///     ad::run_verified(ctx, &hvp, &[v], &mut verified).await?;
/// }
/// # Ok(())
/// # }
/// ```
///
/// Keeping a fact for a table whose keys have changed is the caller's
/// promise broken: a program its check would refuse runs, and rows that
/// share keys get their summed gradient.
pub async fn run_verified<P: Program + ?Sized>(
    ctx: &SessionContext,
    program: &P,
    inputs: &[DataFrame],
    verified: &mut Verified,
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
        execute(ctx, program, verified).await?;
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
