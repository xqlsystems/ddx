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
//! and `jvp(f, table.column, tangent)` is a CTE's output with, beside each
//! column, its tangent along `tangent`, a relation shaped like the table:
//!
//! ```sql
//! WITH loss AS (SELECT SUM(val * val) AS l FROM w),
//!      v AS (SELECT i, 1.0 AS val FROM w)
//! SELECT l, l_tangent FROM jvp(loss, w.val, v)
//! ```
//!
//! Underneath, [`grad`] turns a query whose result is one number into a
//! [`BackwardProgram`], and [`run`] runs it: every step is materialized as a
//! table on the context, the number ends up in [`BackwardProgram::value`], and
//! each `wrt` table's gradient, shaped like the table, in
//! [`BackwardProgram::gradients`]. [`vjp`] does the same for a query with any
//! output, pulling back a cotangent the caller registers as the program's input
//! table ([`BackwardProgram::inputs`]). [`jvp`] pushes a tangent forward instead
//! ([`ForwardProgram`]); both kinds of program share one vocabulary of input and
//! output tables ([`InputTable`], [`OutputTable`]). Each of the three takes SQL,
//! a plan, or a program ([`Differentiable`]), so they compose: `jvp` of a
//! `grad` program gives Hessian-vector products.
//!
//! Every table a program writes is named with a prefix unique to that program,
//! `__ddx_{id}_`, so two programs on one context never read each other's
//! tables, and a user's table is never replaced unless its name starts with
//! `__ddx_`, which is reserved. After [`run`], only the value and the
//! gradients remain on the context; [`release`] drops those too.
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
//! ad::run(&ctx, &program).await?;
//! let grad = &program.gradients[0].step;
//! # let _ = ctx.sql(&format!("SELECT i, val FROM {grad}")).await?.collect().await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::common::{Column, DFSchema, TableReference};
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{FunctionRegistry, SessionState};
use datafusion::logical_expr::{Expr, LogicalPlan, Projection};
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion_substrait::extensions::Extensions;
use datafusion_substrait::logical_plan::consumer::{
    from_project_rel, from_substrait_plan_with_consumer, from_substrait_rel,
    DefaultSubstraitConsumer, SubstraitConsumer,
};
use datafusion_substrait::logical_plan::producer::to_substrait_plan;
use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::{NamedStruct, Rel};
use ddx_ad::substrait::proto::{Plan, ProjectRel};
use ddx_ad::{bind_reads, unbound_reads, Action, RunError};

pub use ddx_ad::{
    AdError, BackwardProgram, Check, ColumnRef, ForwardProgram, InputTable, Of, OutputTable,
    Program, Step, Tangent, Verified,
};

use ddx_ad::Statements;

fn to_df_err(e: AdError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

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

/// Run the SQL statement `sql`, in which, in a `FROM` clause (see
/// [`ddx_ad::sql`]):
///
/// - `grad(f, table.column, …)` is the gradient of the number the CTE `f`
///   computes: a relation shaped like `table`, its dims and the named
///   columns' gradients;
/// - `jvp(f, table.column, …, tangent, …)` is the CTE `f`'s output and,
///   beside each column that has one, its tangent along the tangents named
///   (a CTE or a table per `wrt` table), as `{column}_tangent`.
///
/// Each objective's [`grad`] program runs first, once, however many calls use it,
/// and each distinct `jvp` call's program, and each call is replaced by a
/// query over what it needs before the statement is planned. A statement
/// with no such call is planned as it is. The programs' tables are dropped
/// once the statement is planned; the returned DataFrame keeps what it reads.
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
    let mut ran_jvps: Vec<ForwardProgram> = Vec::with_capacity(planned.jvp_jobs().len());
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
            refuse_what_substrait_loses(&lp)?;
            let program =
                ddx_ad::grad_with(&*to_substrait_plan(&lp, &ctx.state())?, &job.wrt, &options)
                    .map_err(to_df_err)?;
            ran.push(program);
            run(ctx, ran.last().expect("just pushed")).await?;
        }
        for job in planned.jvp_jobs() {
            ran_jvps.push(jvp(ctx, job.query.as_str(), &job.wrt).await?);
            let program = ran_jvps.last().expect("just pushed");
            register_tangents(ctx, program, job).await?;
            run(ctx, program).await?;
        }
        let rewritten = planned
            .rewrite(
                &ran.iter().collect::<Vec<_>>(),
                &ran_jvps.iter().collect::<Vec<_>>(),
            )
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
    for program in &ran_jvps {
        release(ctx, program)?;
        for input in &program.inputs {
            ctx.deregister_table(input.name.as_str())?;
        }
    }
    result
}

/// Register each of `program`'s tangent tables from the tangent `job` names
/// for its table: the columns the program asks for, by name, materialized.
async fn register_tangents(
    ctx: &SessionContext,
    program: &ForwardProgram,
    job: &ddx_ad::JvpJob,
) -> Result<()> {
    for input in &program.inputs {
        let names = input.of.table().unwrap_or_default();
        let tangent = job.tangent_of(names).ok_or_else(|| {
            to_df_err(AdError::Internal(format!(
                "no tangent was given for `{}`",
                names.join(".")
            )))
        })?;
        let columns: Vec<String> = input
            .columns
            .iter()
            .map(|c| format!("\"{}\"", c.replace('"', "\"\"")))
            .collect();
        let df = ctx
            .sql(&format!(
                "SELECT {} FROM ({}) AS __ddx_tangent",
                columns.join(", "),
                tangent.query
            ))
            .await?;
        let schema = Arc::new(df.schema().as_arrow().clone());
        let batches = df.collect().await?;
        let table = MemTable::try_new(schema, vec![batches])?;
        ctx.register_table(input.name.as_str(), Arc::new(table))?;
    }
    Ok(())
}

/// Run `program`: its [`checks`](BackwardProgram::checks), then every step,
/// forward then backward, registering each result on `ctx` under the step's
/// name. Once the gradients are written, the intermediate tables (saved
/// aggregates and cotangents) are dropped; the value and the gradients stay
/// until [`release`] or the next run replaces them.
///
/// A failed check arrives as [`DataFusionError::External`] boxing
/// [`AdError::InvalidWrt`]: a `wrt` table's rows are not what the program
/// assumed, for instance two rows share their dims.
///
/// A program depends on the tables' names and schemas, not their values, so
/// build it once and run it on every training step. One program's runs must
/// not overlap: they write the same tables. Build a program per concurrent
/// caller instead. When many runs read tables whose keys do not change, see
/// [`run_verified`].
pub async fn run<P: Program + ?Sized>(ctx: &SessionContext, program: &P) -> Result<()> {
    run_verified(ctx, program, &mut Verified::new()).await
}

/// [`run`], skipping each check whose fact `verified` holds, and recording
/// every check's fact in it once the run has succeeded (see [`Verified`]).
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
/// for _ in 0..10 {
///     // A new direction each iteration, under the same name: forget it, so
///     // its keys are checked again; w's are checked on the first run only.
///     let v = &hvp.inputs[0].name;
///     ctx.sql(&format!("CREATE OR REPLACE TABLE {v} AS SELECT i, 1.0 AS val FROM w"))
///         .await?
///         .collect()
///         .await?;
///     verified.forget(v);
///     ad::run_verified(ctx, &hvp, &mut verified).await?;
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
    verified: &mut Verified,
) -> Result<()> {
    // The order, and what is dropped when, are ddx_ad::Runner's. The checks
    // read no step and none reads another, so they run at once, each a task
    // of its own (polled together on one task, they ran one after another),
    // and the runner is told how each went in its own order: on a large wrt
    // table they are most of a run (the dims of a table, then of its
    // tangent, each a grouped scan). Only those `verified` does not vouch for.
    let mut checked: Vec<Option<Result<bool>>> = Vec::with_capacity(program.checks().len());
    let tasks: Vec<_> = program
        .checks()
        .iter()
        .map(|c| {
            (!verified.knows(c)).then(|| {
                let (ctx, plan) = (ctx.clone(), c.plan.clone());
                tokio::spawn(async move { returns_rows(&ctx, &plan).await })
            })
        })
        .collect();
    for task in tasks {
        checked.push(match task {
            Some(t) => Some(
                t.await
                    .unwrap_or_else(|e| Err(DataFusionError::External(Box::new(e)))),
            ),
            None => None,
        });
    }
    let mut runner = ddx_ad::Runner::verified(program, verified);
    while let Some(action) = runner.next() {
        match &action {
            Action::Check(i) => runner.checked(
                checked[*i]
                    .take()
                    .expect("the runner asks for each check it runs once"),
            ),
            Action::Materialize(i) => runner.done(run_step(ctx, program.step(*i)).await),
            Action::Drop(name) => runner.done(ctx.deregister_table(name.as_str()).map(|_| ())),
        }
    }
    runner.finish().map_err(|e| match e {
        RunError::Refused(e) => to_df_err(e),
        RunError::Engine(e) => e,
        other => DataFusionError::Internal(other.to_string()),
    })?;
    verified.record_all(program);
    Ok(())
}

/// Run `program`'s checks, failing on the first that returns a row.
pub async fn run_checks(ctx: &SessionContext, program: &BackwardProgram) -> Result<()> {
    for check in &program.checks {
        if returns_rows(ctx, &check.plan).await? {
            return Err(to_df_err(AdError::InvalidWrt(check.message.clone())));
        }
    }
    Ok(())
}

/// Does `plan` (a check) return any row? The returns-rows primitive of
/// [`ddx_ad::Backend`].
pub async fn returns_rows(ctx: &SessionContext, plan: &Plan) -> Result<bool> {
    let lp = logical_plan(ctx, plan).await?;
    let rows: usize = ctx
        .execute_logical_plan(lp)
        .await?
        .limit(0, Some(1))?
        .collect()
        .await?
        .iter()
        .map(|b| b.num_rows())
        .sum();
    Ok(rows > 0)
}

/// Drop every table `program` registered on `ctx`, the value and the
/// gradients included.
pub fn release<P: Program + ?Sized>(ctx: &SessionContext, program: &P) -> Result<()> {
    for i in 0..program.step_count() {
        let step = program.step(i);
        ctx.deregister_table(step.name.as_str())?;
    }
    Ok(())
}

/// Run one step and register its result, replacing a table of that name.
/// Every step it reads must already be registered. Unlike [`run`], this
/// neither runs the checks nor drops anything.
pub async fn run_step(ctx: &SessionContext, step: &Step) -> Result<()> {
    materialize(ctx, &step.name, &step.plan).await
}

/// Run `plan` and register its rows as the table `name`, replacing one: the
/// materialize primitive of [`ddx_ad::Backend`].
pub async fn materialize(ctx: &SessionContext, name: &str, plan: &Plan) -> Result<()> {
    let lp = logical_plan(ctx, plan).await?;
    let df = ctx.execute_logical_plan(lp).await?;
    // The schema of the plan that runs, not the logical one: a step read
    // from an unanalyzed plan can be typed before type coercion (a CASE
    // between BIGINT and DOUBLE branches), and the table must declare the
    // types its batches hold.
    let task = df.task_ctx();
    let physical = df.create_physical_plan().await?;
    let schema = physical.schema();
    let batches = datafusion::physical_plan::collect(physical, Arc::new(task)).await?;
    // A MemTable, not a view: DataFusion's Substrait consumer can pick
    // columns out of a table scan, and a later step reads only some.
    let table = MemTable::try_new(schema, vec![batches])?;
    ctx.deregister_table(name)?;
    ctx.register_table(name, Arc::new(table))?;
    Ok(())
}

/// The DataFusion plan [`run`] executes for one of a program's plans (a step
/// or a check): its reads of earlier steps bound to the tables registered on
/// `ctx`, and consumed with each computed column given a short name.
///
/// DataFusion's own Substrait consumer names a computed column by its whole
/// expression, and ddx's plans compute each column from earlier ones, so a
/// layer that reads its input twice (`sin(v) + 0.1 * v`) doubles every name
/// after it: 14 such layers made a 60 MB plan, and 20 exhausted 13 GB. The
/// names mean nothing to ddx, whose plans refer to columns by position.
pub async fn logical_plan(ctx: &SessionContext, plan: &Plan) -> Result<LogicalPlan> {
    let mut plan = plan.clone();
    let mut schemas = HashMap::new();
    for name in unbound_reads(&plan) {
        let schema = table_schema(ctx, &name).await?;
        schemas.insert(name, schema);
    }
    bind_reads(&mut plan, &mut |name| schemas.get(name).cloned()).map_err(to_df_err)?;
    let state = ctx.state();
    let extensions = Extensions::try_from(&plan.extensions)?;
    let consumer = ShortNames {
        inner: DefaultSubstraitConsumer::new(&extensions, &state),
        state: &state,
        next: AtomicUsize::new(0),
    };
    from_substrait_plan_with_consumer(&consumer, &plan).await
}

/// DataFusion's Substrait consumer, but a projection's computed columns are
/// named `__ddx_c{n}` rather than by their expressions (see [`logical_plan`]).
struct ShortNames<'a> {
    inner: DefaultSubstraitConsumer<'a>,
    state: &'a SessionState,
    next: AtomicUsize,
}

#[async_trait]
impl SubstraitConsumer for ShortNames<'_> {
    async fn resolve_table_ref(
        &self,
        table: &TableReference,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        self.inner.resolve_table_ref(table).await
    }

    fn get_extensions(&self) -> &Extensions {
        self.inner.get_extensions()
    }

    fn get_function_registry(&self) -> &impl FunctionRegistry {
        self.state
    }

    fn push_outer_schema(&self, schema: Arc<DFSchema>) {
        self.inner.push_outer_schema(schema)
    }

    fn pop_outer_schema(&self) {
        self.inner.pop_outer_schema()
    }

    fn get_outer_schema(&self, steps_out: usize) -> Option<Arc<DFSchema>> {
        self.inner.get_outer_schema(steps_out)
    }

    /// A relation with an emit (an output mapping): the relation without it,
    /// then the mapped columns picked out. DataFusion's own consumer applies
    /// an emit through its `project` builder, which re-normalizes every
    /// column by walking the whole plan (see `consume_project`), and ddx's
    /// pruned plans put an emit on most projections.
    async fn consume_rel(&self, rel: &Rel) -> Result<LogicalPlan> {
        let Some(mapping) = emit_of(rel) else {
            return from_substrait_rel(self, rel).await;
        };
        let mut seen = std::collections::HashSet::new();
        if !mapping.iter().all(|i| seen.insert(*i)) {
            // A column emitted twice needs DataFusion's unique naming.
            return from_substrait_rel(self, rel).await;
        }
        // The relation without its emit: the typed consumer, which does not
        // read the emit (cloning the relation to strip it would copy the
        // whole subtree beneath, at every emit).
        let plan = match rel.rel_type.as_ref() {
            Some(RelType::Project(p)) => self.consume_project(p).await?,
            Some(RelType::Filter(f)) => self.consume_filter(f).await?,
            Some(RelType::Fetch(f)) => self.consume_fetch(f).await?,
            Some(RelType::Sort(s)) => self.consume_sort(s).await?,
            Some(RelType::Join(j)) => self.consume_join(j).await?,
            Some(RelType::Cross(c)) => self.consume_cross(c).await?,
            Some(RelType::Aggregate(a)) => self.consume_aggregate(a).await?,
            Some(RelType::Set(s)) => self.consume_set(s).await?,
            Some(RelType::Read(r)) => self.consume_read(r).await?,
            _ => return from_substrait_rel(self, rel).await,
        };
        let pick = |i: i32| usize::try_from(i).ok();
        let plan = match plan {
            LogicalPlan::Projection(p) => {
                let exprs = mapping
                    .iter()
                    .map(|&i| pick(i).and_then(|i| p.expr.get(i).cloned()))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        DataFusionError::Internal("an emit past the projection".into())
                    })?;
                LogicalPlan::Projection(Projection::try_new(exprs, p.input)?)
            }
            other => {
                let schema = Arc::clone(other.schema());
                let exprs = mapping
                    .iter()
                    .map(|&i| {
                        pick(i)
                            .filter(|&i| i < schema.fields().len())
                            .map(|i| Expr::Column(Column::from(schema.qualified_field(i))))
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| DataFusionError::Internal("an emit past the relation".into()))?;
                LogicalPlan::Projection(Projection::try_new(exprs, Arc::new(other))?)
            }
        };
        Ok(plan)
    }

    /// The input's columns, then the projection's expressions under short
    /// names. Built directly, not with DataFusion's `project` builder: that
    /// normalizes each expression's columns by walking the whole input plan,
    /// once per expression, and ddx's projections carry every earlier column,
    /// so a chain of n of them cost about n³ (1.2 s to consume an 80-map
    /// chain whose plan is 18 KB). These columns come from the input's schema
    /// already qualified, so there is nothing to normalize.
    async fn consume_project(&self, rel: &ProjectRel) -> Result<LogicalPlan> {
        let Some(input) = rel.input.as_deref() else {
            return self.consume_project_by_builder(rel).await;
        };
        let input = self.consume_rel(input).await?;
        let schema = Arc::clone(input.schema());
        let mut exprs: Vec<Expr> = (0..schema.fields().len())
            .map(|i| Expr::Column(Column::from(schema.qualified_field(i))))
            .collect();
        for e in &rel.expressions {
            let e = self.consume_expression(e, &schema).await?;
            // A window function needs a Window relation beneath the
            // projection, which DataFusion's own consumer builds.
            if matches!(e, Expr::WindowFunction(_)) {
                return self.consume_project_by_builder(rel).await;
            }
            let n = self.next.fetch_add(1, Ordering::Relaxed);
            exprs.push(e.alias(format!("__ddx_c{n}")));
        }
        Ok(LogicalPlan::Projection(Projection::try_new(
            exprs,
            Arc::new(input),
        )?))
    }
}

impl ShortNames<'_> {
    /// DataFusion's own projection consumer, then the computed columns
    /// renamed: for a projection with a window function.
    async fn consume_project_by_builder(&self, rel: &ProjectRel) -> Result<LogicalPlan> {
        let LogicalPlan::Projection(p) = from_project_rel(self, rel).await? else {
            return Err(DataFusionError::Internal(
                "a Substrait projection consumed as something else".into(),
            ));
        };
        let exprs = p
            .expr
            .into_iter()
            .map(|e| match e {
                Expr::Column(_) => e,
                computed => {
                    let n = self.next.fetch_add(1, Ordering::Relaxed);
                    computed.unalias().alias(format!("__ddx_c{n}"))
                }
            })
            .collect();
        Ok(LogicalPlan::Projection(Projection::try_new(
            exprs, p.input,
        )?))
    }
}

/// `rel`'s emit, if it has one.
fn emit_of(rel: &Rel) -> Option<Vec<i32>> {
    use ddx_ad::substrait::proto::rel_common::EmitKind;
    let common = match rel.rel_type.as_ref()? {
        RelType::Project(p) => p.common.as_ref(),
        RelType::Filter(f) => f.common.as_ref(),
        RelType::Fetch(f) => f.common.as_ref(),
        RelType::Sort(s) => s.common.as_ref(),
        RelType::Join(j) => j.common.as_ref(),
        RelType::Cross(c) => c.common.as_ref(),
        RelType::Aggregate(a) => a.common.as_ref(),
        RelType::Set(s) => s.common.as_ref(),
        RelType::Read(r) => r.common.as_ref(),
        _ => None,
    }?;
    match &common.emit_kind {
        Some(EmitKind::Emit(e)) => Some(e.output_mapping.clone()),
        _ => None,
    }
}

/// The Substrait schema of the registered table `name`, as DataFusion's
/// producer states it: the base schema of the read in `SELECT * FROM name`.
pub async fn table_schema(ctx: &SessionContext, name: &str) -> Result<NamedStruct> {
    let lp = ctx.table(name).await?.into_unoptimized_plan();
    let plan = to_substrait_plan(&lp, &ctx.state())?;
    let root = plan.relations.iter().find_map(|r| match &r.rel_type {
        Some(PlanRelType::Root(root)) => root.input.as_ref(),
        _ => None,
    });
    let mut rel: Option<&Rel> = root;
    while let Some(r) = rel {
        match &r.rel_type {
            Some(RelType::Read(read)) => {
                return read.base_schema.clone().ok_or_else(|| {
                    DataFusionError::Internal(format!("the read of `{name}` has no schema"))
                })
            }
            Some(RelType::Project(p)) => rel = p.input.as_deref(),
            Some(RelType::Filter(f)) => rel = f.input.as_deref(),
            _ => break,
        }
    }
    Err(DataFusionError::Internal(format!(
        "no table read found in the Substrait plan of `{name}`"
    )))
}
