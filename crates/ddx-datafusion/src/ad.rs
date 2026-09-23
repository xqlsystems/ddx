// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Query-level reverse-mode AD (ddx v2) on DataFusion.
//!
//! The simplest way in is SQL itself. [`sql`] runs a statement in which
//! `grad(loss, table.column)` is the gradient of the loss a CTE computes, as a
//! relation shaped like the table:
//!
//! ```sql
//! WITH loss AS (SELECT SUM(val * val) AS l FROM w)
//! SELECT w.i, w.val - 0.1 * g.val AS val
//! FROM w JOIN grad(loss, w.val) g ON w.i = g.i
//! ```
//!
//! Underneath, [`grad`] turns a query whose result is a loss into a
//! [`BackwardProgram`], and [`run`] runs it: every step is materialized as a
//! table on the context, the loss ends up in [`BackwardProgram::value`], and
//! each `wrt` table's gradient, shaped like the table, in the table
//! [`BackwardProgram::gradients`] names. [`vjp`] does the same for a query with
//! any output, pulling back a cotangent the caller registers as [`COTANGENT`].
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
use std::sync::Arc;

use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion_substrait::logical_plan::consumer::from_substrait_plan;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;
use ddx_ad::emit::{bind_reads, unbound_reads};
use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::{NamedStruct, Rel};

pub use ddx_ad::{AdError, BackwardProgram, ColumnRef, Gradient, Step, COTANGENT, VALUE};

use ddx_ad::GradCalls;

fn to_df_err(e: AdError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// The gradient of the loss the SQL query `sql` computes, with respect to the
/// `wrt` columns. The query must return one row and one column.
///
/// The query is planned and optimized by `ctx`, converted to Substrait, and
/// handed to [`ddx_ad::grad`]. A refusal arrives as
/// [`DataFusionError::External`] boxing an [`AdError`].
pub async fn grad(ctx: &SessionContext, sql: &str, wrt: &[ColumnRef]) -> Result<BackwardProgram> {
    grad_plan(ctx, &ctx.sql(sql).await?.into_optimized_plan()?, wrt)
}

/// [`grad`] for a plan already built, for instance with the DataFrame API.
pub fn grad_plan(
    ctx: &SessionContext,
    plan: &LogicalPlan,
    wrt: &[ColumnRef],
) -> Result<BackwardProgram> {
    ddx_ad::grad(&*to_substrait_plan(plan, &ctx.state())?, wrt).map_err(to_df_err)
}

/// The vector-Jacobian product of the SQL query `sql` with respect to the
/// `wrt` columns. Before the program's backward steps run, register the
/// cotangent as the table [`COTANGENT`], with the columns
/// [`BackwardProgram::cotangent`] lists.
pub async fn vjp(ctx: &SessionContext, sql: &str, wrt: &[ColumnRef]) -> Result<BackwardProgram> {
    vjp_plan(ctx, &ctx.sql(sql).await?.into_optimized_plan()?, wrt)
}

/// [`vjp`] for a plan already built.
pub fn vjp_plan(
    ctx: &SessionContext,
    plan: &LogicalPlan,
    wrt: &[ColumnRef],
) -> Result<BackwardProgram> {
    ddx_ad::vjp(&*to_substrait_plan(plan, &ctx.state())?, wrt).map_err(to_df_err)
}

/// Run the SQL statement `sql`, in which `grad(loss, table.column, …)` in a
/// `FROM` clause is the gradient of the loss the CTE `loss` computes: a
/// relation shaped like `table`, its dims and the named columns' gradients.
///
/// Each loss's [`grad`] program runs first, once, however many calls use it,
/// and each call is replaced by a query over the gradient it needs before the
/// statement is planned. A statement with no such call is planned as it is.
pub async fn sql(ctx: &SessionContext, sql: &str) -> Result<DataFrame> {
    let mut frames = sql_all(ctx, &[sql]).await?;
    Ok(frames.pop().expect("one statement in, one frame out"))
}

/// [`sql`] for several statements at once. Statements whose `grad` calls
/// differentiate the same loss query share one run of its program, so a
/// training step can update each parameter table with its own statement and
/// pay for the backward pass once.
pub async fn sql_all(ctx: &SessionContext, statements: &[&str]) -> Result<Vec<DataFrame>> {
    let mut found = Vec::with_capacity(statements.len());
    for sql in statements {
        found.push(GradCalls::find(sql, &GenericDialect {}).map_err(to_df_err)?);
    }

    // One program per distinct loss query, differentiated with respect to
    // every column any statement asks about.
    let mut programs: Vec<(String, Vec<ColumnRef>)> = Vec::new();
    let mut program_of: HashMap<(usize, usize), usize> = HashMap::new();
    for (s, calls) in found.iter().enumerate() {
        let Some(calls) = calls else { continue };
        for (l, loss) in calls.losses.iter().enumerate() {
            let p = match programs.iter().position(|(q, _)| *q == loss.query) {
                Some(p) => p,
                None => {
                    programs.push((loss.query.clone(), Vec::new()));
                    programs.len() - 1
                }
            };
            for w in &loss.wrt {
                if !programs[p].1.contains(w) {
                    programs[p].1.push(w.clone());
                }
            }
            program_of.insert((s, l), p);
        }
    }

    // Run each, keeping its gradients under names of their own: the next
    // program reuses the step names.
    let mut kept: Vec<(usize, Vec<String>, String, Vec<String>)> = Vec::new();
    for (p, (query, wrt)) in programs.iter().enumerate() {
        let program = grad(ctx, query, wrt).await?;
        run(ctx, &program).await?;
        for g in &program.gradients {
            let name = format!("__ddx_grad_{p}_{}", g.table.join("_"));
            let provider = ctx.table_provider(g.step.as_str()).await?;
            ctx.deregister_table(name.as_str())?;
            ctx.register_table(name.as_str(), provider)?;
            kept.push((p, g.table.clone(), name, g.columns.clone()));
        }
    }

    let mut frames = Vec::with_capacity(statements.len());
    for (s, sql) in statements.iter().enumerate() {
        let Some(calls) = &found[s] else {
            frames.push(ctx.sql(sql).await?);
            continue;
        };
        let mut failure = None;
        let rewritten = calls.rewrite(&mut |call| {
            let p = program_of[&(s, call.loss)];
            let found = kept.iter().find(|(q, table, _, _)| {
                *q == p
                    && (table.join(".") == call.table
                        || table
                            .last()
                            .is_some_and(|t| t.eq_ignore_ascii_case(&call.table)))
            });
            let Some((_, _, name, columns)) = found else {
                failure = Some(format!("no gradient was computed for `{}`", call.table));
                return String::new();
            };
            // The table's dims, then the columns this call asked for.
            let values = value_count(columns, &programs[p].1, &call.table);
            let dims = &columns[..columns.len() - values];
            let picked: Vec<String> = dims
                .iter()
                .cloned()
                .chain(call.columns.iter().filter_map(|c| {
                    columns[dims.len()..]
                        .iter()
                        .find(|v| v.eq_ignore_ascii_case(c))
                        .cloned()
                }))
                .map(|c| format!("\"{}\"", c.replace('"', "\"\"")))
                .collect();
            format!("(SELECT {} FROM \"{name}\")", picked.join(", "))
        });
        if let Some(msg) = failure {
            return Err(DataFusionError::Internal(msg));
        }
        frames.push(ctx.sql(&rewritten).await?);
    }
    Ok(frames)
}

/// How many of a gradient's `columns` (dims, then values) are values: the
/// `wrt` columns of `table`.
fn value_count(columns: &[String], wrt: &[ColumnRef], table: &str) -> usize {
    wrt.iter()
        .filter(|w| w.table.eq_ignore_ascii_case(table))
        .filter(|w| columns.iter().any(|c| c.eq_ignore_ascii_case(&w.column)))
        .count()
}

/// Run every step of `program`, forward then backward, registering each
/// result on `ctx` under the step's name, replacing a table of that name.
///
/// A program depends on the tables' names and schemas, not their values, so
/// build it once and run it on every training step.
pub async fn run(ctx: &SessionContext, program: &BackwardProgram) -> Result<()> {
    for step in program.steps() {
        run_step(ctx, step).await?;
    }
    Ok(())
}

/// Run one step and register its result. Every step it reads must already be
/// registered.
pub async fn run_step(ctx: &SessionContext, step: &Step) -> Result<()> {
    let mut plan = step.plan.clone();
    let mut schemas = HashMap::new();
    for name in unbound_reads(&plan) {
        let schema = table_schema(ctx, &name).await?;
        schemas.insert(name, schema);
    }
    bind_reads(&mut plan, &mut |name| schemas.get(name).cloned()).map_err(to_df_err)?;
    let lp = from_substrait_plan(&ctx.state(), &plan).await?;
    let df = ctx.execute_logical_plan(lp).await?;
    let schema = df.schema().inner().clone();
    let batches = df.collect().await?;
    // A MemTable, not a view: DataFusion's Substrait consumer can pick
    // columns out of a table scan, and a later step reads only some.
    let table = MemTable::try_new(schema, vec![batches])?;
    ctx.deregister_table(step.name.as_str())?;
    ctx.register_table(step.name.as_str(), Arc::new(table))?;
    Ok(())
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
