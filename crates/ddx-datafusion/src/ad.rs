// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Query-level reverse-mode AD (ddx v2) on DataFusion.
//!
//! Write the forward pass as one SQL query whose result is a loss. [`grad`]
//! turns it into a [`BackwardProgram`], and [`run`] runs it: every step is
//! materialized as a table on the context, the loss ends up in
//! [`BackwardProgram::value`], and each `wrt` table's gradient, shaped like the
//! table, in the table [`BackwardProgram::gradients`] names. [`vjp`] does the
//! same for a query with any output, pulling back a cotangent the caller
//! registers as [`COTANGENT`].
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
//! let loss = "SELECT SUM(val * val) AS loss FROM w";
//! let program = ad::grad(&ctx, loss, &[ColumnRef::new("w", "val")]).await?;
//! ad::run(&ctx, &program).await?;
//!
//! // d/dval of Σ val² is 2·val.
//! let grad = &program.gradients[0].step;
//! let df = ctx.sql(&format!("SELECT i, val FROM {grad} ORDER BY i")).await?;
//! # let _ = df.collect().await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use datafusion_substrait::logical_plan::consumer::from_substrait_plan;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;
use ddx_ad::emit::{bind_reads, unbound_reads};
use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::{NamedStruct, Rel};

pub use ddx_ad::{AdError, BackwardProgram, ColumnRef, Gradient, Step, COTANGENT, VALUE};

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
