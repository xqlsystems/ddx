// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A program's run on DataFusion, step by step: what [`crate::ad::run`] and
//! [`crate::ad::sql`] are built on. Public only with the `internals`
//! feature, for ddx's own tests and benchmarks; not part of the API.

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
use datafusion_substrait::extensions::Extensions;
use datafusion_substrait::logical_plan::consumer::{
    from_project_rel, from_substrait_plan_with_consumer, from_substrait_rel,
    DefaultSubstraitConsumer, SubstraitConsumer,
};
use datafusion_substrait::logical_plan::producer::to_substrait_plan;
use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::{NamedStruct, Plan, ProjectRel, Rel};
use ddx_ad::{bind_reads, unbound_reads, Action, AdError, Program, RunError, Step};

pub(crate) fn to_df_err(e: AdError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Run `program`'s checks, then its steps, registering each step's result
/// on `ctx` under its name. Its input tables must be registered. The value and the
/// gradients stay registered until [`release`]; the intermediate tables are
/// dropped.
pub async fn execute<P: Program + ?Sized>(ctx: &SessionContext, program: &P) -> Result<()> {
    // The order, and what is dropped when, are ddx_ad::Runner's. The checks
    // read no step and none reads another, so they run at once, each a task
    // of its own (polled together on one task, they ran one after another),
    // and the runner is told how each went in its own order: on a large wrt
    // table they are most of a run (the dims of a table, then of its
    // tangent, each a grouped scan).
    let mut runner = ddx_ad::Runner::new(program);
    let mut checked: Vec<Option<Result<bool>>> = program.checks().iter().map(|_| None).collect();
    let tasks: Vec<_> = program
        .checks()
        .iter()
        .enumerate()
        .map(|(i, check)| {
            let (ctx, plan) = (ctx.clone(), check.plan.clone());
            (
                i,
                tokio::spawn(async move { returns_rows(&ctx, &plan).await }),
            )
        })
        .collect();
    for (i, task) in tasks {
        checked[i] = Some(
            task.await
                .unwrap_or_else(|e| Err(DataFusionError::External(Box::new(e)))),
        );
    }
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
    Ok(())
}

/// Register `df`'s rows as the table `name`, materialized.
pub async fn register(ctx: &SessionContext, name: &str, df: DataFrame) -> Result<()> {
    let schema = Arc::new(df.schema().as_arrow().clone());
    let batches = df.collect().await?;
    ctx.register_table(name, Arc::new(MemTable::try_new(schema, vec![batches])?))?;
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
