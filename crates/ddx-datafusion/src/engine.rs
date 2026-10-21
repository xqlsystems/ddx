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
use datafusion::common::config::ConfigOptions;
use datafusion::common::{Column, DFSchema, TableReference};
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{FunctionRegistry, SessionState};
use datafusion::logical_expr::{Expr, LogicalPlan, Projection};
use datafusion::optimizer::analyzer::type_coercion::TypeCoercion;
use datafusion::optimizer::Analyzer;
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
use ddx_ad::{AdError, Bound, Program, RunError, Step};

pub(crate) fn to_df_err(e: AdError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Run `program`'s checks, then its steps, registering each step's result
/// on `ctx` under its name ([`ddx_ad::run`] on [`DataFusion`]). Its input
/// tables must be registered. The value and the gradients stay registered
/// until [`release`]; the intermediate tables are dropped.
pub async fn execute<P: Program + ?Sized>(ctx: &SessionContext, program: &P) -> Result<()> {
    ddx_ad::run(&mut DataFusion(ctx.clone()), program)
        .await
        .map_err(from_run_error)
}

/// A run's failure as DataFusion's error: a refusal boxed as an
/// [`AdError`], the engine's own as it was.
pub(crate) fn from_run_error(e: RunError<DataFusionError>) -> DataFusionError {
    match e {
        RunError::Refused(e) => to_df_err(e),
        RunError::Engine(e) => e,
        other => DataFusionError::Internal(other.to_string()),
    }
}

/// A DataFusion context as a [`ddx_ad::Backend`].
pub struct DataFusion(pub SessionContext);

impl ddx_ad::Backend for DataFusion {
    type Error = DataFusionError;

    async fn table_schema(&mut self, name: &str) -> Result<NamedStruct> {
        table_schema(&self.0, name).await
    }

    async fn returns_rows(&mut self, plans: &[Bound]) -> Result<Vec<bool>> {
        // The checks read no step and none reads another, so they run at
        // once, each a task of its own (polled together on one task, they
        // ran one after another): on a large wrt table they are most of a
        // run, each a grouped scan.
        let tasks: Vec<_> = plans
            .iter()
            .map(|plan| {
                let (ctx, plan) = (self.0.clone(), (**plan).clone());
                tokio::spawn(async move { returns_rows(&ctx, &plan).await })
            })
            .collect();
        let mut rows = Vec::with_capacity(tasks.len());
        for task in tasks {
            rows.push(
                task.await
                    .map_err(|e| DataFusionError::External(Box::new(e)))??,
            );
        }
        Ok(rows)
    }

    async fn materialize(&mut self, name: &str, plan: &Bound) -> Result<()> {
        materialize(&self.0, name, plan).await
    }

    async fn drop_table(&mut self, name: &str) -> Result<()> {
        self.0.deregister_table(name).map(|_| ())
    }

    async fn plan_sql(&mut self, sql: &str) -> Result<Plan> {
        let lp = self.0.sql(sql).await?.into_optimized_plan()?;
        crate::ad::substrait_of(&self.0, &lp).await.map(|p| *p)
    }

    async fn plan_key(&mut self, sql: &str) -> Result<Option<String>> {
        // DataFusion expands each view, and each view it reads, as it turns
        // SQL into a plan, so the plan before optimization holds every
        // view's definition, and its schemas every table's columns, types
        // and nullability: all an optimized plan's meaning depends on.
        // Its text names tables but not which: a program kept for a view
        // whose table is registered again since would read the new one.
        let df = self.0.sql(sql).await?;
        crate::ad::refuse_replaced_tables(&self.0, df.logical_plan()).await?;
        let key = df.logical_plan().display_indent_schema().to_string();
        Ok(Some(key))
    }

    async fn view_sql(&mut self, name: &str) -> Result<Option<String>> {
        // A name that reads no table, or a table that is not a view (or a
        // view registered from a DataFrame, which has no SQL), has none.
        // The name is resolved as the session's SQL would resolve it: an
        // unquoted part in lower case only if the session normalizes them.
        let normalize = self
            .0
            .state()
            .config()
            .options()
            .sql_parser
            .enable_ident_normalization;
        let reference = TableReference::parse_str_normalized(name, !normalize);
        let Ok(provider) = self.0.table_provider(reference).await else {
            return Ok(None);
        };
        let Some(view) = provider.downcast_ref::<datafusion::catalog::view::ViewTable>() else {
            return Ok(None);
        };
        let Some(sql) = view.definition() else {
            return Ok(None);
        };
        // A view over a table registered again since reads the old table,
        // its SQL the new one.
        if crate::ad::replaced_table(&self.0, view.logical_plan())
            .await?
            .is_some()
        {
            return Ok(None);
        }
        // DataFusion keeps the plan a view was made with, so once a view it
        // reads is redefined, its SQL means something else than what runs:
        // its SQL is given only if it plans as the view's kept plan.
        let Some(probe) = ddx_ad::view_probe(sql) else {
            return Ok(None);
        };
        let Ok(now) = self.0.state().create_logical_plan(&probe).await else {
            return Ok(None);
        };
        // Coerced as DataFusion coerces the plan it keeps for a view (its
        // `create_view`: the type-coercion rule alone, default options), so
        // `val * 2` over a DOUBLE plans alike both times.
        let Ok(now) = Analyzer::with_rules(vec![Arc::new(TypeCoercion::new())]).execute_and_check(
            now,
            &ConfigOptions::default(),
            |_, _| {},
        ) else {
            return Ok(None);
        };
        // Under the projection and alias of `SELECT * FROM (…) AS …`.
        let inner = now
            .inputs()
            .first()
            .and_then(|alias| alias.inputs().first().copied());
        let same = inner.is_some_and(|now| {
            now.display_indent_schema().to_string()
                == view.logical_plan().display_indent_schema().to_string()
        });
        Ok(same.then(|| sql.clone()))
    }
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
    let rows: usize = dataframe(ctx, plan)
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
    let df = dataframe(ctx, plan).await?;
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
    let plan = bound(ctx, plan).await?;
    let state = ctx.state();
    let extensions = Extensions::try_from(&plan.extensions)?;
    let consumer = ShortNames {
        inner: DefaultSubstraitConsumer::new(&extensions, &state),
        state: &state,
        next: AtomicUsize::new(0),
    };
    from_substrait_plan_with_consumer(&consumer, &plan).await
}

/// `plan` with its reads of earlier steps bound to the tables registered on
/// `ctx`.
async fn bound(ctx: &SessionContext, plan: &Plan) -> Result<Plan> {
    let unbound = ddx_ad::Unbound::from_plan(plan.clone());
    let mut schemas = HashMap::new();
    for name in unbound.reads() {
        let schema = table_schema(ctx, &name).await?;
        schemas.insert(name, schema);
    }
    Ok(unbound
        .bind(&mut |name| schemas.get(name).cloned())
        .map_err(to_df_err)?
        .into_plan())
}

/// One of a program's plans, ready to run: consumed as Substrait, or, with
/// the test switch [`steps_as_sql`] on, run as the SQL `ddx_ad::to_sql`
/// writes for it.
async fn dataframe(ctx: &SessionContext, plan: &Plan) -> Result<DataFrame> {
    if steps_as_sql() {
        let sql = ddx_ad::to_sql(&bound(ctx, plan).await?, ddx_ad::Dialect::DataFusion)
            .map_err(to_df_err)?;
        return ctx.sql(&sql).await;
    }
    let lp = logical_plan(ctx, plan).await?;
    ctx.execute_logical_plan(lp).await
}

/// Whether every step and check runs as SQL rather than Substrait: set by
/// `DDX_STEPS_AS_SQL=1`, read only with the `internals` feature. ddx's tests
/// run the whole suite both ways, so every program they build checks
/// `ddx_ad::to_sql`, which engines without a usable Substrait consumer
/// (DuckDB's) run, against DataFusion's consumer.
fn steps_as_sql() -> bool {
    #[cfg(feature = "internals")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("DDX_STEPS_AS_SQL").is_some_and(|v| v == "1"))
    }
    #[cfg(not(feature = "internals"))]
    false
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
