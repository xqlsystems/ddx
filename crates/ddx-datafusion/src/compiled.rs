// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A program's plans, compiled to run again
//! ([`ddx_ad::Backend::compile`]): each a physical plan, built once, that
//! reads its tables as they are on every run.
//!
//! A physical plan holds what its scans read: a `MemTable`'s scan holds its
//! batches. So a compiled plan's every scan is a [`LiveExec`], a leaf that
//! names its table, and each run replaces it with a scan of the table registered
//! under that name then, and resets every other node's state (a hash join's
//! build side, say), as `reset_plan_states` does. A scan that differs from
//! the one the plan was built for, in its schema, partitioning or ordering,
//! would not mean what the plan assumes: the plan is built again.
//!
//! Nothing a compiled plan's first run reads is built into it. Its scans
//! state their tables' statistics as inexact, so the optimizer cannot fold an
//! ungrouped `COUNT(*)` into the first run's number; a filter pushed into a
//! scan is applied again above it; and dynamic filters are off while it is
//! planned, since a reset plan keeps the filter its first run pushed down.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Constraints, Statistics, TableReference};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::execution::{SendableRecordBatchStream, SessionState, TaskContext};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_expr::{Partitioning, PhysicalExpr};
use datafusion::physical_plan::statistics::{StatisticsArgs, StatisticsContext};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use datafusion::prelude::SessionContext;
use ddx_ad::substrait::proto::Plan;

/// A plan, as [`crate::engine::DataFusion`] compiles it: its physical plan,
/// whose scans are [`LiveExec`]s, replaced when its tables' scans have
/// changed (see the module docs).
pub struct Compiled(Mutex<Arc<dyn ExecutionPlan>>);

impl fmt::Debug for Compiled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Compiled")
    }
}

/// `plan`, compiled to run again on `ctx`.
pub(crate) async fn compile(ctx: &SessionContext, plan: &Plan) -> Result<Compiled> {
    Ok(Compiled(Mutex::new(physical(ctx, plan).await?)))
}

/// `compiled` (made of `plan`), ready to run once on `ctx`'s tables as they
/// are: each scan replaced, every node's state reset. Built again from
/// `plan` if a table's scan has changed.
pub(crate) async fn ready(
    ctx: &SessionContext,
    plan: &Plan,
    compiled: &Compiled,
) -> Result<Arc<dyn ExecutionPlan>> {
    let state = ctx.state();
    // A poisoned lock (a panic while it was held) is read as no plan:
    // `plan` is planned again each run.
    let kept = compiled.0.lock().ok().map(|p| Arc::clone(&p));
    if let Some(physical) = kept {
        if let Some(ready) = with_scans(&state, &physical).await? {
            return Ok(ready);
        }
    }
    let physical = self::physical(ctx, plan).await?;
    let ready = with_scans(&state, &physical).await?.ok_or_else(|| {
        DataFusionError::Internal("a compiled plan's scans changed as it was built".into())
    })?;
    if let Ok(mut kept) = compiled.0.lock() {
        *kept = physical;
    }
    Ok(ready)
}

/// `plan`'s physical plan, with a [`LiveExec`] for each scan, and planned
/// without dynamic filters.
async fn physical(ctx: &SessionContext, plan: &Plan) -> Result<Arc<dyn ExecutionPlan>> {
    let lp = crate::engine::logical_plan_reading(ctx, plan, Live::wrap).await?;
    // Not a new default catalog: it would replace the context's own, which
    // the new state shares.
    let mut config = ctx
        .copied_config()
        .with_create_default_catalog_and_schema(false);
    let optimizer = &mut config.options_mut().optimizer;
    optimizer.enable_dynamic_filter_pushdown = false;
    optimizer.enable_join_dynamic_filter_pushdown = false;
    optimizer.enable_topk_dynamic_filter_pushdown = false;
    optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    let state = SessionStateBuilder::new_from_existing(ctx.state())
        .with_config(config)
        .build();
    state.create_physical_plan(&lp).await
}

/// `plan` with each [`LiveExec`] replaced by a scan of its table as
/// registered now, and every other node's state reset; `None` if a scan
/// differs from the one `plan` was built for.
async fn with_scans(
    state: &SessionState,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let mut leaves: Vec<&LiveExec> = Vec::new();
    plan.apply(|node| {
        if let Some(live) = node.downcast_ref::<LiveExec>() {
            leaves.push(live);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    // By address: a leaf may sit in the plan more than once.
    let mut scans: HashMap<usize, Arc<dyn ExecutionPlan>> = HashMap::new();
    for live in leaves {
        let scan = live.scan(state).await?;
        if !live.planned_for(&scan) {
            return Ok(None);
        }
        scans.insert(live as *const LiveExec as usize, scan);
    }
    let ready = Arc::clone(plan)
        .transform_up(|node| {
            let scan = node
                .downcast_ref::<LiveExec>()
                .and_then(|live| scans.get(&(live as *const LiveExec as usize)));
            Ok(Transformed::yes(match scan {
                Some(scan) => Arc::clone(scan),
                None => node.reset_state()?,
            }))
        })?
        .data;
    Ok(Some(ready))
}

/// A table as a compiled plan reads it: by name, when the plan runs (see the
/// module docs). Planned as the table registered when the plan is built.
#[derive(Debug)]
struct Live {
    reference: TableReference,
    table: Arc<dyn TableProvider>,
}

impl Live {
    fn wrap(reference: &TableReference, table: Arc<dyn TableProvider>) -> Arc<dyn TableProvider> {
        Arc::new(Live {
            reference: reference.clone(),
            table,
        })
    }
}

#[async_trait]
impl TableProvider for Live {
    fn schema(&self) -> SchemaRef {
        self.table.schema()
    }

    fn constraints(&self) -> Option<&Constraints> {
        self.table.constraints()
    }

    fn table_type(&self) -> TableType {
        self.table.table_type()
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let scan = self.table.scan(state, projection, filters, limit).await?;
        let stats = StatisticsContext::new();
        let statistics = (0..scan.output_partitioning().partition_count())
            .map(Some)
            .chain([None])
            .map(|partition| {
                let args = StatisticsArgs::new().with_partition(partition);
                let exact = stats.compute(scan.as_ref(), &args)?;
                Ok(Arc::new(Arc::unwrap_or_clone(exact).to_inexact()))
            })
            .collect::<Result<_>>()?;
        Ok(Arc::new(LiveExec {
            reference: self.reference.clone(),
            projection: projection.cloned(),
            filters: filters.to_vec(),
            limit,
            properties: Arc::clone(scan.properties()),
            statistics,
        }))
    }

    /// Each filter the table takes is applied again above its scan: the
    /// table registered when the plan runs may not take it.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(self
            .table
            .supports_filters_pushdown(filters)?
            .into_iter()
            .map(|pushed| match pushed {
                TableProviderFilterPushDown::Unsupported => pushed,
                _ => TableProviderFilterPushDown::Inexact,
            })
            .collect())
    }

    fn statistics(&self) -> Option<Statistics> {
        self.table.statistics().map(Statistics::to_inexact)
    }
}

/// A compiled plan's scan of a table: replaced by the table's own scan, as
/// registered when the plan runs, before it runs (see the module docs).
#[derive(Debug)]
struct LiveExec {
    reference: TableReference,
    projection: Option<Vec<usize>>,
    filters: Vec<Expr>,
    limit: Option<usize>,
    /// The properties of the scan it was planned as.
    properties: Arc<PlanProperties>,
    /// That scan's statistics, inexact: each partition's, then all of them.
    statistics: Vec<Arc<Statistics>>,
}

impl LiveExec {
    /// The scan of the table registered under its name now.
    async fn scan(&self, state: &SessionState) -> Result<Arc<dyn ExecutionPlan>> {
        let table = state
            .schema_for_ref(self.reference.clone())?
            .table(self.reference.table())
            .await?
            .ok_or_else(|| {
                DataFusionError::Plan(format!("table '{}' not found", self.reference))
            })?;
        table
            .scan(state, self.projection.as_ref(), &self.filters, self.limit)
            .await
    }

    /// Whether the plan above this leaf, built for the scan it was planned
    /// as, means the same over `scan`.
    fn planned_for(&self, scan: &Arc<dyn ExecutionPlan>) -> bool {
        let (was, is) = (&*self.properties, scan.properties());
        self.schema() == scan.schema()
            && same_partitioning(&was.partitioning, &is.partitioning)
            && was.output_ordering() == is.output_ordering()
            && was.emission_type == is.emission_type
            && was.boundedness == is.boundedness
    }
}

/// Whether `a` and `b` split rows alike: DataFusion's `==` holds no
/// unknown partitioning equal to another, even into as many partitions.
fn same_partitioning(a: &Partitioning, b: &Partitioning) -> bool {
    match (a, b) {
        (Partitioning::UnknownPartitioning(m), Partitioning::UnknownPartitioning(n)) => m == n,
        _ => a == b,
    }
}

impl DisplayAs for LiveExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LiveExec: {}", self.reference)
    }
}

impl ExecutionPlan for LiveExec {
    fn name(&self) -> &str {
        "LiveExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn apply_expressions(
        &self,
        _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(&self, _: usize, _: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        Err(DataFusionError::Internal(format!(
            "a compiled plan ran its scan of '{}' before the scan was replaced",
            self.reference
        )))
    }

    fn statistics_from_inputs(
        &self,
        _: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        let i = args.partition().unwrap_or(self.statistics.len() - 1);
        Ok(Arc::clone(&self.statistics[i]))
    }
}
