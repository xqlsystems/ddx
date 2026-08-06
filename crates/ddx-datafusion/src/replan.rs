// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Turning a `ddx-core` derivative back into a DataFusion [`Expr`].
//!
//! This is the return leg of the Path B bridge (design.md §3.3). The outbound
//! leg is trivial — DataFusion's `expr_to_sql` emits exactly the
//! [`sqlparser::ast::Expr`] that `ddx-core` consumes — but coming back needs a
//! planner, and a planner needs a [`ContextProvider`].
//!
//! # Why not `SessionState::create_logical_expr`
//!
//! design.md §3.3 names `SessionState::create_logical_expr` as the re-plan
//! seam. It isn't reachable from where we need it: [`AnalyzerRule::analyze`]
//! receives only `(LogicalPlan, &ConfigOptions)` — no `SessionState` — and a
//! rule cannot hold the state it is installed into without a reference cycle.
//!
//! So the bridge plans the expression itself with [`SqlToRel`], over the
//! minimal [`ContextProvider`] below. This is *more* self-contained than the
//! documented seam rather than less: the only thing a scalar expression needs
//! from a context is function resolution, and a re-planned derivative has no
//! table references to resolve — differentiation maps columns to columns.
//!
//! [`AnalyzerRule::analyze`]: datafusion::optimizer::AnalyzerRule::analyze

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::DFSchema;
use datafusion::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::planner::ExprPlanner;
use datafusion::logical_expr::{
    AggregateUDF, Expr, HigherOrderUDF, ScalarUDF, TableSource, WindowUDF,
};
use datafusion::sql::planner::{ContextProvider, PlannerContext, SqlToRel};
use datafusion::sql::TableReference;
use ddx_core::sqlparser::ast as sql_ast;

/// Everything the planner needs to turn a derivative expression back into an
/// [`Expr`]: a function registry, config, and expression planners.
///
/// Scalar functions are the only interesting part. A derivative contains
/// whatever `ddx-core` emitted — `cos`, `sin`, `exp`, `ln`, `power`, `abs`,
/// `sqrt`, a `CASE`-based `sign` — plus whatever a user rule emitted, so the
/// registry is seeded with DataFusion's built-ins and extended with any UDFs
/// the caller passes in.
#[derive(Debug)]
pub(crate) struct ExprContext {
    functions: HashMap<String, Arc<ScalarUDF>>,
    options: ConfigOptions,
    expr_planners: Vec<Arc<dyn ExprPlanner>>,
}

impl ExprContext {
    /// Seed with DataFusion's default scalar functions, then layer `extra` on
    /// top (later entries win, so a caller can override a built-in).
    pub(crate) fn new(extra: impl IntoIterator<Item = Arc<ScalarUDF>>) -> Self {
        let mut functions = HashMap::new();
        for f in datafusion::functions::all_default_functions() {
            functions.insert(f.name().to_ascii_lowercase(), f);
        }
        for f in extra {
            functions.insert(f.name().to_ascii_lowercase(), f);
        }
        ExprContext {
            functions,
            options: ConfigOptions::default(),
            // Exactly the planner list a stock SessionState uses, so an
            // expression re-planned here is planned the same way the engine
            // would have planned it had the user written the derivative out by
            // hand. Anything less risks a subtly different Expr for the same SQL.
            //
            // `SessionStateDefaults::default_expr_planners()` would say this
            // directly but is private, so the list is borrowed from a throwaway
            // default state. Built once, when the analyzer is constructed.
            expr_planners: SessionStateBuilder::new()
                .with_default_features()
                .build()
                .expr_planners()
                .to_vec(),
        }
    }
}

impl ContextProvider for ExprContext {
    fn get_table_source(&self, name: TableReference) -> Result<Arc<dyn TableSource>> {
        // Unreachable for a scalar expression: differentiation maps column
        // references to column references and never introduces a relation. If
        // this ever fires it is a ddx bug, not user error, so say so plainly.
        Err(DataFusionError::Internal(format!(
            "ddx: re-planning a derivative expression tried to resolve the table `{name}`. \
             A differentiated scalar expression must not contain table references — \
             please report this with the query that triggered it."
        )))
    }

    fn get_function_meta(&self, name: &str) -> Option<Arc<ScalarUDF>> {
        self.functions.get(&name.to_ascii_lowercase()).cloned()
    }

    fn get_higher_order_meta(&self, _name: &str) -> Option<Arc<HigherOrderUDF>> {
        None
    }

    fn get_aggregate_meta(&self, _name: &str) -> Option<Arc<AggregateUDF>> {
        // A derivative is a scalar expression. Aggregates in the user's query
        // are outside the marker (design.md §3.1 puts the marker *inside* the
        // aggregate: `AVG(grad(loss, theta))`), so the differentiated fragment
        // never contains one.
        None
    }

    fn get_window_meta(&self, _name: &str) -> Option<Arc<WindowUDF>> {
        None
    }

    fn get_variable_type(&self, _variable_names: &[String]) -> Option<DataType> {
        None
    }

    fn get_expr_planners(&self) -> &[Arc<dyn ExprPlanner>] {
        &self.expr_planners
    }

    fn options(&self) -> &ConfigOptions {
        &self.options
    }

    fn udf_names(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }

    fn higher_order_function_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn udaf_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn udwf_names(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Plan a `sqlparser` expression back into a DataFusion [`Expr`] against
/// `schema`.
///
/// `schema` is the input schema of the plan node the marker was found in, which
/// is what binds the derivative's column references to the same columns the
/// original expression used.
pub(crate) fn replan(ctx: &ExprContext, expr: sql_ast::Expr, schema: &DFSchema) -> Result<Expr> {
    SqlToRel::new(ctx).sql_to_expr(expr, schema, &mut PlannerContext::new())
}
