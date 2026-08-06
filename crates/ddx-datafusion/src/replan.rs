// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Turning a `ddx-core` derivative back into a DataFusion [`Expr`].
//!
//! This is the return leg of the bridge. The outbound
//! leg is trivial — DataFusion's `expr_to_sql` emits exactly the
//! [`sqlparser::ast::Expr`] that `ddx-core` consumes — but coming back needs a
//! planner, and a planner needs a [`ContextProvider`].
//!
//! # Why not `SessionState::create_logical_expr`
//!
//! The obvious way to re-plan an expression is `SessionState::create_logical_expr`.
//! It isn't reachable from here: [`AnalyzerRule::analyze`] receives only
//! `(LogicalPlan, &ConfigOptions)` — no `SessionState` — and a rule cannot hold
//! the state it is installed into without a reference cycle.
//!
//! So the bridge plans the expression itself with [`SqlToRel`], over the minimal
//! [`ContextProvider`] below. That is *more* self-contained rather than less:
//! the only thing a scalar expression needs from a context is function
//! resolution, and a re-planned derivative has no table references to resolve,
//! because differentiation maps column references to column references.
//!
//! [`AnalyzerRule::analyze`]: datafusion::optimizer::AnalyzerRule::analyze

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::DFSchema;
use datafusion::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionStateDefaults;
use datafusion::logical_expr::planner::ExprPlanner;
use datafusion::logical_expr::{
    AggregateUDF, Expr, HigherOrderUDF, ScalarUDF, TableSource, WindowUDF,
};
use datafusion::sql::planner::{ContextProvider, PlannerContext, SqlToRel};
use datafusion::sql::TableReference;
use ddx_core::sqlparser::ast as sql_ast;

/// The expensive half of what the planner needs to turn a derivative back into
/// an [`Expr`]: function registries and expression planners, built once when the
/// analyzer is constructed.
///
/// This is deliberately *not* a [`ContextProvider`] itself — it cannot answer
/// `options()`, which is per-query. [`ScopedExprContext`] borrows it together
/// with the session's config for a single re-plan, and is the only
/// `ContextProvider` here.
///
/// Scalar functions are the only interesting part. A derivative contains
/// whatever `ddx-core` emitted — `cos`, `sin`, `exp`, `ln`, `power`, `abs`,
/// `sqrt`, a `CASE`-based `sign` — *and* whatever survived from the user's own
/// marker body: `d/dx [f(y) * x]` is `f(y)`, so a UDF the user called is still
/// there afterwards — but those arrive per-call via [`ScopedExprContext`],
/// harvested from the marker body, so this base registry only needs the
/// defaults plus anything a custom rule may emit.
#[derive(Debug)]
pub(crate) struct ExprContext {
    functions: HashMap<String, Arc<ScalarUDF>>,
    higher_order: HashMap<String, Arc<HigherOrderUDF>>,
    expr_planners: Vec<Arc<dyn ExprPlanner>>,
}

impl ExprContext {
    /// Seed with the same defaults a stock `SessionState` gets, then layer
    /// `extra` on top (later entries win, so a caller can override a built-in).
    pub(crate) fn new(extra: impl IntoIterator<Item = Arc<ScalarUDF>>) -> Self {
        let mut functions = HashMap::new();
        // `SessionStateDefaults::default_scalar_functions()`, not
        // `functions::all_default_functions()`: the former is the latter *plus*
        // the nested-expression functions, under the default-on
        // `nested_expressions` feature. Taking only the first half would leave
        // this registry quietly narrower than the session's.
        for f in SessionStateDefaults::default_scalar_functions() {
            functions.insert(f.name().to_ascii_lowercase(), f);
        }
        for f in extra {
            functions.insert(f.name().to_ascii_lowercase(), f);
        }
        let higher_order = SessionStateDefaults::default_higher_order_functions()
            .into_iter()
            .map(|f| (f.name().to_ascii_lowercase(), f))
            .collect();

        ExprContext {
            functions,
            higher_order,
            // Exactly the planner list a stock SessionState uses, so an
            // expression re-planned here is planned the same way the engine
            // would have planned it had the user written the derivative out by
            // hand. Anything less risks a subtly different Expr for the same SQL.
            expr_planners: SessionStateDefaults::default_expr_planners(),
        }
    }

    /// Borrow this registry for one re-plan, under the session's own config and
    /// with `local` functions layered on top.
    ///
    /// **Config.** `AnalyzerRule::analyze` is handed the session's
    /// `&ConfigOptions`, and the derivative should be planned under those rather
    /// than under defaults — `enable_ident_normalization`, the parser dialect,
    /// and `parse_float_as_decimal` all steer `SqlToRel`. None is known to
    /// produce a wrong answer today (identifier normalization is masked because the
    /// `Unparser` quotes anything case-sensitive, and `parse_float_as_decimal`
    /// because the replacement is cast to Float64 anyway), so this closes a
    /// latent divergence rather than a live bug.
    ///
    /// **Local functions.** See [`ScopedExprContext`].
    pub(crate) fn scoped<'a>(
        &'a self,
        options: &'a ConfigOptions,
        local: HashMap<String, Arc<ScalarUDF>>,
    ) -> ScopedExprContext<'a> {
        ScopedExprContext {
            inner: self,
            options,
            local,
        }
    }
}

/// [`ExprContext`] borrowed for a single re-plan.
///
/// `local` holds the scalar UDFs harvested from the marker's *own arguments*,
/// and is consulted before the base registry. This is what makes a session UDF
/// work without the analyzer ever seeing the session: a function can only
/// survive into a derivative if it appeared in the body being differentiated
/// (`d/dx [f(y) * x]` is `f(y)`), and a bound `Expr` carries the `Arc<ScalarUDF>`
/// itself, not merely its name. So the definition needed to re-plan is always
/// already in hand.
///
/// That is strictly better than snapshotting the session's registry at install
/// time, which was the obvious fix: it needs no snapshot, cannot go stale, and
/// covers UDFs registered *after* `install`. The base registry still matters for
/// functions ddx-core *introduces* — `cos` for `sin`, the `CASE`-based sign for
/// `abs` — which by definition are not in the body.
pub(crate) struct ScopedExprContext<'a> {
    inner: &'a ExprContext,
    options: &'a ConfigOptions,
    local: HashMap<String, Arc<ScalarUDF>>,
}

impl ContextProvider for ScopedExprContext<'_> {
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
        let name = name.to_ascii_lowercase();
        self.local
            .get(&name)
            .or_else(|| self.inner.functions.get(&name))
            .cloned()
    }

    fn get_higher_order_meta(&self, name: &str) -> Option<Arc<HigherOrderUDF>> {
        self.inner
            .higher_order
            .get(&name.to_ascii_lowercase())
            .cloned()
    }

    fn get_aggregate_meta(&self, _name: &str) -> Option<Arc<AggregateUDF>> {
        // A derivative is a scalar expression. Aggregates in the user's query
        // are outside the marker — the marker goes *inside* the aggregate, as in
        // `AVG(grad(loss, theta))` — so the differentiated fragment never
        // contains one.
        None
    }

    fn get_window_meta(&self, _name: &str) -> Option<Arc<WindowUDF>> {
        None
    }

    fn get_variable_type(&self, _variable_names: &[String]) -> Option<DataType> {
        None
    }

    fn get_expr_planners(&self) -> &[Arc<dyn ExprPlanner>] {
        &self.inner.expr_planners
    }

    fn options(&self) -> &ConfigOptions {
        self.options
    }

    fn udf_names(&self) -> Vec<String> {
        self.inner
            .functions
            .keys()
            .chain(self.local.keys())
            .cloned()
            .collect()
    }

    fn higher_order_function_names(&self) -> Vec<String> {
        self.inner.higher_order.keys().cloned().collect()
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
pub(crate) fn replan(
    ctx: &ExprContext,
    options: &ConfigOptions,
    local: HashMap<String, Arc<ScalarUDF>>,
    expr: sql_ast::Expr,
    schema: &DFSchema,
) -> Result<Expr> {
    SqlToRel::new(&ctx.scoped(options, local)).sql_to_expr(expr, schema, &mut PlannerContext::new())
}

/// Every scalar UDF called anywhere in `exprs`, keyed by lowercase name.
///
/// Harvested from the marker's arguments so a function the user called can be
/// re-planned without the analyzer needing access to the session registry.
pub(crate) fn functions_in(exprs: &[Expr]) -> HashMap<String, Arc<ScalarUDF>> {
    let mut found = HashMap::new();
    for e in exprs {
        let _ = e.apply(|node| {
            if let Expr::ScalarFunction(call) = node {
                found.insert(
                    call.func.name().to_ascii_lowercase(),
                    Arc::clone(&call.func),
                );
            }
            Ok(TreeNodeRecursion::Continue)
        });
    }
    found
}
