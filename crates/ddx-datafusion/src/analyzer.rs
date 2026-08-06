// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Path B — the in-engine plan rewrite (design.md §3.3).
//!
//! An [`AnalyzerRule`] that finds every `grad`/`jvp` marker in a bound
//! [`LogicalPlan`], differentiates its argument through `ddx-core`, and splices
//! the result back as a real DataFusion [`Expr`]. This is what makes bare
//! `grad()` work with no wrapper, across both the SQL and DataFrame APIs.
//!
//! **It is a bridge, not a second rule engine.** All the calculus lives in
//! `ddx-core`; this module only moves expressions across the boundary:
//!
//! ```text
//!   DataFusion Expr --expr_to_sql--> sqlparser::ast::Expr   (same crate version!)
//!                                          |
//!                                    ddx-core differentiate
//!                                          |
//!   DataFusion Expr <----replan----- sqlparser::ast::Expr
//! ```
//!
//! Both hops are type-level, with no SQL string in between — which only works
//! because `ddx-core` and `datafusion` resolve the *identical* `sqlparser`
//! (design.md §6, decision-log G2; enforced by `tests/sqlparser_pin.rs`).
//!
//! # Binding-awareness comes free here
//!
//! The plan is already bound when the rule sees it, so column references arrive
//! qualified. That means the syntactic ambiguity guard of design.md §3.5 —
//! which exists because a *pre-binding* text rewrite cannot tell `a.x` from
//! `b.x` — simply never fires on this path.

use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, DFSchema};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, LogicalPlan, ScalarUDF};
use datafusion::optimizer::AnalyzerRule;
use datafusion::sql::unparser::Unparser;
use ddx_core::sqlparser::ast as sql_ast;
use ddx_core::{ColRef, Ddx};

use crate::error::to_df_err;
use crate::markers::{marker_kind, GRAD};
use crate::replan::{replan, ExprContext};

/// The ddx analyzer rule: rewrites `grad`/`jvp` markers away before execution.
///
/// Install it with [`crate::install`], which also registers the marker UDFs so
/// the calls parse in the first place.
pub struct DdxAnalyzer {
    ddx: Ddx,
    exprs: ExprContext,
}

// `AnalyzerRule` requires `Debug`, but `Ddx` holds a rule registry of function
// pointers and is deliberately not `Debug` itself. Print what is actually
// useful in a plan dump — which rule this is — rather than nothing.
impl std::fmt::Debug for DdxAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DdxAnalyzer")
            .field("rule", &"ddx_markers")
            .finish_non_exhaustive()
    }
}

impl Default for DdxAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl DdxAnalyzer {
    /// A rule driving the built-in rule set.
    pub fn new() -> Self {
        Self::with_engine(Ddx::for_datafusion())
    }

    /// A rule driving a caller-supplied engine — use this to pick up custom
    /// differentiation rules registered via [`Ddx::register`].
    pub fn with_engine(ddx: Ddx) -> Self {
        Self::with_engine_and_functions(ddx, [])
    }

    /// As [`DdxAnalyzer::with_engine`], plus extra scalar functions that may
    /// appear in an emitted derivative.
    ///
    /// Needed only when a custom rule emits a call to a UDF of your own:
    /// DataFusion's built-ins are always available, but the rule re-plans the
    /// derivative through its own function registry and cannot see UDFs
    /// registered on the `SessionContext`.
    pub fn with_engine_and_functions(
        ddx: Ddx,
        functions: impl IntoIterator<Item = Arc<ScalarUDF>>,
    ) -> Self {
        DdxAnalyzer {
            ddx,
            exprs: ExprContext::new(functions),
        }
    }

    /// Rewrite every marker in one expression, bottom-up.
    ///
    /// Bottom-up is what makes higher-order differentiation fall out for free:
    /// the inner `grad` of `grad(grad(f, x), x)` is already an ordinary
    /// expression by the time the outer one is differentiated (design.md §3.1).
    fn rewrite_expr(&self, expr: Expr, schema: &DFSchema) -> Result<Transformed<Expr>> {
        expr.transform_up(|e| {
            let Expr::ScalarFunction(call) = &e else {
                return Ok(Transformed::no(e));
            };
            let Some(kind) = marker_kind(call.func.name()) else {
                return Ok(Transformed::no(e));
            };
            let derivative = self.differentiate_call(kind, &call.args, schema)?;
            Ok(Transformed::yes(derivative))
        })
    }

    /// Differentiate one marker call and return the replacement expression.
    fn differentiate_call(
        &self,
        kind: &'static str,
        args: &[Expr],
        schema: &DFSchema,
    ) -> Result<Expr> {
        let expected = if kind == GRAD { 2 } else { 3 };
        if args.len() != expected {
            return Err(DataFusionError::Plan(format!(
                "ddx: `{kind}` takes {expected} arguments, got {}. \
                 Write `grad(expr, column)` or `jvp(expr, column, tangent)`.",
                args.len()
            )));
        }

        let body = to_sql_ast(&args[0])?;
        let wrt = wrt_colref(kind, &args[1])?;

        let derivative: sql_ast::Expr = match kind {
            GRAD => self.ddx.differentiate(&body, &wrt).map_err(to_df_err)?,
            _ => {
                let tangent = to_sql_ast(&args[2])?;
                self.ddx.jvp(&body, &[(wrt, tangent)]).map_err(to_df_err)?
            }
        };

        replan(&self.exprs, derivative, schema)
    }
}

impl AnalyzerRule for DdxAnalyzer {
    fn name(&self) -> &str {
        "ddx_markers"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        // Skip the whole walk when the plan contains no marker. Same spirit as
        // ddx-core's parse-free pre-gate (design.md §3.2, F5): a query that
        // never mentions ddx should not be touched, and must not be able to
        // fail inside ddx.
        if !plan_has_marker(&plan)? {
            return Ok(plan);
        }

        plan.transform_up(|node| {
            // A node's expressions are resolved against its *inputs*, not its
            // own output schema — that is what binds the derivative's columns
            // to the same columns the original expression used. Leaf nodes
            // (which have no inputs) fall back to their own schema.
            let schema = merged_input_schema(&node)?;

            // Nodes whose expressions *name* output fields. For these the
            // rewrite must not change the field name: an unaliased
            // `AVG(grad(x*x, x))` derives the field name
            // `avg(grad(t.x * t.x,t.x))`, and the parent projection refers to
            // it by exactly that string. Rewriting the expression underneath
            // would rename the field and leave the parent's column reference
            // dangling ("field not found"), so the replacement is aliased back
            // to the original name. Predicates (Filter) and sort keys name
            // nothing, so they are left alone.
            let names_fields = matches!(
                node,
                LogicalPlan::Projection(_) | LogicalPlan::Aggregate(_) | LogicalPlan::Window(_)
            );

            let mut rewrote = false;
            let node = node.map_expressions(|expr| {
                let original_name = names_fields.then(|| expr.schema_name().to_string());
                let out = self.rewrite_expr(expr, &schema)?;
                rewrote |= out.transformed;
                Ok(out.update_data(|e| match original_name {
                    Some(name) if e.schema_name().to_string() != name => e.alias(name),
                    _ => e,
                }))
            })?;

            if rewrote {
                // The replacement has a different type than the marker call it
                // replaced, so the node's schema must be rebuilt.
                node.map_data(|plan| plan.recompute_schema())
            } else {
                Ok(node)
            }
        })
        .map(|t| t.data)
    }
}

/// The schema a node's expressions resolve against: all of its inputs merged.
fn merged_input_schema(plan: &LogicalPlan) -> Result<DFSchema> {
    let inputs = plan.inputs();
    if inputs.is_empty() {
        return Ok(plan.schema().as_ref().clone());
    }
    let mut merged = DFSchema::empty();
    for input in inputs {
        merged.merge(input.schema());
    }
    Ok(merged)
}

/// Does this plan contain a ddx marker anywhere?
fn plan_has_marker(plan: &LogicalPlan) -> Result<bool> {
    let mut found = false;
    plan.apply(|node| {
        node.apply_expressions(|expr| {
            expr.apply(|e| {
                if let Expr::ScalarFunction(call) = e {
                    if marker_kind(call.func.name()).is_some() {
                        found = true;
                        return Ok(TreeNodeRecursion::Stop);
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
    Ok(found)
}

/// Unparse a bound DataFusion expression into the `sqlparser` AST `ddx-core`
/// consumes. This is the load-bearing type identity of design.md §6 (G2): the
/// output here *is* `ddx-core`'s input type, with no string in between.
fn to_sql_ast(expr: &Expr) -> Result<sql_ast::Expr> {
    Unparser::default().expr_to_sql(expr)
}

/// Read the differentiation variable off the marker's second argument.
///
/// Taken straight from the bound [`Column`] rather than unparsed text, so the
/// qualifier is whatever the planner resolved — the reason this path is
/// binding-aware for free.
fn wrt_colref(kind: &str, arg: &Expr) -> Result<ColRef> {
    match arg {
        Expr::Column(Column { relation, name, .. }) => Ok(ColRef {
            qualifier: relation
                .as_ref()
                .map(|r| sql_ast::Ident::new(r.table().to_string())),
            name: sql_ast::Ident::new(name.clone()),
        }),
        other => {
            // Not a column: produce ddx-core's own error, so the message is
            // identical whichever path the user came in on.
            let unparsed = to_sql_ast(other)?;
            ColRef::from_wrt_arg(kind, &unparsed).map_err(to_df_err)
        }
    }
}
