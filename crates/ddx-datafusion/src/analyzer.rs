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

use datafusion::arrow::datatypes::DataType;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::DFSchema;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, ExprSchemable, LogicalPlan, ScalarUDF};
use datafusion::optimizer::analyzer::type_coercion::TypeCoercion;
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
            // A subquery embedded in an expression carries its own
            // `LogicalPlan`, and `LogicalPlan::map_children` does NOT descend
            // into those — it visits only direct relational inputs. Without
            // this, a marker inside `WHERE x > (SELECT AVG(grad(y*y, y)) ...)`
            // is never rewritten and survives to execution.
            if let Some(sub) = subquery_plan(&e) {
                let rewritten = self.rewrite_plan(sub.as_ref().clone())?;
                return Ok(Transformed::yes(with_subquery_plan(e, Arc::new(rewritten))));
            }
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

        let replanned = replan(&self.exprs, derivative, schema)?;

        // Force the replacement to the type the marker UDF declared
        // (`Marker::return_type` → Float64). Two reasons, one of them a bug:
        //
        // 1. Correctness. The marker's declared type is already baked into
        //    every ancestor node's cached schema. `LogicalPlan::map_children`
        //    preserves a parent's `schema` field while swapping its input, so
        //    only the rewritten node gets `recompute_schema` — if the
        //    derivative planned to a different type (`x + x` on an Int64 column
        //    is Int64), ancestors keep a stale Float64 and the optimizer's
        //    invariant check fails with an internal error.
        // 2. Policy. design.md §3.2 (F4/R1b) says derivatives are always
        //    emitted DOUBLE-typed, because differentiation runs pre-binding and
        //    SQL integer division truncates on some engines but not others.
        //    Without this, `grad(x*x, x)` over a BIGINT column returns Int64 —
        //    quietly violating the invariant this crate documents.
        //
        // `cast_to` is a no-op when the type already matches, which is the
        // common case.
        replanned.cast_to(&DataType::Float64, schema).map_err(|e| {
            DataFusionError::Plan(format!(
                "ddx: the derivative of a `{kind}` argument could not be represented as \
                     DOUBLE, which design.md §3.2 requires of every emitted derivative: {e}"
            ))
        })
    }
}

impl AnalyzerRule for DdxAnalyzer {
    fn name(&self) -> &str {
        "ddx_markers"
    }

    fn analyze(&self, plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan> {
        // Skip the whole walk when the plan contains no marker. Same spirit as
        // ddx-core's parse-free pre-gate (design.md §3.2, F5): a query that
        // never mentions ddx should not be touched, and must not be able to
        // fail inside ddx.
        if !plan_has_marker(&plan)? {
            return Ok(plan);
        }

        let plan = self.rewrite_plan(plan)?;

        // Re-run type coercion over the rewritten plan.
        //
        // `add_analyzer_rule` installs this rule to run AFTER DataFusion's own
        // `TypeCoercion` pass, so the expression we splice in has never been
        // coerced — nothing runs after us. That matters because ddx-core
        // deliberately emits DOUBLE-typed literals and casts (design.md §3.2,
        // F4), so differentiating anything over an integer column yields
        // mixed-type arithmetic: `grad(x / 2, x)` on a BIGINT column produced
        // `Float64 / Int64`, which plans fine and then dies at execution with an
        // Arrow error. Coercing here is what the engine would have done had the
        // derivative been written by hand.
        TypeCoercion::new().analyze(plan, config)
    }
}

impl DdxAnalyzer {
    /// Rewrite every marker in one plan (and in any plan embedded in its
    /// expressions). No pre-gate and no type coercion — [`AnalyzerRule::analyze`]
    /// wraps those around the outermost call.
    fn rewrite_plan(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
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
                // Field *names* can change even when the type doesn't, so the
                // node's schema is rebuilt regardless.
                node.map_data(|plan| plan.recompute_schema())
            } else {
                Ok(node)
            }
        })
        .map(|t| t.data)
    }
}

/// The embedded plan of a subquery expression, if `e` is one.
fn subquery_plan(e: &Expr) -> Option<&Arc<LogicalPlan>> {
    match e {
        Expr::ScalarSubquery(sq) => Some(&sq.subquery),
        Expr::InSubquery(is) => Some(&is.subquery.subquery),
        Expr::Exists(ex) => Some(&ex.subquery.subquery),
        _ => None,
    }
}

/// Put `plan` back into a subquery expression produced by [`subquery_plan`].
fn with_subquery_plan(e: Expr, plan: Arc<LogicalPlan>) -> Expr {
    match e {
        Expr::ScalarSubquery(mut sq) => {
            sq.subquery = plan;
            Expr::ScalarSubquery(sq)
        }
        Expr::InSubquery(mut is) => {
            is.subquery.subquery = plan;
            Expr::InSubquery(is)
        }
        Expr::Exists(mut ex) => {
            ex.subquery.subquery = plan;
            Expr::Exists(ex)
        }
        other => other,
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

/// Does this plan contain a ddx marker anywhere — including inside a plan
/// embedded in one of its expressions?
///
/// The subquery case matters: this is a *gate*, so a false negative here means
/// the rewrite is skipped entirely and the marker survives to execution.
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
                if let Some(sub) = subquery_plan(e) {
                    if plan_has_marker(sub)? {
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
/// **The `wrt` must go through the same unparser as the body.** It is tempting
/// to build the `ColRef` directly from the bound [`Column`]'s `relation`/`name`
/// strings, since the planner already resolved them — but that produces
/// *unquoted* idents, while the body's occurrences of the very same column are
/// unparsed by `Unparser`, whose `DefaultDialect` quotes any identifier
/// containing an uppercase letter (or a keyword). `IdentCasing::FoldUnquoted`
/// then folds the unquoted `wrt` to lowercase and preserves the quoted
/// occurrence's case, they compare unequal, every occurrence classifies as
/// `Match::Not`, and the derivative comes back a silent `0` for any capitalized
/// column — exactly the silently-wrong class design principle 5 exists to
/// prevent, and one that hits any Parquet/CSV schema with capitalized headers.
///
/// Unparsing the column keeps the qualifier the planner resolved (so this path
/// stays binding-aware) *and* guarantees both sides share one quoting rule.
fn wrt_colref(kind: &str, arg: &Expr) -> Result<ColRef> {
    let unparsed = to_sql_ast(arg)?;
    ColRef::from_wrt_arg(kind, &unparsed).map_err(to_df_err)
}
