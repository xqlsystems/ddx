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
//!
//! The claim survived a deliberate attempt to break it, and the reason is worth
//! stating: the obvious attack is two bound columns that unparse to the same
//! text, which needs an unaliased self-join — and DataFusion rejects that as
//! ambiguous during planning, before this rule is ever handed the plan. So the
//! guarantee rests on the planner having already refused the ambiguous cases,
//! not merely on qualifiers being present.

use std::collections::HashSet;
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
use crate::replan::{functions_in, replan, ExprContext};

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
    /// appear in a re-planned derivative.
    ///
    /// The rule re-plans derivatives through its own function registry (it
    /// cannot reach the session's — see [`crate::install_with`]), seeded with
    /// DataFusion's defaults. Two things land outside that seed:
    ///
    /// You rarely need this. A UDF from your *own marker body* is handled
    /// automatically — it is harvested from the bound expression itself, so
    /// `grad(my_udf(y) * x, x)` works with no setup regardless of when `my_udf`
    /// was registered. What lands outside that is a UDF **emitted by a custom
    /// differentiation rule** you registered with [`Ddx::register`]: that
    /// function appears only in the *output*, so it cannot be harvested from the
    /// input and must be declared here.
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
    fn rewrite_expr(
        &self,
        expr: Expr,
        schema: &DFSchema,
        options: &ConfigOptions,
    ) -> Result<Transformed<Expr>> {
        expr.transform_up(|e| {
            let Expr::ScalarFunction(call) = &e else {
                return Ok(Transformed::no(e));
            };
            let Some(kind) = marker_kind(call.func.name()) else {
                return Ok(Transformed::no(e));
            };
            let derivative = self.differentiate_call(kind, &call.args, schema, options)?;
            Ok(Transformed::yes(derivative))
        })
    }

    /// Differentiate one marker call and return the replacement expression.
    fn differentiate_call(
        &self,
        kind: &'static str,
        args: &[Expr],
        schema: &DFSchema,
        options: &ConfigOptions,
    ) -> Result<Expr> {
        let expected = if kind == GRAD { 2 } else { 3 };
        if args.len() != expected {
            return Err(DataFusionError::Plan(format!(
                "ddx: `{kind}` takes {expected} arguments, got {}. \
                 Write `grad(expr, column)` or `jvp(expr, column, tangent)`.",
                args.len()
            )));
        }

        // Reject a correlated outer reference *before* unparsing, so the user
        // gets the real constraint instead of a lie about their schema.
        //
        // `Expr::OuterReferenceColumn` does not survive the bridge: it unparses
        // to an ordinary qualified column, and the derivative is then re-planned
        // against the *inner* node's inputs, where by construction that column
        // is absent. The resulting "No field named t.x" is true about the wrong
        // thing — the column exists, it just isn't reachable from here.
        //
        // Failing loudly is correct (design principle 5); this only fixes what
        // the failure blames. It is structurally loud, incidentally: the planner
        // creates an `OuterReferenceColumn` only when the name does *not*
        // resolve in the inner scope, so the unparsed text cannot silently
        // rebind to an inner column of the same name.
        if let Some(outer) = outer_reference_in(&args[..expected.min(args.len())]) {
            return Err(DataFusionError::Plan(format!(
                "ddx: this `{kind}` marker is inside a correlated subquery and references \
                 the outer column `{outer}`. Path B's bridge re-plans the derivative against \
                 the subquery's own inputs, so an outer reference cannot be carried through \
                 it.\n\n\
                 Rewrite the SQL text instead with `ddx_datafusion::ddx_sql(&ctx, sql)` \
                 (design.md §3.3 Path A), which differentiates before planning and is not \
                 subject to this limit."
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

        // Functions the user called are harvested from the marker's own body:
        // anything that survives differentiation was necessarily in there, and
        // a bound Expr carries the ScalarUDF itself. That is why a session UDF
        // works here without the analyzer ever reaching the session registry.
        let local = functions_in(args);
        let replanned = replan(&self.exprs, options, local, derivative, schema)?;

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

        let plan = self.rewrite_plan(plan, config)?;

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
    fn rewrite_plan(&self, plan: LogicalPlan, options: &ConfigOptions) -> Result<LogicalPlan> {
        // `transform_up_with_subqueries`, not `transform_up`: a subquery carries
        // its own `LogicalPlan` inside an *expression*, and the plain walk
        // visits only direct relational inputs. DataFusion already knows every
        // expression variant that can carry one, so this delegates rather than
        // re-deriving the list — a hand-rolled match over an upstream enum has
        // to be re-audited on every bump, and ours was already one arm short
        // (`Expr::SetComparison`, i.e. ANY/ALL/SOME, reached execution).
        plan.transform_up_with_subqueries(|node| {
            // A node's expressions are resolved against its *inputs*, not its
            // own output schema — that is what binds the derivative's columns
            // to the same columns the original expression used. Leaf nodes
            // (which have no inputs) fall back to their own schema.
            let schema = merged_input_schema(&node)?;

            // Which names this node publishes to its parents, read off the
            // node's own schema *before* the rewrite.
            //
            // A rewritten expression must keep its name only where a parent can
            // refer to it by that name: an unaliased `AVG(grad(x*x, x))` derives
            // the field `avg(grad(t.x * t.x,t.x))`, and the projection above it
            // refers to exactly that string, so renaming it underneath leaves
            // the parent dangling. Predicates, sort keys, and join conditions
            // name nothing, and aliasing those would be at best noise and at
            // worst harmful (an alias around a join key can defeat equijoin
            // recognition).
            //
            // This asks the plan rather than matching on node variants. The
            // variant list was `Projection | Aggregate | Window` and was missing
            // `Distinct::On`, which also derives its schema from its
            // expressions — the same staleness that bit the subquery walk above.
            // A schema lookup cannot go stale, because any node that publishes a
            // name necessarily has it in its schema.
            let published: HashSet<String> = node
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();

            let mut rewrote = false;
            let node = node.map_expressions(|expr| {
                let original_name = expr.schema_name().to_string();
                let out = self.rewrite_expr(expr, &schema, options)?;
                rewrote |= out.transformed;
                Ok(out.update_data(|e| {
                    if published.contains(&original_name)
                        && e.schema_name().to_string() != original_name
                    {
                        e.alias(original_name)
                    } else {
                        e
                    }
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
/// This is a *gate*: a false negative skips the rewrite for the whole plan and
/// the marker survives to execution. So it walks with `apply_with_subqueries`,
/// the mirror of the `transform_up_with_subqueries` used for the rewrite — the
/// two must agree on what "anywhere" means, and delegating to DataFusion is the
/// only way to keep them agreeing across upstream changes.
fn plan_has_marker(plan: &LogicalPlan) -> Result<bool> {
    let mut found = false;
    plan.apply_with_subqueries(|node| {
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

/// The first correlated outer reference anywhere in `args`, if there is one.
fn outer_reference_in(args: &[Expr]) -> Option<String> {
    let mut found = None;
    for arg in args {
        let _ = arg.apply(|e| {
            if let Expr::OuterReferenceColumn(_, col) = e {
                found = Some(col.flat_name());
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        });
        if found.is_some() {
            break;
        }
    }
    found
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
