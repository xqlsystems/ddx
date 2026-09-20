// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Everything the transpose rules need to know about a plan before emitting a
//! backward step: its nodes, its columns, which columns carry gradient, and what
//! each gradient-carrying aggregate was tagged as.
//!
//! This is also where *tag, don't infer* (design.md §2, principle 3) is
//! enforced. A marker must sit where its meaning is unambiguous, and an
//! aggregate that carries gradient must say what it is — an untagged `SUM` over
//! a gradient-carrying column is an error, not a guess.

use std::collections::HashMap;

use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::aggregate_rel::Measure;
use substrait::proto::fetch_rel::{CountMode, OffsetMode};
use substrait::proto::rel::RelType;
use substrait::proto::{Expression, Plan, Rel};

use crate::activity::{Activity, ColumnRef};
use crate::columns::{Col, ColumnDef, Columns};
use crate::error::{AdError, Result};
use crate::expr::{value_args, walk, Event};
use crate::index::{NodeId, PlanIndex};
use crate::markers::{Functions, Marker};
use crate::names::AggKind;

/// A plan, analysed with respect to a set of parameters.
#[derive(Debug, Clone)]
pub struct Analysis<'a> {
    pub index: PlanIndex<'a>,
    pub functions: Functions,
    pub columns: Columns<'a>,
    pub activity: Activity,
    measures: HashMap<(NodeId, Col), Marker>,
}

impl<'a> Analysis<'a> {
    /// Analyse `plan` with respect to `wrt`.
    pub fn new(plan: &'a Plan, wrt: &[ColumnRef]) -> Result<Self> {
        let index = PlanIndex::build(plan)?;
        let functions = Functions::from_plan(plan)?;
        let columns = Columns::build(&index)?;
        check_marker_placement(&index, &functions)?;
        let activity = Activity::analyze(&index, &columns, &functions, wrt)?;
        let measures = classify_measures(&index, &columns, &functions, &activity)?;
        Ok(Analysis {
            index,
            functions,
            columns,
            activity,
            measures,
        })
    }

    /// For an aggregate output column that carries gradient, the marker it was
    /// tagged with: [`Marker::Contraction`] or [`Marker::Reduce`]. `None` for a
    /// column that isn't a gradient-carrying measure.
    pub fn measure_marker(&self, node: NodeId, col: Col) -> Option<Marker> {
        self.measures.get(&(node, col)).copied()
    }
}

/// Where in a relation an expression sits — which decides which markers may
/// appear at its root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// The value argument of an aggregate measure.
    MeasureArg,
    /// A projected value.
    Projected,
    /// Anywhere else: conditions, keys, sort fields.
    Other,
}

impl Slot {
    /// How to describe this slot to someone whose SQL didn't obviously put a
    /// marker here. The plan is what ddx sees, and an engine may move a marker
    /// somewhere the SQL didn't suggest: DataFusion plans `SUM(DISTINCT x)` as a
    /// *grouping* on `x` feeding an outer sum, so a marker written inside that
    /// SUM arrives in a grouping key.
    fn describe(self) -> &'static str {
        match self {
            Slot::MeasureArg => "an aggregate's argument",
            Slot::Projected => "a projected value",
            Slot::Other => "a condition, grouping key or sort key",
        }
    }
}

fn check_marker_placement(index: &PlanIndex<'_>, fns: &Functions) -> Result<()> {
    for node in index.nodes() {
        for (e, slot) in slotted_expressions(node.rel) {
            walk(e, fns, &mut |ev| {
                let Event::Marker {
                    marker,
                    call,
                    at_root,
                } = ev
                else {
                    return Ok(());
                };
                #[allow(deprecated)]
                let arity = value_args(&call.arguments).len() + call.args.len();
                if arity != 1 {
                    return Err(AdError::InvalidMarker(format!(
                        "`{}` takes exactly one argument, got {arity}",
                        marker.name()
                    )));
                }
                let placed = match marker {
                    Marker::Contraction | Marker::Reduce => at_root && slot == Slot::MeasureArg,
                    Marker::Route => at_root && slot == Slot::Projected,
                    Marker::StopGradient => true,
                };
                if placed {
                    return Ok(());
                }
                let wanted = match marker {
                    Marker::Contraction => {
                        "the whole argument of a SUM, as in \
                                            SUM(ddx_contract_mark(a.val * b.val))"
                    }
                    Marker::Reduce => {
                        "the whole argument of a SUM, as in \
                                       SUM(ddx_reduce_mark(val))"
                    }
                    _ => "a whole projected value, as in SELECT ddx_route_mark(val) AS val",
                };
                let found = if at_root {
                    format!("in {}", slot.describe())
                } else {
                    format!("nested inside a larger expression in {}", slot.describe())
                };
                Err(AdError::InvalidMarker(format!(
                    "`{}` must be {wanted}; the plan has it {found}",
                    marker.name()
                )))
            })?;
        }
    }
    Ok(())
}

/// Every expression a relation holds, with the slot it sits in.
fn slotted_expressions<'a>(rel: &'a Rel) -> Vec<(&'a Expression, Slot)> {
    use Slot::*;
    let mut out = Vec::new();
    let mut push = |e: Option<&'a Expression>, s| {
        if let Some(e) = e {
            out.push((e, s));
        }
    };
    match rel.rel_type.as_ref() {
        Some(RelType::Read(r)) => {
            push(r.filter.as_deref(), Other);
            push(r.best_effort_filter.as_deref(), Other);
        }
        Some(RelType::Filter(f)) => push(f.condition.as_deref(), Other),
        Some(RelType::Project(p)) => p.expressions.iter().for_each(|e| push(Some(e), Projected)),
        Some(RelType::Join(j)) => {
            push(j.expression.as_deref(), Other);
            push(j.post_join_filter.as_deref(), Other);
        }
        Some(RelType::Aggregate(a)) => {
            #[allow(deprecated)]
            for g in &a.groupings {
                g.grouping_expressions
                    .iter()
                    .for_each(|e| push(Some(e), Other));
            }
            a.grouping_expressions
                .iter()
                .for_each(|e| push(Some(e), Other));
            for m in &a.measures {
                push(m.filter.as_ref(), Other);
                if let Some(f) = &m.measure {
                    value_args(&f.arguments)
                        .into_iter()
                        .for_each(|e| push(Some(e), MeasureArg));
                    f.sorts.iter().for_each(|s| push(s.expr.as_ref(), Other));
                }
            }
        }
        Some(RelType::Window(w)) => {
            for f in &w.window_functions {
                value_args(&f.arguments)
                    .into_iter()
                    .for_each(|e| push(Some(e), Other));
            }
            w.partition_expressions
                .iter()
                .for_each(|e| push(Some(e), Other));
            w.sorts.iter().for_each(|s| push(s.expr.as_ref(), Other));
        }
        Some(RelType::Sort(s)) => s.sorts.iter().for_each(|s| push(s.expr.as_ref(), Other)),
        Some(RelType::Fetch(f)) => {
            if let Some(OffsetMode::OffsetExpr(e)) = &f.offset_mode {
                push(Some(e), Other);
            }
            if let Some(CountMode::CountExpr(e)) = &f.count_mode {
                push(Some(e), Other);
            }
        }
        _ => {}
    }
    out
}

/// Tag every gradient-carrying measure, refusing the ones that aren't tagged.
fn classify_measures(
    index: &PlanIndex<'_>,
    columns: &Columns<'_>,
    fns: &Functions,
    activity: &Activity,
) -> Result<HashMap<(NodeId, Col), Marker>> {
    let mut out = HashMap::new();
    for node in index.nodes() {
        for c in columns.cols(node.id) {
            if let ColumnDef::Measure(m) = columns.def(node.id, c)? {
                if activity.is_active(node.id, c) {
                    out.insert((node.id, c), measure_marker(m, fns)?);
                }
            }
        }
    }
    Ok(out)
}

fn measure_marker(m: &Measure, fns: &Functions) -> Result<Marker> {
    let f = m
        .measure
        .as_ref()
        .ok_or_else(|| AdError::InvalidPlan("an aggregate measure has no function".into()))?;
    let name = fns.base_name(f.function_reference)?;
    let agg = AggKind::from_name(&name);
    let marker = match value_args(&f.arguments).as_slice() {
        [arg] => match &arg.rex_type {
            Some(substrait::proto::expression::RexType::ScalarFunction(call)) => {
                fns.marker(call.function_reference)?
            }
            _ => None,
        },
        _ => None,
    };
    match (agg, marker) {
        (Some(AggKind::Sum), Some(mk @ (Marker::Contraction | Marker::Reduce))) => {
            if f.invocation() == AggregationInvocation::Distinct {
                return Err(AdError::NotImplemented(
                    "SUM(DISTINCT …) over a gradient-carrying column".into(),
                ));
            }
            if m.filter.is_some() {
                return Err(AdError::NotImplemented(
                    "an aggregate FILTER clause on a gradient-carrying measure".into(),
                ));
            }
            Ok(mk)
        }
        (Some(AggKind::Mean), Some(mk @ (Marker::Contraction | Marker::Reduce))) => {
            Err(AdError::InvalidMarker(format!(
                "`{}` must be summed, but it sits inside `{name}`; for a mean, SUM and then \
                 divide by the count as a separate elementwise step (design.md §4.3)",
                mk.name()
            )))
        }
        (_, Some(mk @ (Marker::Contraction | Marker::Reduce))) => {
            Err(AdError::InvalidMarker(format!(
                "`{}` must be the whole argument of a SUM, but it sits inside `{name}`",
                mk.name()
            )))
        }
        (Some(AggKind::Sum), _) => Err(AdError::Untagged(
            "a SUM over a gradient-carrying column must say what it is: \
             SUM(ddx_contract_mark(a.val * b.val)) for a contraction of a join, or \
             SUM(ddx_reduce_mark(val)) for a reduction"
                .into(),
        )),
        (Some(AggKind::Extremum), _) => Err(AdError::Untagged(format!(
            "`{name}` over a gradient-carrying column has no transpose rule. If it is a \
             numerical-stability shift (softmax's max), wrap its use in ddx_stop_gradient; \
             for argmax routing, use the ddx_route_mark idiom"
        ))),
        _ => Err(AdError::NotImplemented(format!(
            "aggregate `{name}` over a gradient-carrying column"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::TableRef;
    use crate::test_plans::*;

    const SUM: u32 = 0;
    const MUL: u32 = 1;
    const CONTRACT: u32 = 2;
    const REDUCE: u32 = 3;
    const STOP: u32 = 4;
    const MAX: u32 = 5;
    const TANH: u32 = 6;
    const ROUTE: u32 = 7;
    const AVG: u32 = 8;

    fn with_fns(root: Rel) -> Plan {
        plan_with_functions(
            root,
            &[
                (SUM, "sum"),
                (MUL, "multiply"),
                (CONTRACT, "ddx_contract_mark"),
                (REDUCE, "ddx_reduce_mark"),
                (STOP, "ddx_stop_gradient"),
                (MAX, "max"),
                (TANH, "tanh"),
                (ROUTE, "ddx_route_mark"),
                (AVG, "avg"),
            ],
        )
    }

    /// The error analysing `root` with respect to `wrt` produces.
    fn refusal(root: Rel, wrt: &[ColumnRef]) -> AdError {
        let p = with_fns(root);
        Analysis::new(&p, wrt).map(|_| ()).unwrap_err()
    }

    fn wrt(t: &str, c: &str) -> Vec<ColumnRef> {
        vec![ColumnRef::new(TableRef::new([t]), c)]
    }

    /// x(i, j, val) ⋈ w(j, k, val), contracted over j:
    /// SELECT i, k, SUM(ddx_contract_mark(x.val * w.val)) GROUP BY i, k
    fn contraction(measure_fn: u32, arg: Expression) -> Rel {
        let j = join(read("x", &["i", "j", "val"]), read("w", &["j", "k", "val"]));
        aggregate(j, &[field(0), field(4)], &[(measure_fn, vec![arg])])
    }

    fn mul_vals() -> Expression {
        call(MUL, vec![field(2), field(5)])
    }

    #[test]
    fn a_contraction_is_tagged_and_its_operands_are_active() {
        let p = with_fns(contraction(SUM, call(CONTRACT, vec![mul_vals()])));
        let a = Analysis::new(&p, &wrt("w", "val")).unwrap();
        let root = a.index.root();
        assert_eq!(a.activity.active(root), [Col::new(2)]);
        assert_eq!(
            a.measure_marker(root, Col::new(2)),
            Some(Marker::Contraction)
        );
        // The join passes both `val`s through, but only w's depends on w.
        assert_eq!(a.activity.active(NodeId::new(1)), [Col::new(5)]);
        // x.val is varied by nothing; w's keys are inactive.
        assert_eq!(a.activity.active(NodeId::new(3)), [Col::new(2)]);
        assert!(a.activity.active(NodeId::new(2)).is_empty());
    }

    #[test]
    fn an_untagged_sum_over_a_parameter_is_refused() {
        let p = with_fns(contraction(SUM, mul_vals()));
        let err = Analysis::new(&p, &wrt("w", "val")).unwrap_err();
        assert!(matches!(err, AdError::Untagged(ref m) if m.contains("must say what it is")));
    }

    #[test]
    fn an_untagged_sum_over_data_alone_is_fine() {
        // SUM(x.val) is untagged, but no parameter reaches it.
        let j = join(read("x", &["i", "j", "val"]), read("w", &["j", "k", "val"]));
        let agg = aggregate(
            j,
            &[field(0)],
            &[
                (SUM, vec![field(2)]),
                (SUM, vec![call(REDUCE, vec![field(5)])]),
            ],
        );
        let p = with_fns(agg);
        let a = Analysis::new(&p, &wrt("w", "val")).unwrap();
        assert_eq!(a.activity.active(a.index.root()), [Col::new(2)]);
        assert_eq!(a.measure_marker(a.index.root(), Col::new(1)), None);
        assert_eq!(
            a.measure_marker(a.index.root(), Col::new(2)),
            Some(Marker::Reduce)
        );
    }

    #[test]
    fn a_marker_must_sit_where_its_meaning_is_unambiguous() {
        // SUM(2 * ddx_contract_mark(…)) — the marker is not the whole argument.
        let nested = call(MUL, vec![lit_f64(2.0), call(CONTRACT, vec![mul_vals()])]);
        let err = refusal(contraction(SUM, nested), &wrt("w", "val"));
        assert!(matches!(err, AdError::InvalidMarker(_)), "{err:?}");

        // A contraction marker in a projection.
        let pr = project(read("w", &["val"]), vec![call(CONTRACT, vec![field(0)])]);
        let err = refusal(pr, &wrt("w", "val"));
        assert!(matches!(err, AdError::InvalidMarker(_)), "{err:?}");

        // A route marker inside a measure.
        let err = refusal(
            contraction(SUM, call(ROUTE, vec![mul_vals()])),
            &wrt("w", "val"),
        );
        assert!(matches!(err, AdError::InvalidMarker(_)), "{err:?}");

        // Two arguments.
        let err = refusal(
            contraction(SUM, call(CONTRACT, vec![field(2), field(5)])),
            &wrt("w", "val"),
        );
        assert!(
            matches!(err, AdError::InvalidMarker(ref m) if m.contains("exactly one")),
            "{err:?}"
        );

        // AVG instead of SUM.
        let err = refusal(
            contraction(AVG, call(CONTRACT, vec![mul_vals()])),
            &wrt("w", "val"),
        );
        assert!(
            matches!(err, AdError::InvalidMarker(ref m) if m.contains("must be summed")),
            "{err:?}"
        );
    }

    /// Softmax's shift: `exp(z - max(z))` with the max stopped. The max is
    /// varied but not useful, so it needs no rule.
    #[test]
    fn stop_gradient_cuts_a_max_out_of_the_backward_pass() {
        // m = SELECT i, MAX(z) FROM t GROUP BY i ; SELECT tanh(t.z - stop(m.m)) FROM t JOIN m
        let t = || read("t", &["i", "z"]);
        let m = aggregate(t(), &[field(0)], &[(MAX, vec![field(1)])]);
        let shifted = |sub: Expression| {
            // A project also passes its input through; emit only the new value.
            let pr = project(
                join(t(), m.clone()),
                vec![call(TANH, vec![call(MUL, vec![field(1), sub])])],
            );
            emit(pr, &[4])
        };
        let ok = with_fns(shifted(call(STOP, vec![field(3)])));
        let a = Analysis::new(&ok, &wrt("t", "z")).unwrap();
        // The max is varied but never reached through a value position.
        let max_node = NodeId::new(3);
        assert!(a.activity.is_varied(max_node, Col::new(1)));
        assert!(!a.activity.is_active(max_node, Col::new(1)));

        // Without the stop, the MAX is active and has no rule: refused, loudly.
        let bad = with_fns(shifted(field(3)));
        let err = Analysis::new(&bad, &wrt("t", "z")).unwrap_err();
        assert!(matches!(err, AdError::Untagged(ref m) if m.contains("ddx_stop_gradient")));
    }

    #[test]
    fn a_column_used_only_as_a_key_is_not_active() {
        // SELECT val FROM w WHERE k = … — `k` is read by the filter, never a value.
        let f = filter(read("w", &["k", "val"]), field(0));
        let p = with_fns(f);
        let a = Analysis::new(&p, &wrt("w", "val")).unwrap();
        assert_eq!(a.activity.active(a.index.root()), [Col::new(1)]);
    }

    #[test]
    fn grouping_by_a_parameter_is_refused() {
        let agg = aggregate(read("w", &["val"]), &[field(0)], &[]);
        let err = refusal(agg, &wrt("w", "val"));
        assert!(matches!(err, AdError::NotImplemented(ref m) if m.contains("grouping key")));
    }

    #[test]
    fn wrt_must_describe_the_plan() {
        let p = with_fns(contraction(SUM, call(CONTRACT, vec![mul_vals()])));
        let err = Analysis::new(&p, &wrt("weights", "val")).unwrap_err();
        assert!(
            matches!(err, AdError::InvalidWrt(ref m) if m.contains("never reads table `weights`") && m.contains("w, x")),
            "{err}"
        );
        let err = Analysis::new(&p, &wrt("w", "vals")).unwrap_err();
        assert!(
            matches!(err, AdError::InvalidWrt(ref m) if m.contains("no column `vals`") && m.contains("j, k, val")),
            "{err}"
        );
        let err = Analysis::new(&p, &[]).unwrap_err();
        assert!(matches!(err, AdError::InvalidWrt(_)), "{err}");
    }

    #[test]
    fn an_output_independent_of_every_parameter_is_refused() {
        // SELECT val FROM w WHERE k … — `w.k` is a real column, but only filters.
        let p = with_fns(emit(filter(read("w", &["k", "val"]), field(0)), &[1]));
        let err = Analysis::new(&p, &wrt("w", "k")).unwrap_err();
        assert!(matches!(err, AdError::InvalidWrt(ref m) if m.contains("doesn't depend")));
    }
}
