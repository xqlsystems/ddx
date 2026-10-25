// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The reduce primitive's derivative: one linear form per aggregate function,
//! which forward mode ([`crate::tangent`]) renders as a tangent and reverse
//! mode ([`crate::transpose`]) as its transpose.
//!
//! JAX defines a linear primitive once (`deflinear`), and that one definition
//! gives both its tangent and its transpose. Every reduction ddx
//! differentiates has a tangent linear in its argument's tangents `ṫ`:
//!
//! ```text
//! tangent(group) = Σ over the group's rows of coef(row) · ṫ(row)
//! coef(row)      = counts(row) / (shared ? number of rows that count : 1)
//! ```
//!
//! where a row counts if its argument is not NULL (an aggregate skips NULL)
//! and, for an extreme, if it attains the extreme. [`Linear`] holds the two
//! choices:
//!
//! | Aggregate | Rows that count | Shared | `coef` |
//! |---|---|---|---|
//! | `SUM` | every row | no | 1 |
//! | `AVG` | every row | yes | 1 / count |
//! | `MAX`, `MIN` | the rows attaining it | yes | 1 / number attaining, on those rows |
//! | `COUNT` | [`Reduce::Constant`]: no tangent | | 0 |
//!
//! Forward mode renders the form as an aggregate over the group: `SUM` of
//! the counted tangents, or `AVG` of them when they are shared. Reverse mode
//! renders its transpose: each row's cotangent is the group's cotangent
//! times `coef`. A window function over a whole partition is the same
//! reduction broadcast back to the partition's rows, so the same form gives
//! its tangent and its transpose (the group's cotangent is then the sum of
//! its rows' cotangents).
//!
//! `MAX` and `MIN` follow `jax.jvp` and `jax.grad` of `jnp.max`, which share
//! a tie evenly. An ordering (`SUM(x ORDER BY y)`) changes none of these
//! values, so it changes no form. `DISTINCT` changes which rows a sum or a
//! mean counts, so those are refused; an extreme is the same over distinct
//! values, and a count does not move either way.

use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::expression::window_function::bound::Kind;
use substrait::proto::expression::window_function::{Bound, BoundsType};
use substrait::proto::expression::WindowFunction;
use substrait::proto::function_argument::ArgType;
use substrait::proto::{Expression, FunctionArgument};

use crate::error::{AdError, Result};
use crate::expr::{call, lit_f64};
use crate::functions::{Extensions, Function, Functions};

/// A reduction's tangent as a linear form in its argument's tangents (see
/// the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Linear {
    /// `Some(MAX or MIN)` if only the rows attaining the group's extreme
    /// count; `None` if every row does.
    pub extreme: Option<Function>,
    /// Whether the rows that count share the tangent evenly (`AVG`, and an
    /// extreme at a tie) rather than each passing it whole (`SUM`).
    pub shared: bool,
}

/// What an aggregate's rule does with its argument.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Reduce<'e> {
    /// It moves with its argument, `.1`, by the linear form `.0`.
    Of(Linear, &'e Expression),
    /// It does not move when its argument does (`COUNT`).
    Constant,
}

/// The rule of the aggregate function `anchor` names, called with
/// `arguments` and `invocation` (as an aggregate or a window function
/// carries them); `None` if it has no rule, which each mode refuses in its
/// own words ([`no_rule`] for an aggregate).
pub(crate) fn reduce_rule<'e>(
    functions: &Functions,
    anchor: u32,
    invocation: i32,
    arguments: &'e [FunctionArgument],
) -> Result<Option<Reduce<'e>>> {
    let name = functions.name(anchor)?;
    let (extreme, shared) = match name {
        "count" => return Ok(Some(Reduce::Constant)),
        "sum" => (None, false),
        "avg" | "mean" => (None, true),
        "max" => (Some(Function::Max), true),
        "min" => (Some(Function::Min), true),
        _ => return Ok(None),
    };
    if invocation == AggregationInvocation::Distinct as i32 && extreme.is_none() {
        return Err(AdError::NotImplemented(format!(
            "{}(DISTINCT …) over a value that depends on a wrt column",
            name.to_uppercase()
        )));
    }
    let args: Vec<&Expression> = arguments
        .iter()
        .filter_map(|a| match &a.arg_type {
            Some(ArgType::Value(e)) => Some(e),
            _ => None,
        })
        .collect();
    match args.as_slice() {
        [arg] => Ok(Some(Reduce::Of(Linear { extreme, shared }, arg))),
        _ => Err(AdError::InvalidPlan(format!(
            "{} with {} arguments",
            name.to_uppercase(),
            args.len()
        ))),
    }
}

/// Does window function `w` read each whole partition, with no frame
/// narrower than it (an unset bound is the partition's end)? Then it is its
/// reduction over the partition, broadcast back to the partition's rows,
/// and its value does not depend on how a partition's rows are ordered.
/// A `RANGE` frame with no `ORDER BY` makes every row of a partition a peer
/// of the current one, so `CURRENT ROW` is its end too: DuckDB writes
/// `OVER ()` as `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.
pub(crate) fn whole_partition(w: &WindowFunction) -> bool {
    let peers = w.sorts.is_empty() && w.bounds_type == BoundsType::Range as i32;
    let unbounded = |b: &Option<Bound>| {
        b.as_ref().is_none_or(|b| match b.kind {
            None | Some(Kind::Unbounded(_)) => true,
            Some(Kind::CurrentRow(_)) => peers,
            _ => false,
        })
    };
    unbounded(&w.lower_bound) && unbounded(&w.upper_bound)
}

/// Does a row's argument `arg` attain its group's extreme `extreme`? Equal
/// to it, or, when `tolerant` and the extreme is finite, within 8 ulps of
/// it. An argument computed by a sum or a mean (`tolerant`) can differ in
/// its last bits from run to run, since a grouped sum over several
/// partitions adds in arrival order, so two groups that tie in exact
/// arithmetic would give the whole tangent to whichever rounded higher on
/// that run. Within the tolerance they share it, as at an exact tie, the
/// same way every run. An argument computed from table values alone is the
/// same every run, and two of its values a few ulps apart do not tie
/// (MAX(1, 1 + 2 ulps) has gradient (0, 1), as jax.grad gives).
///
/// Forward and reverse mode both count rows with this test, so `jvp` along
/// `t` is ⟨∇L, t⟩ at a near-tie too (#179).
pub(crate) fn attains(
    ext: &mut Extensions,
    arg: Expression,
    extreme: Expression,
    tolerant: bool,
) -> Expression {
    let equal = ext.anchor(Function::Equal);
    if !tolerant {
        return call(equal, vec![arg, extreme]);
    }
    let or = ext.anchor(Function::Or);
    let and = ext.anchor(Function::And);
    let lte = ext.anchor(Function::Lte);
    let abs = ext.anchor(Function::Abs);
    let subtract = ext.anchor(Function::Subtract);
    let multiply = ext.anchor(Function::Multiply);
    let size = call(abs, vec![extreme.clone()]);
    let finite = call(lte, vec![size.clone(), lit_f64(f64::MAX)]);
    let gap = call(
        abs,
        vec![call(subtract, vec![arg.clone(), extreme.clone()])],
    );
    let close = call(
        lte,
        vec![gap, call(multiply, vec![lit_f64(8.0 * f64::EPSILON), size])],
    );
    call(
        or,
        vec![
            call(equal, vec![arg, extreme]),
            call(and, vec![finite, close]),
        ],
    )
}

/// The refusal of an aggregate `name` with no rule.
pub(crate) fn no_rule(name: &str) -> AdError {
    AdError::NotImplemented(format!(
        "the aggregate `{name}` over a value that depends on a wrt column; ddx has rules for \
         SUM, AVG, MAX, MIN and COUNT"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::field;
    use crate::functions::tests::plan_declaring;

    fn rule(name: &str, invocation: AggregationInvocation) -> Result<Option<Linear>> {
        let functions = Functions::from_plan(&plan_declaring(&[(0, name)])).unwrap();
        let args = [FunctionArgument {
            arg_type: Some(ArgType::Value(field(0))),
        }];
        Ok(
            match reduce_rule(&functions, 0, invocation as i32, &args)? {
                Some(Reduce::Of(rule, _)) => Some(rule),
                Some(Reduce::Constant) | None => None,
            },
        )
    }

    #[test]
    fn distinct_changes_a_sum_or_a_mean_and_neither_an_extreme_nor_a_count() {
        // DataFusion plans a DISTINCT aggregate as a GROUP BY of its argument,
        // so only a producer that writes the DISTINCT invocation reaches this.
        use AggregationInvocation::{All, Distinct};
        for name in ["sum:fp64", "avg:fp64"] {
            assert!(rule(name, Distinct).is_err(), "{name}");
            assert!(rule(name, All).unwrap().is_some(), "{name}");
        }
        for (name, extreme) in [("max:fp64", Function::Max), ("min:fp64", Function::Min)] {
            let form = rule(name, Distinct)
                .unwrap()
                .expect("an extreme has a rule");
            assert_eq!(form.extreme, Some(extreme), "{name}");
        }
        let functions = Functions::from_plan(&plan_declaring(&[(0, "count")])).unwrap();
        assert!(matches!(
            reduce_rule(&functions, 0, Distinct as i32, &[]),
            Ok(Some(Reduce::Constant))
        ));
    }

    #[test]
    fn an_aggregate_with_no_rule_has_none() {
        assert_eq!(rule("stddev", AggregationInvocation::All).unwrap(), None);
    }

    #[test]
    fn a_range_frame_with_no_order_by_is_its_whole_partition() {
        use substrait::proto::expression::window_function::bound::{CurrentRow, Preceding};
        use substrait::proto::SortField;
        let bound = |kind| Some(Bound { kind: Some(kind) });
        let current = || bound(Kind::CurrentRow(CurrentRow {}));
        let frame = |bounds_type: BoundsType, upper, sorted: bool| WindowFunction {
            bounds_type: bounds_type as i32,
            upper_bound: upper,
            sorts: if sorted {
                vec![SortField::default()]
            } else {
                vec![]
            },
            ..Default::default()
        };
        // DuckDB's OVER (): RANGE up to the current row, which every row of
        // the partition is a peer of.
        assert!(whole_partition(&frame(BoundsType::Range, current(), false)));
        assert!(whole_partition(&frame(BoundsType::Rows, None, false)));
        // A running frame, and a ROWS frame up to the row itself, are not.
        assert!(!whole_partition(&frame(BoundsType::Range, current(), true)));
        assert!(!whole_partition(&frame(BoundsType::Rows, current(), false)));
        let one_back = bound(Kind::Preceding(Preceding { offset: 1 }));
        assert!(!whole_partition(&frame(BoundsType::Range, one_back, false)));
    }
}
