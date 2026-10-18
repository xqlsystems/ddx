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
use substrait::proto::expression::window_function::Bound;
use substrait::proto::expression::WindowFunction;
use substrait::proto::function_argument::ArgType;
use substrait::proto::{Expression, FunctionArgument};

use crate::error::{AdError, Result};
use crate::functions::{Function, Functions};

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
pub(crate) fn whole_partition(w: &WindowFunction) -> bool {
    let unbounded = |b: &Option<Bound>| {
        b.as_ref()
            .is_none_or(|b| matches!(b.kind, None | Some(Kind::Unbounded(_))))
    };
    unbounded(&w.lower_bound) && unbounded(&w.upper_bound)
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
}
