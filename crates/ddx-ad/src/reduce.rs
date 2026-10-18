// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The reduce primitive's rules: one per aggregate function, read by forward
//! mode ([`crate::tangent`]) and reverse mode ([`crate::transpose`]) alike.
//!
//! JAX defines each primitive's derivative once, in a table every
//! transformation reads. ddx's two modes apply their rules separately, but
//! which rule an aggregate has, and which aggregates have none, is decided
//! here, once, so an aggregate one mode differentiates the other does too:
//!
//! | Aggregate | Rule | Tangent | Transpose |
//! |---|---|---|---|
//! | `SUM` | [`Rule::Sum`] | the same sum, of the tangents | every summed row gets the group's cotangent |
//! | `AVG` | [`Rule::Mean`] | the same mean, of the tangents | the same, over the group's count |
//! | `MAX`, `MIN` | [`Rule::Extreme`] | the mean tangent of the rows attaining it | the rows attaining it share the group's cotangent |
//! | `COUNT` | [`Reduce::Constant`] | 0 | 0 |
//!
//! `SUM` and `AVG` are linear, so each is its own tangent. `MAX` and `MIN`
//! follow `jax.jvp` and `jax.grad` of `jnp.max`, which share a tie evenly.
//!
//! An ordering (`SUM(x ORDER BY y)`) changes none of these values, so it
//! changes no rule. `DISTINCT` changes which rows a sum or a mean counts, so
//! those are refused; an extreme is the same over distinct values, and a
//! count does not move either way.

use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::function_argument::ArgType;
use substrait::proto::{Expression, FunctionArgument};

use crate::error::{AdError, Result};
use crate::functions::{Function, Functions};

/// How a cotangent or tangent passes through an aggregate's argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rule {
    /// `SUM`.
    Sum,
    /// `AVG`: a sum over the group's count.
    Mean,
    /// `MAX` or `MIN` (the function): through the rows attaining it.
    Extreme(Function),
}

/// What an aggregate's rule does with its argument.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Reduce<'e> {
    /// It moves with its argument, `.1`, by the rule.
    Of(Rule, &'e Expression),
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
    let rule = match name {
        "count" => return Ok(Some(Reduce::Constant)),
        "sum" => Rule::Sum,
        "avg" | "mean" => Rule::Mean,
        "max" => Rule::Extreme(Function::Max),
        "min" => Rule::Extreme(Function::Min),
        _ => return Ok(None),
    };
    if invocation == AggregationInvocation::Distinct as i32 && !matches!(rule, Rule::Extreme(_)) {
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
        [arg] => Ok(Some(Reduce::Of(rule, arg))),
        _ => Err(AdError::InvalidPlan(format!(
            "{} with {} arguments",
            name.to_uppercase(),
            args.len()
        ))),
    }
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

    fn rule(name: &str, invocation: AggregationInvocation) -> Result<Option<Rule>> {
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
        assert_eq!(
            rule("max:fp64", Distinct).unwrap(),
            Some(Rule::Extreme(Function::Max))
        );
        assert_eq!(
            rule("min:fp64", Distinct).unwrap(),
            Some(Rule::Extreme(Function::Min))
        );
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
