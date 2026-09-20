// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A walk over one Substrait expression, reporting the two things analysis
//! needs from it: which input fields it reads, and where the ddx markers are.
//!
//! Each read is reported with the kind of position it sits in, because the two
//! halves of activity analysis need different answers (see [`Reads`]):
//!
//! - a **value** position contributes to the expression's value, so a cotangent
//!   can flow back into it;
//! - a **control** position only selects or orders rows — a `CASE` condition, a
//!   window's `PARTITION BY`/`ORDER BY`. The result is piecewise constant in
//!   such a read, so its derivative is zero almost everywhere.
//!
//! Anything the walk can't see into — a subquery, a lambda, a nested-field
//! reference — is [`AdError::NotImplemented`] rather than "reads nothing": a
//! dependency the walk missed is a gradient it would silently drop.

use substrait::proto::consistent_partition_window_rel::WindowRelFunction;
use substrait::proto::expression::field_reference::{ReferenceType, RootType};
use substrait::proto::expression::reference_segment::ReferenceType as Segment;
use substrait::proto::expression::{FieldReference, RexType, ScalarFunction};
use substrait::proto::function_argument::ArgType;
use substrait::proto::{AggregateFunction, Expression, FunctionArgument, SortField};

use crate::columns::Field;
use crate::error::{AdError, Result};
use crate::index::MAX_DEPTH;
use crate::markers::{Functions, Marker};

/// Which reads a caller wants back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reads {
    /// Only value positions: where a cotangent can flow. This is what *useful*
    /// is defined as (design.md §4.4) — reaching the output "not only through a
    /// join condition, filter, grouping key or sort".
    Value,
    /// Value *and* control positions: anything the expression's result could
    /// depend on. This is the *varied* side, which deliberately
    /// over-approximates.
    Varied,
}

impl Reads {
    fn wants(self, control: bool) -> bool {
        self == Reads::Varied || !control
    }
}

/// Something the walk found.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Event<'a> {
    /// A read of a position in the node's input row.
    Read {
        field: Field,
        /// Inside a `ddx_stop_gradient` call, where no cotangent flows.
        stopped: bool,
        /// In a position that only selects or orders rows and never contributes
        /// to the value: a `CASE` condition, a window `PARTITION BY`/`ORDER BY`.
        control: bool,
    },
    /// A call to a marker. `at_root` is true when the call is the whole
    /// expression walked, not a sub-expression of it.
    Marker {
        marker: Marker,
        call: &'a ScalarFunction,
        at_root: bool,
    },
}

/// Walk `e`, calling `f` for every field read and marker call in it.
pub(crate) fn walk<'a>(
    e: &'a Expression,
    fns: &Functions,
    f: &mut impl FnMut(Event<'a>) -> Result<()>,
) -> Result<()> {
    Walker { fns, f }.expr(e, Where::default(), 0)
}

/// The input-row fields `e` reads, outside any `ddx_stop_gradient`, in the
/// positions `which` asks for.
pub(crate) fn refs(e: &Expression, fns: &Functions, which: Reads) -> Result<Vec<Field>> {
    let mut out = Vec::new();
    walk(e, fns, &mut |ev| {
        if let Event::Read {
            field,
            stopped: false,
            control,
        } = ev
        {
            if which.wants(control) {
                out.push(field);
            }
        }
        Ok(())
    })?;
    Ok(out)
}

/// [`refs`] over each value argument of a function call.
pub(crate) fn args_refs(
    args: &[FunctionArgument],
    fns: &Functions,
    which: Reads,
) -> Result<Vec<Field>> {
    let mut out = Vec::new();
    for e in value_args(args) {
        out.extend(refs(e, fns, which)?);
    }
    Ok(out)
}

/// Every value expression an aggregate function reads.
///
/// Both argument forms, because `arguments` superseded `args` without removing
/// it: a producer still emitting the deprecated field would otherwise look like
/// an aggregate over nothing — not varied, not active, and so never checked for
/// a tag, which is a silent zero gradient rather than a refusal. The deprecated
/// forms of a scalar call, a window call and a grouping are handled the same way.
pub(crate) fn agg_value_exprs(f: &AggregateFunction) -> Vec<&Expression> {
    #[allow(deprecated)]
    value_args(&f.arguments)
        .into_iter()
        .chain(f.args.iter())
        .collect()
}

/// The input-row fields an aggregate function reads.
pub(crate) fn agg_refs(f: &AggregateFunction, fns: &Functions, which: Reads) -> Result<Vec<Field>> {
    let mut out = Vec::new();
    for e in agg_value_exprs(f) {
        out.extend(refs(e, fns, which)?);
    }
    Ok(out)
}

/// The input-row fields a window relation's function reads: its own arguments,
/// plus — for [`Reads::Varied`] — the partition and sort expressions the
/// *relation* holds.
///
/// A `WindowRelFunction` has no partitions or sorts of its own; a
/// consistent-partition window shares them across every function in it. Reading
/// only `arguments` would make `ROW_NUMBER() OVER (ORDER BY val)` unvaried here
/// while the same window written as an expression inside a projection is varied.
/// Same SQL, same answer, whichever form the producer chose.
pub(crate) fn window_rel_refs(
    w: &WindowRelFunction,
    partitions: &[Expression],
    sorts: &[SortField],
    fns: &Functions,
    which: Reads,
) -> Result<Vec<Field>> {
    let mut out = args_refs(&w.arguments, fns, which)?;
    if which == Reads::Varied {
        for e in partitions {
            out.extend(refs(e, fns, which)?);
        }
        for s in sorts {
            if let Some(e) = &s.expr {
                out.extend(refs(e, fns, which)?);
            }
        }
    }
    Ok(out)
}

/// `e` with any casts peeled off its root.
///
/// An engine's type coercion wraps a marker rather than replacing it: summing a
/// `REAL` column makes DataFusion plan `sum(CAST(ddx_reduce_mark(fval) AS
/// Float64))`, where the marker is no longer the root of the measure's argument
/// even though the user wrote it there. A cast around an identity marker is still
/// that marker, so placement checks compare against the peeled expression.
pub(crate) fn peel_casts(e: &Expression) -> &Expression {
    let mut e = e;
    for _ in 0..MAX_DEPTH {
        match e.rex_type.as_ref() {
            Some(RexType::Cast(c)) => match c.input.as_deref() {
                Some(inner) => e = inner,
                None => return e,
            },
            _ => return e,
        }
    }
    e
}

/// The value arguments of a function call, in order.
pub(crate) fn value_args(args: &[FunctionArgument]) -> Vec<&Expression> {
    args.iter()
        .filter_map(|a| match &a.arg_type {
            Some(ArgType::Value(e)) => Some(e),
            _ => None,
        })
        .collect()
}

/// Where in an expression the walk currently is.
#[derive(Debug, Clone, Copy, Default)]
struct Where {
    stopped: bool,
    control: bool,
}

impl Where {
    fn stopped(self) -> Self {
        Where {
            stopped: true,
            ..self
        }
    }

    fn control(self) -> Self {
        Where {
            control: true,
            ..self
        }
    }
}

struct Walker<'f, F> {
    fns: &'f Functions,
    f: &'f mut F,
}

impl<'a, F: FnMut(Event<'a>) -> Result<()>> Walker<'_, F> {
    fn expr(&mut self, e: &'a Expression, at: Where, depth: usize) -> Result<()> {
        if depth > MAX_DEPTH {
            return Err(AdError::InvalidPlan(format!(
                "an expression nests deeper than {MAX_DEPTH} levels"
            )));
        }
        let d = depth + 1;
        let rex = e
            .rex_type
            .as_ref()
            .ok_or_else(|| AdError::InvalidPlan("an expression has no type".into()))?;
        match rex {
            RexType::Literal(_) | RexType::DynamicParameter(_) => Ok(()),
            RexType::Selection(r) => {
                let field = root_field(r)?;
                (self.f)(Event::Read {
                    field,
                    stopped: at.stopped,
                    control: at.control,
                })
            }
            RexType::ScalarFunction(call) => {
                let marker = self.fns.marker(call.function_reference)?;
                if let Some(marker) = marker {
                    (self.f)(Event::Marker {
                        marker,
                        call,
                        at_root: depth == 0,
                    })?;
                }
                let at = if marker == Some(Marker::StopGradient) {
                    at.stopped()
                } else {
                    at
                };
                self.args(&call.arguments, at, d)?;
                #[allow(deprecated)]
                for a in &call.args {
                    self.expr(a, at, d)?;
                }
                Ok(())
            }
            RexType::WindowFunction(w) => {
                self.fns.name(w.function_reference)?;
                self.args(&w.arguments, at, d)?;
                #[allow(deprecated)]
                for a in &w.args {
                    self.expr(a, at, d)?;
                }
                // A window's frame selects and orders rows; what the function
                // returns is piecewise constant in those keys.
                for p in &w.partitions {
                    self.expr(p, at.control(), d)?;
                }
                for s in &w.sorts {
                    self.opt(s.expr.as_ref(), at.control(), d)?;
                }
                Ok(())
            }
            RexType::Cast(c) => self.opt(c.input.as_deref(), at, d),
            RexType::IfThen(i) => {
                for clause in &i.ifs {
                    // The condition picks a branch; only the branches are values.
                    self.opt(clause.r#if.as_ref(), at.control(), d)?;
                    self.opt(clause.then.as_ref(), at, d)?;
                }
                self.opt(i.r#else.as_deref(), at, d)
            }
            RexType::SwitchExpression(s) => {
                self.opt(s.r#match.as_deref(), at.control(), d)?;
                for clause in &s.ifs {
                    self.opt(clause.then.as_ref(), at, d)?;
                }
                self.opt(s.r#else.as_deref(), at, d)
            }
            RexType::SingularOrList(s) => {
                // `x IN (…)` yields a boolean, so every read in it is control.
                self.opt(s.value.as_deref(), at.control(), d)?;
                for o in &s.options {
                    self.expr(o, at.control(), d)?;
                }
                Ok(())
            }
            other => Err(AdError::NotImplemented(format!(
                "a `{}` expression",
                variant_name(other)
            ))),
        }
    }

    fn opt(&mut self, e: Option<&'a Expression>, at: Where, depth: usize) -> Result<()> {
        e.map_or(Ok(()), |e| self.expr(e, at, depth))
    }

    fn args(&mut self, args: &'a [FunctionArgument], at: Where, depth: usize) -> Result<()> {
        for a in args {
            if let Some(ArgType::Value(e)) = &a.arg_type {
                self.expr(e, at, depth)?;
            }
        }
        Ok(())
    }
}

/// The input-row position a plain `root_reference` field selection names.
pub(crate) fn root_field(r: &FieldReference) -> Result<Field> {
    match &r.root_type {
        Some(RootType::RootReference(_)) => {}
        Some(RootType::OuterReference(_)) => {
            return Err(AdError::NotImplemented(
                "a correlated (outer) column reference".into(),
            ))
        }
        _ => {
            return Err(AdError::NotImplemented(
                "a field reference rooted in something other than the input row".into(),
            ))
        }
    }
    let field = match &r.reference_type {
        Some(ReferenceType::DirectReference(seg)) => match &seg.reference_type {
            Some(Segment::StructField(sf)) if sf.child.is_none() => sf.field,
            _ => {
                return Err(AdError::NotImplemented(
                    "a reference into a nested field".into(),
                ))
            }
        },
        _ => return Err(AdError::NotImplemented("a masked field reference".into())),
    };
    usize::try_from(field)
        .map(Field::new)
        .map_err(|_| AdError::InvalidPlan(format!("a field reference to position {field}")))
}

/// The variant name of an expression, read off its `Debug` form.
fn variant_name(r: &RexType) -> String {
    let dbg = format!("{r:?}");
    match dbg.split_once('(') {
        Some((name, _)) => name.to_string(),
        None => dbg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_plans::*;

    fn fns() -> Functions {
        functions(&[
            (0, "multiply"),
            (1, "ddx_stop_gradient"),
            (2, "ddx_contract_mark"),
            (3, "gt"),
        ])
    }

    #[test]
    fn finds_fields_and_markers() {
        // ddx_contract_mark(f0 * ddx_stop_gradient(f1))
        let e = call(2, vec![call(0, vec![field(0), call(1, vec![field(1)])])]);
        let mut events = Vec::new();
        walk(&e, &fns(), &mut |ev| {
            events.push(match ev {
                Event::Read {
                    field,
                    stopped,
                    control,
                } => format!(
                    "f{}{}{}",
                    field.index(),
                    if stopped { "!" } else { "" },
                    if control { "?" } else { "" }
                ),
                Event::Marker {
                    marker, at_root, ..
                } => format!("{}@{at_root}", marker.name()),
            });
            Ok(())
        })
        .unwrap();
        assert_eq!(
            events,
            [
                "ddx_contract_mark@true",
                "f0",
                "ddx_stop_gradient@false",
                "f1!"
            ]
        );
        assert_eq!(refs(&e, &fns(), Reads::Value).unwrap(), [Field::new(0)]);
    }

    /// A `CASE` condition selects a branch rather than contributing a value, and
    /// the two halves of activity analysis want different answers about it.
    #[test]
    fn a_condition_is_a_control_read_not_a_value_read() {
        let e = if_then(
            call(3, vec![field(0), lit_f64(0.0)]),
            field(1),
            lit_f64(0.0),
        );
        assert_eq!(refs(&e, &fns(), Reads::Value).unwrap(), [Field::new(1)]);
        assert_eq!(
            refs(&e, &fns(), Reads::Varied).unwrap(),
            [Field::new(0), Field::new(1)]
        );
    }

    #[test]
    fn refuses_what_it_cannot_see_into() {
        let sub = Expression {
            rex_type: Some(RexType::Subquery(Box::default())),
        };
        assert!(matches!(
            refs(&sub, &fns(), Reads::Value),
            Err(AdError::NotImplemented(_))
        ));
        let undeclared = call(9, vec![field(0)]);
        assert!(matches!(
            refs(&undeclared, &fns(), Reads::Value),
            Err(AdError::InvalidPlan(_))
        ));
    }
}
