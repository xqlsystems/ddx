// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A walk over one Substrait expression, reporting the two things analysis
//! needs from it: which input fields it reads, and where the ddx markers are.
//!
//! Anything the walk can't see into — a subquery, a lambda, a nested-field
//! reference — is [`AdError::NotImplemented`] rather than "reads nothing": a
//! dependency the walk missed is a gradient it would silently drop.

use substrait::proto::expression::field_reference::{ReferenceType, RootType};
use substrait::proto::expression::reference_segment::ReferenceType as Segment;
use substrait::proto::expression::{FieldReference, RexType, ScalarFunction};
use substrait::proto::function_argument::ArgType;
use substrait::proto::{Expression, FunctionArgument};

use crate::columns::Field;
use crate::error::{AdError, Result};
use crate::index::MAX_DEPTH;
use crate::markers::{Functions, Marker};

/// Something the walk found.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Event<'a> {
    /// A read of a position in the node's input row. `stopped` is true under a
    /// `ddx_stop_gradient` call, where no cotangent flows.
    Read { field: Field, stopped: bool },
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
    Walker { fns, f }.expr(e, false, 0)
}

/// The input-row fields `e` reads outside any `ddx_stop_gradient`: the ones a
/// cotangent on `e` can flow back into.
pub(crate) fn differentiable_refs(e: &Expression, fns: &Functions) -> Result<Vec<Field>> {
    let mut out = Vec::new();
    walk(e, fns, &mut |ev| {
        if let Event::Read {
            field,
            stopped: false,
        } = ev
        {
            out.push(field);
        }
        Ok(())
    })?;
    Ok(out)
}

/// [`differentiable_refs`] over each value argument of a function call.
pub(crate) fn args_refs(args: &[FunctionArgument], fns: &Functions) -> Result<Vec<Field>> {
    let mut out = Vec::new();
    for a in args {
        if let Some(ArgType::Value(e)) = &a.arg_type {
            out.extend(differentiable_refs(e, fns)?);
        }
    }
    Ok(out)
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

struct Walker<'f, F> {
    fns: &'f Functions,
    f: &'f mut F,
}

impl<'a, F: FnMut(Event<'a>) -> Result<()>> Walker<'_, F> {
    fn expr(&mut self, e: &'a Expression, stopped: bool, depth: usize) -> Result<()> {
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
                (self.f)(Event::Read { field, stopped })
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
                let stopped = stopped || marker == Some(Marker::StopGradient);
                self.args(&call.arguments, stopped, d)?;
                #[allow(deprecated)]
                for a in &call.args {
                    self.expr(a, stopped, d)?;
                }
                Ok(())
            }
            RexType::WindowFunction(w) => {
                self.fns.name(w.function_reference)?;
                self.args(&w.arguments, stopped, d)?;
                #[allow(deprecated)]
                for a in &w.args {
                    self.expr(a, stopped, d)?;
                }
                for p in &w.partitions {
                    self.expr(p, stopped, d)?;
                }
                for s in &w.sorts {
                    self.opt(s.expr.as_ref(), stopped, d)?;
                }
                Ok(())
            }
            RexType::Cast(c) => self.opt(c.input.as_deref(), stopped, d),
            RexType::IfThen(i) => {
                for clause in &i.ifs {
                    self.opt(clause.r#if.as_ref(), stopped, d)?;
                    self.opt(clause.then.as_ref(), stopped, d)?;
                }
                self.opt(i.r#else.as_deref(), stopped, d)
            }
            RexType::SwitchExpression(s) => {
                self.opt(s.r#match.as_deref(), stopped, d)?;
                for clause in &s.ifs {
                    self.opt(clause.then.as_ref(), stopped, d)?;
                }
                self.opt(s.r#else.as_deref(), stopped, d)
            }
            RexType::SingularOrList(s) => {
                self.opt(s.value.as_deref(), stopped, d)?;
                for o in &s.options {
                    self.expr(o, stopped, d)?;
                }
                Ok(())
            }
            other => Err(AdError::NotImplemented(format!(
                "a `{}` expression",
                variant_name(other)
            ))),
        }
    }

    fn opt(&mut self, e: Option<&'a Expression>, stopped: bool, depth: usize) -> Result<()> {
        e.map_or(Ok(()), |e| self.expr(e, stopped, depth))
    }

    fn args(&mut self, args: &'a [FunctionArgument], stopped: bool, depth: usize) -> Result<()> {
        for a in args {
            if let Some(ArgType::Value(e)) = &a.arg_type {
                self.expr(e, stopped, depth)?;
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
    dbg.split('(').next().unwrap_or(&dbg).to_string()
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
        ])
    }

    #[test]
    fn finds_fields_and_markers() {
        // ddx_contract_mark(f0 * ddx_stop_gradient(f1))
        let e = call(2, vec![call(0, vec![field(0), call(1, vec![field(1)])])]);
        let mut events = Vec::new();
        walk(&e, &fns(), &mut |ev| {
            events.push(match ev {
                Event::Read { field, stopped } => {
                    format!("f{}{}", field.index(), if stopped { "!" } else { "" })
                }
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
        assert_eq!(differentiable_refs(&e, &fns()).unwrap(), [Field::new(0)]);
    }

    #[test]
    fn refuses_what_it_cannot_see_into() {
        let sub = Expression {
            rex_type: Some(RexType::Subquery(Box::default())),
        };
        assert!(matches!(
            differentiable_refs(&sub, &fns()),
            Err(AdError::NotImplemented(_))
        ));
        let undeclared = call(9, vec![field(0)]);
        assert!(matches!(
            differentiable_refs(&undeclared, &fns()),
            Err(AdError::InvalidPlan(_))
        ));
    }
}
