// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Building and taking apart Substrait expressions.
//!
//! Substrait's generated types are verbose: a column reference is five nested
//! messages. These helpers build the few shapes ddx emits and recognize the few
//! it reads.

use substrait::proto::expression::field_reference::{ReferenceType, RootReference, RootType};
use substrait::proto::expression::literal::LiteralType;
use substrait::proto::expression::reference_segment::{self, StructField};
use substrait::proto::expression::{
    cast, if_then, FieldReference, IfThen, Literal, ReferenceSegment, RexType, ScalarFunction,
};
use substrait::proto::function_argument::ArgType;
use substrait::proto::r#type::{Fp64, Kind, Nullability};
use substrait::proto::{Expression, FunctionArgument, Type};

use crate::error::{AdError, Result};

/// A reference to field `i` of the input row.
pub fn field(i: usize) -> Expression {
    Expression {
        rex_type: Some(RexType::Selection(Box::new(FieldReference {
            reference_type: Some(ReferenceType::DirectReference(ReferenceSegment {
                reference_type: Some(reference_segment::ReferenceType::StructField(Box::new(
                    StructField {
                        field: i as i32,
                        child: None,
                    },
                ))),
            })),
            root_type: Some(RootType::RootReference(RootReference {})),
        }))),
    }
}

/// The input field `e` refers to, if it is a plain reference to one.
pub fn as_field(e: &Expression) -> Option<usize> {
    match &e.rex_type {
        Some(RexType::Selection(r)) => plain_field(r),
        _ => None,
    }
}

fn plain_field(r: &FieldReference) -> Option<usize> {
    match (&r.reference_type, &r.root_type) {
        (Some(ReferenceType::DirectReference(seg)), Some(RootType::RootReference(_))) => {
            match &seg.reference_type {
                Some(reference_segment::ReferenceType::StructField(sf)) if sf.child.is_none() => {
                    usize::try_from(sf.field).ok()
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// The DOUBLE literal `v`.
pub fn lit_f64(v: f64) -> Expression {
    Expression {
        rex_type: Some(RexType::Literal(Literal {
            nullable: false,
            type_variation_reference: 0,
            literal_type: Some(LiteralType::Fp64(v)),
        })),
    }
}

/// A NULL of type DOUBLE.
pub fn null_f64() -> Expression {
    Expression {
        rex_type: Some(RexType::Literal(Literal {
            nullable: true,
            type_variation_reference: 0,
            literal_type: Some(LiteralType::Null(fp64())),
        })),
    }
}

/// The value of a numeric literal, if `e` is one.
pub fn as_number(e: &Expression) -> Option<f64> {
    match &e.rex_type {
        Some(RexType::Literal(Literal {
            literal_type: Some(t),
            ..
        })) => match t {
            LiteralType::Fp64(v) => Some(*v),
            LiteralType::Fp32(v) => Some(*v as f64),
            LiteralType::I8(v) => Some(*v as f64),
            LiteralType::I16(v) => Some(*v as f64),
            LiteralType::I32(v) => Some(*v as f64),
            LiteralType::I64(v) => Some(*v as f64),
            _ => None,
        },
        _ => None,
    }
}

/// The nullable DOUBLE type.
pub fn fp64() -> Type {
    Type {
        kind: Some(Kind::Fp64(Fp64 {
            type_variation_reference: 0,
            nullability: Nullability::Nullable as i32,
        })),
    }
}

/// A call to the scalar function declared at `anchor`.
pub fn call(anchor: u32, args: Vec<Expression>) -> Expression {
    Expression {
        rex_type: Some(RexType::ScalarFunction(ScalarFunction {
            function_reference: anchor,
            arguments: args.into_iter().map(value_arg).collect(),
            ..Default::default()
        })),
    }
}

/// `CAST(e AS ty)`.
pub fn cast(e: Expression, ty: Type) -> Expression {
    Expression {
        rex_type: Some(RexType::Cast(Box::new(
            substrait::proto::expression::Cast {
                r#type: Some(ty),
                input: Some(Box::new(e)),
                failure_behavior: cast::FailureBehavior::Unspecified as i32,
            },
        ))),
    }
}

/// `CASE WHEN c1 THEN r1 … ELSE otherwise END`.
pub fn if_then(clauses: Vec<(Expression, Expression)>, otherwise: Expression) -> Expression {
    Expression {
        rex_type: Some(RexType::IfThen(Box::new(IfThen {
            ifs: clauses
                .into_iter()
                .map(|(c, r)| if_then::IfClause {
                    r#if: Some(c),
                    then: Some(r),
                })
                .collect(),
            r#else: Some(Box::new(otherwise)),
        }))),
    }
}

fn value_arg(e: Expression) -> FunctionArgument {
    FunctionArgument {
        arg_type: Some(ArgType::Value(e)),
    }
}

/// The value arguments of a function call.
///
/// Refuses enum and type arguments: none of the functions ddx differentiates
/// or emits take one, so seeing one means the call is not what ddx thinks.
pub fn value_args(arguments: &[FunctionArgument]) -> Result<Vec<&Expression>> {
    arguments
        .iter()
        .map(|a| match &a.arg_type {
            Some(ArgType::Value(e)) => Ok(e),
            other => Err(AdError::NotImplemented(format!(
                "a function argument that is not a value: {other:?}"
            ))),
        })
        .collect()
}

/// The value arguments of a scalar function call. Falls back to the
/// deprecated `args` field, which older producers fill instead.
pub fn scalar_args(f: &ScalarFunction) -> Result<Vec<&Expression>> {
    #[allow(deprecated)]
    if f.arguments.is_empty() && !f.args.is_empty() {
        #[allow(deprecated)]
        return Ok(f.args.iter().collect());
    }
    value_args(&f.arguments)
}

/// Rewrite every input-field reference in `e` through `f`.
///
/// Used when an expression moves onto a different input row: `f` maps an old
/// field index to the new one. Refuses subqueries and lambdas, whose field
/// references are not all rooted in the input row.
pub fn map_fields(e: &Expression, f: &mut dyn FnMut(usize) -> Result<usize>) -> Result<Expression> {
    let mut out = e.clone();
    walk_fields(&mut out, &mut |r: &mut FieldReference| {
        let old = plain_field(r).ok_or_else(|| {
            AdError::NotImplemented(format!("a field reference ddx cannot follow: {r:?}"))
        })?;
        *r = match field(f(old)?).rex_type {
            Some(RexType::Selection(new)) => *new,
            _ => unreachable!("field() builds a selection"),
        };
        Ok(())
    })?;
    Ok(out)
}

/// Every input field `e` references, in order of appearance.
pub fn fields_of(e: &Expression) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    map_fields(e, &mut |i| {
        out.push(i);
        Ok(i)
    })?;
    Ok(out)
}

/// Visit every field reference in `e`, recursing into every expression kind
/// that can contain one.
fn walk_fields(
    e: &mut Expression,
    f: &mut dyn FnMut(&mut FieldReference) -> Result<()>,
) -> Result<()> {
    let Some(rex) = e.rex_type.as_mut() else {
        return Err(AdError::InvalidPlan("an expression with no content".into()));
    };
    match rex {
        RexType::Literal(_) | RexType::DynamicParameter(_) => Ok(()),
        RexType::Selection(r) => {
            // An expression-rooted reference (a field of a struct-valued
            // expression) recurses into that expression; ddx does not produce
            // or differentiate those, so it is refused by the caller.
            f(r)
        }
        RexType::ScalarFunction(s) => {
            #[allow(deprecated)]
            for a in s.args.iter_mut() {
                walk_fields(a, f)?;
            }
            walk_arguments(&mut s.arguments, f)
        }
        RexType::WindowFunction(w) => {
            #[allow(deprecated)]
            for a in w.args.iter_mut() {
                walk_fields(a, f)?;
            }
            walk_arguments(&mut w.arguments, f)?;
            for p in w.partitions.iter_mut() {
                walk_fields(p, f)?;
            }
            for s in w.sorts.iter_mut() {
                if let Some(x) = s.expr.as_mut() {
                    walk_fields(x, f)?;
                }
            }
            Ok(())
        }
        RexType::IfThen(it) => {
            for c in it.ifs.iter_mut() {
                if let Some(x) = c.r#if.as_mut() {
                    walk_fields(x, f)?;
                }
                if let Some(x) = c.then.as_mut() {
                    walk_fields(x, f)?;
                }
            }
            if let Some(x) = it.r#else.as_mut() {
                walk_fields(x, f)?;
            }
            Ok(())
        }
        RexType::SwitchExpression(sw) => {
            if let Some(x) = sw.r#match.as_mut() {
                walk_fields(x, f)?;
            }
            for c in sw.ifs.iter_mut() {
                if let Some(x) = c.then.as_mut() {
                    walk_fields(x, f)?;
                }
            }
            if let Some(x) = sw.r#else.as_mut() {
                walk_fields(x, f)?;
            }
            Ok(())
        }
        RexType::SingularOrList(sl) => {
            if let Some(x) = sl.value.as_mut() {
                walk_fields(x, f)?;
            }
            for o in sl.options.iter_mut() {
                walk_fields(o, f)?;
            }
            Ok(())
        }
        RexType::MultiOrList(ml) => {
            for v in ml.value.iter_mut() {
                walk_fields(v, f)?;
            }
            for rec in ml.options.iter_mut() {
                for v in rec.fields.iter_mut() {
                    walk_fields(v, f)?;
                }
            }
            Ok(())
        }
        RexType::Cast(c) => match c.input.as_mut() {
            Some(x) => walk_fields(x, f),
            None => Err(AdError::InvalidPlan("a cast with no input".into())),
        },
        other => Err(AdError::NotImplemented(format!(
            "this kind of expression inside a differentiated query: {}",
            rex_name(other)
        ))),
    }
}

fn walk_arguments(
    args: &mut [FunctionArgument],
    f: &mut dyn FnMut(&mut FieldReference) -> Result<()>,
) -> Result<()> {
    for a in args.iter_mut() {
        if let Some(ArgType::Value(x)) = a.arg_type.as_mut() {
            walk_fields(x, f)?;
        }
    }
    Ok(())
}

/// The direct subexpressions of `e`. A subquery's relation is not included.
pub fn children(e: &Expression) -> Vec<&Expression> {
    let mut out = Vec::new();
    let Some(rex) = &e.rex_type else {
        return out;
    };
    fn args<'e>(a: &'e [FunctionArgument], out: &mut Vec<&'e Expression>) {
        for x in a {
            if let Some(ArgType::Value(v)) = &x.arg_type {
                out.push(v);
            }
        }
    }
    match rex {
        RexType::ScalarFunction(s) => {
            args(&s.arguments, &mut out);
            #[allow(deprecated)]
            out.extend(s.args.iter());
        }
        RexType::WindowFunction(w) => {
            args(&w.arguments, &mut out);
            #[allow(deprecated)]
            out.extend(w.args.iter());
            out.extend(w.partitions.iter());
            out.extend(w.sorts.iter().filter_map(|s| s.expr.as_ref()));
        }
        RexType::IfThen(it) => {
            for c in &it.ifs {
                out.extend(c.r#if.iter());
                out.extend(c.then.iter());
            }
            out.extend(it.r#else.as_deref());
        }
        RexType::SwitchExpression(sw) => {
            out.extend(sw.r#match.as_deref());
            out.extend(sw.ifs.iter().filter_map(|c| c.then.as_ref()));
            out.extend(sw.r#else.as_deref());
        }
        RexType::SingularOrList(sl) => {
            out.extend(sl.value.as_deref());
            out.extend(sl.options.iter());
        }
        RexType::MultiOrList(ml) => {
            out.extend(ml.value.iter());
            for rec in &ml.options {
                out.extend(rec.fields.iter());
            }
        }
        RexType::Cast(c) => out.extend(c.input.as_deref()),
        _ => {}
    }
    out
}

/// Does `e`, or any expression inside it, satisfy `pred`?
pub fn contains(e: &Expression, pred: &dyn Fn(&Expression) -> bool) -> bool {
    pred(e) || children(e).into_iter().any(|c| contains(c, pred))
}

/// A short name for an expression kind, for error messages.
pub fn rex_name(r: &RexType) -> &'static str {
    match r {
        RexType::Literal(_) => "literal",
        RexType::Selection(_) => "field reference",
        RexType::ScalarFunction(_) => "scalar function",
        RexType::WindowFunction(_) => "window function",
        RexType::IfThen(_) => "CASE",
        RexType::SwitchExpression(_) => "CASE on a value",
        RexType::SingularOrList(_) => "IN list",
        RexType::MultiOrList(_) => "multi-column IN list",
        RexType::Cast(_) => "cast",
        RexType::Subquery(_) => "subquery",
        RexType::Nested(_) => "struct/list/map constructor",
        RexType::DynamicParameter(_) => "query parameter",
        RexType::Lambda(_) => "lambda",
        RexType::LambdaInvocation(_) => "lambda invocation",
        #[allow(deprecated)]
        RexType::Enum(_) => "enum",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_field_reference_round_trips() {
        assert_eq!(as_field(&field(7)), Some(7));
        assert_eq!(as_field(&lit_f64(1.0)), None);
    }

    #[test]
    fn fields_are_found_and_remapped_everywhere() {
        let e = call(
            0,
            vec![
                field(1),
                cast(call(1, vec![field(3), lit_f64(2.0)]), fp64()),
                if_then(vec![(field(4), field(1))], field(5)),
            ],
        );
        assert_eq!(fields_of(&e).unwrap(), vec![1, 3, 4, 1, 5]);
        let moved = map_fields(&e, &mut |i| Ok(i + 10)).unwrap();
        assert_eq!(fields_of(&moved).unwrap(), vec![11, 13, 14, 11, 15]);
    }

    #[test]
    fn numbers_are_read_off_literals() {
        assert_eq!(as_number(&lit_f64(2.5)), Some(2.5));
        assert_eq!(as_number(&field(0)), None);
    }
}
