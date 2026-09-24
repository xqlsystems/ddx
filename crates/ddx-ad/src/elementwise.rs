// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The map primitive's local derivatives: partial derivatives of one Substrait
//! expression, computed by `ddx-core`.
//!
//! A projection maps each row to a new row, one scalar expression per new
//! column. Its transpose needs, for each output column, the partial derivative
//! with respect to each input column of the same row. This is the seam
//! between v1 and v2 (design.md §4.3): v1 already differentiates scalar
//! expressions. So the rule translates the Substrait
//! expression into the `sqlparser` expression `ddx-core` takes, asks for one
//! derivative per input column, and translates each answer back.
//!
//! Three details keep the translation honest:
//!
//! - **Columns become identifiers.** Input field `i` is the identifier `ci`,
//!   and the derivative with respect to it is `ddx-core`'s derivative with
//!   respect to column `ci`.
//! - **Constants become opaque.** A subexpression that depends on no varied
//!   column (see below) is replaced by a placeholder identifier `kn` and
//!   restored verbatim on the way back. Its derivative is zero, so `ddx-core`
//!   never needs to understand it: `CASE WHEN label = out THEN 1.0 ELSE 0.0 END`
//!   in a softmax loss has no rule in `ddx-core`, and needs none. Numeric
//!   literals stay literals, because `ddx-core` reads them (`power(x, 2)` has a
//!   rule only because the exponent is a known constant).
//! - **Stop-gradient.** `ddx_stop_gradient(x)` is a constant, whatever `x` is.
//!
//! A column is **varied** when it depends on a column the gradient is taken
//! with respect to. The caller says which input fields are varied; a
//! derivative is computed only for those.

use std::collections::BTreeSet;

use ddx_core::build::{finite_num, func};
use ddx_core::sqlparser::ast::{
    BinaryOperator, CaseWhen, CastKind, DataType, ExactNumberInfo, Expr as Sql, FunctionArg,
    FunctionArgExpr, FunctionArguments, Ident, ObjectNamePart, UnaryOperator, Value,
};
use ddx_core::{ColRef, Ddx};
use substrait::proto::expression::RexType;
use substrait::proto::r#type::{Fp32, Fp64, Kind, Nullability, I16, I32, I64, I8};
use substrait::proto::{Expression, Type};

use crate::error::{AdError, Result};
use crate::expr::{
    as_field, as_number, call, cast, field, fields_of, if_then, lit_f64, null_f64, rex_name,
    scalar_args,
};
use crate::functions::{Extensions, Functions};

/// Partial derivatives of Substrait expressions.
pub struct Elementwise<'a> {
    ddx: &'a Ddx,
    functions: &'a Functions,
}

impl<'a> Elementwise<'a> {
    /// Differentiate with `ddx`'s rules, reading function names from
    /// `functions`, the table of the plan the expressions come from.
    pub fn new(ddx: &'a Ddx, functions: &'a Functions) -> Self {
        Elementwise { ddx, functions }
    }

    /// `∂e/∂field` for every varied input field `e` depends on, skipping the
    /// ones that are zero.
    ///
    /// Each derivative is an expression over the same input row as `e`. New
    /// functions it calls are declared in `ext`.
    pub fn partials(
        &self,
        e: &Expression,
        varied: &dyn Fn(usize) -> bool,
        ext: &mut Extensions,
    ) -> Result<Vec<(usize, Expression)>> {
        let mut to_sql = ToSql {
            functions: self.functions,
            varied,
            placeholders: Vec::new(),
            columns: BTreeSet::new(),
        };
        let sql = to_sql.expr(e)?;
        let mut out = Vec::new();
        for &i in &to_sql.columns {
            let d = self.ddx.differentiate(&sql, &ColRef::bare(column(i)))?;
            if is_zero(&d) {
                continue;
            }
            let back = FromSql {
                placeholders: &to_sql.placeholders,
                ext: &mut *ext,
            }
            .expr(&d)?;
            out.push((i, back));
        }
        Ok(out)
    }

    /// Does `e` depend on a varied field, outside `ddx_stop_gradient`?
    pub fn depends(&self, e: &Expression, varied: &dyn Fn(usize) -> bool) -> Result<bool> {
        depends(self.functions, e, varied)
    }
}

fn depends(functions: &Functions, e: &Expression, varied: &dyn Fn(usize) -> bool) -> Result<bool> {
    if let Some(RexType::ScalarFunction(f)) = &e.rex_type {
        if functions.is_stop_gradient(f.function_reference)? {
            return Ok(false);
        }
        for a in scalar_args(f)? {
            if depends(functions, a, varied)? {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    if let Some(RexType::Cast(c)) = &e.rex_type {
        if let Some(input) = &c.input {
            return depends(functions, input, varied);
        }
    }
    // Any other kind: count every field it references. That can call an
    // expression varied because of a stop-gradient buried in a CASE, which
    // leads to a refusal rather than a wrong number.
    Ok(fields_of(e)?.into_iter().any(varied))
}

/// The identifier for input field `i`.
fn column(i: usize) -> String {
    format!("c{i}")
}

fn ident(name: String) -> Sql {
    Sql::Identifier(Ident::new(name))
}

fn is_zero(e: &Sql) -> bool {
    match e {
        Sql::Value(v) => matches!(&v.value, Value::Number(n, _) if n.parse::<f64>() == Ok(0.0)),
        Sql::Nested(inner) => is_zero(inner),
        _ => false,
    }
}

/// Substrait → `sqlparser`.
struct ToSql<'a> {
    functions: &'a Functions,
    varied: &'a dyn Fn(usize) -> bool,
    /// Constant subexpressions, by placeholder number.
    placeholders: Vec<Expression>,
    /// The varied fields the translation references.
    columns: BTreeSet<usize>,
}

impl ToSql<'_> {
    fn expr(&mut self, e: &Expression) -> Result<Sql> {
        if let Some(v) = as_number(e) {
            return Ok(finite_num(v)?);
        }
        if !depends(self.functions, e, self.varied)? {
            self.placeholders.push(e.clone());
            return Ok(ident(format!("k{}", self.placeholders.len() - 1)));
        }
        let rex = e
            .rex_type
            .as_ref()
            .expect("depends() refused an empty expression");
        match rex {
            RexType::Selection(_) => {
                let i = as_field(e).ok_or_else(|| {
                    AdError::NotImplemented(format!("a field reference ddx cannot follow: {e:?}"))
                })?;
                self.columns.insert(i);
                Ok(ident(column(i)))
            }
            RexType::ScalarFunction(f) => {
                let name = self.functions.name(f.function_reference)?.to_string();
                let args = scalar_args(f)?;
                let mut sql = args
                    .into_iter()
                    .map(|a| self.expr(a))
                    .collect::<Result<Vec<_>>>()?;
                if let Some(op) = binary_op(&name) {
                    if sql.len() == 2 {
                        let right = sql.pop().unwrap();
                        let left = sql.pop().unwrap();
                        return Ok(Sql::BinaryOp {
                            left: Box::new(left),
                            op,
                            right: Box::new(right),
                        });
                    }
                }
                if matches!(name.as_str(), "negate" | "negative") && sql.len() == 1 {
                    return Ok(Sql::UnaryOp {
                        op: UnaryOperator::Minus,
                        expr: Box::new(sql.pop().unwrap()),
                    });
                }
                if is_logical(&name) {
                    return Err(AdError::NotImplemented(format!(
                        "`{name}` of a value that carries gradient is not differentiable; if \
                         the gradient should not flow through it, wrap the operand in \
                         ddx_stop_gradient(...)"
                    )));
                }
                Ok(func(&name, sql))
            }
            RexType::Cast(c) => {
                let input = c
                    .input
                    .as_ref()
                    .ok_or_else(|| AdError::InvalidPlan("a cast with no input".into()))?;
                let ty = c
                    .r#type
                    .as_ref()
                    .ok_or_else(|| AdError::InvalidPlan("a cast with no type".into()))?;
                Ok(Sql::Cast {
                    kind: CastKind::Cast,
                    expr: Box::new(self.expr(input)?),
                    data_type: sql_type(ty)?,
                    array: false,
                    format: None,
                })
            }
            other => Err(AdError::NotImplemented(format!(
                "a {} over a value that carries gradient is not differentiable; if the \
                 gradient should not flow through it, wrap it in ddx_stop_gradient(...)",
                rex_name(other)
            ))),
        }
    }
}

/// The SQL operator a Substrait arithmetic function name stands for.
fn binary_op(name: &str) -> Option<BinaryOperator> {
    Some(match name {
        "add" => BinaryOperator::Plus,
        "subtract" => BinaryOperator::Minus,
        "multiply" => BinaryOperator::Multiply,
        "divide" => BinaryOperator::Divide,
        _ => return None,
    })
}

fn is_logical(name: &str) -> bool {
    matches!(
        name,
        "equal"
            | "not_equal"
            | "lt"
            | "lte"
            | "gt"
            | "gte"
            | "and"
            | "or"
            | "not"
            | "is_null"
            | "is_not_null"
            | "is_not_distinct_from"
            | "is_distinct_from"
    )
}

/// The Substrait name of a SQL operator ddx-core emits.
fn function_name(op: &BinaryOperator) -> Option<&'static str> {
    Some(match op {
        BinaryOperator::Plus => "add",
        BinaryOperator::Minus => "subtract",
        BinaryOperator::Multiply => "multiply",
        BinaryOperator::Divide => "divide",
        BinaryOperator::Gt => "gt",
        BinaryOperator::Lt => "lt",
        BinaryOperator::GtEq => "gte",
        BinaryOperator::LtEq => "lte",
        BinaryOperator::Eq => "equal",
        _ => return None,
    })
}

/// The SQL type of a cast over a value that carries gradient: a float type.
///
/// A cast to an integer type truncates, so its derivative is zero almost
/// everywhere and undefined at the steps. `ddx-core` would treat it as the
/// identity (derivative 1), a silently wrong answer, so it is refused here.
fn sql_type(t: &Type) -> Result<DataType> {
    Ok(match &t.kind {
        Some(Kind::Fp64(_)) => DataType::Double(ExactNumberInfo::None),
        Some(Kind::Fp32(_)) => DataType::Real,
        Some(Kind::I64(_) | Kind::I32(_) | Kind::I16(_) | Kind::I8(_)) => {
            return Err(AdError::NotImplemented(
                "a cast to an integer type over a value that carries gradient: it truncates, \
                 so its derivative is zero almost everywhere, not the identity. Cast to a \
                 float type, or wrap the cast in ddx_stop_gradient(...) if no gradient \
                 should flow through it"
                    .into(),
            ))
        }
        other => {
            return Err(AdError::NotImplemented(format!(
                "a cast to {other:?} over a value that carries gradient"
            )))
        }
    })
}

fn substrait_type(t: &DataType) -> Result<Type> {
    let n = Nullability::Nullable as i32;
    let kind = match t {
        DataType::Double(_) | DataType::DoublePrecision | DataType::Float64 => Kind::Fp64(Fp64 {
            type_variation_reference: 0,
            nullability: n,
        }),
        DataType::Real | DataType::Float4 | DataType::Float32 => Kind::Fp32(Fp32 {
            type_variation_reference: 0,
            nullability: n,
        }),
        DataType::BigInt(_) => Kind::I64(I64 {
            type_variation_reference: 0,
            nullability: n,
        }),
        DataType::Int(_) => Kind::I32(I32 {
            type_variation_reference: 0,
            nullability: n,
        }),
        DataType::SmallInt(_) => Kind::I16(I16 {
            type_variation_reference: 0,
            nullability: n,
        }),
        DataType::TinyInt(_) => Kind::I8(I8 {
            type_variation_reference: 0,
            nullability: n,
        }),
        other => {
            return Err(AdError::Internal(format!(
                "ddx-core emitted a cast to {other}, which ddx-ad cannot translate"
            )))
        }
    };
    Ok(Type { kind: Some(kind) })
}

/// `sqlparser` → Substrait, for the derivatives `ddx-core` returns.
struct FromSql<'a> {
    placeholders: &'a [Expression],
    ext: &'a mut Extensions,
}

impl FromSql<'_> {
    fn expr(&mut self, e: &Sql) -> Result<Expression> {
        match e {
            Sql::Identifier(id) => self.identifier(&id.value),
            Sql::Nested(inner) => self.expr(inner),
            Sql::Value(v) => match &v.value {
                Value::Number(n, _) => n
                    .parse::<f64>()
                    .map(lit_f64)
                    .map_err(|_| AdError::Internal(format!("ddx-core emitted the number `{n}`"))),
                // The ELSE of `abs`'s sign: reached only by a NULL input.
                Value::Null => Ok(null_f64()),
                other => Err(AdError::Internal(format!(
                    "ddx-core emitted the literal `{other}`"
                ))),
            },
            Sql::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => {
                let inner = self.expr(expr)?;
                if let Some(v) = as_number(&inner) {
                    return Ok(lit_f64(-v));
                }
                Ok(call(self.ext.anchor("negate"), vec![inner]))
            }
            Sql::UnaryOp {
                op: UnaryOperator::Plus,
                expr,
            } => self.expr(expr),
            Sql::BinaryOp { left, op, right } => {
                let name = function_name(op).ok_or_else(|| {
                    AdError::Internal(format!("ddx-core emitted the operator `{op}`"))
                })?;
                let args = vec![self.expr(left)?, self.expr(right)?];
                Ok(call(self.ext.anchor(name), args))
            }
            Sql::Function(f) => {
                let name = match f.name.0.as_slice() {
                    [ObjectNamePart::Identifier(id)] => id.value.to_ascii_lowercase(),
                    _ => {
                        return Err(AdError::Internal(format!(
                            "ddx-core emitted the qualified call `{f}`"
                        )))
                    }
                };
                let FunctionArguments::List(list) = &f.args else {
                    return Err(AdError::Internal(format!("ddx-core emitted `{f}`")));
                };
                let args = list
                    .args
                    .iter()
                    .map(|a| match a {
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(x)) => self.expr(x),
                        _ => Err(AdError::Internal(format!("ddx-core emitted `{f}`"))),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(call(self.ext.anchor(&name), args))
            }
            Sql::Cast {
                expr, data_type, ..
            } => Ok(cast(self.expr(expr)?, substrait_type(data_type)?)),
            Sql::Case {
                operand: None,
                conditions,
                else_result,
                ..
            } => {
                let clauses = conditions
                    .iter()
                    .map(|CaseWhen { condition, result }| {
                        Ok((self.expr(condition)?, self.expr(result)?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let otherwise = match else_result {
                    Some(x) => self.expr(x)?,
                    None => {
                        return Err(AdError::Internal(
                            "ddx-core emitted a CASE with no ELSE".into(),
                        ))
                    }
                };
                Ok(if_then(clauses, otherwise))
            }
            other => Err(AdError::Internal(format!(
                "ddx-core emitted `{other}`, which ddx-ad cannot translate"
            ))),
        }
    }

    fn identifier(&self, name: &str) -> Result<Expression> {
        let parse = |prefix: char| {
            name.strip_prefix(prefix)
                .and_then(|n| n.parse::<usize>().ok())
        };
        if let Some(i) = parse('c') {
            return Ok(field(i));
        }
        if let Some(n) = parse('k') {
            if let Some(p) = self.placeholders.get(n) {
                return Ok(p.clone());
            }
        }
        Err(AdError::Internal(format!(
            "ddx-core emitted the identifier `{name}`, which ddx-ad did not give it"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::normalize;
    use crate::functions::tests::plan_declaring;
    use std::collections::HashMap;

    /// A plan's function table, plus a name → anchor lookup for building
    /// expressions in tests.
    struct Fixture {
        functions: Functions,
        anchors: HashMap<&'static str, u32>,
    }

    const NAMES: [&str; 15] = [
        "add",
        "subtract",
        "multiply",
        "divide",
        "negate",
        "tanh",
        "exp",
        "ln",
        "power",
        "sqrt",
        "gt",
        "ddx_stop_gradient",
        "cos",
        "sin",
        "abs",
    ];

    fn fixture() -> Fixture {
        let decls: Vec<(u32, &str)> = NAMES
            .iter()
            .enumerate()
            .map(|(i, n)| (i as u32, *n))
            .collect();
        Fixture {
            functions: Functions::from_plan(&plan_declaring(&decls)).unwrap(),
            anchors: decls.iter().map(|(a, n)| (*n, *a)).collect(),
        }
    }

    impl Fixture {
        fn f(&self, name: &str, args: Vec<Expression>) -> Expression {
            call(self.anchors[name], args)
        }
    }

    /// Evaluate `e` on `row`, looking function names up in `ext`'s
    /// declarations.
    fn eval(e: &Expression, row: &[f64], names: &HashMap<u32, String>) -> f64 {
        if let Some(v) = as_number(e) {
            return v;
        }
        if let Some(i) = as_field(e) {
            return row[i];
        }
        if let Some(RexType::Literal(_)) = &e.rex_type {
            return f64::NAN; // the typed NULL
        }
        match e.rex_type.as_ref().unwrap() {
            RexType::ScalarFunction(f) => {
                let a: Vec<f64> = scalar_args(f)
                    .unwrap()
                    .into_iter()
                    .map(|x| eval(x, row, names))
                    .collect();
                match names[&f.function_reference].as_str() {
                    "add" => a[0] + a[1],
                    "subtract" => a[0] - a[1],
                    "multiply" => a[0] * a[1],
                    "divide" => a[0] / a[1],
                    "negate" => -a[0],
                    "tanh" => a[0].tanh(),
                    "exp" => a[0].exp(),
                    "ln" => a[0].ln(),
                    "power" => a[0].powf(a[1]),
                    "sqrt" => a[0].sqrt(),
                    "sin" => a[0].sin(),
                    "cos" => a[0].cos(),
                    "abs" => a[0].abs(),
                    "gt" => (a[0] > a[1]) as u8 as f64,
                    "lt" => (a[0] < a[1]) as u8 as f64,
                    "equal" => (a[0] == a[1]) as u8 as f64,
                    "ddx_stop_gradient" => a[0],
                    other => panic!("no test evaluator for {other}"),
                }
            }
            RexType::Cast(c) => eval(c.input.as_ref().unwrap(), row, names),
            RexType::IfThen(it) => {
                for c in &it.ifs {
                    if eval(c.r#if.as_ref().unwrap(), row, names) != 0.0 {
                        return eval(c.then.as_ref().unwrap(), row, names);
                    }
                }
                eval(it.r#else.as_ref().unwrap(), row, names)
            }
            other => panic!("no test evaluator for {}", rex_name(other)),
        }
    }

    fn names_of(ext: &Extensions) -> HashMap<u32, String> {
        let plan = substrait::proto::Plan {
            extensions: ext.declarations(),
            ..Default::default()
        };
        let f = Functions::from_plan(&plan).unwrap();
        (0..64)
            .filter_map(|a| f.name(a).ok().map(|n| (a, normalize(n))))
            .collect()
    }

    /// Check every partial against a central difference at `row`, and that
    /// exactly the fields in `expect` have a nonzero partial.
    fn check(fx: &Fixture, e: &Expression, varied: &[usize], row: &[f64], expect: &[usize]) {
        let ddx = Ddx::new();
        let mut ext = Extensions::new(&fx.functions);
        let ew = Elementwise::new(&ddx, &fx.functions);
        let partials = ew.partials(e, &|i| varied.contains(&i), &mut ext).unwrap();
        let names = names_of(&ext);
        let got: Vec<usize> = partials.iter().map(|(i, _)| *i).collect();
        assert_eq!(got, expect);
        for (i, d) in &partials {
            let h = 1e-6;
            let (mut up, mut down) = (row.to_vec(), row.to_vec());
            up[*i] += h;
            down[*i] -= h;
            let fd = (eval(e, &up, &names) - eval(e, &down, &names)) / (2.0 * h);
            let ad = eval(d, row, &names);
            assert!(
                (fd - ad).abs() < 1e-6,
                "∂/∂c{i}: {ad} vs finite difference {fd}"
            );
        }
    }

    #[test]
    fn the_chain_and_product_rules_come_from_ddx_core() {
        let fx = fixture();
        // tanh(c0 * c1 + c2)
        let e = fx.f(
            "tanh",
            vec![fx.f(
                "add",
                vec![fx.f("multiply", vec![field(0), field(1)]), field(2)],
            )],
        );
        check(&fx, &e, &[0, 1, 2], &[0.3, -0.7, 0.2], &[0, 1, 2]);
    }

    #[test]
    fn only_varied_fields_get_a_partial() {
        let fx = fixture();
        // c0 * c1 / c2 with only c1 varied
        let e = fx.f(
            "divide",
            vec![fx.f("multiply", vec![field(0), field(1)]), field(2)],
        );
        check(&fx, &e, &[1], &[2.0, 3.0, 5.0], &[1]);
    }

    #[test]
    fn a_stop_gradient_is_a_constant() {
        let fx = fixture();
        // exp(c0 - ddx_stop_gradient(c0)): the derivative is exp(0) = 1,
        // not the 0 of the unmarked expression.
        let e = fx.f(
            "exp",
            vec![fx.f(
                "subtract",
                vec![field(0), fx.f("ddx_stop_gradient", vec![field(0)])],
            )],
        );
        let ddx = Ddx::new();
        let mut ext = Extensions::new(&fx.functions);
        let p = Elementwise::new(&ddx, &fx.functions)
            .partials(&e, &|_| true, &mut ext)
            .unwrap();
        assert_eq!(p.len(), 1);
        let v = eval(&p[0].1, &[1.7], &names_of(&ext));
        assert!((v - 1.0).abs() < 1e-12, "{v}");
    }

    #[test]
    fn a_constant_subexpression_is_carried_through_unread() {
        let fx = fixture();
        // c0 * CASE WHEN c1 > 0 THEN 2 ELSE 3 END, c1 not varied: ddx-core has
        // no CASE rule, and needs none here.
        let case = if_then(
            vec![(fx.f("gt", vec![field(1), lit_f64(0.0)]), lit_f64(2.0))],
            lit_f64(3.0),
        );
        let e = fx.f("multiply", vec![field(0), case]);
        check(&fx, &e, &[0], &[1.5, 1.0], &[0]);
        check(&fx, &e, &[0], &[1.5, -1.0], &[0]);
    }

    #[test]
    fn a_varied_comparison_is_refused() {
        let fx = fixture();
        let e = fx.f("gt", vec![field(0), lit_f64(0.0)]);
        let ddx = Ddx::new();
        let mut ext = Extensions::new(&fx.functions);
        let err = Elementwise::new(&ddx, &fx.functions)
            .partials(&e, &|_| true, &mut ext)
            .unwrap_err();
        assert!(matches!(err, AdError::NotImplemented(_)), "{err}");
        assert!(err.to_string().contains("ddx_stop_gradient"), "{err}");
    }

    #[test]
    fn casts_are_seen_through() {
        let fx = fixture();
        let e = fx.f(
            "sin",
            vec![cast(
                fx.f("multiply", vec![field(0), field(1)]),
                crate::expr::fp64(),
            )],
        );
        check(&fx, &e, &[0, 1], &[2.0, -3.0], &[0, 1]);
    }

    #[test]
    fn a_cast_to_an_integer_is_refused_not_treated_as_the_identity() {
        let fx = fixture();
        let bigint = Type {
            kind: Some(Kind::I64(I64 {
                type_variation_reference: 0,
                nullability: Nullability::Nullable as i32,
            })),
        };
        let ddx = Ddx::new();
        let mut ext = Extensions::new(&fx.functions);
        let ew = Elementwise::new(&ddx, &fx.functions);
        let e = cast(field(0), bigint.clone());
        let err = ew.partials(&e, &|_| true, &mut ext).unwrap_err();
        assert!(matches!(err, AdError::NotImplemented(_)), "{err}");
        assert!(err.to_string().contains("truncates"), "{err}");
        // Over a constant it is a placeholder, and fine.
        let e = fx.f("multiply", vec![field(0), cast(field(1), bigint)]);
        assert_eq!(ew.partials(&e, &|i| i == 0, &mut ext).unwrap().len(), 1);
    }

    #[test]
    fn constant_exponents_and_the_abs_kink_translate_back() {
        let fx = fixture();
        let e = fx.f(
            "add",
            vec![
                fx.f("power", vec![field(0), lit_f64(3.0)]),
                fx.f("abs", vec![field(1)]),
            ],
        );
        check(&fx, &e, &[0, 1], &[1.3, -0.4], &[0, 1]);
        check(&fx, &e, &[0, 1], &[1.3, 0.4], &[0, 1]);
    }

    #[test]
    fn an_unknown_function_is_refused_by_ddx_core() {
        let fx = fixture();
        let decls = [(0, "atan2")];
        let functions = Functions::from_plan(&plan_declaring(&decls)).unwrap();
        let e = call(0, vec![field(0), field(1)]);
        let ddx = Ddx::new();
        let mut ext = Extensions::new(&functions);
        let err = Elementwise::new(&ddx, &functions)
            .partials(&e, &|_| true, &mut ext)
            .unwrap_err();
        assert!(matches!(err, AdError::Diff(_)), "{err}");
        let _ = fx;
    }
}
