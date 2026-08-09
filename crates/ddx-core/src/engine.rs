// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The differentiation engine: forward-mode linearization over
//! `sqlparser::ast::Expr`, with a name-keyed, user-extensible rule registry.
//!
//! The approach mirrors JAX's per-primitive rule registry: every expression
//! node has a differentiation rule and the chain rule composes them as the tree
//! is walked. Because each row of a relational table is an independent
//! evaluation point, differentiating a column expression and letting the engine
//! evaluate it per row is the relational equivalent of `jax.vmap(jax.grad(f))`
//! (design.md §1). Both [`differentiate`] (one partial derivative) and [`jvp`]
//! (a directional derivative) are thin wrappers over [`linearize`] that differ
//! only in their *leaf rule* — the tangent assigned to each column.

use std::collections::HashMap;
use std::f64::consts::{LN_10, LN_2};
use std::sync::Arc;

use sqlparser::ast::{
    BinaryOperator, DataType, Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments,
    ObjectNamePart, UnaryOperator,
};

use crate::colref::{ColRef, IdentCasing, Match};
use crate::constructors::{
    add, as_const, div, finite_num, func, func1, is_zero, mul, neg, num, one, sign, square, sub,
    zero,
};
use crate::error::{DiffError, Result};

/// A one-line summary of what v1 differentiates, appended to "unsupported"
/// errors so a user reading the message learns what *is* supported and can act.
const SUPPORTED: &str = "ddx differentiates the operators + - * /, unary calls to \
sin/cos/tan/asin/acos/atan/exp/ln/log2/log10/sqrt/sinh/cosh/tanh/abs, power(...) with a \
constant base or exponent, casts to a numeric type, and column/literal leaves";

/// A differentiation rule for a unary primitive `f(u)`: given the argument
/// expression `u`, it returns the *outer* derivative `f'(u)`. The engine
/// multiplies by `du` (the chain rule) itself, so a user rule supplies only
/// the local factor (design.md §3.2).
pub type Rule = Arc<dyn Fn(&Expr) -> Result<Expr> + Send + Sync>;

/// A registry of differentiation rules, keyed by (lower-cased) function name.
///
/// Built-ins populate it; users extend it with [`RuleRegistry::register`]
/// (design.md §3.2 — the "extensible rule registry" decision).
#[derive(Clone)]
pub struct RuleRegistry {
    unary: HashMap<String, Rule>,
}

impl Default for RuleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrap an infallible closure as a [`Rule`].
fn rule(f: impl Fn(&Expr) -> Expr + Send + Sync + 'static) -> Rule {
    Arc::new(move |u| Ok(f(u)))
}

impl RuleRegistry {
    /// A registry populated with the built-in v1 rule set: `+ - * /`, the unary
    /// chain rule for the trig / inverse-trig / exp / log / hyperbolic set plus
    /// `abs`, and `power` with a constant base or exponent (design.md §3.6).
    pub fn new() -> Self {
        let mut unary: HashMap<String, Rule> = HashMap::new();

        // Trigonometric.
        unary.insert("sin".into(), rule(|u| func1("cos", u.clone())));
        unary.insert("cos".into(), rule(|u| neg(func1("sin", u.clone()))));
        unary.insert(
            "tan".into(),
            rule(|u| div(one(), square(func1("cos", u.clone())))),
        );
        // Inverse trigonometric.
        unary.insert(
            "asin".into(),
            rule(|u| div(one(), func1("sqrt", sub(one(), square(u.clone()))))),
        );
        unary.insert(
            "acos".into(),
            rule(|u| neg(div(one(), func1("sqrt", sub(one(), square(u.clone())))))),
        );
        unary.insert(
            "atan".into(),
            rule(|u| div(one(), add(one(), square(u.clone())))),
        );
        // Exponential / logarithmic.
        unary.insert("exp".into(), rule(|u| func1("exp", u.clone())));
        unary.insert("ln".into(), rule(|u| div(one(), u.clone())));
        unary.insert(
            "log2".into(),
            rule(|u| div(one(), mul(u.clone(), num(LN_2)))),
        );
        unary.insert(
            "log10".into(),
            rule(|u| div(one(), mul(u.clone(), num(LN_10)))),
        );
        unary.insert(
            "sqrt".into(),
            rule(|u| div(one(), mul(num(2.0), func1("sqrt", u.clone())))),
        );
        // Hyperbolic.
        unary.insert("sinh".into(), rule(|u| func1("cosh", u.clone())));
        unary.insert("cosh".into(), rule(|u| func1("sinh", u.clone())));
        unary.insert(
            "tanh".into(),
            rule(|u| sub(one(), square(func1("tanh", u.clone())))),
        );
        // Piecewise-linear: d/du |u| = sign(u), emitted as a portable CASE that
        // pins abs'(0) = 0 on every engine (design.md §5, F12). It deliberately
        // does NOT emit signum()/sign(): DuckDB has no signum (only sign),
        // DataFusion has no sign (only signum), and signum(0) = 1 — so a bare
        // builtin would be non-portable AND violate the pinned convention. Note
        // this pins ddx's own convention; jax.grad(abs)(0) uses a different one.
        unary.insert("abs".into(), rule(|u| sign(u.clone())));

        RuleRegistry { unary }
    }

    /// The unary function names this registry has a rule for, sorted.
    ///
    /// Read out of the registry rather than restated, so it cannot fall behind
    /// what is implemented — and so a test asking "is every rule covered?" is
    /// asking the engine instead of a second list that has to be maintained
    /// alongside the first. Includes rules a caller registered.
    pub fn unary_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.unary.keys().cloned().collect();
        names.sort();
        names
    }

    /// Register (or override) a unary differentiation rule under `name`. The
    /// name is matched case-insensitively.
    pub fn register(&mut self, name: &str, rule: Rule) {
        self.unary.insert(name.to_ascii_lowercase(), rule);
    }

    fn lookup(&self, name: &str) -> Option<&Rule> {
        self.unary.get(name)
    }
}

/// A *leaf rule*: the tangent seed for a column occurrence. Returns an error
/// when the occurrence's identity against the differentiation variable can't be
/// pinned syntactically (the ambiguity guard, F2).
type Leaf<'a> = dyn Fn(&ColRef) -> Result<Expr> + 'a;

/// Differentiate `expr` with respect to the column `wrt`.
///
/// Forward-mode with a one-hot seed: `1` on `wrt`, `0` on every other column.
pub fn differentiate(
    expr: &Expr,
    wrt: &ColRef,
    casing: IdentCasing,
    reg: &RuleRegistry,
) -> Result<Expr> {
    let leaf = |c: &ColRef| match c.classify(wrt, casing) {
        Match::Is => Ok(one()),
        Match::Not => Ok(zero()),
        Match::Ambiguous => Err(DiffError::AmbiguousColumn(format!(
            "occurrence of `{}` cannot be matched against differentiation \
             variable `{}` — fully qualify it",
            c.display(),
            wrt.display()
        ))),
    };
    linearize(expr, &leaf, reg)
}

/// Forward-mode directional derivative: the tangent of `expr` given a tangent
/// (`seeds`) for each seeded input column; unseeded columns are constant.
///
/// The marker form `jvp(expr, column, tangent)` seeds a single column; a
/// multi-input directional derivative is a sum of `jvp` terms (design.md §3.6).
pub fn jvp(
    expr: &Expr,
    seeds: &[(ColRef, Expr)],
    casing: IdentCasing,
    reg: &RuleRegistry,
) -> Result<Expr> {
    let leaf = |c: &ColRef| {
        for (col, tangent) in seeds {
            match c.classify(col, casing) {
                Match::Is => return Ok(tangent.clone()),
                Match::Ambiguous => {
                    return Err(DiffError::AmbiguousColumn(format!(
                        "occurrence of `{}` cannot be matched against seeded \
                         column `{}` — fully qualify it",
                        c.display(),
                        col.display()
                    )))
                }
                Match::Not => continue,
            }
        }
        Ok(zero())
    };
    linearize(expr, &leaf, reg)
}

/// Push tangents from the leaves up through `expr` via the chain rule.
fn linearize(expr: &Expr, leaf: &Leaf, reg: &RuleRegistry) -> Result<Expr> {
    match expr {
        // Leaves: the leaf rule decides a column's tangent.
        Expr::Identifier(_) | Expr::CompoundIdentifier(_) => {
            let cr = ColRef::from_expr(expr)
                .ok_or_else(|| DiffError::Internal("column expr yielded no ColRef".into()))?;
            leaf(&cr)
        }

        // Constants have zero tangent.
        Expr::Value(_) => Ok(zero()),

        // Parentheses are transparent to differentiation; the smart
        // constructors re-introduce any precedence parentheses the result needs.
        Expr::Nested(inner) => linearize(inner, leaf, reg),

        // A cast to a numeric type is locally linear: tangent of cast(u) =
        // cast(du) to the same type. A cast to a non-numeric type (VARCHAR,
        // DATE, BOOLEAN, …) is not differentiable — differentiating through it
        // would emit a nonsensical `CAST(1.0 AS VARCHAR)`, so it is a typed
        // error rather than a silently-wrong derivative (principle 5).
        Expr::Cast {
            kind,
            expr: inner,
            data_type,
            array,
            format,
        } => {
            if !is_numeric_type(data_type) {
                return Err(DiffError::NotImplemented(format!(
                    "differentiation through a cast to non-numeric type `{data_type}` \
                     is not supported"
                )));
            }
            let du = linearize(inner, leaf, reg)?;
            Ok(Expr::Cast {
                kind: kind.clone(),
                expr: Box::new(du),
                data_type: data_type.clone(),
                array: *array,
                format: format.clone(),
            })
        }

        // tangent of -u = -(du); unary plus is transparent.
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr: inner,
        } => Ok(neg(linearize(inner, leaf, reg)?)),
        Expr::UnaryOp {
            op: UnaryOperator::Plus,
            expr: inner,
        } => linearize(inner, leaf, reg),

        Expr::BinaryOp { left, op, right } => linearize_binary(left, op, right, leaf, reg),

        Expr::Function(f) => linearize_function(f, leaf, reg),

        other => Err(DiffError::NotImplemented(format!(
            "this expression cannot be differentiated: `{other}`. {SUPPORTED}; CASE, \
             comparisons, subqueries, window functions, and string/temporal expressions are \
             not differentiable"
        ))),
    }
}

/// Linearize a binary arithmetic expression via the sum/product/quotient rules.
fn linearize_binary(
    left: &Expr,
    op: &BinaryOperator,
    right: &Expr,
    leaf: &Leaf,
    reg: &RuleRegistry,
) -> Result<Expr> {
    let da = linearize(left, leaf, reg)?;
    let db = linearize(right, leaf, reg)?;
    match op {
        // tangent of (a + b) = da + db
        BinaryOperator::Plus => Ok(add(da, db)),
        // tangent of (a - b) = da - db
        BinaryOperator::Minus => Ok(sub(da, db)),
        // tangent of (a * b) = da*b + a*db   (product rule)
        BinaryOperator::Multiply => Ok(add(mul(da, right.clone()), mul(left.clone(), db))),
        // tangent of (a / b) = (da*b - a*db) / b^2   (quotient rule)
        BinaryOperator::Divide => {
            let numerator = sub(mul(da, right.clone()), mul(left.clone(), db));
            Ok(div(numerator, square(right.clone())))
        }
        other => Err(DiffError::NotImplemented(format!(
            "the operator `{other}` is not differentiable. {SUPPORTED}"
        ))),
    }
}

/// Linearize a scalar-function call via the chain rule.
fn linearize_function(f: &Function, leaf: &Leaf, reg: &RuleRegistry) -> Result<Expr> {
    let name = simple_func_name(f).ok_or_else(|| {
        DiffError::NotImplemented(format!(
            "cannot differentiate the call `{f}`: only an unqualified function name has a \
             differentiation rule (a schema-qualified or otherwise complex name is left alone)"
        ))
    })?;
    let args = positional_args(f).ok_or_else(|| {
        DiffError::NotImplemented(format!(
            "function `{name}` has non-positional arguments, which are not differentiable"
        ))
    })?;

    // `power(base, exponent)` / `pow(...)` is the one binary primitive.
    if name == "power" || name == "pow" {
        return linearize_power(&name, &args, leaf, reg);
    }

    if args.len() != 1 {
        return Err(DiffError::NotImplemented(format!(
            "no differentiation rule for `{name}` with {} arguments: the built-in function \
             rules are unary, and `power` is the only two-argument rule",
            args.len()
        )));
    }
    let u = args[0];
    let du = linearize(u, leaf, reg)?;
    // Chain-rule short-circuit: a zero inner tangent kills the whole term.
    if is_zero(&du) {
        return Ok(zero());
    }
    let outer = reg.lookup(&name).ok_or_else(|| {
        DiffError::NotImplemented(format!(
            "no differentiation rule for function `{name}`. {SUPPORTED}. Register a custom \
             rule with `Ddx::register(\"{name}\", ...)`"
        ))
    })?(u)?;
    Ok(mul(outer, du))
}

/// Linearize `power(base, exponent)` (design.md §3.6).
///
/// * Constant exponent `c`: `c * base^(c-1) * d(base)`.
/// * Constant base `a`: `a^u * ln(a) * d(u)`.
/// * Both variable (`u^v`): not supported yet (needs the exp/log trick).
fn linearize_power(name: &str, args: &[&Expr], leaf: &Leaf, reg: &RuleRegistry) -> Result<Expr> {
    if args.len() != 2 {
        return Err(DiffError::NotImplemented(format!(
            "{name}() expects exactly two arguments"
        )));
    }
    let base = args[0];
    let exponent = args[1];
    match (as_const(base), as_const(exponent)) {
        // Constant exponent (covers x^2, x^0.5, x^-2, ...).
        (_, Some(c)) => {
            let dbase = linearize(base, leaf, reg)?;
            if is_zero(&dbase) {
                return Ok(zero());
            }
            // `finite_num` fails loud on a non-finite constant (e.g. an
            // out-of-range literal `1e400` → inf), but *only here at emission* —
            // after the zero short-circuit above — so a wrt-independent base
            // still differentiates to `0` rather than erroring (#49/F1, #33).
            let outer = mul(
                finite_num(c)?,
                func("power", vec![base.clone(), finite_num(c - 1.0)?]),
            );
            Ok(mul(outer, dbase))
        }
        // Constant base, variable exponent.
        (Some(a), None) => {
            let dexp = linearize(exponent, leaf, reg)?;
            if is_zero(&dexp) {
                return Ok(zero());
            }
            // The derivative is `a^u · ln(a) · du`; `ln(a)` is non-finite for a
            // non-positive (or infinite) base — `finite_num` fails loud there
            // rather than emit an `inf`/`NaN` token (#33).
            let outer = mul(
                func("power", vec![base.clone(), exponent.clone()]),
                finite_num(a.ln())?,
            );
            Ok(mul(outer, dexp))
        }
        // General u^v — deferred (design.md §3.6 roadmap).
        (None, None) => Err(DiffError::NotImplemented(
            "cannot differentiate `power(base, exponent)` when both the base and the exponent \
             depend on the differentiation variable; ddx handles it only when one side is a \
             constant (e.g. `power(x, 2)` or `power(2, x)`). For the general u^v case, rewrite \
             it as `exp(exponent * ln(base))` when the base is positive"
                .into(),
        )),
    }
}

/// The lower-cased name of a function call — but only for an **unqualified**
/// call (a single-identifier name), mirroring the marker path's strict
/// `len() == 1` (F8). A schema-qualified call like `myschema.sin(x)` may be an
/// unrelated user function, so it must not silently match the built-in `sin`
/// rule ("tag explicitly, never infer" — principle 3; round-3 review #47).
fn simple_func_name(f: &Function) -> Option<String> {
    match f.name.0.as_slice() {
        [ObjectNamePart::Identifier(id)] => Some(id.value.to_ascii_lowercase()),
        _ => None,
    }
}

/// The positional (unnamed) argument expressions of a function call, or `None`
/// if it uses any non-positional argument form (named args, wildcards, a
/// subquery).
pub(crate) fn positional_args(f: &Function) -> Option<Vec<&Expr>> {
    match &f.args {
        FunctionArguments::List(list) => {
            let mut out = Vec::with_capacity(list.args.len());
            for arg in &list.args {
                match arg {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => out.push(e),
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// True if `dt` is a numeric type — the only kind of cast that is locally
/// linear (and so differentiable). The list is exhaustive for the pinned
/// `sqlparser` version; a `sqlparser` bump is already a breaking release of
/// `ddx-core` (design.md §6, G2), at which point this is re-checked.
pub(crate) fn is_numeric_type(dt: &DataType) -> bool {
    matches!(
        dt,
        // Floating-point / fixed-point.
        DataType::Numeric(_)
            | DataType::Decimal(_)
            | DataType::BigNumeric(_)
            | DataType::BigDecimal(_)
            | DataType::Dec(_)
            | DataType::Float(_)
            | DataType::FloatUnsigned(_)
            | DataType::Float4
            | DataType::Float32
            | DataType::Float64
            | DataType::Real
            | DataType::RealUnsigned
            | DataType::Float8
            | DataType::Double(_)
            | DataType::DoubleUnsigned(_)
            | DataType::DoublePrecision
            | DataType::DoublePrecisionUnsigned
            // Integers (signed / unsigned / width-tagged aliases).
            | DataType::TinyInt(_)
            | DataType::TinyIntUnsigned(_)
            | DataType::UTinyInt
            | DataType::Int2(_)
            | DataType::Int2Unsigned(_)
            | DataType::SmallInt(_)
            | DataType::SmallIntUnsigned(_)
            | DataType::USmallInt
            | DataType::MediumInt(_)
            | DataType::MediumIntUnsigned(_)
            | DataType::Int(_)
            | DataType::Int4(_)
            | DataType::Int8(_)
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Int128
            | DataType::Int256
            | DataType::Integer(_)
            | DataType::IntUnsigned(_)
            | DataType::Int4Unsigned(_)
            | DataType::IntegerUnsigned(_)
            | DataType::HugeInt
            | DataType::UHugeInt
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::UInt128
            | DataType::UInt256
            | DataType::BigInt(_)
            | DataType::BigIntUnsigned(_)
            | DataType::UBigInt
            | DataType::Int8Unsigned(_)
            | DataType::Signed
            | DataType::SignedInteger
            | DataType::Unsigned
            | DataType::UnsignedInteger
    )
}
