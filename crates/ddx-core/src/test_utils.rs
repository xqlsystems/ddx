// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Shared simulation harness for ddx's property/fuzz suites.
//!
//! **Test support, not public API.** Gated behind the `test-utils` feature, off
//! by default, and exempt from semver — it exists so every crate in the
//! workspace can fuzz against *one* expression generator and *one* reference
//! evaluator instead of forking them. When `ddx-datafusion` and `ddx-core`
//! disagree about what "a random derivable expression" means, a cross-crate
//! agreement test stops proving anything.
//!
//! What lives here is the *setup*: a deterministic PRNG, a generator over the
//! derivable v1 grammar, a reference interpreter for that grammar, the numeric
//! conditioning gates that keep a float oracle honest, statement-level
//! generators for `rewrite_sql`, and the failure-reporting loop. The
//! *properties* stay with the crate that owns the behaviour they assert.
//!
//! Everything is dependency-free and seed-reproducible: a failure reported with
//! its seed replays exactly.

use std::ops::ControlFlow;

use crate::sqlparser::ast::{
    BinaryOperator, Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments,
    ObjectNamePart, Statement, UnaryOperator, Value, Visit, Visitor,
};
use crate::sqlparser::dialect::GenericDialect;
use crate::sqlparser::parser::Parser;
use crate::{ColRef, Ddx};

// ---------------------------------------------------------------------------
// A tiny deterministic PRNG (SplitMix64) — reproducible, no dependencies.
// ---------------------------------------------------------------------------

/// A deterministic SplitMix64 generator.
///
/// Deliberately not a real RNG crate: the core is `sqlparser`-only by design,
/// and a fixed algorithm means a seed reported in a failure reproduces it
/// forever, across machines and toolchains.
pub struct Rng(u64);

impl Rng {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    /// The next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A `u64` in `[0, n)`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// A float in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }

    /// Pick one element of a non-empty slice.
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

/// The seed schedule the bounded suites share, so a seed index means the same
/// thing everywhere and a `salt` keeps different properties on different ground.
pub fn seeded(seed: u64, salt: u64) -> Rng {
    Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ salt)
}

/// The differentiation variable a check runs against.
///
/// [`gen_expr`] emits exactly two free variables, `x` and `y`, so a numeric
/// oracle only ever has to perturb one of two directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Var {
    /// The variable `x`.
    X,
    /// The variable `y`.
    Y,
}

impl Var {
    /// The variable's name as it appears in generated SQL.
    pub fn name(self) -> &'static str {
        match self {
            Var::X => "x",
            Var::Y => "y",
        }
    }

    /// The variable as a bare [`ColRef`].
    pub fn col(self) -> ColRef {
        ColRef::bare(self.name())
    }
}

// ---------------------------------------------------------------------------
// Random expression generator over the *derivable* v1 grammar.
// ---------------------------------------------------------------------------
//
// Everything produced here is inside the engine's supported surface, so
// `differentiate` never returns `NotImplemented`: vars {x, y}, numeric
// literals, `+ - * /`, unary minus, a numeric `CAST`, the unary-rule function
// set, and `power` with exactly one constant side. It emits SQL text
// (parenthesized to fix structure), which both exercises the parser and gives
// readable failures.

/// Unary functions that have a differentiation rule (design.md §3.6).
pub const UNARY_FNS: &[&str] = &[
    "sin", "cos", "tan", "asin", "acos", "atan", "exp", "ln", "log2", "log10", "sqrt", "sinh",
    "cosh", "tanh", "abs",
];

/// A *non-negative* constant, safe to place under a generated unary minus
/// without producing a `--` line-comment in the source text. (Negative literals
/// still appear — as `power` exponents, below — where they are direct function
/// arguments, and via the engine's own `num()` output.)
fn gen_const(rng: &mut Rng) -> String {
    rng.pick(&["2", "3", "0.5", "1.5", "2.5"]).to_string()
}

/// A constant `power` exponent, which *may* be negative — passed as a direct
/// call argument (`power(x, -2)`), never wrapped in a unary minus, so it never
/// forms a `--` in the generated text.
fn gen_exponent(rng: &mut Rng) -> String {
    rng.pick(&["2", "3", "0.5", "1.5", "-1", "-2", "-0.5", "2.5"])
        .to_string()
}

/// A random expression over the derivable v1 grammar, as SQL text.
///
/// `depth` bounds the tree height; leaves appear early with ~30% probability at
/// every level, so the size distribution is broad rather than uniform.
pub fn gen_expr(rng: &mut Rng, depth: u32) -> String {
    if depth == 0 || rng.below(100) < 30 {
        // Leaf: a variable or a constant.
        return match rng.below(5) {
            0 | 1 => "x".to_string(),
            2 => "y".to_string(),
            _ => gen_const(rng),
        };
    }
    match rng.below(11) {
        0 => format!(
            "({} + {})",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        1 => format!(
            "({} - {})",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        2 => format!(
            "({} * {})",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        3 => format!(
            "({} / {})",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        4 => format!("(-{})", gen_expr(rng, depth - 1)),
        5 | 6 => {
            let f = rng.pick(UNARY_FNS);
            format!("{f}({})", gen_expr(rng, depth - 1))
        }
        7 => format!("power({}, {})", gen_expr(rng, depth - 1), gen_exponent(rng)),
        8 => {
            // power(positive-const-base, variable-exponent)
            let base = rng.pick(&["2", "3", "1.5", "0.5"]);
            format!("power({base}, {})", gen_expr(rng, depth - 1))
        }
        _ => format!("CAST({} AS DOUBLE)", gen_expr(rng, depth - 1)),
    }
}

/// A random derivable expression plus a random differentiation variable.
pub fn gen_expr_and_wrt(rng: &mut Rng) -> (String, Var) {
    let depth = 2 + rng.below(4) as u32;
    let text = gen_expr(rng, depth);
    let wrt = if rng.below(2) == 0 { Var::X } else { Var::Y };
    (text, wrt)
}

// ---------------------------------------------------------------------------
// Parsing helpers.
// ---------------------------------------------------------------------------

/// Parse a scalar expression, returning the parser's message on failure.
pub fn try_parse(text: &str) -> Result<Expr, String> {
    Parser::new(&GenericDialect {})
        .try_with_sql(text)
        .and_then(|mut p| p.parse_expr())
        .map_err(|e| e.to_string())
}

/// [`try_parse`], panicking on failure — for text a generator just produced,
/// where a parse failure is a harness bug worth surfacing immediately.
pub fn parse_expr(text: &str) -> Expr {
    try_parse(text).unwrap_or_else(|e| panic!("reparse of `{text}` failed: {e}"))
}

/// Parse a whole statement (not just an expression).
pub fn try_parse_stmt(sql: &str) -> Result<Vec<Statement>, String> {
    Parser::parse_sql(&GenericDialect {}, sql).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// A reference interpreter for the emitted grammar (primal *and* derivative).
// ---------------------------------------------------------------------------

/// Evaluate a scalar expression at `(x, y)`.
///
/// Returns `None` for anything outside the numeric grammar, so an unexpected
/// node fails a comparison loudly rather than silently returning a bogus number.
pub fn eval(e: &Expr, x: f64, y: f64) -> Option<f64> {
    eval_mag(e, x, y).map(|(v, _)| v)
}

/// The largest absolute value taken by any subexpression of `e` at `(x, y)` —
/// the "how big did the intermediates get" probe.
///
/// A point where this is huge is unfit for a float oracle: f64 can no longer
/// resolve an O(1) perturbation against it (a huge additive term cancels the
/// perturbation away; a huge argument to `sin`/`cos` aliases), so *neither* a
/// finite difference *nor* the symbolic value is meaningful there — the point
/// must be skipped.
pub fn max_intermediate_mag(e: &Expr, x: f64, y: f64) -> Option<f64> {
    eval_mag(e, x, y).map(|(_, m)| m)
}

/// Evaluate `e`, returning `(value, max_abs_intermediate)`.
pub fn eval_mag(e: &Expr, x: f64, y: f64) -> Option<(f64, f64)> {
    let here = |v: f64| Some((v, v.abs()));
    match e {
        Expr::Value(v) => match &v.value {
            Value::Number(s, _) => here(s.parse::<f64>().ok()?),
            _ => None,
        },
        Expr::Identifier(id) => match id.value.to_ascii_lowercase().as_str() {
            "x" => here(x),
            "y" => here(y),
            _ => None,
        },
        Expr::CompoundIdentifier(parts) => {
            match parts.last()?.value.to_ascii_lowercase().as_str() {
                "x" => here(x),
                "y" => here(y),
                _ => None,
            }
        }
        Expr::Nested(inner) => eval_mag(inner, x, y),
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => {
            let (v, m) = eval_mag(expr, x, y)?;
            Some((-v, m.max(v.abs())))
        }
        Expr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => eval_mag(expr, x, y),
        Expr::BinaryOp { left, op, right } => {
            let (a, ma) = eval_mag(left, x, y)?;
            let (b, mb) = eval_mag(right, x, y)?;
            let r = match op {
                BinaryOperator::Plus => a + b,
                BinaryOperator::Minus => a - b,
                BinaryOperator::Multiply => a * b,
                BinaryOperator::Divide => a / b,
                _ => return None,
            };
            Some((r, ma.max(mb).max(r.abs())))
        }
        // Numeric casts are the identity on f64.
        Expr::Cast { expr, .. } => eval_mag(expr, x, y),
        Expr::Function(f) => eval_function(f, x, y),
        // The `sign` CASE (the only CASE the engine emits).
        Expr::Case {
            operand: None,
            conditions,
            else_result,
            ..
        } => {
            let mut m = 0.0f64;
            for w in conditions {
                // Track the compared operand's magnitude too.
                if let Expr::BinaryOp { left, .. } = &w.condition {
                    if let Some((_, lm)) = eval_mag(left, x, y) {
                        m = m.max(lm);
                    }
                }
                if eval_bool(&w.condition, x, y)? {
                    let (v, rm) = eval_mag(&w.result, x, y)?;
                    return Some((v, m.max(rm)));
                }
            }
            let (v, rm) = eval_mag(else_result.as_deref()?, x, y)?;
            Some((v, m.max(rm)))
        }
        _ => None,
    }
}

fn eval_bool(e: &Expr, x: f64, y: f64) -> Option<bool> {
    if let Expr::BinaryOp { left, op, right } = e {
        let a = eval(left, x, y)?;
        let b = eval(right, x, y)?;
        return match op {
            BinaryOperator::Gt => Some(a > b),
            BinaryOperator::Lt => Some(a < b),
            BinaryOperator::GtEq => Some(a >= b),
            BinaryOperator::LtEq => Some(a <= b),
            BinaryOperator::Eq => Some(a == b),
            BinaryOperator::NotEq => Some(a != b),
            _ => None,
        };
    }
    None
}

fn eval_function(f: &Function, x: f64, y: f64) -> Option<(f64, f64)> {
    let [ObjectNamePart::Identifier(id)] = f.name.0.as_slice() else {
        return None;
    };
    let name = id.value.to_ascii_lowercase();
    let FunctionArguments::List(list) = &f.args else {
        return None;
    };
    let mut args = Vec::new();
    let mut argmag = 0.0f64;
    for a in &list.args {
        match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => {
                let (v, m) = eval_mag(e, x, y)?;
                args.push(v);
                argmag = argmag.max(m);
            }
            _ => return None,
        }
    }
    let a0 = *args.first()?;
    let v = match name.as_str() {
        "sin" => a0.sin(),
        "cos" => a0.cos(),
        "tan" => a0.tan(),
        "asin" => a0.asin(),
        "acos" => a0.acos(),
        "atan" => a0.atan(),
        "exp" => a0.exp(),
        "ln" => a0.ln(),
        "log2" => a0.log2(),
        "log10" => a0.log10(),
        "sqrt" => a0.sqrt(),
        "sinh" => a0.sinh(),
        "cosh" => a0.cosh(),
        "tanh" => a0.tanh(),
        "abs" => a0.abs(),
        "power" | "pow" => {
            let e1 = *args.get(1)?;
            a0.powf(e1)
        }
        _ => return None,
    };
    Some((v, argmag.max(v.abs())))
}

/// The smallest distance from any *restricted-domain* function call in `e` to
/// its domain boundary at `(x, y)` — `f64::INFINITY` if there are none.
///
/// The derivative of a restricted-domain primitive is singular *at* the
/// boundary even where the primal is finite (design.md §5, "domain-widening"):
/// `acos`/`asin` have `±1` (`d = ∓1/√(1−u²)`), `sqrt`/`ln`/`log` have `0`
/// (`d = 1/(2√u)`, `1/u`), and division has a `0` denominator. Near such a
/// boundary the symbolic derivative is a `0·∞`/`∞` form that f64 evaluates to
/// garbage — e.g. `sqrt(acos(x·0.5^log2(x)))` where the argument is *identically
/// 1*. Numeric oracles skip these points rather than mistake a
/// numerically-singular (but symbolically correct) derivative for a bug.
pub fn min_domain_margin(e: &Expr, x: f64, y: f64) -> Option<f64> {
    match e {
        Expr::Value(_) | Expr::Identifier(_) | Expr::CompoundIdentifier(_) => Some(f64::INFINITY),
        Expr::Nested(i) => min_domain_margin(i, x, y),
        Expr::UnaryOp { expr, .. } | Expr::Cast { expr, .. } => min_domain_margin(expr, x, y),
        Expr::BinaryOp { left, op, right } => {
            let mut m = min_domain_margin(left, x, y)?.min(min_domain_margin(right, x, y)?);
            if matches!(op, BinaryOperator::Divide) {
                m = m.min(eval(right, x, y)?.abs());
            }
            Some(m)
        }
        Expr::Function(Function {
            name,
            args: FunctionArguments::List(list),
            ..
        }) => {
            let mut arg_exprs = Vec::new();
            let mut m = f64::INFINITY;
            for a in &list.args {
                let FunctionArg::Unnamed(FunctionArgExpr::Expr(ae)) = a else {
                    return Some(f64::INFINITY);
                };
                m = m.min(min_domain_margin(ae, x, y)?);
                arg_exprs.push(ae);
            }
            let fname = match name.0.as_slice() {
                [ObjectNamePart::Identifier(id)] => id.value.to_ascii_lowercase(),
                _ => return Some(m),
            };
            if let Some(a0) = arg_exprs.first() {
                let v = eval(a0, x, y)?;
                m = m.min(match fname.as_str() {
                    "asin" | "acos" => 1.0 - v.abs(),
                    "sqrt" | "ln" | "log2" | "log10" => v,
                    _ => f64::INFINITY,
                });
            }
            Some(m)
        }
        Expr::Case {
            conditions,
            else_result,
            ..
        } => {
            let mut m = f64::INFINITY;
            for w in conditions {
                m = m.min(min_domain_margin(&w.condition, x, y)?);
                m = m.min(min_domain_margin(&w.result, x, y)?);
            }
            if let Some(er) = else_result {
                m = m.min(min_domain_margin(er, x, y)?);
            }
            Some(m)
        }
        _ => Some(f64::INFINITY),
    }
}

/// A central finite difference of `f` in the `wrt` direction at `(x0, y0)`,
/// step `h`.
pub fn central_diff(f: &Expr, x0: f64, y0: f64, wrt: Var, h: f64) -> Option<f64> {
    let (fp, fm) = match wrt {
        Var::X => (eval(f, x0 + h, y0)?, eval(f, x0 - h, y0)?),
        Var::Y => (eval(f, x0, y0 + h)?, eval(f, x0, y0 - h)?),
    };
    Some((fp - fm) / (2.0 * h))
}

// ---------------------------------------------------------------------------
// Numeric conditioning: is this point fit to compare floats at?
// ---------------------------------------------------------------------------

/// The conditioning gates a float oracle needs to stay honest.
///
/// These are not tolerances — they decide whether a *point* is comparable at
/// all. Skipping an unfit point is the difference between an oracle with teeth
/// and one that reports correct derivatives as bugs (#54).
#[derive(Clone, Copy, Debug)]
pub struct Conditioning {
    /// Skip points within this distance of a restricted-domain boundary.
    pub domain_eps: f64,
    /// Skip points where any intermediate exceeds this magnitude.
    pub mag_cap: f64,
}

impl Default for Conditioning {
    fn default() -> Self {
        Conditioning {
            domain_eps: 1e-3,
            mag_cap: 1e8,
        }
    }
}

impl Conditioning {
    /// Is `(x, y)` a point at which `exprs` can be meaningfully compared?
    ///
    /// Every expression must evaluate, stay finite, keep its intermediates
    /// under [`Conditioning::mag_cap`], and stay [`Conditioning::domain_eps`]
    /// clear of every restricted-domain boundary.
    pub fn admits(&self, exprs: &[&Expr], x: f64, y: f64) -> bool {
        exprs.iter().all(|e| {
            matches!(min_domain_margin(e, x, y), Some(m) if m >= self.domain_eps)
                && matches!(max_intermediate_mag(e, x, y), Some(m) if m.is_finite() && m <= self.mag_cap)
                && matches!(eval(e, x, y), Some(v) if v.is_finite())
        })
    }
}

/// Do two floats agree to within `atol + rtol · scale`?
///
/// **Tolerance is relative to the computation scale, not the result** — the #54
/// lesson. Two expressions that compute the same value by different
/// *associations* agree only up to ≈ `ε · (magnitude of the intermediates)`,
/// which explodes past a result-relative tolerance at a cancellation or
/// near-singular point. A real bug perturbs the result by a finite fraction of
/// its own scale, so it still exceeds `rtol · scale` at well-conditioned
/// points — the check keeps its teeth while shedding float-noise false
/// positives.
pub fn close_at_scale(a: f64, b: f64, scale: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * scale.abs().max(1.0)
}

/// Compare a symbolic `lhs` expression to a `rhs` value closure at random
/// well-conditioned points, returning the first genuine disagreement as
/// `(x, y, lhs_value, rhs_value)`.
///
/// Used for exact metamorphic identities (jvp↔grad, linearity) — as exact
/// identities, not finite differences, they need no majority vote.
pub fn metamorphic_mismatch(
    rng: &mut Rng,
    gate: &[&Expr],
    lhs: &Expr,
    rhs: impl Fn(f64, f64) -> Option<f64>,
) -> Option<(f64, f64, f64, f64)> {
    const RTOL: f64 = 1e-7;
    const ATOL: f64 = 1e-9;
    for _ in 0..40 {
        let x0 = rng.range(0.2, 1.8);
        let y0 = rng.range(0.2, 1.8);
        // Scale = the largest intermediate magnitude across every expression
        // involved; skip the point if any is non-finite or overflowing.
        let mut scale = 0.0f64;
        let mut ok = true;
        for e in gate.iter().chain(std::iter::once(&lhs)) {
            match max_intermediate_mag(e, x0, y0) {
                Some(m) if m.is_finite() && m < 1e300 => scale = scale.max(m),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let (Some(a), Some(b)) = (eval(lhs, x0, y0), rhs(x0, y0)) else {
            continue;
        };
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        if (a - b).abs() > ATOL + RTOL * scale {
            return Some((x0, y0, a, b));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Statement-level generators (the `rewrite_sql` / marker-placement surface).
// ---------------------------------------------------------------------------

/// Valid SELECT-list prefixes to place before a marker. Several carry multibyte
/// characters (in string literals / comments) *before* the marker, so the
/// marker's character-column no longer equals its byte offset — the case a
/// char→byte conversion (G3) must get right.
pub const STMT_PREFIXES: &[&str] = &[
    "SELECT ",
    "SELECT x, ",
    "SELECT 'héllo', ",
    "SELECT 'naïve café ☕' AS greeting, ",
    "SELECT /* café ☕ */ ",
    "SELECT   ",
    "SELECT y AS why, ",
];

/// Valid statement tails (ASCII identifiers only — unicode kept to string
/// literals/comments, since unquoted-identifier unicode support is dialect-
/// dependent and not what this fuzz is testing).
pub const STMT_SUFFIXES: &[&str] = &[
    " FROM t",
    " AS d FROM t",
    " FROM data",
    " AS d FROM t WHERE label <> 'niño'",
    "",
];

/// Valid separators between two markers — as sibling select items (`, `) or
/// inside one arithmetic select item (` + `, ` * `).
pub const STMT_MIDS: &[&str] = &[", ", " + ", " * ", ", z, "];

/// Build one marker call and the exact text `rewrite_sql` must splice in its
/// place (`(derivative)`), or `None` if the marker's derivative is undefined
/// (in which case `rewrite_sql` must error on the whole statement).
pub fn gen_marker_segment(rng: &mut Rng, ddx: &Ddx) -> (String, Option<String>) {
    let wrt = if rng.below(2) == 0 { "x" } else { "y" };
    let wrt_col = ColRef::bare(wrt);
    let depth = 2 + rng.below(2) as u32;
    let expr_text = gen_expr(rng, depth);
    let expr = parse_expr(&expr_text);

    match rng.below(3) {
        // Nested higher-order: grad(grad(expr, wrt), wrt).
        0 => {
            let marker = format!("grad(grad({expr_text}, {wrt}), {wrt})");
            let repl = ddx
                .differentiate(&expr, &wrt_col)
                .and_then(|d1| ddx.differentiate(&d1, &wrt_col))
                .ok()
                .map(|dd| format!("({dd})"));
            (marker, repl)
        }
        // jvp(expr, wrt, tangent).
        1 => {
            let tan_depth = 1 + rng.below(2) as u32;
            let tan_text = gen_expr(rng, tan_depth);
            let tan = parse_expr(&tan_text);
            let marker = format!("jvp({expr_text}, {wrt}, {tan_text})");
            match ddx.jvp(&expr, &[(wrt_col, tan)]) {
                Ok(v) => (marker, Some(format!("({v})"))),
                Err(_) => (marker, None),
            }
        }
        // grad(expr, wrt)
        _ => {
            let marker = format!("grad({expr_text}, {wrt})");
            match ddx.differentiate(&expr, &wrt_col) {
                Ok(d) => (marker, Some(format!("({d})"))),
                Err(_) => (marker, None),
            }
        }
    }
}

/// Assemble a random marker-bearing statement: 1–3 markers wrapped in random
/// (Unicode-bearing) scaffolding. Returns `(input, expected)` where `expected`
/// is the exact byte-for-byte `rewrite_sql` output, or `None` if some marker's
/// derivative is undefined (in which case the whole rewrite must error).
pub fn gen_marker_statement(rng: &mut Rng, ddx: &Ddx) -> (String, Option<String>) {
    let n = 1 + rng.below(3) as usize;
    let prefix = *rng.pick(STMT_PREFIXES);
    let suffix = *rng.pick(STMT_SUFFIXES);

    let mut input = String::from(prefix);
    let mut expected = String::from(prefix);
    let mut any_undefined = false;
    for i in 0..n {
        if i > 0 {
            let mid = *rng.pick(STMT_MIDS);
            input.push_str(mid);
            expected.push_str(mid);
        }
        let (marker, repl) = gen_marker_segment(rng, ddx);
        input.push_str(&marker);
        match repl {
            Some(r) => expected.push_str(&r),
            None => any_undefined = true,
        }
    }
    input.push_str(suffix);
    expected.push_str(suffix);

    (input, if any_undefined { None } else { Some(expected) })
}

/// A marker-free statement — some deliberately containing a `grad(`/`jvp(`
/// substring inside a string literal, a comment, or a *qualified* call, so a
/// pre-gate's substring filter hits and the statement is parsed but no real
/// marker is found. Every one must come back byte-identical.
pub fn gen_marker_free_stmt(rng: &mut Rng) -> String {
    match rng.below(6) {
        0 => {
            let depth = 2 + rng.below(2) as u32;
            format!("SELECT {} FROM t", gen_expr(rng, depth))
        }
        1 => "SELECT 'grad(x, x)' AS s FROM t".to_string(),
        2 => "SELECT x /* grad(y, y) */ FROM t".to_string(),
        3 => "SELECT myschema.grad(x, x) AS d FROM t".to_string(),
        4 => "SELECT 'jvp(sin(x), x, dx)' AS label, x FROM t".to_string(),
        _ => format!(
            "SELECT {} AS val FROM t WHERE label <> 'grad('",
            gen_expr(rng, 2)
        ),
    }
}

/// Unicode strings used to stress char↔byte offset arithmetic.
pub const UNICODE_STRESSORS: &[&str] = &[
    "héllo",
    "☕🔥",
    "naïve",
    "Ωμέγα",
    "🇺🇸",
    "e\u{0301}",
    "\u{202E}rtl",
    "𝕏𝕐",
];

/// Adversarial / malformed inputs for the never-panic invariant.
pub fn gen_adversarial_sql(rng: &mut Rng) -> String {
    match rng.below(8) {
        // Malformed marker arities / shapes — must be typed errors, not panics.
        0 => rng
            .pick(&[
                "SELECT grad() FROM t",
                "SELECT grad(x) FROM t",
                "SELECT grad(x, y, z) FROM t",
                "SELECT jvp(x, x) FROM t",
                "SELECT jvp(x) FROM t",
                "SELECT grad(x, 1 + 2) FROM t",
                "SELECT grad(*, x) FROM t",
                "SELECT grad(x, ) FROM t",
            ])
            .to_string(),
        // A valid marker behind a Unicode-heavy prefix (stresses offset math).
        1 => {
            let u = *rng.pick(UNICODE_STRESSORS);
            let depth = 2 + rng.below(2) as u32;
            format!(
                "SELECT '{u}' AS c, grad({}, x) FROM t",
                gen_expr(rng, depth)
            )
        }
        // Deeply nested markers.
        2 => {
            let k = 1 + rng.below(12) as usize;
            let mut s = String::from("SELECT ");
            for _ in 0..k {
                s.push_str("grad(");
            }
            s.push('x');
            for _ in 0..k {
                s.push_str(", x)");
            }
            s.push_str(" FROM t");
            s
        }
        // A valid marker statement truncated at a random char boundary.
        3 => {
            let full = "SELECT café AS c, grad(sin(x) * y, x) FROM t";
            let cut = full
                .char_indices()
                .nth(1 + rng.below(full.chars().count() as u64) as usize)
                .map(|(i, _)| i)
                .unwrap_or(full.len());
            full[..cut].to_string()
        }
        // Marker with a Unicode identifier argument.
        4 => {
            let u = *rng.pick(UNICODE_STRESSORS);
            format!("SELECT grad({u}, x) FROM t")
        }
        // Unicode injected into a valid marker statement at a char boundary.
        5 => {
            let u = *rng.pick(UNICODE_STRESSORS);
            let base = "SELECT grad(sin(x), x) FROM t";
            let at = base
                .char_indices()
                .nth(rng.below(base.chars().count() as u64) as usize)
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut s = String::from(&base[..at]);
            s.push_str(u);
            s.push_str(&base[at..]);
            s
        }
        // Odd whitespace/comments around the marker (the #52 family).
        6 => "SELECT grad\t(\n x , x )/* ☕ */ FROM t".to_string(),
        // A deep but valid marker payload.
        _ => {
            let depth = 3 + rng.below(3) as u32;
            format!("SELECT grad({}, x) FROM t", gen_expr(rng, depth))
        }
    }
}

// ---------------------------------------------------------------------------
// Residual-marker detection.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct MarkerScan {
    found: bool,
}

impl Visitor for MarkerScan {
    type Break = ();
    fn pre_visit_expr(&mut self, e: &Expr) -> ControlFlow<()> {
        if let Expr::Function(Function { name, .. }) = e {
            if let [ObjectNamePart::Identifier(id)] = name.0.as_slice() {
                let n = id.value.to_ascii_lowercase();
                if n == "grad" || n == "jvp" {
                    self.found = true;
                    return ControlFlow::Break(());
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// Does an AST still contain an *unqualified* `grad`/`jvp` call — a marker that
/// should have been rewritten away?
pub fn has_residual_marker(stmts: &[Statement]) -> bool {
    let mut scan = MarkerScan::default();
    for s in stmts {
        let _ = Visit::visit(s, &mut scan);
        if scan.found {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Reporting.
// ---------------------------------------------------------------------------

/// How many failure reports a bounded run prints before truncating.
pub const REPORT_LIMIT: usize = 15;

/// Collected failures from one bounded run, each tagged with the seed that
/// reproduces it.
#[derive(Default, Debug)]
pub struct Failures {
    reports: Vec<String>,
    tested: u32,
}

impl Failures {
    /// An empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a failure found at `seed`.
    pub fn push(&mut self, seed: u64, report: impl Into<String>) {
        self.reports.push(format!("seed={seed}: {}", report.into()));
    }

    /// Count one exercised (non-skipped) case.
    pub fn tested(&mut self) {
        self.tested += 1;
    }

    /// How many cases were exercised, as opposed to skipped.
    pub fn tested_count(&self) -> u32 {
        self.tested
    }

    /// How many failures were recorded.
    pub fn len(&self) -> usize {
        self.reports.len()
    }

    /// Were there no failures?
    pub fn is_empty(&self) -> bool {
        self.reports.is_empty()
    }

    /// Panic with every recorded failure (truncated to [`REPORT_LIMIT`]) if any
    /// were found, and assert that at least `min_tested` cases were exercised
    /// so a silently-skipping generator cannot masquerade as a pass.
    #[track_caller]
    pub fn assert_clean(&self, label: &str, min_tested: u32) {
        assert!(
            self.tested >= min_tested,
            "{label}: only {} case(s) were exercised (expected at least {min_tested}) — \
             the generator or its conditioning gates are skipping almost everything, \
             so a pass here would prove nothing",
            self.tested
        );
        assert!(
            self.is_empty(),
            "{label} found {} failure(s) out of {} case(s) tested:\n\n{}{}",
            self.len(),
            self.tested,
            self.reports
                .iter()
                .take(REPORT_LIMIT)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n\n"),
            if self.len() > REPORT_LIMIT {
                format!("\n\n… and {} more", self.len() - REPORT_LIMIT)
            } else {
                String::new()
            },
        );
    }
}

/// Run a per-seed check over `0..n`, collecting every failure, then assert the
/// run was clean.
///
/// The check returns `Some(report)` for a failure and `None` for a pass or a
/// skip; it is handed a freshly seeded [`Rng`] so each iteration replays alone.
pub fn run_bounded<F>(label: &str, n: u64, salt: u64, mut check: F)
where
    F: FnMut(&mut Rng) -> Option<String>,
{
    let mut failures = Failures::new();
    for seed in 0..n {
        let mut rng = seeded(seed, salt);
        failures.tested();
        if let Some(report) = check(&mut rng) {
            failures.push(seed, report);
        }
    }
    failures.assert_clean(label, 1);
}
