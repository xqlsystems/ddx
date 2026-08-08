// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Simulation / property-based tests for the v1 differentiation engine
//! (design.md §5: "numeric agreement" + "round-trip property tests").
//!
//! These are adversarial: instead of hand-picked expressions, they generate
//! random derivable SQL scalar expressions and hold the engine to three
//! properties any correct symbolic differentiator must satisfy:
//!
//! 1. **Numeric agreement (the finite-difference oracle).** The single
//!    strongest check on a derivative: for a random `f`, the symbolic `d/dv f`
//!    evaluated at a point must equal a central finite difference of `f` in the
//!    `v` direction there. A wrong rule (a sign flip, a missing chain factor, a
//!    bad power exponent) disagrees at *every* well-conditioned point, so it is
//!    caught even though a kink artifact (from `abs`) is tolerated as a lone
//!    outlier. Proven to have teeth by mutation testing (a corrupted `cos` rule
//!    fails with 8/8-points-disagree).
//! 2. **Render fidelity.** `reparse(render(d))` must be *value-equal* to `d`.
//!    This is the correctness-relevant form of the §5 round-trip invariant: a
//!    purely structural "== d modulo Nested" check is imprecise for `*`/`/`
//!    associativity (issue #50), but a value comparison still catches the G1
//!    precedence bug (`(a+b)*c` losing its parens → `a+b*c`).
//! 3. **Self-consumption / higher-order stability.** The engine must re-parse
//!    and re-differentiate its *own* text output repeatedly without panicking,
//!    erroring, or emitting unparseable SQL (e.g. a `--` line comment).
//!
//! No external fuzzing crate is used: the core is deliberately `sqlparser`-only,
//! and a dependency-free, deterministic generator keeps every failure perfectly
//! reproducible (each is reported with the seed that produced it).
//!
//! # Soak mode
//!
//! [`soak_continuous_property_fuzz`] is a long-running, `#[ignore]`-d variant
//! that explores far past the bounded tests' fixed seed ranges. It runs for a
//! wall-clock budget and keeps generating fresh expressions, so it can be left
//! running to hunt for rare bugs. Drive it with env vars:
//!
//! ```text
//! DDX_SOAK_SECS=300   cargo test -p ddx-core --test simulation \
//! DDX_SOAK_BASE=0       -- --ignored --nocapture soak_continuous_property_fuzz
//! DDX_SOAK_LOG=/path/to/soak.log
//! ```
//!
//! * `DDX_SOAK_SECS` — wall-clock budget in seconds (default 15).
//! * `DDX_SOAK_BASE` — starting seed offset; bump it between runs to cover new
//!   ground (default 0).
//! * `DDX_SOAK_LOG`  — if set, failures are appended immediately and a heartbeat
//!   line is written ~once a second, so a background run can be tailed live.

use std::fmt::Write as _;
use std::io::Write as _;

use ddx_core::sqlparser::ast::Expr;
use ddx_core::sqlparser::dialect::GenericDialect;
use ddx_core::test_utils::{
    central_diff, divides_by_noise, eval, gen_adversarial_sql, gen_expr, gen_expr_and_wrt,
    gen_marker_free_stmt, gen_marker_statement, has_residual_marker, max_intermediate_mag,
    metamorphic_mismatch, min_domain_margin, parse_expr, run_bounded, seeded, try_parse,
    try_parse_stmt, Rng, Var,
};
use ddx_core::{ColRef, Ddx, DiffError};

// ---------------------------------------------------------------------------
// The three property checks, as reusable helpers.
// ---------------------------------------------------------------------------

/// Property 1: symbolic `d` vs a central finite difference of `f` (`expr_text`)
/// in the `wrt` direction. Returns `Some(report)` when a strong majority of
/// well-conditioned points disagree (a real rule bug), tolerating a lone
/// `abs`-kink outlier.
///
/// **Richardson self-consistency gate.** A finite difference is only trusted at
/// a point where halving the step barely moves it (`fd(h) ≈ fd(h/2)`). This is
/// what makes the oracle sound at depth 5–6, where the generator reaches
/// pathological shapes a plain central difference mis-handles — proven necessary
/// by an earlier soak that flagged 16 *correct* derivatives (#54). It kills two
/// false-positive families. Catastrophic cancellation: `power(3, y…) + x`, where
/// the `3^96 ≈ 1e45` term swamps the `+x`, so `f(x+h) − f(x−h)` loses it to
/// float rounding (fd wrongly reads `0`) — halving `h` doubles that error, so
/// the two disagree and the point is skipped. Truncation / aliasing:
/// `sin(exp(9+x))` oscillates with period ≈ `h`, so the central difference is
/// out of its asymptotic regime — halving `h` changes it materially, so the
/// point is skipped. Only points where the difference is in its convergent
/// regime are compared to the symbolic derivative, so a surviving disagreement
/// is a real rule bug.
fn fd_failure(rng: &mut Rng, expr_text: &str, d: &Expr, wrt: Var) -> Option<String> {
    const H: f64 = 1e-4;
    const RTOL: f64 = 2e-3;
    const ATOL: f64 = 1e-5;
    const COND_CAP: f64 = 1e5; // skip near-singular points (huge slope)
                               // Max relative gap between fd(h) and fd(h/2) for the difference to count as
                               // "in its convergent regime" and therefore trustworthy as an oracle.
    const RICHARDSON_TOL: f64 = 1e-4;
    // Above this, some intermediate value is too large for f64 to resolve an
    // O(1) perturbation against — the point is unfit for numeric comparison
    // (total cancellation passes Richardson because *both* fd(h) and fd(h/2)
    // collapse to the same wrong value, so this magnitude gate is what catches
    // it — #54).
    const MAG_CAP: f64 = 1e8;
    // Skip points within this distance of a restricted-domain boundary
    // (`acos`/`asin` at ±1, `sqrt`/`ln`/`log` at 0, division at 0), where the
    // symbolic derivative is a singular `0·∞` form f64 can't evaluate.
    const DOMAIN_EPS: f64 = 1e-3;

    let f = parse_expr(expr_text);
    let mut comparable = 0u32;
    let mut disagree = 0u32;
    let mut first_bad = String::new();

    for _ in 0..80 {
        if comparable >= 8 {
            break;
        }
        let x0 = rng.range(0.2, 1.8);
        let y0 = rng.range(0.2, 1.8);
        // Domain-edge gate: skip if the primal is near a restricted-domain
        // boundary anywhere in the finite-difference window (design.md §5).
        let near_edge = [
            (x0, y0),
            (x0 + H, y0),
            (x0 - H, y0),
            (x0, y0 + H),
            (x0, y0 - H),
        ]
        .iter()
        .any(|&(px, py)| matches!(min_domain_margin(&f, px, py), Some(m) if m < DOMAIN_EPS));
        if near_edge {
            continue;
        }
        // Magnitude gate: skip points where f (or its derivative) exercises an
        // intermediate too large for f64 to resolve a perturbation against.
        let fmag = max_intermediate_mag(&f, x0, y0);
        let dmag = max_intermediate_mag(d, x0, y0);
        match (fmag, dmag) {
            (Some(fm), Some(dm)) if fm <= MAG_CAP && dm <= MAG_CAP => {}
            _ => continue,
        }
        let (Some(fd_h), Some(fd_h2), Some(dv)) = (
            central_diff(&f, x0, y0, wrt, H),
            central_diff(&f, x0, y0, wrt, H / 2.0),
            eval(d, x0, y0),
        ) else {
            continue;
        };
        if !fd_h.is_finite() || !fd_h2.is_finite() || !dv.is_finite() {
            continue;
        }
        if fd_h.abs() > COND_CAP || fd_h2.abs() > COND_CAP || dv.abs() > COND_CAP {
            continue;
        }
        // Richardson gate: skip points where the finite difference is not yet in
        // its convergent regime (cancellation- or truncation-dominated).
        if (fd_h - fd_h2).abs() > RICHARDSON_TOL * fd_h2.abs().max(1.0) {
            continue;
        }
        comparable += 1;
        // fd(h/2) is the more accurate estimate at a convergent point.
        let fd = fd_h2;
        if (fd - dv).abs() > ATOL + RTOL * dv.abs().max(fd.abs()) {
            disagree += 1;
            if first_bad.is_empty() {
                first_bad = format!(
                    "x={x0:.6} y={y0:.6}: symbolic d/d{} = {dv:.8}, finite-diff = {fd:.8}",
                    wrt.name()
                );
            }
        }
    }

    if comparable >= 4 && disagree >= 2 && disagree * 2 > comparable {
        return Some(format!(
            "[finite-diff] d/d{} {expr_text}\n  => {d}\n  {disagree}/{comparable} points disagree; e.g. {first_bad}",
            wrt.name()
        ));
    }
    None
}

/// Property 2: `reparse(render(d))` computes the same value as `d` (immune to
/// benign `*`/`/` reassociation; catches a value-changing paren-drop).
fn fidelity_failure(rng: &mut Rng, expr_text: &str, d: &Expr, wrt: Var) -> Option<String> {
    const RTOL: f64 = 1e-9;
    const ATOL: f64 = 1e-11;
    let rendered = d.to_string();
    if rendered.contains("--") {
        return Some(format!(
            "[render] emitted a `--` comment: d/d{} {expr_text} => {rendered}",
            wrt.name()
        ));
    }
    let reparsed = match try_parse(&rendered) {
        Ok(rp) => rp,
        Err(e) => {
            return Some(format!(
                "[render] engine emitted unparseable SQL: d/d{} {expr_text} => {rendered} ({e})",
                wrt.name()
            ))
        }
    };
    let mut compared = 0u32;
    for _ in 0..40 {
        if compared >= 6 {
            break;
        }
        let x0 = rng.range(0.2, 1.8);
        let y0 = rng.range(0.2, 1.8);
        // Tolerance relative to the computation *scale*, not the result (#54's
        // lesson, generalized — see `metamorphic_mismatch`). AST-vs-reparse
        // differ only by float *association* (the `a·(b/c)` → `(a·b)/c` reprint,
        // issue #50), which agrees to ≈ ε·scale — so at a huge-magnitude point
        // (`sinh(16/…)`, deriv ≈ 1e62) or a cancellation/near-pole point
        // (`tan(exp(…))`) a result-relative tolerance false-positives. Skip the
        // point only if the scale is non-finite/overflowing.
        let scale = match (
            max_intermediate_mag(d, x0, y0),
            max_intermediate_mag(&reparsed, x0, y0),
        ) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() && a.max(b) < 1e300 => a.max(b),
            _ => continue,
        };
        let (Some(va), Some(vb)) = (eval(d, x0, y0), eval(&reparsed, x0, y0)) else {
            continue;
        };
        if !va.is_finite() || !vb.is_finite() {
            continue;
        }
        compared += 1;
        if (va - vb).abs() > ATOL + RTOL * scale {
            return Some(format!(
                "[render] render changed the value: d/d{} {expr_text}\n  rendered = {rendered}\n  at x={x0:.4} y={y0:.4}: AST = {va:.10}, reparsed = {vb:.10}",
                wrt.name()
            ));
        }
    }
    None
}

/// Property 3: the engine re-consumes its own text output for up to 4 rounds of
/// higher-order differentiation without panicking, erroring unexpectedly, or
/// emitting unparseable SQL.
fn self_consumption_failure(ddx: &Ddx, wrt: &ColRef, original: &str) -> Option<String> {
    let mut current = original.to_string();
    for round in 0..4 {
        let parsed = match try_parse(&current) {
            Ok(p) => p,
            // Hitting the parser's *depth* budget is the expression-swell wall,
            // not malformed output: repeated differentiation grows the tree
            // super-linearly, and by the third or fourth round a derivative can
            // nest deeper than `sqlparser`'s default recursion limit. The text
            // is still well-formed SQL — a parser configured with a larger
            // budget accepts it — so this is where the swell stops being usable,
            // which the design already documents as a known limit rather than a
            // defect. Stop the chain here and let the earlier rounds stand.
            //
            // Every *other* parse error still fails the property. That is the
            // distinction worth keeping: "we emitted something unparseable" is a
            // bug; "we emitted something enormous" is a documented cost.
            Err(e) if e.contains("recursion limit exceeded") => break,
            Err(e) => {
                return Some(format!(
                    "[self-consumption] round {round}: engine's own output did not reparse: `{current}` ({e}) [from {original}]"
                ))
            }
        };
        match ddx.differentiate(&parsed, wrt) {
            Ok(d) => {
                let rendered = d.to_string();
                if rendered.contains("--") {
                    return Some(format!(
                        "[self-consumption] round {round}: emitted `--` comment: `{rendered}` [from {original}]"
                    ));
                }
                current = rendered;
            }
            // Re-differentiating can legitimately reach a non-finite constant
            // (e.g. an overflowing exponent) — a *typed* error by design.
            Err(DiffError::NotImplemented(_)) => break,
            Err(e) => {
                return Some(format!(
                    "[self-consumption] round {round}: unexpected error re-differentiating `{current}`: {e} [from {original}]"
                ))
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Property 4: `rewrite_sql` splice fidelity (design.md §3.2, G3/F5).
// ---------------------------------------------------------------------------
//
// The three properties above drive `differentiate` on bare expressions; none of
// them exercise `rewrite_sql` — the parse-free pre-gate, the UTF-8-aware source
// span → byte-offset splice, multiple/nested markers, or the marker-free
// identity guarantee. That subsystem is exactly where bug #52 lived. The
// invariant here is *structural* (byte-level), not numeric: rewriting a marker
// statement must replace **only** each marker's span with `(derivative)` and
// leave every other byte identical.

/// Property 4a: assemble a statement with 1–3 markers wrapped in random
/// (Unicode-bearing) scaffolding and assert `rewrite_sql` splices each marker
/// exactly, byte-for-byte, leaving all surrounding text untouched. If any
/// marker's derivative is undefined, the whole rewrite must error instead.
fn splice_failure(rng: &mut Rng, ddx: &Ddx) -> Option<String> {
    let (input, expected) = gen_marker_statement(rng, ddx);
    let got = ddx.rewrite_sql(&input, &GenericDialect {});
    let Some(expected) = expected else {
        // At least one marker's derivative is undefined → the whole rewrite must
        // fail loud, never partially rewrite.
        return match got {
            Err(_) => None,
            Ok(o) => Some(format!(
                "[splice] expected an error (a marker derivative is undefined) but got Ok:\n  input  = {input}\n  output = {o}"
            )),
        };
    };
    match got {
        Ok(o) if o == expected => None,
        Ok(o) => Some(format!(
            "[splice] rewrite_sql splice mismatch:\n  input    = {input}\n  expected = {expected}\n  actual   = {o}"
        )),
        Err(e) => Some(format!(
            "[splice] rewrite_sql errored on a valid marker statement:\n  input = {input}\n  error = {e}"
        )),
    }
}

/// Property 4b: a marker-free statement is returned byte-identical.
fn marker_free_failure(rng: &mut Rng, ddx: &Ddx) -> Option<String> {
    let s = gen_marker_free_stmt(rng);
    match ddx.rewrite_sql(&s, &GenericDialect {}) {
        Ok(o) if o == s => None,
        Ok(o) => Some(format!(
            "[identity] marker-free statement was modified:\n  input  = {s}\n  output = {o}"
        )),
        Err(e) => Some(format!(
            "[identity] marker-free statement errored:\n  input = {s}\n  error = {e}"
        )),
    }
}

// ---------------------------------------------------------------------------
// Additional invariants (issue #58).
// ---------------------------------------------------------------------------

/// Invariant 1: the SQL `rewrite_sql` emits must always re-parse and contain no
/// residual marker. A *broad* net over the rewrite path — it needs no predicted
/// output, so it tolerates arbitrary scaffolding, and it independently catches
/// corruption bugs like #57 (the corrupt output fails to re-parse).
fn rewrite_validity_failure(rng: &mut Rng, ddx: &Ddx) -> Option<String> {
    let (input, expected) = gen_marker_statement(rng, ddx);
    let out = match ddx.rewrite_sql(&input, &GenericDialect {}) {
        Ok(o) => o,
        // An undefined-derivative marker legitimately errors (fail loud).
        Err(_) if expected.is_none() => return None,
        Err(e) => {
            return Some(format!(
                "[validity] rewrite_sql errored on a valid marker statement:\n  input = {input}\n  error = {e}"
            ))
        }
    };
    match try_parse_stmt(&out) {
        Err(e) => Some(format!(
            "[validity] rewrite_sql emitted unparseable SQL:\n  input  = {input}\n  output = {out}\n  parse error = {e}"
        )),
        Ok(stmts) if has_residual_marker(&stmts) => Some(format!(
            "[validity] rewrite_sql left a residual grad/jvp marker:\n  input  = {input}\n  output = {out}"
        )),
        Ok(_) => None,
    }
}

/// Invariant 5: `rewrite_sql` is idempotent — a second pass over its own output
/// is a no-op (no markers remain to rewrite, and the text is stable).
fn idempotence_failure(rng: &mut Rng, ddx: &Ddx) -> Option<String> {
    let (input, _) = gen_marker_statement(rng, ddx);
    let once = match ddx.rewrite_sql(&input, &GenericDialect {}) {
        Ok(o) => o,
        Err(_) => return None, // undefined-derivative markers error; fine here
    };
    match ddx.rewrite_sql(&once, &GenericDialect {}) {
        Ok(twice) if twice == once => None,
        Ok(twice) => Some(format!(
            "[idempotence] rewrite_sql is not idempotent:\n  input = {input}\n  once  = {once}\n  twice = {twice}"
        )),
        Err(e) => Some(format!(
            "[idempotence] rewrite_sql errored on its own output:\n  input = {input}\n  once  = {once}\n  error = {e}"
        )),
    }
}

/// Invariant 2: `rewrite_sql` never *panics* — for any input, adversarial or
/// malformed, it returns `Ok` or a typed `DiffError` (design principle 5: fail
/// loud, never crash). The prime suspects are the UTF-8 `locate` math and the
/// span logic.
fn panic_failure(rng: &mut Rng, ddx: &Ddx) -> Option<String> {
    let input = gen_adversarial_sql(rng);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Return a bool so nothing non-UnwindSafe crosses the boundary.
        ddx.rewrite_sql(&input, &GenericDialect {}).is_ok()
    }));
    match result {
        Ok(_) => None,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".to_string());
            Some(format!(
                "[panic] rewrite_sql PANICKED (must return a typed error instead):\n  input = {input:?}\n  panic = {msg}"
            ))
        }
    }
}

/// Invariant 4: `d/dv` of an expression that does not mention `v` is exactly
/// zero. Differentiating w.r.t. the fresh variable `w` (which the generator
/// never emits) must fold to a value-0 derivative — a crisp check of the
/// 0-folding smart constructors and the leaf `Match::Not` classification.
fn zero_derivative_failure(rng: &mut Rng, ddx: &Ddx, text: &str) -> Option<String> {
    let f = parse_expr(text);
    let d = match ddx.differentiate(&f, &ColRef::bare("w")) {
        Ok(d) => d,
        Err(_) => return None,
    };
    for _ in 0..12 {
        let x0 = rng.range(0.2, 1.8);
        let y0 = rng.range(0.2, 1.8);
        if let Some(v) = eval(&d, x0, y0) {
            if v.is_finite() && v.abs() > 1e-12 {
                return Some(format!(
                    "[zero-deriv] d/dw {text} is not zero (w is absent):\n  => {d}\n  at x={x0:.4} y={y0:.4}: value = {v}"
                ));
            }
        }
    }
    None
}

/// Invariant 6: the engine never emits an `inf`/`nan` token in a derivative
/// (design principle 5 / #33 — a non-finite constant is a typed error, never an
/// invalid literal in the output text).
fn no_inf_nan_failure(text: &str, d: &Expr) -> Option<String> {
    let rendered = d.to_string();
    let low = rendered.to_ascii_lowercase();
    if low.contains("inf") || low.contains("nan") {
        return Some(format!(
            "[inf-nan] derivative text contains an inf/nan token:\n  d/d? {text}\n  => {rendered}"
        ));
    }
    None
}

/// Invariant 3: `jvp(f, wrt, t)` equals `t · grad(f, wrt)` (forward mode is
/// linear in the seed). Ties the two forward-mode entry points; `jvp` is where
/// #57 lived.
fn jvp_consistency_failure(rng: &mut Rng, ddx: &Ddx, text: &str, wrt: Var) -> Option<String> {
    let f = parse_expr(text);
    let wrt_col = ColRef::bare(wrt.name());
    let tan_depth = 1 + rng.below(2) as u32;
    let t_text = gen_expr(rng, tan_depth);
    let t = parse_expr(&t_text);
    let grad_e = ddx.differentiate(&f, &wrt_col).ok()?;
    let jvp_e = ddx.jvp(&f, &[(wrt_col, t.clone())]).ok()?;
    let gate = [&f, &t, &grad_e];
    if let Some((x0, y0, a, b)) = metamorphic_mismatch(rng, &gate, &jvp_e, |x, y| {
        Some(eval(&t, x, y)? * eval(&grad_e, x, y)?)
    }) {
        return Some(format!(
            "[jvp≠t·grad] jvp({text}, {w}, {t_text}) ≠ tangent·grad:\n  jvp  => {jvp_e}\n  grad => {grad_e}\n  at x={x0:.4} y={y0:.4}: jvp = {a}, t·grad = {b}",
            w = wrt.name()
        ));
    }
    None
}

/// Invariant 7: linearity and the product rule as exact metamorphic identities,
/// `d(f+g) = d(f)+d(g)` and `d(f·g) = d(f)·g + f·d(g)`, value-checked. An exact
/// algebraic cross-check independent of the finite-difference oracle — it holds
/// even at the high-magnitude points the FD oracle skips.
fn linearity_failure(rng: &mut Rng, ddx: &Ddx, f_text: &str, wrt: Var) -> Option<String> {
    let wrt_col = ColRef::bare(wrt.name());
    let f = parse_expr(f_text);
    let g_depth = 2 + rng.below(2) as u32;
    let g_text = gen_expr(rng, g_depth);
    let g = parse_expr(&g_text);
    let df = ddx.differentiate(&f, &wrt_col).ok()?;
    let dg = ddx.differentiate(&g, &wrt_col).ok()?;

    // Sum rule.
    let sum = parse_expr(&format!("({f_text}) + ({g_text})"));
    let dsum = ddx.differentiate(&sum, &wrt_col).ok()?;
    let gate_sum = [&f, &g, &df, &dg];
    if let Some((x0, y0, a, b)) = metamorphic_mismatch(rng, &gate_sum, &dsum, |x, y| {
        Some(eval(&df, x, y)? + eval(&dg, x, y)?)
    }) {
        return Some(format!(
            "[linearity] d(f+g) ≠ d(f)+d(g):\n  f = {f_text}\n  g = {g_text}\n  d(f+g) => {dsum}\n  at x={x0:.4} y={y0:.4}: lhs = {a}, rhs = {b}"
        ));
    }

    // Product rule.
    let prod = parse_expr(&format!("({f_text}) * ({g_text})"));
    let dprod = ddx.differentiate(&prod, &wrt_col).ok()?;
    let gate_prod = [&f, &g, &df, &dg];
    if let Some((x0, y0, a, b)) = metamorphic_mismatch(rng, &gate_prod, &dprod, |x, y| {
        Some(eval(&df, x, y)? * eval(&g, x, y)? + eval(&f, x, y)? * eval(&dg, x, y)?)
    }) {
        return Some(format!(
            "[product-rule] d(f*g) ≠ d(f)*g + f*d(g):\n  f = {f_text}\n  g = {g_text}\n  d(f*g) => {dprod}\n  at x={x0:.4} y={y0:.4}: lhs = {a}, rhs = {b}"
        ));
    }
    None
}

/// Parse, differentiate, and run every property on one generated expression.
/// Returns each failure report (empty ⇒ all properties held).
fn run_all_checks(rng: &mut Rng, ddx: &Ddx, text: &str, wrt: Var) -> Vec<String> {
    let mut out = Vec::new();
    let parsed = match try_parse(text) {
        Ok(p) => p,
        Err(e) => {
            out.push(format!(
                "[generator] produced unparseable text `{text}` ({e})"
            ));
            return out;
        }
    };
    let wrt_col = ColRef::bare(wrt.name());
    let d = match ddx.differentiate(&parsed, &wrt_col) {
        Ok(d) => d,
        Err(DiffError::NotImplemented(_)) => return out, // outside surface; skip
        Err(e) => {
            out.push(format!("[differentiate] unexpected error on `{text}`: {e}"));
            return out;
        }
    };
    if let Some(f) = fd_failure(rng, text, &d, wrt) {
        out.push(f);
    }
    if let Some(f) = fidelity_failure(rng, text, &d, wrt) {
        out.push(f);
    }
    if let Some(f) = self_consumption_failure(ddx, &wrt_col, text) {
        out.push(f);
    }
    // Expression-level metamorphic / structural invariants (#58).
    if let Some(f) = no_inf_nan_failure(text, &d) {
        out.push(f);
    }
    if let Some(f) = zero_derivative_failure(rng, ddx, text) {
        out.push(f);
    }
    if let Some(f) = jvp_consistency_failure(rng, ddx, text, wrt) {
        out.push(f);
    }
    if let Some(f) = linearity_failure(rng, ddx, text, wrt) {
        out.push(f);
    }
    // Statement-level rewrite_sql properties (self-generating; the `text`/`wrt`
    // above are for the expression-level checks).
    if let Some(f) = splice_failure(rng, ddx) {
        out.push(f);
    }
    if let Some(f) = marker_free_failure(rng, ddx) {
        out.push(f);
    }
    if let Some(f) = rewrite_validity_failure(rng, ddx) {
        out.push(f);
    }
    if let Some(f) = idempotence_failure(rng, ddx) {
        out.push(f);
    }
    if let Some(f) = panic_failure(rng, ddx) {
        out.push(f);
    }
    out
}

// ---------------------------------------------------------------------------
// Bounded tests (run every `cargo test`).
// ---------------------------------------------------------------------------

#[test]
fn finite_difference_agreement_over_random_expressions() {
    let ddx = Ddx::new();
    let wrt = ColRef::bare("x");
    let mut failures: Vec<String> = Vec::new();
    let mut tested = 0u32;

    for seed in 0..4000u64 {
        let mut rng = seeded(seed, 0);
        let depth = 2 + (seed % 3) as u32;
        let text = gen_expr(&mut rng, depth);
        let parsed = parse_expr(&text);
        let d = match ddx.differentiate(&parsed, &wrt) {
            Ok(d) => d,
            Err(DiffError::NotImplemented(_)) => continue,
            Err(e) => {
                failures.push(format!("UNEXPECTED ERROR on `{text}`: {e}"));
                continue;
            }
        };
        tested += 1;
        if let Some(report) = fd_failure(&mut rng, &text, &d, Var::X) {
            failures.push(report);
        }
    }

    assert!(
        tested > 500,
        "generator produced too few derivable cases: {tested}"
    );
    assert!(
        failures.is_empty(),
        "finite-difference oracle found {} disagreement(s) out of {} tested:\n\n{}",
        failures.len(),
        tested,
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[test]
fn render_reparse_is_value_preserving() {
    let ddx = Ddx::new();
    let wrt = ColRef::bare("x");
    let mut failures: Vec<String> = Vec::new();

    for seed in 0..5000u64 {
        let mut rng = seeded(seed, 0xDEAD_BEEF);
        let depth = 2 + (seed % 4) as u32;
        let text = gen_expr(&mut rng, depth);
        let parsed = parse_expr(&text);
        let d = match ddx.differentiate(&parsed, &wrt) {
            Ok(d) => d,
            Err(_) => continue,
        };
        if let Some(report) = fidelity_failure(&mut rng, &text, &d, Var::X) {
            failures.push(report);
        }
    }

    assert!(
        failures.is_empty(),
        "render-fidelity fuzz found {} failure(s):\n\n{}",
        failures.len(),
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[test]
fn higher_order_self_consumption_is_stable() {
    let ddx = Ddx::new();
    let wrt = ColRef::bare("x");
    let mut failures: Vec<String> = Vec::new();

    for seed in 0..2000u64 {
        let mut rng = seeded(seed, 0x1234_5678);
        let depth = 2 + (seed % 3) as u32;
        let original = gen_expr(&mut rng, depth);
        if let Some(report) = self_consumption_failure(&ddx, &wrt, &original) {
            failures.push(report);
        }
    }

    assert!(
        failures.is_empty(),
        "self-consumption fuzz found {} failure(s):\n\n{}",
        failures.len(),
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[test]
fn rewrite_sql_splice_is_byte_faithful() {
    // Statement-level fuzz of `rewrite_sql`: markers wrapped in random
    // (Unicode-bearing) scaffolding must be spliced exactly, leaving every other
    // byte identical (design.md §3.2, G3/F5).
    let ddx = Ddx::new();
    let mut failures: Vec<String> = Vec::new();

    for seed in 0..4000u64 {
        let mut rng = seeded(seed, 0x5719_C0DE);
        if let Some(report) = splice_failure(&mut rng, &ddx) {
            failures.push(report);
        }
    }

    assert!(
        failures.is_empty(),
        "splice-fidelity fuzz found {} failure(s):\n\n{}",
        failures.len(),
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[test]
fn splice_handles_marker_with_cast_or_nested_tail() {
    // The splice must cover the *whole* marker call. When the last argument's
    // tail is a CAST (span excludes ` AS <type>`) or a Nested `( … )` (span
    // excludes the closing `)`), rewrite_sql currently stops early and leaves
    // trailing bytes behind, producing unbalanced/corrupt SQL (#57).
    let ddx = Ddx::new();
    // jvp(sin(x), x, CAST(y AS DOUBLE)) — tangent tail is a CAST.
    assert_eq!(
        ddx.rewrite_sql(
            "SELECT jvp(sin(x), x, CAST(y AS DOUBLE)) FROM t",
            &GenericDialect {}
        )
        .unwrap(),
        "SELECT (cos(x) * CAST(y AS DOUBLE)) FROM t"
    );
    // jvp(x, x, (y + z)) — tangent tail is a Nested `( … )`.
    assert_eq!(
        ddx.rewrite_sql("SELECT jvp(x, x, (y + z)) FROM t", &GenericDialect {})
            .unwrap(),
        "SELECT ((y + z)) FROM t"
    );
}

#[test]
fn marker_free_statements_are_byte_identical() {
    // The pre-gate / no-marker guarantee: a statement with no real marker —
    // including one whose text carries a `grad(`/`jvp(` substring in a string,
    // comment, or qualified call — comes back byte-identical (design.md §3.2).
    let ddx = Ddx::new();
    let mut failures: Vec<String> = Vec::new();

    for seed in 0..2000u64 {
        let mut rng = seeded(seed, 0x1DE0_7175);
        if let Some(report) = marker_free_failure(&mut rng, &ddx) {
            failures.push(report);
        }
    }

    assert!(
        failures.is_empty(),
        "marker-free identity fuzz found {} failure(s):\n\n{}",
        failures.len(),
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[test]
fn rewrite_sql_output_is_valid_and_marker_free() {
    let ddx = Ddx::new();
    run_bounded("rewrite validity fuzz", 4000, 0x5A11_D000, |rng| {
        rewrite_validity_failure(rng, &ddx)
    });
}

#[test]
fn rewrite_sql_never_panics_on_adversarial_input() {
    let ddx = Ddx::new();
    run_bounded("never-panic fuzz", 5000, 0x9A11_C000, |rng| {
        panic_failure(rng, &ddx)
    });
}

#[test]
fn jvp_equals_tangent_times_grad() {
    let ddx = Ddx::new();
    run_bounded("jvp↔grad consistency fuzz", 4000, 0x0F5E_ED00, |rng| {
        let (text, wrt) = gen_expr_and_wrt(rng);
        jvp_consistency_failure(rng, &ddx, &text, wrt)
    });
}

#[test]
fn derivative_of_absent_variable_is_zero() {
    let ddx = Ddx::new();
    run_bounded("zero-derivative fuzz", 4000, 0x2E50_1000, |rng| {
        let depth = 2 + rng.below(4) as u32;
        let text = gen_expr(rng, depth);
        zero_derivative_failure(rng, &ddx, &text)
    });
}

#[test]
fn rewrite_sql_is_idempotent() {
    let ddx = Ddx::new();
    run_bounded("idempotence fuzz", 4000, 0x1DE1_1000, |rng| {
        idempotence_failure(rng, &ddx)
    });
}

#[test]
fn no_inf_or_nan_token_is_ever_emitted() {
    let ddx = Ddx::new();
    run_bounded("inf/nan-token fuzz", 4000, 0x1FFF_F000, |rng| {
        let (text, wrt) = gen_expr_and_wrt(rng);
        let d = ddx
            .differentiate(&parse_expr(&text), &ColRef::bare(wrt.name()))
            .ok()?;
        no_inf_nan_failure(&text, &d)
    });
}

#[test]
fn differentiation_is_linear_and_obeys_the_product_rule() {
    let ddx = Ddx::new();
    run_bounded("linearity/product-rule fuzz", 4000, 0x114E_A200, |rng| {
        let (text, wrt) = gen_expr_and_wrt(rng);
        linearity_failure(rng, &ddx, &text, wrt)
    });
}

// ---------------------------------------------------------------------------
// Soak test — long-running, #[ignore]-d, driven by env vars (see module docs).
// ---------------------------------------------------------------------------

/// The divisor-conditioning gate must be narrow: it exists to skip points where
/// a denominator has cancelled to rounding noise, and nothing else.
///
/// Both directions matter. If it never fires, the properties keep comparing
/// meaningless garbage and reporting it as a defect. If it fires too eagerly, it
/// silently blinds every property that uses it — a far worse outcome, because
/// the suite would still look green.
#[test]
fn divisor_gate_skips_only_annihilated_denominators() {
    // `sqrt(1/x) - x^-0.5` is identically zero, so the denominator here is pure
    // rounding residue and the quotient is meaningless.
    let annihilated = parse_expr("power(y, -0.5) / (sqrt(power(x, -1)) - power(x, -0.5))");
    assert!(
        divides_by_noise(&annihilated, 0.9299, 0.8170, 1e-7),
        "a denominator that cancels to zero must be gated out"
    );

    // Ordinary divisions must not be gated, including one whose denominator is
    // genuinely small but carries full significance.
    for healthy in [
        "x / y",
        "1.0 / (x + y)",
        "sin(x) / (x * x)",
        "x / 0.0001",
        "x / (y - 0.19)", // small at y≈0.2, but not a cancellation
    ] {
        let e = parse_expr(healthy);
        assert!(
            !divides_by_noise(&e, 0.9299, 0.8170, 1e-7),
            "`{healthy}` is well-conditioned and must not be gated out"
        );
    }
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "soak: long-running continuous fuzz; run explicitly with DDX_SOAK_SECS set"]
fn soak_continuous_property_fuzz() {
    use std::time::Instant;

    let budget_secs = env_u64("DDX_SOAK_SECS", 15);
    let base = env_u64("DDX_SOAK_BASE", 0);
    let log_path = std::env::var("DDX_SOAK_LOG").ok();

    let mut log = log_path.as_ref().map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .unwrap_or_else(|e| panic!("cannot open DDX_SOAK_LOG `{p}`: {e}"))
    });
    let mut logline = |s: &str| {
        eprintln!("{s}");
        if let Some(f) = log.as_mut() {
            let _ = writeln!(f, "{s}");
            let _ = f.flush();
        }
    };

    let ddx = Ddx::new();
    let start = Instant::now();
    let deadline = budget_secs;
    let mut iters: u64 = 0;
    let mut failures: u64 = 0;
    let mut last_beat = 0u64;

    logline(&format!(
        "SOAK start: budget={budget_secs}s base={base} log={:?}",
        log_path
    ));

    loop {
        let elapsed = start.elapsed().as_secs();
        if elapsed >= deadline {
            break;
        }

        // A fresh, reproducible seed for this iteration.
        let seed = base.wrapping_add(iters);
        let mut rng = seeded(seed, 0xA5A5_5A5A);
        // Deeper trees than the bounded tests, to reach rarer shapes.
        let depth = 2 + (rng.below(5) as u32); // 2..=6
        let wrt = if rng.below(2) == 0 { Var::X } else { Var::Y };
        let text = gen_expr(&mut rng, depth);

        let reports = run_all_checks(&mut rng, &ddx, &text, wrt);
        if reports.is_empty() {
            // A skip (outside-surface) vs a real pass are indistinguishable
            // here; count both as progress.
        } else {
            for r in &reports {
                failures += 1;
                logline(&format!(
                    "\nFAILURE (seed={seed}, base={base}, depth={depth}, wrt={}):\n{r}",
                    wrt.name()
                ));
            }
        }

        iters += 1;

        // Heartbeat ~once a second.
        if elapsed != last_beat {
            last_beat = elapsed;
            logline(&format!(
                "HEARTBEAT elapsed={elapsed}s iters={iters} failures={failures} rate={}/s",
                iters / elapsed.max(1)
            ));
        }
    }

    let mut summary = String::new();
    let _ = write!(
        summary,
        "SOAK done: elapsed={}s iters={iters} failures={failures} base={base} next_base={}",
        start.elapsed().as_secs(),
        base.wrapping_add(iters)
    );
    logline(&summary);

    assert_eq!(
        failures, 0,
        "soak found {failures} property failure(s) — see the FAILURE lines above (each has a reproducing seed)"
    );
}
