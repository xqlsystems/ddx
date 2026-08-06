// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Simulation / property tests for the DataFusion adapter.
//!
//! `ddx-core`'s suite already proves the *calculus* against a finite-difference
//! oracle. Nothing here re-litigates that. What this crate adds is a **bridge**
//! — unparse a bound `Expr`, differentiate, re-plan, coerce, splice — and every
//! bug found in it so far (#77, #79, #80) has been a bridge bug that left the
//! calculus untouched: a column that stopped matching after a quoting round
//! trip, a type that stopped coercing, a field that got renamed, a plan node
//! the walk never reached.
//!
//! So the properties here are all of one shape: **a rewrite must be invisible
//! except for the value it computes.** Change something that cannot matter —
//! the column's name, its storage type, where in the plan the marker sits, what
//! else is selected alongside it — and the answer must not move. That is
//! metamorphic testing, and it is the right tool: it needs no oracle for the
//! derivative itself, only the insistence that two runs agree.
//!
//! Generators, the reference interpreter, and the numeric conditioning gates
//! come from [`ddx_core::test_utils`] — the *same* ones `ddx-core` fuzzes
//! against, so "Path A and Path B agree" is a statement about the two paths and
//! not about two generators.
//!
//! # Currently failing
//!
//! Two properties fail, both against filed bugs. They are checked in red on
//! purpose:
//!
//! * [`an_installed_marker_never_reaches_execution`] — #79, markers inside
//!   `ANY`/`ALL` subqueries are invisible to the rewrite.
//! * [`marker_columns_keep_their_name_across_plan_shapes`] — #80, `DISTINCT ON`
//!   renames the field it rewrites.

use datafusion::arrow::array::{Array, Float64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::error::{DataFusionError, Result};
use datafusion::prelude::SessionContext;
use ddx_core::test_utils::{
    close_at_scale, eval, gen_adversarial_sql, gen_expr, max_intermediate_mag, seeded,
    Conditioning, Failures, Rng, Var,
};
use ddx_core::{ColRef, Ddx};

// ---------------------------------------------------------------------------
// Fixture: three contexts over identical data.
// ---------------------------------------------------------------------------

/// How many rows the fixture tables carry. Enough that a per-row disagreement
/// has somewhere to hide from a lucky sample, small enough to stay fast.
const ROWS: usize = 16;

/// Seeds per bounded property. Each seed is one generated expression and at
/// least one engine query, so this is the knob that trades coverage for CI time.
const SEEDS: u64 = 120;

/// Tolerances for comparing two runs of the same mathematics.
///
/// Both sides compute the same value, but not necessarily by the same
/// *association* — Path A hands the engine ddx-core's rendered text while Path B
/// hands it a re-planned `Expr`, and the engine is free to fold, reassociate and
/// coerce each differently. Agreement is therefore to within float noise at the
/// scale of the intermediates, not at the scale of the result (the #54 lesson,
/// which `close_at_scale` encodes).
const RTOL: f64 = 1e-9;
const ATOL: f64 = 1e-12;

/// The three contexts a property needs, plus the exact rows they hold.
struct Sim {
    /// ddx installed: marker UDFs + the analyzer rule. The system under test.
    ddx: SessionContext,
    /// No ddx at all. Runs Path A (which rewrites text before the engine sees
    /// it) and serves as the oracle for "what this query does without ddx".
    plain: SessionContext,
    /// Marker UDFs registered but *no* analyzer rule, so a marker survives
    /// planning. Never executed — it exists only to read the field names a plan
    /// has when the rewrite has not happened, which is the oracle for
    /// [`marker_columns_keep_their_name_across_plan_shapes`].
    names: SessionContext,
    /// The `(x, y)` of each row of `t`, in `i` order.
    pts: Vec<(f64, f64)>,
    /// The `(x, y)` of each row of `ti*` — integral, so every storage type
    /// represents them exactly.
    ipts: Vec<(f64, f64)>,
}

impl Sim {
    async fn new() -> Result<Sim> {
        let mut rng = Rng::new(0x5EED_D47A);
        let pts: Vec<(f64, f64)> = (0..ROWS)
            .map(|_| (rng.range(0.2, 1.8), rng.range(0.2, 1.8)))
            .collect();
        // Integral points for the storage-type property: exactly representable
        // as BIGINT, DECIMAL and DOUBLE alike, so any disagreement between them
        // is the rewrite's doing and not a rounding artifact.
        let ipts: Vec<(f64, f64)> = (0..ROWS)
            .map(|k| (1.0 + (k % 4) as f64, 1.0 + (k / 4) as f64))
            .collect();

        let ddx = SessionContext::new();
        ddx_datafusion::install(&ddx);
        let plain = SessionContext::new();
        let names = SessionContext::new();
        names.register_udf(ddx_datafusion::grad_udf());
        names.register_udf(ddx_datafusion::jvp_udf());

        for ctx in [&ddx, &plain, &names] {
            seed_tables(ctx, &pts, &ipts).await?;
        }
        Ok(Sim {
            ddx,
            plain,
            names,
            pts,
            ipts,
        })
    }
}

/// Create the fixture tables on one context.
async fn seed_tables(ctx: &SessionContext, pts: &[(f64, f64)], ipts: &[(f64, f64)]) -> Result<()> {
    let values = |ps: &[(f64, f64)]| {
        ps.iter()
            .enumerate()
            .map(|(i, (x, y))| format!("({i}, {x:?}, {y:?})"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    for ddl in [
        format!(
            "CREATE TABLE t AS SELECT * FROM (VALUES {}) AS v(i, x, y)",
            values(pts)
        ),
        // Same values, columns renamed. Quoted-uppercase is what any Parquet or
        // CSV file with capitalized headers produces; the reserved words and the
        // embedded space are the other two ways an identifier stops being a bare
        // lowercase token.
        "CREATE TABLE t_upper AS SELECT i, x AS \"X\", y AS \"Y\" FROM t".into(),
        "CREATE TABLE t_kw AS SELECT i, x AS \"order\", y AS \"select\" FROM t".into(),
        "CREATE TABLE t_space AS SELECT i, x AS \"my x\", y AS \"my y\" FROM t".into(),
        // Integral values in three storage types.
        format!(
            "CREATE TABLE ti_double AS SELECT * FROM (VALUES {}) AS v(i, x, y)",
            values(ipts)
        ),
        "CREATE TABLE ti_bigint AS SELECT i, CAST(x AS BIGINT) AS x, CAST(y AS BIGINT) AS y \
         FROM ti_double"
            .into(),
        "CREATE TABLE ti_decimal AS SELECT i, CAST(x AS DECIMAL(20, 6)) AS x, \
         CAST(y AS DECIMAL(20, 6)) AS y FROM ti_double"
            .into(),
    ] {
        ctx.sql(&ddl).await?.collect().await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Running one query and reading its `d` column.
// ---------------------------------------------------------------------------

/// Run `sql` and return its `d` column in row order, plus the column's type.
///
/// Every value query in this file projects exactly `i` and `d` and orders by
/// `i`, so the returned vector lines up with `Sim::pts` index for index. `None`
/// is a SQL `NULL`, which is a value like any other here: a rewrite that turns a
/// number into a `NULL` has changed the answer.
///
/// A `d` column that is *not* `Float64` violates design.md §3.2 (F4/R1b), but
/// this returns the type with an empty value list rather than an error, so a
/// caller that skips on `Err` cannot accidentally skip that bug: the type check
/// fires, and the empty column also trips `compare_rows`'s length check.
async fn d_column(ctx: &SessionContext, sql: &str) -> Result<(Vec<Option<f64>>, DataType)> {
    let batches = ctx.sql(sql).await?.collect().await?;
    let Some(first) = batches.first() else {
        return Err(DataFusionError::Execution(format!(
            "query returned no batches: {sql}"
        )));
    };
    let idx = first.schema().index_of("d")?;
    let ty = first.schema().field(idx).data_type().clone();
    if ty != DataType::Float64 {
        return Ok((Vec::new(), ty));
    }
    let mut out = Vec::new();
    for b in &batches {
        let a = b
            .column(idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution(format!("the `d` column changed type mid-stream: {sql}"))
            })?;
        out.extend((0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i))));
    }
    Ok((out, ty))
}

/// Compare two `d` columns row by row at the points a numeric comparison is
/// meaningful at, returning the first genuine disagreement.
///
/// `gate` names the expressions whose conditioning decides whether a row counts:
/// a row where the primal sits on a domain boundary, or where some intermediate
/// has outgrown f64's ability to resolve it, is skipped rather than compared.
/// Returns `(compared_rows, first_disagreement)`.
fn compare_rows(
    pts: &[(f64, f64)],
    gate: &[&ddx_core::sqlparser::ast::Expr],
    lhs: &[Option<f64>],
    rhs: &[Option<f64>],
) -> (u32, Option<String>) {
    let cond = Conditioning::default();
    let mut compared = 0u32;
    if lhs.len() != rhs.len() {
        return (
            0,
            Some(format!("row counts differ: {} vs {}", lhs.len(), rhs.len())),
        );
    }
    for (row, &(x, y)) in pts.iter().enumerate().take(lhs.len()) {
        if !cond.admits(gate, x, y) {
            continue;
        }
        let scale = gate
            .iter()
            .filter_map(|e| max_intermediate_mag(e, x, y))
            .fold(1.0f64, f64::max);
        match (lhs[row], rhs[row]) {
            (None, None) => compared += 1,
            (Some(a), Some(b)) if a.is_nan() && b.is_nan() => compared += 1,
            (Some(a), Some(b)) => {
                compared += 1;
                if !close_at_scale(a, b, scale, RTOL, ATOL) {
                    return (
                        compared,
                        Some(format!(
                            "row {row} (x={x:.6}, y={y:.6}): {a} vs {b} \
                             (scale {scale:.3e}, allowed {:.3e})",
                            ATOL + RTOL * scale.max(1.0)
                        )),
                    );
                }
            }
            (a, b) => {
                return (
                    compared,
                    Some(format!(
                        "row {row} (x={x:.6}, y={y:.6}): NULL mismatch — {a:?} vs {b:?}"
                    )),
                )
            }
        }
    }
    (compared, None)
}

// ---------------------------------------------------------------------------
// Generating a marker over the fixture.
// ---------------------------------------------------------------------------

/// One generated case: the primal text, the differentiation variable, and
/// ddx-core's own derivative of it (the expression whose *conditioning* decides
/// which rows are comparable).
struct Case {
    primal: String,
    wrt: Var,
    d: ddx_core::sqlparser::ast::Expr,
    f: ddx_core::sqlparser::ast::Expr,
}

impl Case {
    /// `grad(<primal>, <wrt>)` as SQL text.
    fn marker(&self) -> String {
        format!("grad({}, {})", self.primal, self.wrt.name())
    }
}

/// Generate a derivable expression and differentiate it with `ddx-core`.
///
/// `None` when the generator lands outside the supported surface, which the
/// bounded runs treat as a skip rather than a pass.
fn gen_case(rng: &mut Rng, ddx: &Ddx, depth: u32) -> Option<Case> {
    let primal = gen_expr(rng, depth);
    let wrt = if rng.below(2) == 0 { Var::X } else { Var::Y };
    let f = ddx_core::test_utils::try_parse(&primal).ok()?;
    let d = ddx.differentiate(&f, &ColRef::bare(wrt.name())).ok()?;
    Some(Case { primal, wrt, d, f })
}

/// Token-aware rename of the free variables in generated SQL text.
///
/// A naive `replace("x", …)` would corrupt `exp(`; this walks identifier runs
/// and only rewrites a run that *is* `x` or `y`. Generated text is ASCII, so
/// byte indexing is safe.
fn rename_vars(text: &str, xname: &str, yname: &str) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            match &text[start..i] {
                "x" => out.push_str(xname),
                "y" => out.push_str(yname),
                word => out.push_str(word),
            }
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Property 1 — the two paths compute the same numbers.
// ---------------------------------------------------------------------------

/// Path A (rewrite the text, hand plain SQL to a stock engine) and Path B (bind
/// first, rewrite the plan) must agree on every generated expression.
///
/// They drive the *same* `ddx-core`, so this isolates the bridge: everything
/// Path B does that Path A does not — unparse, re-plan, re-coerce, cast to
/// Float64 — has to be value-preserving. `regressions.rs` asserts this over five
/// hand-picked queries; the generator is what turns that into a property.
#[tokio::test]
async fn path_a_and_path_b_agree_on_random_expressions() -> Result<()> {
    let sim = Sim::new().await?;
    let ddx = Ddx::for_datafusion();
    let mut fail = Failures::new();

    for seed in 0..SEEDS {
        let mut rng = seeded(seed, 0x9A7B_0001);
        let Some(case) = gen_case(&mut rng, &ddx, 2 + (seed % 3) as u32) else {
            continue;
        };
        let sql = format!("SELECT i, {} AS d FROM t ORDER BY i", case.marker());

        let via_b = d_column(&sim.ddx, &sql).await;
        let via_a = match ddx_datafusion::ddx_sql(&sim.plain, &sql).await {
            Ok(df) => df.collect().await.and_then(|batches| {
                let idx = batches[0].schema().index_of("d")?;
                let mut out = Vec::new();
                for b in &batches {
                    let a = b
                        .column(idx)
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| {
                            DataFusionError::Execution("path A `d` is not Float64".into())
                        })?;
                    out.extend((0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i))));
                }
                Ok(out)
            }),
            Err(e) => Err(e),
        };

        match (via_a, via_b) {
            (Ok(a), Ok((b, ty))) => {
                if ty != DataType::Float64 {
                    fail.push(seed, format!("Path B returned {ty:?}, not Float64: {sql}"));
                    continue;
                }
                let (compared, bad) = compare_rows(&sim.pts, &[&case.f, &case.d], &a, &b);
                if let Some(bad) = bad {
                    fail.push(seed, format!("paths disagree on `{sql}`\n  {bad}"));
                } else if compared > 0 {
                    fail.tested();
                }
            }
            (Err(a), Err(b)) => {
                // Both refusing is consistent behaviour; only a one-sided
                // failure is a divergence.
                let _ = (a, b);
            }
            (Ok(_), Err(e)) => fail.push(
                seed,
                format!("Path A ran but Path B failed on `{sql}`: {e}"),
            ),
            (Err(e), Ok(_)) => fail.push(
                seed,
                format!("Path B ran but Path A failed on `{sql}`: {e}"),
            ),
        }
    }
    fail.assert_clean("path A vs path B", 40);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 2 — the engine agrees with the reference interpreter.
// ---------------------------------------------------------------------------

/// What DataFusion computes for a marker must equal what `ddx-core`'s
/// derivative evaluates to in plain Rust f64.
///
/// This is the independent-implementation check: `ddx_core::test_utils::eval`
/// knows nothing about Arrow kernels, type coercion, or expression planners, so
/// a bridge that quietly changes the meaning of the expression it splices —
/// binding a column to the wrong one, dropping a cast, truncating an integer
/// division — shows up here even though both Path A and Path B would agree with
/// each other about it.
#[tokio::test]
async fn the_engine_agrees_with_the_reference_interpreter() -> Result<()> {
    let sim = Sim::new().await?;
    let ddx = Ddx::for_datafusion();
    let mut fail = Failures::new();

    for seed in 0..SEEDS {
        let mut rng = seeded(seed, 0x9A7B_0002);
        let Some(case) = gen_case(&mut rng, &ddx, 2 + (seed % 3) as u32) else {
            continue;
        };
        let sql = format!("SELECT i, {} AS d FROM t ORDER BY i", case.marker());
        let Ok((engine, _)) = d_column(&sim.ddx, &sql).await else {
            continue;
        };
        let reference: Vec<Option<f64>> =
            sim.pts.iter().map(|&(x, y)| eval(&case.d, x, y)).collect();

        let (compared, bad) = compare_rows(&sim.pts, &[&case.f, &case.d], &reference, &engine);
        if let Some(bad) = bad {
            fail.push(
                seed,
                format!(
                    "engine disagrees with the reference interpreter\n  primal = {}\n  \
                     d/d{} = {}\n  {bad}",
                    case.primal,
                    case.wrt.name(),
                    case.d
                ),
            );
        } else if compared > 0 {
            fail.tested();
        }
    }
    fail.assert_clean("engine vs reference interpreter", 40);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 3 — renaming a column cannot change a derivative.
// ---------------------------------------------------------------------------

/// The derivative must not depend on what the columns are *called*.
///
/// This is #77/F1 stated as a property instead of a fixture. That bug —
/// `grad("X" * "X", "X")` silently returning `0.0` because the body was quoted
/// by the unparser while the `wrt` was not — was a *name* changing an *answer*,
/// and it hit every Parquet or CSV file with capitalized headers. The three
/// renamings are the three ways an identifier stops being a bare lowercase
/// token: capitalized, a reserved word, and one containing a space.
#[tokio::test]
async fn a_derivative_does_not_depend_on_the_column_name() -> Result<()> {
    let sim = Sim::new().await?;
    let ddx = Ddx::for_datafusion();
    let mut fail = Failures::new();

    let renamings: &[(&str, &str, &str)] = &[
        ("t_upper", "\"X\"", "\"Y\""),
        ("t_kw", "\"order\"", "\"select\""),
        ("t_space", "\"my x\"", "\"my y\""),
    ];

    for seed in 0..SEEDS {
        let mut rng = seeded(seed, 0x9A7B_0003);
        let Some(case) = gen_case(&mut rng, &ddx, 2 + (seed % 3) as u32) else {
            continue;
        };
        let base_sql = format!("SELECT i, {} AS d FROM t ORDER BY i", case.marker());
        let Ok((base, _)) = d_column(&sim.ddx, &base_sql).await else {
            continue;
        };
        let mut counted = false;
        for &(table, xn, yn) in renamings {
            let renamed = format!(
                "SELECT i, grad({}, {}) AS d FROM {table} ORDER BY i",
                rename_vars(&case.primal, xn, yn),
                match case.wrt {
                    Var::X => xn,
                    Var::Y => yn,
                }
            );
            match d_column(&sim.ddx, &renamed).await {
                Err(e) => fail.push(
                    seed,
                    format!("renaming to {table} broke a working query: {renamed}\n  {e}"),
                ),
                Ok((got, ty)) => {
                    if ty != DataType::Float64 {
                        fail.push(seed, format!("{table} returned {ty:?}, not Float64"));
                        continue;
                    }
                    let (compared, bad) = compare_rows(&sim.pts, &[&case.f, &case.d], &base, &got);
                    if let Some(bad) = bad {
                        fail.push(
                            seed,
                            format!(
                                "renaming the columns changed the derivative\n  \
                                 base    = {base_sql}\n  renamed = {renamed}\n  {bad}"
                            ),
                        );
                    } else if compared > 0 && !counted {
                        counted = true;
                        fail.tested();
                    }
                }
            }
        }
    }
    fail.assert_clean("column-rename invariance", 40);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 4 — storage type cannot change a derivative.
// ---------------------------------------------------------------------------

/// The rows at which two runs agree, as indices — the only rows at which a
/// *derived* quantity is entitled to agree.
fn agreeing_rows(
    pts: &[(f64, f64)],
    gate: &[&ddx_core::sqlparser::ast::Expr],
    a: &[Option<f64>],
    b: &[Option<f64>],
) -> Vec<usize> {
    let cond = Conditioning::default();
    let mut keep = Vec::new();
    for (row, &(x, y)) in pts.iter().enumerate() {
        if row >= a.len() || row >= b.len() || !cond.admits(gate, x, y) {
            continue;
        }
        let scale = gate
            .iter()
            .filter_map(|e| max_intermediate_mag(e, x, y))
            .fold(1.0f64, f64::max);
        let same = match (a[row], b[row]) {
            (None, None) => true,
            (Some(p), Some(q)) if p.is_nan() && q.is_nan() => true,
            (Some(p), Some(q)) => close_at_scale(p, q, scale, RTOL, ATOL),
            _ => false,
        };
        if same {
            keep.push(row);
        }
    }
    keep
}

/// **Conditional** on the primal meaning the same thing: the same values stored
/// as `DOUBLE`, `BIGINT` and `DECIMAL` must give the same derivative, always
/// typed `Float64`.
///
/// #77/F2 and F3 were both this: the rule runs *after* `TypeCoercion` and
/// nothing runs after it, so ddx-core's DOUBLE literals met an `Int64` column
/// and Arrow refused the op; and when the derivative did plan, it planned to
/// `Int64` and left every ancestor holding a stale `Float64` schema. The points
/// are integral so all three storage types represent them exactly.
///
/// # Why the property is conditional
///
/// The unconditional version — "same values, same derivative" — is *false*, and
/// finding out why is issue **#87**. On a `BIGINT` column `2 / x` is integer
/// division, so `ln(2 / x)` at `x = 3` is `ln(0) = -inf` where the `DOUBLE`
/// column gives `-0.405`; on `DECIMAL(20, 6)` an intermediate is truncated to
/// six places. In both cases the *primal* already means something different, so
/// its derivative is entitled to differ too — ddx is not wrong about the
/// function, the function is not the one the user thinks they wrote.
///
/// So the property first asks the engine what the primal evaluates to in each
/// storage type and compares derivatives only at the rows where those agree.
/// That keeps the invariant honest and keeps its teeth: at every row where the
/// primal is genuinely the same function, the derivative must be the same
/// number. Path A and Path B agree exactly on the excluded rows, which is how
/// #87 was identified as a model gap rather than a bridge bug.
///
/// # Why `DECIMAL` is type-checked but not value-checked
///
/// `DECIMAL(20, 6)` quantizes *every* intermediate to six places, and a
/// derivative's intermediates are not the primal's. `grad(power(x + x, 1.5), x)`
/// at `x = 4` is `1.5 · power(x + x, 0.5) · 2`: the primal's quantization error
/// lands at 2e-9, well inside tolerance, while `sqrt(8)`'s lands at 1.2e-7 and
/// then gets multiplied by three. An agreeing primal therefore does *not* bound
/// the derivative's error, and no fixed tolerance honestly can — the
/// amplification is the expression's condition number, which the generator is
/// free to make arbitrary.
///
/// Rather than invent a tolerance that would either hide bugs or flake, the
/// decimal column asserts only what is actually invariant: the query must
/// **run** (a `DECIMAL` column meeting ddx-core's `DOUBLE` literals is exactly
/// the #77/F2 coercion hazard) and the result must be **`Float64`** (#77/F3).
/// `BIGINT` carries the value claim, where the values are exact and the question
/// is genuinely about the rewrite.
#[tokio::test]
async fn a_derivative_does_not_depend_on_the_column_storage_type() -> Result<()> {
    let sim = Sim::new().await?;
    let ddx = Ddx::for_datafusion();
    let mut fail = Failures::new();

    for seed in 0..SEEDS {
        let mut rng = seeded(seed, 0x9A7B_0004);
        let Some(case) = gen_case(&mut rng, &ddx, 2 + (seed % 3) as u32) else {
            continue;
        };
        let grad_query =
            |table: &str| format!("SELECT i, {} AS d FROM {table} ORDER BY i", case.marker());
        // `CAST(… AS DOUBLE)` on the *outside* only: the primal still evaluates
        // in the column's own arithmetic, and the cast just makes the result
        // readable as f64.
        let primal_query = |table: &str| {
            format!(
                "SELECT i, CAST(({}) AS DOUBLE) AS d FROM {table} ORDER BY i",
                case.primal
            )
        };

        let (Ok((base, base_ty)), Ok((base_primal, _))) = (
            d_column(&sim.ddx, &grad_query("ti_double")).await,
            d_column(&sim.ddx, &primal_query("ti_double")).await,
        ) else {
            continue;
        };
        if base_ty != DataType::Float64 {
            fail.push(seed, format!("ti_double returned {base_ty:?}"));
            continue;
        }

        let mut counted = false;
        // `compare_values` is false for the fixed-scale type: see the doc
        // comment — quantization error is amplified by the derivative's
        // condition number, so only the "it runs, and it is DOUBLE" half of the
        // invariant is honestly assertable there.
        for (table, compare_values) in [("ti_bigint", true), ("ti_decimal", false)] {
            let Ok((primal, _)) = d_column(&sim.ddx, &primal_query(table)).await else {
                continue;
            };
            // #87: only the rows where the primal is the same function.
            let keep = agreeing_rows(&sim.ipts, &[&case.f], &base_primal, &primal);
            if keep.is_empty() {
                continue;
            }

            match d_column(&sim.ddx, &grad_query(table)).await {
                Err(e) => fail.push(
                    seed,
                    format!(
                        "{table} failed a query that works on ti_double: {}\n  {e}",
                        grad_query(table)
                    ),
                ),
                Ok((got, ty)) => {
                    if ty != DataType::Float64 {
                        fail.push(
                            seed,
                            format!(
                                "{table} returned {ty:?}, not Float64 — every derivative is \
                                 DOUBLE (design.md §3.2, F4/R1b): {}",
                                grad_query(table)
                            ),
                        );
                        continue;
                    }
                    if !compare_values {
                        // The type and "it planned and ran at all" half held,
                        // which is the whole claim for this table.
                        if !counted {
                            counted = true;
                            fail.tested();
                        }
                        continue;
                    }
                    let pts: Vec<(f64, f64)> = keep.iter().map(|&r| sim.ipts[r]).collect();
                    let pick = |v: &[Option<f64>]| -> Vec<Option<f64>> {
                        keep.iter().map(|&r| v.get(r).copied().flatten()).collect()
                    };
                    let (compared, bad) =
                        compare_rows(&pts, &[&case.f, &case.d], &pick(&base), &pick(&got));
                    if let Some(bad) = bad {
                        fail.push(
                            seed,
                            format!(
                                "storage type changed the derivative at a row where the \
                                 primal did NOT change (ti_double vs {table})\n  {}\n  {bad}",
                                grad_query(table)
                            ),
                        );
                    } else if compared > 0 && !counted {
                        counted = true;
                        fail.tested();
                    }
                }
            }
        }
    }
    fail.assert_clean("storage-type invariance", 20);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 5 — where the marker sits cannot change its value.
// ---------------------------------------------------------------------------

/// The value-bearing plan shapes a marker can sit in, each projecting `i` and
/// `d` so the results line up with the bare-projection baseline.
fn value_placements(marker: &str) -> Vec<(&'static str, String)> {
    vec![
        (
            "nested subquery",
            format!("SELECT i, d FROM (SELECT i, {marker} AS d FROM t) ORDER BY i"),
        ),
        (
            "case arm",
            format!(
                "SELECT i, CASE WHEN i >= 0 THEN {marker} ELSE NULL END AS d FROM t ORDER BY i"
            ),
        ),
        (
            "group-by key",
            format!("SELECT i, {marker} AS d FROM t GROUP BY i, {marker} ORDER BY i"),
        ),
        (
            "window over a unique partition",
            format!("SELECT i, max({marker}) OVER (PARTITION BY i) AS d FROM t ORDER BY i"),
        ),
        (
            "aliased DISTINCT ON",
            format!("SELECT i, d FROM (SELECT DISTINCT ON (i) i, {marker} AS d FROM t) ORDER BY i"),
        ),
        (
            "join key passthrough",
            format!(
                "SELECT a.i AS i, a.d AS d FROM (SELECT i, {marker} AS d FROM t) a \
                 JOIN t b ON a.i = b.i ORDER BY a.i"
            ),
        ),
    ]
}

/// A marker's value must not depend on where in the plan it sits.
///
/// The rewrite walks a `LogicalPlan` node by node, and each node variant it
/// handles is hand-enumerated — which node it is should be irrelevant to the
/// arithmetic, and this asserts exactly that. Every placement here is one the
/// crate claims to support, so a failure is either a walk that missed a node or
/// a splice that changed meaning at one.
#[tokio::test]
async fn marker_values_do_not_depend_on_plan_placement() -> Result<()> {
    let sim = Sim::new().await?;
    let ddx = Ddx::for_datafusion();
    let mut fail = Failures::new();

    for seed in 0..SEEDS {
        let mut rng = seeded(seed, 0x9A7B_0005);
        let Some(case) = gen_case(&mut rng, &ddx, 2 + (seed % 2) as u32) else {
            continue;
        };
        let marker = case.marker();
        let base_sql = format!("SELECT i, {marker} AS d FROM t ORDER BY i");
        let Ok((base, _)) = d_column(&sim.ddx, &base_sql).await else {
            continue;
        };
        let mut counted = false;
        for (label, sql) in value_placements(&marker) {
            match d_column(&sim.ddx, &sql).await {
                Err(e) => fail.push(
                    seed,
                    format!("placement `{label}` failed a query the bare projection runs:\n  {sql}\n  {e}"),
                ),
                Ok((got, ty)) => {
                    if ty != DataType::Float64 {
                        fail.push(seed, format!("placement `{label}` returned {ty:?}: {sql}"));
                        continue;
                    }
                    let (compared, bad) =
                        compare_rows(&sim.pts, &[&case.f, &case.d], &base, &got);
                    if let Some(bad) = bad {
                        fail.push(
                            seed,
                            format!("placement `{label}` changed the value:\n  {sql}\n  {bad}"),
                        );
                    } else if compared > 0 && !counted {
                        counted = true;
                        fail.tested();
                    }
                }
            }
        }
    }
    fail.assert_clean("plan-placement value invariance", 40);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 6 — a marker must never survive to execution. (FAILS: #79)
// ---------------------------------------------------------------------------

/// Every plan shape a marker can be written into, value-bearing or not.
///
/// The tail of this list is the part the value properties cannot reach:
/// predicates, sort keys, and the four kinds of subquery a `LogicalPlan` can
/// carry inside an *expression*.
fn all_placements(marker: &str) -> Vec<(&'static str, String)> {
    let mut v: Vec<(&'static str, String)> = vec![
        ("projection", format!("SELECT i, {marker} AS d FROM t")),
        ("where", format!("SELECT i FROM t WHERE {marker} > 0")),
        (
            "having",
            format!("SELECT i FROM t GROUP BY i HAVING sum({marker}) > 0"),
        ),
        ("order by", format!("SELECT i FROM t ORDER BY {marker}")),
        ("distinct", format!("SELECT DISTINCT {marker} AS d FROM t")),
        (
            "distinct on",
            format!("SELECT DISTINCT ON (i) {marker} AS d FROM t"),
        ),
        (
            "union branch",
            format!("SELECT {marker} AS d FROM t UNION ALL SELECT y AS d FROM t"),
        ),
        (
            "scalar subquery",
            format!("SELECT i FROM t WHERE x > (SELECT avg({marker}) FROM t)"),
        ),
        (
            "IN subquery",
            format!("SELECT i FROM t WHERE x IN (SELECT {marker} FROM t)"),
        ),
        (
            "EXISTS subquery",
            format!("SELECT i FROM t WHERE EXISTS (SELECT 1 FROM t WHERE {marker} > 0)"),
        ),
        // NOTE: no recursive-CTE placement here, deliberately.
        //
        // The only safe spelling needs two columns — a step counter, because a
        // derivative may oscillate (`grad(sin(x), x)` is `cos(x)`, which never
        // leaves [-1, 1]) and a value predicate would then never terminate. But
        // a two-column recursive CTE is broken in DataFusion 54 *independently
        // of ddx*: the identical query with `cos(x)` written in place of the
        // marker fails the same way, on a context with no ddx installed (#88).
        // Putting a known-broken engine shape in a property suite buys noise,
        // not coverage. Path B's recursive-CTE support is pinned by a
        // terminating single-column example in `review_r4.rs` instead.
    ];
    // The quantified-comparison family: `= ANY (…)`, `> ALL (…)`, `SOME (…)`.
    // DataFusion carries these as `Expr::SetComparison`, the fourth
    // expression-embedded plan carrier, and the rewrite's hand-rolled recursion
    // knows only the first three (#79).
    for op in ["ALL", "ANY", "SOME"] {
        v.push((
            match op {
                "ALL" => "ALL subquery",
                "ANY" => "ANY subquery",
                _ => "SOME subquery",
            },
            format!("SELECT i FROM t WHERE x > {op} (SELECT {marker} FROM t)"),
        ));
    }
    v
}

/// **Currently fails — #79.**
///
/// With the analyzer installed, no marker may reach execution, anywhere.
///
/// This is the one invariant in this file that cannot false-positive on
/// unsupported mathematics. A placement is free to fail: the planner may reject
/// the shape, ddx-core may have no rule for the expression, a type may not
/// coerce. All of that is tolerated. The *one* outcome that is always a bug is
/// the marker UDF's own "reached execution" error, because `markers.rs` says so
/// in as many words — reaching execution "never happens in a correct rewrite".
/// When it does happen, the rewrite did not reach the marker, and the message
/// the user gets leads with the one hypothesis that is definitely wrong here:
/// that ddx is not installed.
#[tokio::test]
async fn an_installed_marker_never_reaches_execution() -> Result<()> {
    let sim = Sim::new().await?;
    let ddx = Ddx::for_datafusion();
    let mut fail = Failures::new();

    for seed in 0..SEEDS {
        let mut rng = seeded(seed, 0x9A7B_0006);
        let Some(case) = gen_case(&mut rng, &ddx, 1 + (seed % 2) as u32) else {
            continue;
        };
        let mut counted = false;
        for (label, sql) in all_placements(&case.marker()) {
            let err = match sim.ddx.sql(&sql).await {
                Err(e) => Some(e),
                Ok(df) => df.collect().await.err(),
            };
            if !counted {
                counted = true;
                fail.tested();
            }
            if let Some(e) = err {
                let msg = e.to_string();
                if msg.contains("reached execution") {
                    fail.push(
                        seed,
                        format!(
                            "a marker survived the rewrite in placement `{label}`:\n  {sql}\n  \
                             the rewrite never reached it, so it ran as a row function and \
                             errored"
                        ),
                    );
                }
            }
        }
    }
    fail.assert_clean("marker reachability", 40);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 7 — the rewrite must not disturb anything else. (FAILS: #80)
// ---------------------------------------------------------------------------

/// Plan shapes in which an *unaliased* marker names an output field.
fn naming_shapes(marker: &str) -> Vec<(&'static str, String)> {
    vec![
        ("projection", format!("SELECT {marker} FROM t")),
        ("aggregate", format!("SELECT avg({marker}) FROM t")),
        (
            "window",
            format!("SELECT sum({marker}) OVER (ORDER BY i) FROM t"),
        ),
        (
            "distinct on",
            format!("SELECT DISTINCT ON (i) {marker} FROM t"),
        ),
        ("distinct", format!("SELECT DISTINCT {marker} FROM t")),
    ]
}

/// **Currently fails — #80.**
///
/// Rewriting a marker must not rename the field it produces.
///
/// The oracle needs no hard-coded strings: a context with the marker UDFs
/// registered but *no* analyzer rule plans the same SQL with the marker intact,
/// and its schema is by definition the name the user asked for. The rewritten
/// plan must produce the same one. Anything else silently renames the user's
/// column — and, when a parent refers to that field by name, dangles it into a
/// "No field named …" error.
///
/// `analyzer.rs` already aliases the replacement back to its original
/// `schema_name`, for `Projection | Aggregate | Window`. `Distinct::On` also
/// names output fields and is not on that list.
#[tokio::test]
async fn marker_columns_keep_their_name_across_plan_shapes() -> Result<()> {
    let sim = Sim::new().await?;
    let ddx = Ddx::for_datafusion();
    let mut fail = Failures::new();

    for seed in 0..SEEDS {
        let mut rng = seeded(seed, 0x9A7B_0007);
        let Some(case) = gen_case(&mut rng, &ddx, 1 + (seed % 2) as u32) else {
            continue;
        };
        let mut counted = false;
        for (label, sql) in naming_shapes(&case.marker()) {
            // The un-rewritten plan's field name: what the user wrote, as the
            // planner derives it. Never executed — the marker would error.
            let Ok(want) = sim.names.sql(&sql).await else {
                continue;
            };
            let want = want.schema().field(0).name().clone();

            let Ok(df) = sim.ddx.sql(&sql).await else {
                continue;
            };
            let Ok(batches) = df.collect().await else {
                continue;
            };
            let Some(first) = batches.first() else {
                continue;
            };
            let got = first.schema().field(0).name().clone();
            if !counted {
                counted = true;
                fail.tested();
            }
            if got != want {
                fail.push(
                    seed,
                    format!(
                        "the rewrite renamed the output field in placement `{label}`:\n  \
                         {sql}\n  expected `{want}`\n  got      `{got}`"
                    ),
                );
            }
        }
    }
    fail.assert_clean("field-name preservation", 40);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 8 — the rewrite must leave the rest of the query alone.
// ---------------------------------------------------------------------------

/// Columns that are not markers must come back bit-identical to what the same
/// query returns with no ddx involved at all.
///
/// The rule does two things to the *whole* plan once it finds any marker: it
/// re-runs `TypeCoercion` over everything, and it calls `recompute_schema` on
/// every node it rewrote. Both are global operations justified by a local need,
/// which is exactly the shape of change that quietly perturbs a bystander
/// column. Exact equality is the right assertion here — no tolerance, because
/// nothing about these columns should have been touched.
#[tokio::test]
async fn the_rewrite_leaves_non_marker_columns_untouched() -> Result<()> {
    let sim = Sim::new().await?;
    let ddx = Ddx::for_datafusion();
    let mut fail = Failures::new();

    for seed in 0..SEEDS {
        let mut rng = seeded(seed, 0x9A7B_0008);
        let Some(case) = gen_case(&mut rng, &ddx, 2 + (seed % 3) as u32) else {
            continue;
        };
        // A bystander expression that shares no structure with the marker.
        let bystander = "(x * 3.0 - y / 7.0)";
        let with_marker = format!(
            "SELECT i, {bystander} AS d, {} AS g FROM t WHERE y > 0.0 ORDER BY i",
            case.marker()
        );
        let without = format!("SELECT i, {bystander} AS d FROM t WHERE y > 0.0 ORDER BY i");

        let Ok((got, got_ty)) = d_column(&sim.ddx, &with_marker).await else {
            continue;
        };
        let (want, want_ty) = d_column(&sim.plain, &without).await?;
        fail.tested();
        if got_ty != want_ty {
            fail.push(
                seed,
                format!("the bystander column's type changed: {want_ty:?} -> {got_ty:?}"),
            );
        }
        if got != want {
            fail.push(
                seed,
                format!(
                    "a column that contains no marker changed when a marker was added \
                     elsewhere in the same query:\n  with    = {with_marker}\n  \
                     without = {without}\n  {want:?}\n  {got:?}"
                ),
            );
        }
    }
    fail.assert_clean("bystander-column invariance", 40);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 9 — jvp is grad scaled by the tangent, on the engine.
// ---------------------------------------------------------------------------

/// `jvp(f, v, t)` must equal `t · grad(f, v)` when both are computed by the
/// engine.
///
/// `differentiate_call` handles `jvp` on a separate branch from `grad` — it
/// unparses a third argument and routes through `Ddx::jvp` — so the two markers
/// reach the bridge by different code even though forward-mode linearity says
/// they must land in the same place. ddx-core asserts this identity
/// symbolically; this asserts it after a round trip through the engine.
#[tokio::test]
async fn jvp_equals_the_tangent_times_grad_on_the_engine() -> Result<()> {
    let sim = Sim::new().await?;
    let ddx = Ddx::for_datafusion();
    let mut fail = Failures::new();

    for seed in 0..SEEDS {
        let mut rng = seeded(seed, 0x9A7B_0009);
        let Some(case) = gen_case(&mut rng, &ddx, 2 + (seed % 2) as u32) else {
            continue;
        };
        let tangent_depth = 1 + rng.below(2) as u32;
        let tangent = gen_expr(&mut rng, tangent_depth);
        let Ok(tangent_expr) = ddx_core::test_utils::try_parse(&tangent) else {
            continue;
        };

        let jvp_sql = format!(
            "SELECT i, jvp({}, {}, {tangent}) AS d FROM t ORDER BY i",
            case.primal,
            case.wrt.name()
        );
        let scaled_sql = format!(
            "SELECT i, ({tangent}) * {} AS d FROM t ORDER BY i",
            case.marker()
        );

        let (Ok((jvp, jvp_ty)), Ok((scaled, _))) = (
            d_column(&sim.ddx, &jvp_sql).await,
            d_column(&sim.ddx, &scaled_sql).await,
        ) else {
            continue;
        };
        if jvp_ty != DataType::Float64 {
            fail.push(seed, format!("jvp returned {jvp_ty:?}: {jvp_sql}"));
            continue;
        }
        let (compared, bad) =
            compare_rows(&sim.pts, &[&case.f, &case.d, &tangent_expr], &jvp, &scaled);
        if let Some(bad) = bad {
            fail.push(
                seed,
                format!("jvp ≠ tangent · grad:\n  {jvp_sql}\n  {scaled_sql}\n  {bad}"),
            );
        } else if compared > 0 {
            fail.tested();
        }
    }
    fail.assert_clean("jvp = tangent · grad", 40);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 10 — nesting is repeated differentiation.
// ---------------------------------------------------------------------------

/// `grad(grad(f, v), v)` must equal the engine's evaluation of ddx-core's
/// twice-differentiated expression.
///
/// design.md §3.1 says higher-order "falls out for free" from rewriting
/// bottom-up: the inner marker is an ordinary expression by the time the outer
/// one is differentiated. That is a claim about the *order* the walk visits
/// expression nodes in, and it is exactly the kind of claim that holds on the
/// one example anybody tries.
#[tokio::test]
async fn nesting_a_marker_is_repeated_differentiation() -> Result<()> {
    let sim = Sim::new().await?;
    let ddx = Ddx::for_datafusion();
    let mut fail = Failures::new();

    for seed in 0..SEEDS {
        let mut rng = seeded(seed, 0x9A7B_000A);
        let Some(case) = gen_case(&mut rng, &ddx, 2 + (seed % 2) as u32) else {
            continue;
        };
        let Ok(dd) = ddx.differentiate(&case.d, &ColRef::bare(case.wrt.name())) else {
            continue;
        };

        let nested_sql = format!(
            "SELECT i, grad({}, {w}) AS d FROM t ORDER BY i",
            case.marker(),
            w = case.wrt.name()
        );
        let direct_sql = format!("SELECT i, ({dd}) AS d FROM t ORDER BY i");

        let (Ok((nested, ty)), Ok((direct, _))) = (
            d_column(&sim.ddx, &nested_sql).await,
            d_column(&sim.plain, &direct_sql).await,
        ) else {
            continue;
        };
        if ty != DataType::Float64 {
            fail.push(seed, format!("nested grad returned {ty:?}: {nested_sql}"));
            continue;
        }
        let (compared, bad) = compare_rows(&sim.pts, &[&case.f, &case.d, &dd], &nested, &direct);
        if let Some(bad) = bad {
            fail.push(
                seed,
                format!(
                    "grad(grad(f)) ≠ the twice-differentiated expression:\n  {nested_sql}\n  \
                     {direct_sql}\n  {bad}"
                ),
            );
        } else if compared > 0 {
            fail.tested();
        }
    }
    fail.assert_clean("higher-order nesting", 30);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 11 — the analyzer never panics.
// ---------------------------------------------------------------------------

/// No input may make the analyzer panic; every failure must be a typed
/// `DataFusionError`.
///
/// Reuses ddx-core's adversarial generator — malformed arities, markers behind
/// Unicode prefixes that break char↔byte arithmetic, twelve-deep marker
/// nesting, statements truncated mid-token. On this path the same inputs meet a
/// completely different consumer: the analyzer's plan walk, the unparser, and
/// `SqlToRel`, none of which ddx-core's own never-panic test exercises.
/// Not `#[tokio::test]`: catching a panic requires driving the future from
/// inside [`std::panic::catch_unwind`], and blocking on a runtime from a thread
/// that runtime already owns is itself a panic. Owning the runtime here keeps
/// the only unwinding in the test to the one being measured.
#[test]
fn the_analyzer_never_panics_on_adversarial_sql() -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a current-thread runtime");
    let sim = rt.block_on(Sim::new())?;
    let mut fail = Failures::new();

    for seed in 0..SEEDS * 4 {
        let mut rng = seeded(seed, 0x9A7B_000B);
        let sql = gen_adversarial_sql(&mut rng);
        fail.tested();
        // `sql()` plans; `collect()` optimizes — which is where the analyzer rule
        // actually runs — and executes. Both must return, not unwind. The bool is
        // there so nothing non-`UnwindSafe` crosses the boundary.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.block_on(async {
                match sim.ddx.sql(&sql).await {
                    Ok(df) => df.collect().await.is_ok(),
                    Err(_) => false,
                }
            })
        }));
        if let Err(payload) = outcome {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".to_string());
            fail.push(
                seed,
                format!("PANICKED (must be a typed DataFusionError) on {sql:?}\n  panic = {msg}"),
            );
        }
    }
    fail.assert_clean("analyzer never-panic", 100);
    Ok(())
}
