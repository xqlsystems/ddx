// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Forward mode on the soak's cases: `jvp` of the loss and of the root
//! relation, and `jvp_program` of the loss's `grad` program (H·t).
//!
//! Reverse mode is already checked against finite differences, so it is the
//! oracle here, exactly, with no step size: for any tangent `t`,
//!
//! - `jvp(L)(t) = ⟨∇L, t⟩`, and the jvp's primal is the loss;
//! - `jvp` is linear in `t`; a row left out of the tangent table is a row
//!   whose tangent is 0; a FLOAT tangent column means what a DOUBLE one does;
//! - for an output relation `R`, `⟨J t, c⟩ = ⟨t, Jᵀ c⟩` with `vjp`, and the
//!   jvp's primal rows are `R`'s rows;
//! - `H·t` from forward over reverse is symmetric, `⟨H t₁, t₂⟩ = ⟨t₁, H t₂⟩`,
//!   its primal is the gradient, its value tangent is `jvp(L)(t)`, and it
//!   matches a finite difference of the gradient where that is smooth;
//! - a tangent table with a repeated dim tuple is refused before any step
//!   runs, and nothing is left registered.
//!
//! Where `jvp` and `grad` disagree, a finite difference of the loss along
//! `t` says which one is wrong, in the failure message.

use ddx_ad::{ForwardProgram, InputTable};

use super::*;

/// A tangent of the `wrt` tables: for each table, each row's coefficient,
/// or `None` for a row the tangent table leaves out (tangent 0).
#[derive(Clone, Debug)]
struct Tan(BTreeMap<String, Vec<Option<f64>>>);

impl Tan {
    /// Coefficients are multiples of 1/64 in [-1, 1], exact in FLOAT too.
    /// With `sparse`, about one row in five is left out.
    fn random(rng: &mut Rng, case: &Case, sparse: bool) -> Tan {
        let mut out = BTreeMap::new();
        for t in &case.wrt {
            let n = case.table(t).vals.len();
            let coefs = (0..n)
                .map(|_| {
                    if sparse && rng.below(5) == 0 {
                        None
                    } else {
                        Some((rng.below(129) as f64 - 64.0) / 64.0)
                    }
                })
                .collect();
            out.insert(t.clone(), coefs);
        }
        Tan(out)
    }

    fn map(&self, f: impl Fn(Option<f64>) -> Option<f64>) -> Tan {
        Tan(self
            .0
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().map(|c| f(*c)).collect()))
            .collect())
    }

    fn plus(&self, other: &Tan) -> Tan {
        Tan(self
            .0
            .iter()
            .map(|(k, v)| {
                let o = &other.0[k];
                (
                    k.clone(),
                    v.iter()
                        .zip(o)
                        .map(|(a, b)| match (a, b) {
                            (None, None) => None,
                            _ => Some(a.unwrap_or(0.0) + b.unwrap_or(0.0)),
                        })
                        .collect(),
                )
            })
            .collect())
    }

    /// `⟨g, t⟩` and `Σ |g · t|`, over rows whose gradient is not NULL.
    fn dot(&self, case: &Case, g: &BTreeMap<String, Grad>) -> (f64, f64) {
        let (mut dot, mut abs) = (0.0, 0.0);
        for (t, coefs) in &self.0 {
            let Some(gt) = g.get(t) else { continue };
            for (key, c) in case.table(t).keys.iter().zip(coefs) {
                if let (Some(c), Some(Some(gv))) = (c, gt.get(key)) {
                    dot += gv * c;
                    abs += (gv * c).abs();
                }
            }
        }
        (dot, abs)
    }

    fn direction(&self, case: &Case) -> Direction {
        let mut d = Vec::new();
        for (t, coefs) in &self.0 {
            for (r, c) in coefs.iter().enumerate() {
                if let Some(c) = c.filter(|c| *c != 0.0) {
                    if case.table(t).vals[r].is_some() {
                        d.push((t.clone(), r, c));
                    }
                }
            }
        }
        d
    }
}

/// Register each tangent table a program asks for, from `tan`, its values
/// as FLOAT when `f32`.
fn register_tangents(
    ctx: &SessionContext,
    case: &Case,
    tables: &[InputTable],
    tan: &Tan,
    f32: bool,
    duplicate_a_row: bool,
) -> Result<(), String> {
    for tt in tables {
        let name = tt
            .of
            .table()
            .unwrap_or_default()
            .last()
            .cloned()
            .unwrap_or_default();
        let Some(tb) = case.tables.iter().find(|t| t.name == name) else {
            return Err(format!(
                "[jvp-shape] a tangent table for `{name}`, which is not a wrt table"
            ));
        };
        let coefs = &tan.0[&name];
        let mut rows: Vec<usize> = (0..tb.keys.len()).filter(|&r| coefs[r].is_some()).collect();
        if duplicate_a_row {
            if let Some(&r) = rows.first() {
                rows.push(r);
            }
        }
        let mut fields = Vec::new();
        let mut cols: Vec<ArrayRef> = Vec::new();
        for c in &tt.columns {
            if let Some(k) = tb.dims.iter().position(|d| d == c) {
                fields.push(Field::new(c, tb.key_type.data_type(), true));
                cols.push(
                    tb.key_type
                        .array(rows.iter().map(|&r| tb.keys[r][k]).collect()),
                );
            } else if c == "val" {
                let v: Vec<Option<f64>> = rows.iter().map(|&r| coefs[r]).collect();
                if f32 {
                    fields.push(Field::new(c, DataType::Float32, true));
                    cols.push(Arc::new(Float32Array::from(
                        v.iter().map(|x| x.map(|x| x as f32)).collect::<Vec<_>>(),
                    )));
                } else {
                    fields.push(Field::new(c, DataType::Float64, true));
                    cols.push(Arc::new(Float64Array::from(v)));
                }
            } else {
                return Err(format!(
                    "[jvp-shape] the tangent table of {name} asks for column `{c}`; the table has \
                     dims {:?} and val",
                    tb.dims
                ));
            }
        }
        let batch =
            RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).map_err(|e| e.to_string())?;
        register_batch(ctx, &tt.name, batch)?;
    }
    Ok(())
}

fn deregister_tangents(ctx: &SessionContext, tables: &[InputTable]) {
    for tt in tables {
        let _ = ctx.deregister_table(tt.name.as_str());
    }
}

/// What a run of a jvp program gives: its output's rows, and each
/// gradient's (for `jvp_program`).
struct JvpRun {
    output: Rows,
    gradients: Vec<Rows>,
}

/// Run `program` along `tan`, read what it computes, and release it.
async fn run_jvp(
    ctx: &SessionContext,
    case: &Case,
    program: &ForwardProgram,
    tan: &Tan,
    f32: bool,
) -> Result<JvpRun, String> {
    register_tangents(ctx, case, &program.inputs, tan, f32, false)?;
    let ran = ddx_datafusion::engine::execute(ctx, program, &mut ad::Verified::new()).await;
    let result = match ran {
        Err(e) => Err(format!("[jvp-run] ad::run of a jvp program ddx built: {e}")),
        Ok(()) => {
            let read = |step: String| async move {
                query(ctx, &format!("SELECT * FROM \"{step}\""))
                    .await
                    .map_err(|e| format!("[jvp-run] cannot read step {step}: {e}"))
            };
            let output = read(program.value.step.clone()).await?;
            let mut gradients = Vec::new();
            for g in &program.gradients {
                gradients.push(read(g.step.clone()).await?);
            }
            Ok(JvpRun { output, gradients })
        }
    };
    let _ = ddx_datafusion::engine::release(ctx, program);
    deregister_tangents(ctx, &program.inputs);
    result
}

/// The rows' columns must be the output's, in order.
fn check_columns(label: &str, rows: &Rows, columns: &[String]) -> Result<(), String> {
    if rows.names != columns {
        return Err(format!(
            "[{label}] the step's columns are {:?}, the program says {:?}",
            rows.names, columns
        ));
    }
    Ok(())
}

/// Column positions of each tangent: `(value column, tangent column)`.
fn tangent_positions(o: &ddx_ad::OutputTable) -> Vec<(usize, usize)> {
    o.tangents
        .iter()
        .filter_map(|t| {
            let c = o.columns.iter().position(|n| *n == t.column)?;
            let j = o.columns.iter().position(|n| *n == t.tangent)?;
            Some((c, j))
        })
        .collect()
}

/// A scalar jvp's (primal, tangent).
fn scalar(label: &str, program: &ForwardProgram, r: &JvpRun) -> Result<(f64, f64), String> {
    check_columns(label, &r.output, &program.value.columns)?;
    let pos = tangent_positions(&program.value);
    let [(c, j)] = pos.as_slice() else {
        return Err(format!(
            "[{label}] a scalar loss should have one tangent, has {:?}",
            program.value.tangents
        ));
    };
    match r.output.rows.as_slice() {
        [row] => match (row[*c], row[*j]) {
            (Some(v), Some(t)) => Ok((v, t)),
            (Some(v), None) => Err(format!(
                "[{label}] the loss is {v} but its tangent is NULL (NULL is for a NULL value)"
            )),
            (v, t) => Err(format!("[{label}] the loss is {v:?}, its tangent {t:?}")),
        },
        rows => Err(format!(
            "[{label}] a scalar loss's jvp has {} rows",
            rows.len()
        )),
    }
}

fn close(a: f64, b: f64, scale: f64, rtol: f64) -> bool {
    a == b
        || (a.is_nan() && b.is_nan())
        || (a - b).abs() <= rtol * scale.max(a.abs()).max(b.abs()) + 1e-12
}

/// A central difference of the loss along `tan`, Richardson-extrapolated,
/// or `None` where the loss is not smooth along it.
async fn fd_along(
    ctx: &SessionContext,
    case: &Case,
    sql: &str,
    l0: f64,
    tan: &Tan,
) -> Result<Option<f64>, String> {
    let dir = tan.direction(case);
    let h = 1e-3;
    let mut at = BTreeMap::new();
    for k in [-2i32, -1, 1, 2] {
        match loss_at(ctx, case, sql, &dir, h * k as f64 / 2.0).await? {
            Some(v) => {
                at.insert(k, v);
            }
            None => return Ok(None),
        }
    }
    let d1 = (at[&2] - at[&-2]) / (2.0 * h);
    let d2 = (at[&1] - at[&-1]) / h;
    let a_h = at[&2] - 2.0 * l0 + at[&-2];
    let a_h2 = at[&1] - 2.0 * l0 + at[&-1];
    let noise = 64.0 * f64::EPSILON * (l0.abs() + at[&2].abs() + at[&-2].abs());
    if (a_h - 4.0 * a_h2).abs() > 16.0 * noise + 1e-7 * d2.abs().max(1e-6) * h
        || (d1 - d2).abs() > 1e-3 * d2.abs().max(1e-6)
    {
        return Ok(None);
    }
    Ok(Some((4.0 * d2 - d1) / 3.0))
}

/// Who a finite difference sides with, for a failure message.
async fn arbitrate(
    ctx: &SessionContext,
    case: &Case,
    sql: &str,
    l0: f64,
    tan: &Tan,
    jvp: f64,
    grad: f64,
) -> String {
    if case.exact_ties() || case.modes.ulps || case.modes.extreme.is_some() {
        return "no finite difference in this mode".into();
    }
    match fd_along(ctx, case, sql, l0, tan).await {
        Ok(Some(fd)) => {
            let (dj, dg) = ((fd - jvp).abs(), (fd - grad).abs());
            let who = if dj < dg {
                "grad is wrong"
            } else {
                "jvp is wrong"
            };
            format!("finite difference {fd:.12e}: {who}")
        }
        Ok(None) => "the loss is not smooth along t".into(),
        Err(e) => format!("finite difference failed: {e}"),
    }
}

/// Classify a refusal of jvp where grad accepted.
fn refused(label: &str, e: DataFusionError, out: &mut Outcome) {
    match classify(e) {
        Refusal::Bug(b) => out.fail(format!("[{label}] {b}")),
        Refusal::Allowed(why) => {
            out.fail(format!("[{label}] refused a query grad accepted: {why}"))
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn jvp_checks(
    rng: &mut Rng,
    ctx: &SessionContext,
    case: &Case,
    sql: &str,
    wrt: &[ColumnRef],
    l0: f64,
    grads: &BTreeMap<String, Grad>,
    program: &BackwardProgram,
    out: &mut Outcome,
) -> Result<(), String> {
    let jp = match ad::jvp(ctx, sql, wrt).await {
        Ok(p) => p,
        // A loss that reads wrt only through conditions (a CASE's test, a
        // LIMIT's ordering) is piecewise constant: grad gives zeros, jvp
        // refuses. Tallied, not failed (seed 12046).
        Err(e) if e.to_string().contains("no output column depends") && scale_of(grads) == 0.0 => {
            out.engine
                .push("jvp: NotScalar where grad is all zeros (piecewise constant loss)".into());
            return Ok(());
        }
        Err(e) => {
            refused("jvp", e, out);
            return Ok(());
        }
    };
    // One tangent table per wrt table the loss reads, its dims then val.
    for tt in &jp.inputs {
        let name = tt
            .of
            .table()
            .unwrap_or_default()
            .last()
            .cloned()
            .unwrap_or_default();
        let tb = case.table(&name);
        let mut want = tb.dims.clone();
        want.push("val".into());
        if tt.columns != want {
            out.fail(format!(
                "[jvp-shape] the tangent table of {name} has columns {:?}, expected {want:?}",
                tt.columns
            ));
            return Ok(());
        }
    }

    let t1 = Tan::random(rng, case, true);
    let r1 = match run_jvp(ctx, case, &jp, &t1, false).await {
        Ok(r) => r,
        Err(e) => {
            out.fail(e);
            return Ok(());
        }
    };
    let (v1, j1) = match scalar("jvp", &jp, &r1) {
        Ok(x) => x,
        Err(e) => {
            out.fail(e);
            return Ok(());
        }
    };
    if !close(v1, l0, 0.0, 1e-9) {
        out.fail(format!(
            "[jvp-primal] the jvp's primal is {v1}, the loss {l0}"
        ));
        return Ok(());
    }
    let (dot, abs) = t1.dot(case, grads);
    out.meta_compared += 1;
    let finite = dot.is_finite() && j1.is_finite();
    if !finite {
        if case.modes.extreme.is_none() && !(dot.is_nan() && j1.is_nan()) && dot != j1 {
            out.fail(format!(
                "[jvp-vs-grad] jvp along t is {j1}, ⟨∇L, t⟩ is {dot}"
            ));
        }
    } else if !close(j1, dot, abs, 1e-8) {
        let why = arbitrate(ctx, case, sql, l0, &t1, j1, dot).await;
        out.fail(format!(
            "[jvp-vs-grad] jvp along t is {j1:.15e}, ⟨∇L, t⟩ is {dot:.15e} (Σ|g·t| {abs:.3e}); \
             {why}"
        ));
        return Ok(());
    }

    // Linear in t; a missing row is a zero row; FLOAT is DOUBLE.
    let dense = t1.map(|c| Some(c.unwrap_or(0.0) * 2.5));
    match run_jvp(ctx, case, &jp, &dense, true).await {
        Ok(r) => match scalar("jvp-linear", &jp, &r) {
            Ok((_, j)) => {
                out.meta_compared += 1;
                if finite && !close(j, 2.5 * j1, abs * 2.5, 1e-9) {
                    out.fail(format!(
                        "[jvp-linear] along 2.5·t (dense, FLOAT) the tangent is {j:.15e}, \
                         2.5 × {j1:.15e} = {:.15e}",
                        2.5 * j1
                    ));
                }
            }
            Err(e) => out.fail(e),
        },
        Err(e) => out.fail(e),
    }
    let t3 = Tan::random(rng, case, true);
    let both = t1.plus(&t3);
    let (dot3, abs3) = t3.dot(case, grads);
    let mut j3 = None;
    for (label, t, want, scale) in [
        ("jvp-t3", &t3, dot3, abs3),
        ("jvp-additive", &both, dot + dot3, abs + abs3),
    ] {
        match run_jvp(ctx, case, &jp, t, false).await {
            Ok(r) => match scalar(label, &jp, &r) {
                Ok((_, j)) => {
                    out.meta_compared += 1;
                    if label == "jvp-t3" {
                        j3 = Some(j);
                    }
                    if finite && want.is_finite() && !close(j, want, scale, 1e-8) {
                        out.fail(format!(
                            "[{label}] the tangent is {j:.15e}, ⟨∇L, t⟩ {want:.15e}"
                        ));
                    }
                }
                Err(e) => out.fail(e),
            },
            Err(e) => out.fail(e),
        }
    }

    // The same jvp, however the query is planned.
    let unopt = ctx
        .sql(sql)
        .await
        .map_err(|e| e.to_string())?
        .into_unoptimized_plan();
    let variants: Vec<(&str, Result<ForwardProgram, DataFusionError>)> = vec![
        ("jvp-inline", ad::jvp(ctx, &case.inline_sql(), wrt).await),
        ("jvp-unoptimized", ad::jvp(ctx, &unopt, wrt).await),
    ];
    for (label, p) in variants {
        match p {
            Ok(p) => match run_jvp(ctx, case, &p, &t1, false).await {
                Ok(r) => match scalar(label, &p, &r) {
                    Ok((_, j)) => {
                        out.meta_compared += 1;
                        if finite && !close(j, j1, abs, 1e-8) {
                            out.fail(format!(
                                "[{label}] the tangent is {j:.15e}, the optimized plan's {j1:.15e}"
                            ));
                        }
                    }
                    Err(e) => out.fail(e),
                },
                Err(e) => out.fail(e),
            },
            Err(e) => {
                if let Refusal::Bug(b) = classify(e) {
                    out.fail(format!("[{label}] {b}"));
                }
            }
        }
    }

    // A repeated dim tuple in a tangent table is refused before any step
    // runs, and leaves nothing registered.
    if rng.below(3) == 0 {
        let before = ddx_tables(ctx);
        register_tangents(ctx, case, &jp.inputs, &t1, false, true)?;
        let repeated = jp.inputs.iter().any(|tt| {
            t1.0[tt.of.table().unwrap_or_default().last().unwrap()]
                .iter()
                .any(Option::is_some)
        });
        match ddx_datafusion::engine::execute(ctx, &jp, &mut ad::Verified::new()).await {
            Ok(()) if repeated => {
                out.fail("[jvp-contract] a tangent table with a repeated dim tuple was not refused")
            }
            Ok(()) => {
                let _ = ddx_datafusion::engine::release(ctx, &jp);
            }
            Err(e) if !e.to_string().contains("tangent table") => out.fail(format!(
                "[jvp-contract] a repeated tangent row failed with {e}, not the tangent \
                 table's check"
            )),
            Err(_) => {}
        }
        deregister_tangents(ctx, &jp.inputs);
        let after = ddx_tables(ctx);
        let left: Vec<_> = after.difference(&before).collect();
        if !left.is_empty() {
            out.fail(format!("[jvp-contract] a refused run left tables {left:?}"));
        }
    }

    relation_checks(rng, ctx, case, wrt, &t1, out).await?;

    if let Some(j3) = j3 {
        hvp_checks(
            rng,
            ctx,
            case,
            sql,
            wrt,
            l0,
            grads,
            program,
            (&t1, j1),
            (&t3, j3),
            out,
        )
        .await?;
    }
    Ok(())
}

/// The root relation's jvp: its primal rows are the relation's, and with
/// `vjp`, `⟨J t, c⟩ = ⟨t, Jᵀ c⟩`.
async fn relation_checks(
    rng: &mut Rng,
    ctx: &SessionContext,
    case: &Case,
    wrt: &[ColumnRef],
    t: &Tan,
    out: &mut Outcome,
) -> Result<(), String> {
    let root = &case.nodes[case.root];
    let dims = root.dims.clone();
    let out_sql = format!(
        "WITH {} SELECT {}v FROM r{} c",
        case.ctes(),
        lead(&dims, ""),
        case.root
    );
    let jp = match ad::jvp(ctx, &out_sql, wrt).await {
        Ok(p) => p,
        Err(e) => {
            // The relation may not depend on wrt at all (only the head
            // does), or reach a refusal the loss's head avoided.
            if let Refusal::Bug(b) = classify(e) {
                out.fail(format!("[jvp-relation] {b}"));
            }
            return Ok(());
        }
    };
    let r = match run_jvp(ctx, case, &jp, t, false).await {
        Ok(r) => r,
        Err(e) => {
            out.fail(e);
            return Ok(());
        }
    };
    if let Err(e) = check_columns("jvp-relation", &r.output, &jp.value.columns) {
        out.fail(e);
        return Ok(());
    }
    let direct = query(ctx, &out_sql).await?;
    let w = dims.len() + 1;
    let pos = tangent_positions(&jp.value);
    let tcol = pos.iter().find(|(c, _)| *c == dims.len()).map(|p| p.1);
    // The primal rows, as a multiset.
    let key = |row: &[Option<f64>]| -> Vec<u64> {
        row.iter()
            .map(|v| v.map_or(u64::MAX, |x| if x == 0.0 { 0 } else { x.to_bits() }))
            .collect()
    };
    let mut got: Vec<Vec<Option<f64>>> = r.output.rows.iter().map(|x| x[..w].to_vec()).collect();
    let mut want = direct.rows.clone();
    if got.len() != want.len() {
        out.fail(format!(
            "[jvp-relation] the jvp has {} rows, the relation {}",
            got.len(),
            want.len()
        ));
        return Ok(());
    }
    got.sort_by_key(|x| key(x));
    want.sort_by_key(|x| key(x));
    out.meta_compared += 1;
    for (g, wv) in got.iter().zip(&want) {
        let same = g.iter().zip(wv).all(|(a, b)| match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => close(*a, *b, 0.0, 1e-9),
            _ => false,
        });
        if !same {
            out.fail(format!(
                "[jvp-relation] primal row {g:?} where the relation has {wv:?}"
            ));
            return Ok(());
        }
    }
    // A NULL value has a NULL tangent; a value that is not NULL has one.
    if let Some(tc) = tcol {
        for row in &r.output.rows {
            if row[dims.len()].is_none() != row[tc].is_none() {
                out.fail(format!(
                    "[jvp-null] a row {:?} of the relation has tangent {:?}",
                    &row[..w],
                    row[tc]
                ));
                return Ok(());
            }
        }
    }

    // ⟨J t, c⟩ = ⟨t, Jᵀ c⟩, where the dims identify the rows.
    let Some(tc) = tcol else { return Ok(()) };
    if !root.unique
        || r.output
            .rows
            .iter()
            .any(|x| x[..dims.len()].iter().any(Option::is_none))
    {
        return Ok(());
    }
    let program = match ad::vjp(ctx, &out_sql, wrt).await {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };
    let kt = case.tables[0].key_type;
    let c: Vec<f64> = r
        .output
        .rows
        .iter()
        .map(|_| round6(rng.range(-1.0, 1.0)))
        .collect();
    let mut fields = Vec::new();
    let mut cols: Vec<ArrayRef> = Vec::new();
    let cotangent = &program.inputs[0];
    for name in &cotangent.columns {
        if let Some(k) = dims.iter().position(|d| d == name) {
            fields.push(Field::new(name, kt.data_type(), false));
            cols.push(
                kt.array(
                    r.output
                        .rows
                        .iter()
                        .map(|x| x[k].unwrap_or(0.0) as i64)
                        .collect(),
                ),
            );
        } else {
            fields.push(Field::new(name, DataType::Float64, true));
            cols.push(Arc::new(Float64Array::from(c.clone())));
        }
    }
    let batch =
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).map_err(|e| e.to_string())?;
    register_batch(ctx, &cotangent.name, batch)?;
    let g = run_and_read(ctx, &program).await;
    let _ = ddx_datafusion::engine::release(ctx, &program);
    ctx.deregister_table(cotangent.name.as_str())
        .map_err(|e| e.to_string())?;
    let g = match g {
        Ok(g) => g,
        Err(e) => {
            out.fail(format!("[jvp-dot] vjp: {e}"));
            return Ok(());
        }
    };
    let (rhs, rabs) = t.dot(case, &g);
    let (mut lhs, mut labs) = (0.0, 0.0);
    for (row, ck) in r.output.rows.iter().zip(&c) {
        let jt = row[tc].unwrap_or(0.0);
        lhs += jt * ck;
        labs += (jt * ck).abs();
    }
    out.meta_compared += 1;
    if lhs.is_finite() && rhs.is_finite() && !close(lhs, rhs, labs.max(rabs), 1e-8) {
        out.fail(format!(
            "[jvp-dot] ⟨J t, c⟩ = {lhs:.15e} but ⟨t, Jᵀ c⟩ = {rhs:.15e} (scale {:.3e})",
            labs.max(rabs)
        ));
    }
    Ok(())
}

/// A gradient-shaped result of `jvp_program`: dim tuple → (primal, tangent).
type Hv = BTreeMap<String, BTreeMap<Vec<i64>, (Option<f64>, Option<f64>)>>;

fn read_hv(
    case: &Case,
    program: &BackwardProgram,
    jp: &ForwardProgram,
    r: &JvpRun,
) -> Result<Hv, String> {
    let mut out = Hv::new();
    for ((g, o), rows) in program
        .gradients
        .iter()
        .zip(&jp.gradients)
        .zip(&r.gradients)
    {
        let table =
            g.of.table()
                .unwrap_or_default()
                .last()
                .cloned()
                .unwrap_or_default();
        check_columns("hvp", rows, &o.columns)?;
        let k = case.table(&table).dims.len();
        let pos = tangent_positions(o);
        let (c, j) = match pos.as_slice() {
            [p] => *p,
            [] => {
                // No tangent: the gradient does not move along any t (a
                // linear loss). Its H·t is 0.
                (k, usize::MAX)
            }
            _ => return Err(format!("[hvp] {table}'s gradient has tangents {pos:?}")),
        };
        let mut m = BTreeMap::new();
        for row in &rows.rows {
            let key: Vec<i64> = row[..k]
                .iter()
                .map(|v| v.unwrap_or(f64::NAN) as i64)
                .collect();
            let tan = if j == usize::MAX {
                row[c].map(|_| 0.0)
            } else {
                row[j]
            };
            m.insert(key, (row[c], tan));
        }
        out.insert(table, m);
    }
    Ok(out)
}

/// `⟨H·t, s⟩` and its absolute scale.
fn hv_dot(case: &Case, hv: &Hv, s: &Tan) -> (f64, f64) {
    let g: BTreeMap<String, Grad> = hv
        .iter()
        .map(|(t, m)| (t.clone(), m.iter().map(|(k, v)| (k.clone(), v.1)).collect()))
        .collect();
    s.dot(case, &g)
}

#[allow(clippy::too_many_arguments)]
async fn hvp_checks(
    _rng: &mut Rng,
    ctx: &SessionContext,
    case: &Case,
    sql: &str,
    wrt: &[ColumnRef],
    l0: f64,
    grads: &BTreeMap<String, Grad>,
    program: &BackwardProgram,
    (t1, j1): (&Tan, f64),
    (t3, j3): (&Tan, f64),
    out: &mut Outcome,
) -> Result<(), String> {
    let hp = match ddx_ad::jvp(program, wrt) {
        Ok(p) => p,
        Err(e) => {
            out.fail(format!(
                "[hvp] jvp_program refused a program grad built: {e}"
            ));
            return Ok(());
        }
    };
    if hp.gradients.len() != program.gradients.len() {
        out.fail(format!(
            "[hvp] {} gradients, the program has {}",
            hp.gradients.len(),
            program.gradients.len()
        ));
        return Ok(());
    }
    let mut hvs = Vec::new();
    for (t, j) in [(t1, j1), (t3, j3)] {
        let r = match run_jvp(ctx, case, &hp, t, false).await {
            Ok(r) => r,
            Err(e) => {
                out.fail(e.replace("[jvp-run]", "[hvp-run]"));
                return Ok(());
            }
        };
        match scalar("hvp-value", &hp, &r) {
            Ok((v, jv)) => {
                out.meta_compared += 1;
                if !close(v, l0, 0.0, 1e-9) {
                    out.fail(format!("[hvp-value] the value is {v}, the loss {l0}"));
                }
                if j.is_finite() && jv.is_finite() && !close(jv, j, j.abs(), 1e-8) {
                    out.fail(format!(
                        "[hvp-value] the value's tangent is {jv:.15e}, jvp's {j:.15e}"
                    ));
                }
            }
            Err(e) => out.fail(e),
        }
        let hv = match read_hv(case, program, &hp, &r) {
            Ok(h) => h,
            Err(e) => {
                out.fail(e);
                return Ok(());
            }
        };
        // The primal beside each H·t is the gradient.
        let primal: BTreeMap<String, Grad> = hv
            .iter()
            .map(|(t, m)| (t.clone(), m.iter().map(|(k, v)| (k.clone(), v.0)).collect()))
            .collect();
        out.meta_compared += 1;
        if let Some(f) = compare("hvp-primal", grads, &primal, 1.0, 1e-8) {
            out.fail(f);
            return Ok(());
        }
        for (t, m) in &hv {
            for (k, (p, h)) in m {
                if p.is_some() != h.is_some() {
                    out.fail(format!(
                        "[hvp-null] {t}{k:?}: the gradient is {p:?} but H·t is {h:?}"
                    ));
                    return Ok(());
                }
            }
        }
        hvs.push(hv);
    }

    // Symmetry: ⟨H t₁, t₃⟩ = ⟨t₁, H t₃⟩.
    let (a, aa) = hv_dot(case, &hvs[0], t3);
    let (b, ba) = hv_dot(case, &hvs[1], t1);
    out.meta_compared += 1;
    if a.is_finite() && b.is_finite() && !close(a, b, aa.max(ba), 1e-7) {
        out.fail(format!(
            "[hvp-symmetric] ⟨H t₁, t₃⟩ = {a:.15e} but ⟨t₁, H t₃⟩ = {b:.15e} (scale {:.3e})",
            aa.max(ba)
        ));
    }

    // Against a finite difference of the gradient along t₁, where smooth.
    if case.exact_ties() || case.modes.ulps || case.modes.extreme.is_some() {
        return Ok(());
    }
    let dir = t1.direction(case);
    let h = 1e-4;
    let mut at: BTreeMap<i32, BTreeMap<String, Grad>> = BTreeMap::new();
    for k in [-2i32, -1, 1, 2] {
        let s = h * k as f64 / 2.0;
        let mut moved: BTreeMap<&str, Table> = BTreeMap::new();
        for (name, row, d) in &dir {
            let tb = moved
                .entry(name.as_str())
                .or_insert_with(|| case.table(name).clone());
            if let Some(v) = tb.vals[*row].as_mut() {
                *v += s * d;
            }
        }
        for tb in moved.values() {
            tb.register(ctx).map_err(|e| e.to_string())?;
        }
        let g = grad_of(ctx, sql, wrt).await;
        for name in moved.keys() {
            case.table(name).register(ctx).map_err(|e| e.to_string())?;
        }
        match g {
            Ok((p, g)) => {
                let _ = ddx_datafusion::engine::release(ctx, &p);
                at.insert(k, g);
            }
            Err(_) => return Ok(()),
        }
    }
    let scale = hvs[0]
        .values()
        .flat_map(|m| m.values())
        .filter_map(|v| v.1)
        .fold(0.0f64, |m, v| m.max(v.abs()));
    let gscale = scale_of(grads);
    let mut worst: Option<String> = None;
    let mut compared = 0;
    for (t, m) in &hvs[0] {
        for (key, (_, hvv)) in m {
            let Some(hvv) = hvv else { continue };
            let get = |k: i32| at[&k].get(t).and_then(|g| g.get(key)).copied().flatten();
            let (Some(m2), Some(m1), Some(p1), Some(p2)) = (get(-2), get(-1), get(1), get(2))
            else {
                continue;
            };
            let d1 = (p2 - m2) / (2.0 * h);
            let d2 = (p1 - m1) / h;
            // Rounding in the gradient is relative to the intermediates it
            // is computed from, not to the gradient: a softmax's is ~1e-17
            // of residue, which moves by ~1e-13 over a step (seed 12100296).
            let noise = 64.0 * f64::EPSILON * gscale.max(1.0) / h;
            // A kink of the gradient within the step: no derivative to
            // compare with.
            if (d1 - d2).abs() > 1e-3 * d2.abs().max(scale).max(1e-9) + 4.0 * noise {
                continue;
            }
            let fd = (4.0 * d2 - d1) / 3.0;
            compared += 1;
            let tol = 1e-5 * scale.max(fd.abs()).max(1e-9) + 8.0 * noise + 2.0 * (d1 - d2).abs();
            if (fd - hvv).abs() > tol && worst.is_none() {
                worst = Some(format!(
                    "[hvp-fd] (H·t) of {t}{key:?} is {hvv:.12e}; the gradient moves at \
                     {fd:.12e} along t (tol {tol:.3e})"
                ));
            }
        }
    }
    if compared > 0 {
        out.fd_compared += 1;
    }
    if let Some(w) = worst {
        out.fail(w);
    }
    Ok(())
}

/// Where grad refuses, forward mode may still accept (a UNION ALL, a window
/// over a partition without every dim, the NULL-extended side of an outer
/// join), and nothing else in the soak checks it. The oracle there is a
/// finite difference of the loss along t, and along a single row.
#[allow(clippy::too_many_arguments)]
pub(super) async fn jvp_only_checks(
    rng: &mut Rng,
    ctx: &SessionContext,
    case: &Case,
    sql: &str,
    wrt: &[ColumnRef],
    l0: f64,
    out: &mut Outcome,
) -> Result<(), String> {
    let jp = match ad::jvp(ctx, sql, wrt).await {
        Ok(p) => p,
        Err(e) => {
            match classify(e) {
                Refusal::Bug(b) => out.fail(format!("[jvp-only] {b}")),
                Refusal::Allowed(_) => out.engine.push("jvp-only: refused too".into()),
            }
            return Ok(());
        }
    };
    out.engine.push("jvp-only: accepted".into());
    let fd_ok = !(case.exact_ties() || case.modes.ulps || case.modes.extreme.is_some());
    let mut single = Tan::random(rng, case, false).map(|_| None);
    if let Some((t, r)) = case
        .wrt
        .iter()
        .flat_map(|t| {
            (0..case.table(t).vals.len())
                .filter(|&r| case.table(t).vals[r].is_some())
                .map(move |r| (t.clone(), r))
        })
        .collect::<Vec<_>>()
        .first()
        .cloned()
    {
        single.0.get_mut(&t).unwrap()[r] = Some(1.0);
    }
    for (label, tan) in [
        ("jvp-only", Tan::random(rng, case, true)),
        ("jvp-only-row", single),
    ] {
        let r = match run_jvp(ctx, case, &jp, &tan, false).await {
            Ok(r) => r,
            Err(e) => {
                out.fail(e.replace("[jvp-run]", "[jvp-only-run]"));
                return Ok(());
            }
        };
        let (v, j) = match scalar(label, &jp, &r) {
            Ok(x) => x,
            Err(e) => {
                out.fail(e);
                return Ok(());
            }
        };
        if !close(v, l0, 0.0, 1e-9) {
            out.fail(format!("[{label}] the jvp's primal is {v}, the loss {l0}"));
            return Ok(());
        }
        if !fd_ok || !j.is_finite() {
            continue;
        }
        match fd_along(ctx, case, sql, l0, &tan).await? {
            Some(fd) => {
                out.fd_compared += 1;
                let noise = 8.0 * 64.0 * f64::EPSILON * l0.abs().max(1e-300) / 1e-3;
                let tol = 1e-6 * fd.abs().max(j.abs()).max(1e-6) + noise;
                if (fd - j).abs() > tol {
                    out.fail(format!(
                        "[{label}] jvp along t is {j:.12e}; the loss moves at {fd:.12e} \
                         (tol {tol:.3e}), and grad refused this query"
                    ));
                    return Ok(());
                }
            }
            None => out.fd_screened += 1,
        }
    }
    Ok(())
}
