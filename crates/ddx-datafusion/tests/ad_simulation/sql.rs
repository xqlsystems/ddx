// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad`, `vjp` and `jvp` as columns of a `SELECT` (`ddx_ad::rewrite`), on
//! the soak's cases.
//!
//! The oracle is the programmatic API, which the rest of the soak checks
//! against finite differences: SQL is a surface over the same programs, so
//! what it returns must be what they compute, row for row. Each check
//! writes a statement the way a user might and compares:
//!
//! - **placement**: `grad(loss, val)` inside expressions (an SGD step, a
//!   `CASE`, `COALESCE`, twice in one expression), in an aggregate, in a
//!   `WHERE`, under `GROUP BY`; each is the expression of the program's
//!   gradient. Spelled in any case, qualified or not, with comments and
//!   multibyte text before it on its line.
//! - **restriction**: a `WHERE` over the table's dims (the comparisons ddx
//!   pushes into the program, and many it must not: `OR`, `NOT`, an
//!   expression of a dim, a comparison of `val`) keeps the rows a plain
//!   `WHERE` keeps, each with the gradient an unrestricted program gives.
//!   Two statements of one batch, restricted differently, share a program.
//! - **vjp**: `vjp(out, val, cotangent)` is the program's vjp along it.
//! - **jvp**: of the loss, along a tangent table or a number, is ⟨∇L, t⟩;
//!   of the root relation, row by row, the program's jvp; its `ORDER BY` and
//!   `LIMIT` keep their meaning.
//! - **nesting**: `jvp` of a `SELECT` over `grad` is `H·t` from
//!   `jvp(grad(…))`.
//! - **the rewrite keeps the rest**: the user's SQL around a call (the
//!   case's CTEs, re-rendered by `sqlparser`) computes what it did, and a
//!   statement with no call comes back byte for byte.
//! - **release**: nothing of ddx's is left registered.

use std::collections::BTreeMap;

use datafusion::prelude::DataFrame;
use datafusion::sql::sqlparser::dialect::GenericDialect;

use super::*;

/// The rows of `df`.
async fn rows_of(df: DataFrame) -> Result<Rows, String> {
    let schema = df.schema().inner().clone();
    let batches = df.collect().await.map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            rows.push((0..b.num_columns()).map(|c| cell(b.column(c), r)).collect());
        }
    }
    Ok(Rows {
        names: schema.fields().iter().map(|f| f.name().clone()).collect(),
        types: schema
            .fields()
            .iter()
            .map(|f| f.data_type().clone())
            .collect(),
        rows,
    })
}

/// A dim tuple as the harness keys it.
fn key_of(row: &[Option<f64>], k: usize) -> Vec<i64> {
    row[..k]
        .iter()
        .map(|v| v.map_or(NULL_KEY, |x| x as i64))
        .collect()
}

/// `rows` keyed by their first `k` columns; a repeated key is an error.
fn keyed(
    label: &str,
    rows: &Rows,
    k: usize,
) -> Result<BTreeMap<Vec<i64>, Vec<Option<f64>>>, String> {
    let mut out = BTreeMap::new();
    for r in &rows.rows {
        if out.insert(key_of(r, k), r.clone()).is_some() {
            return Err(format!("[{label}] two rows for dims {:?}", key_of(r, k)));
        }
    }
    Ok(out)
}

fn same(a: Option<f64>, b: Option<f64>, scale: f64, rtol: f64) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            x == y
                || (x.is_nan() && y.is_nan())
                || (x - y).abs() <= rtol * scale.max(x.abs()).max(y.abs()) + 1e-11
        }
        _ => false,
    }
}

/// What a statement failed with: a refusal ddx is entitled to (tallied by
/// message), or anything else (a failure).
fn failed(label: &str, sql: &str, e: &str, out: &mut Outcome) {
    let refused = [
        "not supported by ddx-ad yet",
        "NotImplemented",
        "not implemented",
        "NotScalar",
        "does not depend on",
        "no output column depends",
    ];
    if refused.iter().any(|r| e.contains(r)) {
        let words: Vec<&str> = e.split_whitespace().take(24).collect();
        out.engine
            .push(format!("sql refused [{label}]: {}", words.join(" ")));
    } else if let Some(kind) = engine_fault(e) {
        out.engine.push(kind);
    } else {
        out.fail(format!("[{label}] {e}\n  {sql}"));
    }
}

/// `grad` spelled as a user might: any case, `val` bare or qualified by
/// `qualify` (`w.`, an alias's `p.`, or none), with a comment (multibyte,
/// holding parentheses) before it on its line.
fn grad_call(rng: &mut Rng, qualify: &str) -> String {
    let name = *rng.pick(&["grad", "GRAD", "Grad"]);
    let col = if rng.below(2) == 0 {
        "val".to_string()
    } else {
        format!("{qualify}val")
    };
    let pre = *rng.pick(&["", "", " /* ☕ é (x) */ ", "/*ü*/"]);
    let inner = *rng.pick(&["", " ", "\t"]);
    format!("{pre}{name}({inner}loss,{inner}{col})")
}

/// An expression of the gradient `g` and the value `v` at a row: as SQL
/// over a call and `val`, and as what it computes.
struct Template {
    sql: fn(&str) -> String,
    f: fn(Option<f64>, Option<f64>) -> Option<f64>,
}

const TEMPLATES: &[Template] = &[
    Template {
        sql: |g| g.to_string(),
        f: |g, _| g,
    },
    Template {
        sql: |g| format!("val - 0.25 * {g}"),
        f: |g, v| Some(v? - 0.25 * g?),
    },
    Template {
        sql: |g| format!("CASE WHEN {g} >= 0 THEN {g} ELSE -1.0 * {g} END"),
        f: |g, _| g.map(f64::abs),
    },
    Template {
        sql: |g| format!("{g} * {g} + 1.0"),
        f: |g, _| g.map(|g| g * g + 1.0),
    },
    Template {
        sql: |g| format!("COALESCE({g}, -7.0)"),
        f: |g, _| Some(g.unwrap_or(-7.0)),
    },
    Template {
        sql: |g| format!("{g} + {g}"),
        f: |g, _| g.map(|g| 2.0 * g),
    },
];

/// A random predicate over `tb`'s dims: comparisons ddx may push into its
/// program, mixed with ones it must not (`OR`, `NOT`, an expression of a
/// dim, a comparison of `val`).
fn predicate(rng: &mut Rng, tb: &Table, qualify: &str) -> String {
    let lit = |rng: &mut Rng, d: usize| -> String {
        let present: Vec<i64> = tb.keys.iter().map(|k| k[d]).filter(|&k| k >= 0).collect();
        let k = if present.is_empty() || rng.below(6) == 0 {
            rng.below(4)
        } else {
            *rng.pick(&present) as u64
        };
        tb.key_type.lit(k)
    };
    let atom = |rng: &mut Rng| -> String {
        if tb.dims.is_empty() || rng.below(8) == 0 {
            return rng
                .pick(&[
                    format!("{qualify}val > 0"),
                    format!("{qualify}val IS NOT NULL"),
                    "1 = 1".to_string(),
                ])
                .clone();
        }
        let d = rng.below(tb.dims.len() as u64) as usize;
        let c = format!("{qualify}{}", tb.dims[d]);
        let (a, b) = (lit(rng, d), lit(rng, d));
        match rng.below(12) {
            0 => format!("{c} = {a}"),
            1 => format!("{c} <> {a}"),
            2 => format!("{c} < {a}"),
            3 => format!("{c} >= {a}"),
            4 => format!("{c} IN ({a}, {b})"),
            5 => format!("{c} BETWEEN {a} AND {b}"),
            6 => format!("NOT ({c} = {a})"),
            7 => format!("{c} NOT IN ({a})"),
            8 => format!("{a} > {c}"),
            9 => format!("({c} = {a} OR {c} = {b})"),
            10 => format!("{c} IS NOT NULL"),
            _ => format!("({c} = {a} OR {qualify}val > 0)"),
        }
    };
    let n = 1 + rng.below(3);
    (0..n).map(|_| atom(rng)).collect::<Vec<_>>().join(" AND ")
}

/// A tangent of `tb`'s values: multiples of 1/64, a row in five left out.
fn tangent_coefs(rng: &mut Rng, tb: &Table) -> Vec<Option<f64>> {
    (0..tb.keys.len())
        .map(|_| (rng.below(5) != 0).then(|| (rng.below(129) as f64 - 64.0) / 64.0))
        .collect()
}

/// Register `name` as `tb`'s dims and `coefs` as `val`, the rows `coefs`
/// leaves out left out.
fn register_tangent(
    ctx: &SessionContext,
    name: &str,
    tb: &Table,
    coefs: &[Option<f64>],
) -> Result<(), String> {
    let rows: Vec<usize> = (0..coefs.len()).filter(|&r| coefs[r].is_some()).collect();
    let mut fields: Vec<Field> = tb
        .dims
        .iter()
        .map(|d| Field::new(d, tb.key_type.data_type(), true))
        .collect();
    fields.push(Field::new("val", DataType::Float64, true));
    let mut cols: Vec<ArrayRef> = (0..tb.dims.len())
        .map(|k| {
            tb.key_type
                .array(rows.iter().map(|&r| tb.keys[r][k]).collect())
        })
        .collect();
    cols.push(Arc::new(Float64Array::from(
        rows.iter().map(|&r| coefs[r]).collect::<Vec<_>>(),
    )));
    let batch =
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).map_err(|e| e.to_string())?;
    register_batch(ctx, name, batch)
}

/// `⟨g, coefs⟩` over rows whose gradient is not NULL, and `Σ |g · c|`.
fn dot(tb: &Table, g: &Grad, coefs: &[Option<f64>]) -> (f64, f64) {
    let (mut d, mut a) = (0.0, 0.0);
    for (key, c) in tb.keys.iter().zip(coefs) {
        if let (Some(c), Some(Some(gv))) = (c, g.get(key)) {
            d += gv * c;
            a += (gv * c).abs();
        }
    }
    (d, a)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn sql_checks(
    rng: &mut Rng,
    ctx: &SessionContext,
    case: &Case,
    l0: f64,
    grads: &BTreeMap<String, Grad>,
    out: &mut Outcome,
) -> Result<(), String> {
    let before = ddx_tables(ctx);
    let lp = loss_prefix(case);
    let scale = scale_of(grads).max(1e-300);

    // A statement with no call comes back byte for byte.
    let plain = format!("{lp} SELECT loss FROM loss");
    let mut engine = ddx_datafusion::engine::DataFusion(ctx.clone());
    let programs = ddx_ad::Programs::new();
    match ddx_ad::rewrite(
        &mut engine,
        &programs,
        &[plain.as_str()],
        &GenericDialect {},
    )
    .await
    {
        Ok(r) => {
            out.meta_compared += 1;
            if r.statements != [plain.clone()] || !r.tables.is_empty() {
                out.fail(format!(
                    "[sql-passthrough] a statement with no call came back as {:?} (tables {:?})\n  \
                     {plain}",
                    r.statements, r.tables
                ));
            }
        }
        Err(e) => out.fail(format!("[sql-passthrough] {e}\n  {plain}")),
    }

    for t in &case.wrt {
        let tb = case.table(t);
        let g = &grads[t];
        let k = tb.dims.len();
        let dims = tb.dims.iter().map(String::as_str).collect::<Vec<_>>();
        let lead_dims = lead(&dims, "");

        // Placement: an expression of the gradient, beside the loss read
        // through the re-rendered CTEs.
        let tpl = rng.pick(TEMPLATES);
        let call = grad_call(rng, &format!("{t}."));
        let expr = (tpl.sql)(&call);
        let sql =
            format!("{lp} SELECT {lead_dims}{expr} AS g, (SELECT loss FROM loss) AS lv FROM {t}");
        match query_sql(ctx, &sql).await {
            Err(e) => failed("sql-placement", &sql, &e, out),
            Ok(r) => match keyed("sql-placement", &r, k) {
                Err(e) => out.fail(format!("{e}\n  {sql}")),
                Ok(m) => {
                    out.meta_compared += 1;
                    if m.len() != tb.keys.len() {
                        out.fail(format!(
                            "[sql-placement] {} rows, {t} has {}\n  {sql}",
                            m.len(),
                            tb.keys.len()
                        ));
                    }
                    for (key, v) in tb.keys.iter().zip(&tb.vals) {
                        let Some(row) = m.get(key) else {
                            out.fail(format!("[sql-placement] no row for {t}{key:?}\n  {sql}"));
                            break;
                        };
                        let want = (tpl.f)(g.get(key).copied().flatten(), *v);
                        // G · G scales as the gradient squared.
                        let gs = (scale + 1.0).powi(2);
                        if !same(row[k], want, gs, 1e-8) {
                            out.fail(format!(
                                "[sql-placement] {t}{key:?}: `{}` is {:?}, the program's \
                                 gradient {:?} gives {want:?}\n  {sql}",
                                (tpl.sql)("G"),
                                row[k],
                                g.get(key)
                            ));
                            break;
                        }
                        if !same(row[k + 1], Some(l0), 0.0, 1e-9) {
                            out.fail(format!(
                                "[sql-rendering] the loss read beside a call is {:?}, alone \
                                 {l0}: the rewritten statement computes something else\n  {sql}",
                                row[k + 1]
                            ));
                            break;
                        }
                    }
                }
            },
        }

        // An aggregate of the gradient, and the rows a WHERE on it keeps.
        let call = grad_call(rng, &format!("{t}."));
        let sql = format!("{lp} SELECT SUM({call} * val) AS s, COUNT({call}) AS n FROM {t}");
        let (mut s, mut sa, mut n) = (0.0, 0.0, 0.0);
        for (key, v) in tb.keys.iter().zip(&tb.vals) {
            if let (Some(Some(gv)), Some(v)) = (g.get(key), v) {
                s += gv * v;
                sa += (gv * v).abs();
            }
            if let Some(Some(_)) = g.get(key) {
                n += 1.0;
            }
        }
        match query_sql(ctx, &sql).await {
            Err(e) => failed("sql-aggregate", &sql, &e, out),
            Ok(r) => {
                out.meta_compared += 1;
                let row = r.rows.first().cloned().unwrap_or_default();
                let want_s = (n > 0.0).then_some(s);
                if r.rows.len() != 1
                    || !same(row.first().copied().flatten(), want_s, sa, 1e-8)
                    || row.get(1).copied().flatten() != Some(n)
                {
                    out.fail(format!(
                        "[sql-aggregate] SUM(grad · val), COUNT(grad) is {:?}, the program's \
                         gradient gives ({want_s:?}, {n})\n  {sql}",
                        r.rows
                    ));
                }
            }
        }
        if k > 0 {
            let call = grad_call(rng, &format!("{t}."));
            let sql = format!("{lp} SELECT {} FROM {t} WHERE {call} > 0", dims.join(", "));
            // A gradient within rounding of 0 (a softmax's, about 1e-17 of
            // residue) may land on either side of it in another program.
            let floor = (1e-7 * scale).max(1e-12);
            match query_sql(ctx, &sql).await {
                Err(e) => failed("sql-where-grad", &sql, &e, out),
                Ok(r) => {
                    out.meta_compared += 1;
                    let got: BTreeSet<Vec<i64>> = r.rows.iter().map(|x| key_of(x, k)).collect();
                    for key in &tb.keys {
                        let gv = g.get(key).copied().flatten();
                        // A gradient within rounding of 0 may land either
                        // side of it in another program.
                        if gv.is_some_and(|x| x.abs() <= floor) {
                            continue;
                        }
                        let want = gv.is_some_and(|x| x > 0.0);
                        if got.contains(key) != want {
                            out.fail(format!(
                                "[sql-where-grad] {t}{key:?} with gradient {gv:?} is {} the \
                                 rows `WHERE grad > 0` keeps\n  {sql}",
                                if want { "missing from" } else { "among" }
                            ));
                            break;
                        }
                    }
                }
            }
            // Under GROUP BY the first dim.
            let call = grad_call(rng, &format!("{t}."));
            let d0 = &tb.dims[0];
            let sql = format!("{lp} SELECT {d0}, SUM({call}) AS s FROM {t} GROUP BY {d0}");
            let mut want: BTreeMap<i64, (Option<f64>, f64)> = BTreeMap::new();
            for key in &tb.keys {
                let e = want.entry(key[0]).or_insert((None, 0.0));
                if let Some(Some(gv)) = g.get(key) {
                    e.0 = Some(e.0.unwrap_or(0.0) + gv);
                    e.1 += gv.abs();
                }
            }
            match query_sql(ctx, &sql).await {
                Err(e) => failed("sql-group-by", &sql, &e, out),
                Ok(r) => {
                    out.meta_compared += 1;
                    for row in &r.rows {
                        let d = row[0].map_or(NULL_KEY, |x| x as i64);
                        let (w, wa) = want.get(&d).copied().unwrap_or((None, 0.0));
                        if !same(row[1], w, wa, 1e-8) {
                            out.fail(format!(
                                "[sql-group-by] SUM(grad) for {d0} = {d} is {:?}, the program's \
                                 gradient gives {w:?}\n  {sql}",
                                row[1]
                            ));
                            break;
                        }
                    }
                }
            }
        }

        // Restriction: rows a plain WHERE keeps, each with its gradient;
        // two statements of one batch, restricted differently, sharing one
        // program.
        let (q, alias) = if rng.below(3) == 0 {
            (format!("{t}."), String::new())
        } else if rng.below(2) == 0 {
            ("p.".to_string(), " p".to_string())
        } else {
            (String::new(), String::new())
        };
        let preds = [predicate(rng, tb, &q), predicate(rng, tb, &q)];
        let mut statements = Vec::new();
        let mut wants = Vec::new();
        for p in &preds {
            let keep = query(
                ctx,
                &format!("SELECT {lead_dims}val FROM {t}{alias} WHERE {p}"),
            )
            .await;
            let Ok(keep) = keep else { continue };
            let kept: BTreeSet<Vec<i64>> = keep.rows.iter().map(|x| key_of(x, k)).collect();
            statements.push(format!(
                "{lp} SELECT {lead_dims}{} AS g FROM {t}{alias} WHERE {p}",
                grad_call(rng, &q)
            ));
            wants.push(kept);
        }
        if !statements.is_empty() {
            let texts: Vec<&str> = statements.iter().map(String::as_str).collect();
            match ad::sql_all(ctx, &texts).await {
                Err(e) => failed(
                    "sql-restrict",
                    &statements.join(";\n  "),
                    &e.to_string(),
                    out,
                ),
                Ok(frames) => {
                    for ((df, kept), sql) in frames.into_iter().zip(&wants).zip(&statements) {
                        let r = match rows_of(df).await {
                            Ok(r) => r,
                            Err(e) => {
                                failed("sql-restrict", sql, &e, out);
                                continue;
                            }
                        };
                        let m = match keyed("sql-restrict", &r, k) {
                            Ok(m) => m,
                            Err(e) => {
                                out.fail(format!("{e}\n  {sql}"));
                                continue;
                            }
                        };
                        out.meta_compared += 1;
                        let got: BTreeSet<Vec<i64>> = m.keys().cloned().collect();
                        if &got != kept {
                            out.fail(format!(
                                "[sql-restrict] the statement keeps rows {got:?}, a plain WHERE \
                                 keeps {kept:?}\n  {sql}"
                            ));
                            continue;
                        }
                        for (key, row) in &m {
                            let want = g.get(key).copied().flatten();
                            if !same(row[k], want, scale, 1e-8) {
                                out.fail(format!(
                                    "[sql-restrict] {t}{key:?}: gradient {:?} under the WHERE, \
                                     {want:?} without it\n  {sql}",
                                    row[k]
                                ));
                                break;
                            }
                        }
                    }
                }
            }
        }

        // jvp of the loss, along a tangent table and along a number.
        let coefs = tangent_coefs(rng, tb);
        let tan = format!("tan_user_{t}");
        register_tangent(ctx, &tan, tb, &coefs)?;
        let (want, abs) = dot(tb, g, &coefs);
        let ones: Vec<Option<f64>> = tb.vals.iter().map(|_| Some(2.5)).collect();
        let (want_n, abs_n) = dot(tb, g, &ones);
        for (label, tangent, want, abs) in [
            ("sql-jvp-loss", tan.clone(), want, abs),
            ("sql-jvp-number", "2.5".to_string(), want_n, abs_n),
        ] {
            let sql = format!("{lp} SELECT loss, jvp(loss, {t}.val, {tangent}) AS dl FROM loss");
            match query_sql(ctx, &sql).await {
                Err(e) => failed(label, &sql, &e, out),
                Ok(r) => {
                    out.meta_compared += 1;
                    let row = r.rows.first().cloned().unwrap_or_default();
                    if r.rows.len() != 1
                        || !same(row.first().copied().flatten(), Some(l0), 0.0, 1e-9)
                        || !(want.is_finite()
                            && same(row.get(1).copied().flatten(), Some(want), abs, 1e-8)
                            || !want.is_finite())
                    {
                        out.fail(format!(
                            "[{label}] (loss, jvp) is {:?}; the loss is {l0} and ⟨∇L, t⟩ \
                             {want:.15e}\n  {sql}",
                            r.rows
                        ));
                    }
                }
            }
        }

        // jvp of the root relation, row by row, against the program's.
        relation_jvp(rng, ctx, case, t, &tan, out).await?;

        // Nesting: jvp of a SELECT over grad is H·t.
        nested_hvp(ctx, case, t, &tan, &coefs, out).await?;

        // vjp of the root relation along a cotangent table.
        relation_vjp(rng, ctx, case, t, out).await?;

        let _ = ctx.deregister_table(tan.as_str());
    }

    let after = ddx_tables(ctx);
    let left: Vec<_> = after.difference(&before).collect();
    if !left.is_empty() {
        out.fail(format!("[sql-release] statements left ddx tables {left:?}"));
    }
    Ok(())
}

/// The root relation's `WITH … SELECT dims, v FROM r{root}`.
fn root_sql(case: &Case) -> (String, Vec<&'static str>) {
    let dims = case.nodes[case.root].dims.clone();
    (
        format!(
            "WITH {} SELECT {}v FROM r{} c",
            case.ctes(),
            lead(&dims, ""),
            case.root
        ),
        dims,
    )
}

/// Rows sorted by every column, NULL last.
fn sorted(mut rows: Vec<Vec<Option<f64>>>) -> Vec<Vec<Option<f64>>> {
    rows.sort_by(|a, b| {
        for (x, y) in a.iter().zip(b) {
            let o = match (x, y) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, _) => std::cmp::Ordering::Greater,
                (_, None) => std::cmp::Ordering::Less,
                (Some(x), Some(y)) => x.total_cmp(y),
            };
            if o != std::cmp::Ordering::Equal {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    });
    rows
}

async fn relation_jvp(
    rng: &mut Rng,
    ctx: &SessionContext,
    case: &Case,
    t: &str,
    tan: &str,
    out: &mut Outcome,
) -> Result<(), String> {
    let (root, dims) = root_sql(case);
    let w = dims.len() + 1;
    // The program's jvp along the tangent table.
    let p = match ad::jvp(ctx, root.as_str(), &[ColumnRef::new(t, "val")]).await {
        Ok(p) => p,
        Err(_) => return Ok(()), // refused or constant: not SQL's question
    };
    let tdf = ctx
        .sql(&format!("SELECT * FROM {tan}"))
        .await
        .map_err(|e| e.to_string())?;
    let want = match ad::run(ctx, &p, &[tdf]).await {
        Ok(o) => rows_of(o.value).await?,
        Err(e) => {
            out.fail(format!("[sql-jvp-relation] the program's jvp fails: {e}"));
            return Ok(());
        }
    };
    if want.names.len() != w + 1 {
        return Ok(()); // v has no tangent
    }
    let sql = format!(
        "WITH {}, tg AS (SELECT * FROM {tan}) SELECT {}v, jvp(v, {t}.val, tg) AS dv FROM r{} c",
        case.ctes(),
        lead(&dims, ""),
        case.root
    );
    let got = match query_sql(ctx, &sql).await {
        Ok(r) => r,
        Err(e) => {
            failed("sql-jvp-relation", &sql, &e, out);
            return Ok(());
        }
    };
    out.meta_compared += 1;
    let (g, wv) = (sorted(got.rows.clone()), sorted(want.rows.clone()));
    if g.len() != wv.len() {
        out.fail(format!(
            "[sql-jvp-relation] {} rows, the program's jvp {}\n  {sql}",
            g.len(),
            wv.len()
        ));
        return Ok(());
    }
    let scale = wv
        .iter()
        .filter_map(|r| r[w])
        .fold(0.0f64, |m, x| m.max(x.abs()));
    for (a, b) in g.iter().zip(&wv) {
        let ok = (0..w).all(|c| same(a[c], b[c], 0.0, 1e-9)) && same(a[w], b[w], scale, 1e-8);
        if !ok {
            out.fail(format!(
                "[sql-jvp-relation] row {a:?} where the program's jvp has {b:?}\n  {sql}"
            ));
            return Ok(());
        }
    }
    // ORDER BY the tangent, LIMIT: the top rows of the same.
    let n = 1 + rng.below(3) as usize;
    let sql_top = format!("{sql} ORDER BY dv DESC NULLS LAST LIMIT {n}");
    match query_sql(ctx, &sql_top).await {
        Err(e) => failed("sql-jvp-order", &sql_top, &e, out),
        Ok(r) => {
            out.meta_compared += 1;
            let mut all: Vec<f64> = wv.iter().filter_map(|r| r[w]).collect();
            all.sort_by(|a, b| b.total_cmp(a));
            let mut want_top: Vec<Option<f64>> = all.into_iter().map(Some).take(n).collect();
            while want_top.len() < n.min(wv.len()) {
                want_top.push(None);
            }
            let got_top: Vec<Option<f64>> = r.rows.iter().map(|x| x[w]).collect();
            let ok = got_top.len() == want_top.len()
                && got_top
                    .iter()
                    .zip(&want_top)
                    .all(|(a, b)| same(*a, *b, scale, 1e-8));
            if !ok {
                out.fail(format!(
                    "[sql-jvp-order] ORDER BY dv DESC LIMIT {n} gives tangents {got_top:?}, \
                     the top of the program's are {want_top:?}\n  {sql_top}"
                ));
            }
        }
    }
    Ok(())
}

async fn nested_hvp(
    ctx: &SessionContext,
    case: &Case,
    t: &str,
    tan: &str,
    _coefs: &[Option<f64>],
    out: &mut Outcome,
) -> Result<(), String> {
    let tb = case.table(t);
    let k = tb.dims.len();
    let dims: Vec<&str> = tb.dims.iter().map(String::as_str).collect();
    let loss = format!("{} SELECT loss FROM loss", loss_prefix(case));
    let wrt = [ColumnRef::new(t, "val")];
    let program = match ad::grad(ctx, loss.as_str(), &wrt).await {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };
    let hp = match ad::jvp(ctx, &program, &wrt).await {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };
    let tdf = ctx
        .sql(&format!("SELECT * FROM {tan}"))
        .await
        .map_err(|e| e.to_string())?;
    let want = match ad::run(ctx, &hp, &[tdf]).await {
        Ok(o) => match o.gradients.into_iter().next() {
            Some(df) => rows_of(df).await?,
            None => return Ok(()),
        },
        Err(e) => {
            out.fail(format!("[sql-hvp] the program's H·t fails: {e}"));
            return Ok(());
        }
    };
    // dims, val, val_tangent.
    if want.names.len() != k + 2 {
        return Ok(());
    }
    let sql = format!(
        "{}, g AS (SELECT {}grad(loss, val) AS val FROM {t}), tg AS (SELECT * FROM {tan}) \
         SELECT {}jvp(val, {t}.val, tg) AS hv FROM g",
        loss_prefix(case),
        lead(&dims, ""),
        lead(&dims, "")
    );
    let got = match query_sql(ctx, &sql).await {
        Ok(r) => r,
        Err(e) => {
            failed("sql-hvp", &sql, &e, out);
            return Ok(());
        }
    };
    let (Ok(g), Ok(w)) = (keyed("sql-hvp", &got, k), keyed("sql-hvp", &want, k)) else {
        out.fail(format!("[sql-hvp] repeated dims\n  {sql}"));
        return Ok(());
    };
    out.meta_compared += 1;
    let scale = w
        .values()
        .filter_map(|r| r[k + 1])
        .fold(0.0f64, |m, x| m.max(x.abs()));
    if g.len() != w.len() {
        out.fail(format!(
            "[sql-hvp] {} rows, the program's H·t {}\n  {sql}",
            g.len(),
            w.len()
        ));
        return Ok(());
    }
    for (key, row) in &w {
        let got_v = g.get(key).map(|r| r[k]);
        if !same(got_v.flatten(), row[k + 1], scale, 1e-7) {
            out.fail(format!(
                "[sql-hvp] {t}{key:?}: jvp of a SELECT over grad is {got_v:?}, the program's \
                 H·t {:?}\n  {sql}",
                row[k + 1]
            ));
            return Ok(());
        }
    }
    Ok(())
}

async fn relation_vjp(
    rng: &mut Rng,
    ctx: &SessionContext,
    case: &Case,
    t: &str,
    out: &mut Outcome,
) -> Result<(), String> {
    let root = &case.nodes[case.root];
    if !root.unique {
        return Ok(());
    }
    let (root_q, dims) = root_sql(case);
    let rows = query(ctx, &root_q).await?;
    if rows
        .rows
        .iter()
        .any(|r| r[..dims.len()].iter().any(Option::is_none))
    {
        return Ok(());
    }
    let wrt = [ColumnRef::new(t, "val")];
    let p = match ad::vjp(ctx, root_q.as_str(), &wrt).await {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };
    let kt = case.tables[0].key_type;
    let c: Vec<f64> = rows
        .rows
        .iter()
        .map(|_| round6(rng.range(-1.0, 1.0)))
        .collect();
    let mut fields: Vec<Field> = dims
        .iter()
        .map(|d| Field::new(*d, kt.data_type(), false))
        .collect();
    fields.push(Field::new("v", DataType::Float64, true));
    let mut cols: Vec<ArrayRef> = (0..dims.len())
        .map(|d| {
            kt.array(
                rows.rows
                    .iter()
                    .map(|r| r[d].unwrap_or(0.0) as i64)
                    .collect(),
            )
        })
        .collect();
    cols.push(Arc::new(Float64Array::from(c)));
    let batch =
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).map_err(|e| e.to_string())?;
    register_batch(ctx, "cot_user", batch)?;
    let cdf = ctx
        .sql("SELECT * FROM cot_user")
        .await
        .map_err(|e| e.to_string())?;
    let want = match ad::run(ctx, &p, &[cdf]).await {
        Ok(o) => rows_of(o.gradients[0].clone()).await?,
        Err(e) => {
            out.fail(format!("[sql-vjp] the program's vjp fails: {e}"));
            let _ = ctx.deregister_table("cot_user");
            return Ok(());
        }
    };
    let tb = case.table(t);
    let k = tb.dims.len();
    let tdims: Vec<&str> = tb.dims.iter().map(String::as_str).collect();
    let sql = format!(
        "WITH {}, out AS (SELECT {}v FROM r{} c) SELECT {}vjp(out, val, cot_user) AS g FROM {t}",
        case.ctes(),
        lead(&dims, ""),
        case.root,
        lead(&tdims, "")
    );
    let got = query_sql(ctx, &sql).await;
    let _ = ctx.deregister_table("cot_user");
    let got = match got {
        Ok(r) => r,
        Err(e) => {
            failed("sql-vjp", &sql, &e, out);
            return Ok(());
        }
    };
    let (Ok(g), Ok(w)) = (keyed("sql-vjp", &got, k), keyed("sql-vjp", &want, k)) else {
        out.fail(format!("[sql-vjp] repeated dims\n  {sql}"));
        return Ok(());
    };
    out.meta_compared += 1;
    let scale = w
        .values()
        .filter_map(|r| r[k])
        .fold(0.0f64, |m, x| m.max(x.abs()));
    for (key, row) in &w {
        let got_v = g.get(key).and_then(|r| r[k]);
        if !same(got_v, row[k], scale, 1e-8) {
            out.fail(format!(
                "[sql-vjp] {t}{key:?}: vjp in SQL is {got_v:?}, the program's {:?}\n  {sql}",
                row[k]
            ));
            return Ok(());
        }
    }
    if g.len() != tb.keys.len() {
        out.fail(format!(
            "[sql-vjp] {} rows, {t} has {}\n  {sql}",
            g.len(),
            tb.keys.len()
        ));
    }
    Ok(())
}
