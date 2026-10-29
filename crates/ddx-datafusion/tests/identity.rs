// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A property test for identity: which rows, which names, whose tables.
//!
//! An adversarial review of the first query-level AD stack found its defects
//! in identity, none in the derivatives. Each has its own regression test,
//! and the soak (`ad_simulation.rs`) varies each axis on its own. This test
//! varies them together, on generated cases:
//!
//! - `wrt` tables under a bare name, a schema, a quoted mixed-case name,
//!   names whose parts join alike (`a_b.c` and `a.b_c`) or end alike (`s1.v`
//!   and `s3.v`), each spelled in any case;
//! - NULL values in `wrt` and other columns, and tables the loss doesn't
//!   read;
//! - tables whose dims repeat, which a run must refuse, never sum;
//! - several programs and SQL statements on one `SessionContext` at once
//!   (`tokio::spawn`), beside user tables named like ddx's own.
//!
//! The oracle is each program run alone on a context of its own, checked
//! against a finite difference of its loss. On the shared context every
//! program must give the same gradients, or the same refusal, and leave the
//! catalog as it found it.

use std::collections::BTreeMap;

use datafusion::arrow::array::{Array, AsArray, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Float64Type, Int64Type};
use datafusion::prelude::{SessionConfig, SessionContext};
use ddx_core::test_utils::{seeded, Rng};
use ddx_datafusion::ad::{self, ColumnRef};

/// How SQL names a table, and how `wrt` does.
struct Name {
    sql: &'static str,
    wrt: &'static str,
    schema: Option<&'static str>,
}

const NAMES: &[Name] = &[
    Name {
        sql: "w",
        wrt: "w",
        schema: None,
    },
    Name {
        sql: "s1.v",
        wrt: "s1.v",
        schema: Some("s1"),
    },
    Name {
        sql: "\"Wq\"",
        wrt: "Wq",
        schema: None,
    },
    // Their parts join alike: a_b_c.
    Name {
        sql: "a_b.c",
        wrt: "a_b.c",
        schema: Some("a_b"),
    },
    Name {
        sql: "a.b_c",
        wrt: "a.b_c",
        schema: Some("a"),
    },
    Name {
        sql: "s2.\"MiXed\"",
        wrt: "s2.MiXed",
        schema: Some("s2"),
    },
    // The same last part as s1.v, which a step's readable name is.
    Name {
        sql: "s3.v",
        wrt: "s3.v",
        schema: Some("s3"),
    },
];

/// Pairs of names a step could confuse: their parts join alike, or end alike.
const TWINS: &[(usize, usize)] = &[(3, 4), (1, 6)];

/// Tables a user already has, named as ddx names its own.
const DECOYS: &[&str] = &["__ddx_value", "__ddx_0_grad_0_w", "__ddx_cotangent"];

/// A table of the case: `i`, `j` when it has two, and `aux` are its dims,
/// since ddx takes every column `wrt` doesn't name as one.
#[derive(Clone)]
struct Table {
    name: usize,
    two_dims: bool,
    /// `(i, j, val, aux)`; `val` is the `wrt` column.
    rows: Vec<(i64, i64, Option<f64>, Option<f64>)>,
}

impl Table {
    fn sql(&self) -> &'static str {
        NAMES[self.name].sql
    }

    /// Whether two of its rows share their dims.
    fn repeats(&self) -> bool {
        let mut keys: Vec<Key> = self.rows.iter().map(|r| key(r.0, r.1, r.3)).collect();
        keys.sort();
        keys.windows(2).any(|w| w[0] == w[1])
    }

    fn dims(&self) -> &'static str {
        if self.two_dims {
            "i, j, aux"
        } else {
            "i, aux"
        }
    }

    async fn create(&self, ctx: &SessionContext) {
        if let Some(s) = NAMES[self.name].schema {
            exec(ctx, &format!("CREATE SCHEMA IF NOT EXISTS {s}")).await;
        }
        let num = |v: Option<f64>| v.map_or("NULL".to_string(), |v| format!("{v:?}"));
        let rows: Vec<String> = self
            .rows
            .iter()
            .map(|&(i, j, val, aux)| {
                if self.two_dims {
                    format!("({i}, {j}, {}, {})", num(val), num(aux))
                } else {
                    format!("({i}, {}, {})", num(val), num(aux))
                }
            })
            .collect();
        let j = if self.two_dims { "j BIGINT, " } else { "" };
        exec(
            ctx,
            &format!(
                "CREATE OR REPLACE TABLE {} (i BIGINT, {j}val DOUBLE, aux DOUBLE) AS VALUES {}",
                self.sql(),
                rows.join(", ")
            ),
        )
        .await;
    }
}

/// A loss over some of the case's tables, and the tables it is
/// differentiated with respect to: some it reads, and maybe one it doesn't,
/// which `grad` must refuse.
#[derive(Clone)]
struct Loss {
    sql: String,
    read: Vec<usize>,
    wrt: Vec<usize>,
    /// How `wrt` and each table's `grad` in SQL spell the column and, for
    /// each `wrt` table, the table: in any case, which must not matter.
    column: String,
    wrt_names: Vec<String>,
    from_names: Vec<String>,
}

/// `name` in some case: an unquoted name means the same in any, and
/// `wrt` matches any; a quoted one keeps its own in SQL.
fn spell(rng: &mut Rng, name: &str) -> String {
    let quoted = name.contains('"');
    match rng.below(3) {
        0 => name.to_string(),
        1 if !quoted => name.to_uppercase(),
        _ if !quoted => {
            let mut up = false;
            name.chars()
                .map(|c| {
                    up = !up;
                    if up {
                        c.to_ascii_uppercase()
                    } else {
                        c
                    }
                })
                .collect()
        }
        _ => name.to_string(),
    }
}

struct Case {
    tables: Vec<Table>,
    losses: Vec<Loss>,
}

fn gen_case(rng: &mut Rng) -> Case {
    let mut names: Vec<usize> = (0..NAMES.len()).collect();
    for k in (1..names.len()).rev() {
        names.swap(k, rng.below(k as u64 + 1) as usize);
    }
    let n = 2 + rng.below(3) as usize;
    // Often a pair of twins (see TWINS), both in the case.
    let twin = (rng.below(100) < 50).then(|| *rng.pick(TWINS));
    if let Some((x, y)) = twin {
        for (k, name) in [x, y].into_iter().enumerate() {
            let at = names.iter().position(|&n| n == name).unwrap();
            names.swap(k, at);
        }
    }
    let maybe = |rng: &mut Rng, pct: u64| {
        (rng.below(100) >= pct).then(|| (rng.range(-1.0, 1.0) * 1e4).round() / 1e4)
    };
    let tables: Vec<Table> = names[..n]
        .iter()
        .map(|&name| {
            let two_dims = rng.below(2) == 0;
            let mut rows = Vec::new();
            // Some keys are missing, so a join leaves rows of the other side
            // that no gradient reaches.
            for i in 0..3 {
                for j in 0..if two_dims { 2 } else { 1 } {
                    if rng.below(100) < 15 {
                        continue;
                    }
                    let val = maybe(rng, 12);
                    let aux = maybe(rng, 12);
                    rows.push((i, j, val, aux));
                }
            }
            if rows.is_empty() {
                rows.push((0, 0, maybe(rng, 0), maybe(rng, 0)));
            }
            // Another row with the same i and j: the same dims when it
            // copies aux too, which a run must refuse.
            if rng.below(100) < 15 {
                let (i, j, _, aux) = rows[rng.below(rows.len() as u64) as usize];
                let aux = if rng.below(2) == 0 {
                    aux
                } else {
                    maybe(rng, 0)
                };
                rows.push((i, j, maybe(rng, 0), aux));
            }
            for k in (1..rows.len()).rev() {
                rows.swap(k, rng.below(k as u64 + 1) as usize);
            }
            Table {
                name,
                two_dims,
                rows,
            }
        })
        .collect();
    // With twins, the first loss joins them and is differentiated with
    // respect to both: their gradients must not share a step.
    let twins = twin.map(|(x, y)| {
        let at = |name| tables.iter().position(|t| t.name == name).unwrap();
        (at(x), at(y))
    });
    let losses = (0..2 + rng.below(2))
        .map(|k| gen_loss(rng, &tables, twins.filter(|_| k == 0)))
        .collect();
    Case { tables, losses }
}

/// A loss over `tables`; with `pair`, one whose first term joins the two and
/// that is differentiated with respect to both.
fn gen_loss(rng: &mut Rng, tables: &[Table], pair: Option<(usize, usize)>) -> Loss {
    let n = tables.len();
    let mut terms = Vec::new();
    let mut read = Vec::new();
    if let Some((t, u)) = pair {
        terms.push(format!(
            "SELECT SUM(p.val * sin(q.val) * q.aux) AS l FROM {} p JOIN {} q ON p.i = q.i",
            spell(rng, tables[t].sql()),
            spell(rng, tables[u].sql())
        ));
        read.extend([t, u]);
    }
    for _ in 0..1 + rng.below(3) {
        let t = rng.below(n as u64) as usize;
        let term = match rng.below(7) {
            // The value read where no gradient reaches it: its gradient is 0.
            6 => format!(
                "SELECT COUNT(t.val) AS l FROM {} t",
                spell(rng, tables[t].sql())
            ),
            0 | 1 => format!(
                "SELECT SUM(tanh(t.val) * t.aux) AS l FROM {} t",
                spell(rng, tables[t].sql())
            ),
            2 | 3 => format!(
                "SELECT SUM(t.val * t.val + COALESCE(t.aux, 0.5) * t.val) AS l FROM {} t",
                spell(rng, tables[t].sql())
            ),
            _ => {
                let u = rng.below(n as u64) as usize;
                read.push(u);
                format!(
                    "SELECT SUM(p.val * sin(q.val) * q.aux) AS l FROM {} p JOIN {} q ON p.i = q.i",
                    spell(rng, tables[t].sql()),
                    spell(rng, tables[u].sql())
                )
            }
        };
        read.push(t);
        terms.push(term);
    }
    let sum: Vec<String> = (0..terms.len()).map(|k| format!("a{k}.l")).collect();
    let from: Vec<String> = terms
        .iter()
        .enumerate()
        .map(|(k, t)| format!("({t}) a{k}"))
        .collect();
    let sql = format!(
        "SELECT {} AS loss FROM {}",
        sum.join(" + "),
        from.join(" CROSS JOIN ")
    );
    read.sort();
    read.dedup();
    let mut wrt: Vec<usize> = read
        .iter()
        .copied()
        .filter(|_| rng.below(100) < 80)
        .collect();
    if let Some((t, u)) = pair {
        wrt.extend([t, u]);
    }
    if wrt.is_empty() || rng.below(100) < 15 {
        wrt.push(rng.below(n as u64) as usize);
    }
    wrt.sort();
    wrt.dedup();
    let column = spell(rng, "val");
    let wrt_names = wrt
        .iter()
        .map(|&t| spell(rng, NAMES[tables[t].name].wrt).replace('"', ""))
        .collect();
    let from_names = wrt.iter().map(|&t| spell(rng, tables[t].sql())).collect();
    Loss {
        sql,
        read,
        wrt,
        column,
        wrt_names,
        from_names,
    }
}

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
}

async fn batches(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn cell(a: &dyn Array, r: usize) -> Option<f64> {
    if a.is_null(r) {
        return None;
    }
    Some(match a.data_type() {
        DataType::Float64 => a.as_primitive::<Float64Type>().value(r),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(r) as f64,
        other => panic!("a column of type {other}"),
    })
}

/// A row's dims: `i`, `j` (0 for a table without one) and `aux`'s bits.
type Key = (i64, i64, Option<u64>);

fn key(i: i64, j: i64, aux: Option<f64>) -> Key {
    (i, j, aux.map(f64::to_bits))
}

/// A gradient, by its row's dims.
type Grad = BTreeMap<Key, Option<f64>>;

/// Rows of `bs` whose columns are the dims, `i`, `j` and `aux`, then the
/// gradient. Two rows with the same dims are an error: a table's gradient
/// has a row per row of the table.
fn grad_of(bs: &[RecordBatch], two_dims: bool) -> Result<Grad, String> {
    let mut g = Grad::new();
    for b in bs {
        let last = b.num_columns() - 1;
        for r in 0..b.num_rows() {
            let i = cell(b.column(0), r).expect("a dim") as i64;
            let j = if two_dims {
                cell(b.column(1), r).expect("a dim") as i64
            } else {
                0
            };
            let k = key(i, j, cell(b.column(last - 1), r));
            if let Some(old) = g.insert(k, cell(b.column(last), r)) {
                return Err(format!(
                    "two gradient rows at {k:?}: {old:?} and {:?}",
                    g[&k]
                ));
            }
        }
    }
    Ok(g)
}

fn wrt_of(loss: &Loss) -> Vec<ColumnRef> {
    loss.wrt_names
        .iter()
        .map(|t| ColumnRef::new(t.as_str(), loss.column.as_str()))
        .collect()
}

/// What a program gives: each `wrt` table's gradient, or a refusal.
type Answer = Result<Vec<Grad>, String>;

/// Build `loss`'s program on `ctx`, run it, read its gradients, release it.
async fn answer(ctx: &SessionContext, case: &Case, loss: &Loss) -> Answer {
    let program = ad::grad(ctx, loss.sql.as_str(), &wrt_of(loss))
        .await
        .map_err(|e| format!("grad: {e}"))?;
    let ran = ddx_datafusion::engine::execute(ctx, &program).await;
    let out = match ran {
        Ok(()) => {
            let mut grads = Ok(Vec::new());
            // In `loss.wrt`'s order: the program's is the plan's.
            for &t in &loss.wrt {
                let name = NAMES[case.tables[t].name].wrt;
                let g = program
                    .gradients
                    .iter()
                    .find(|g| {
                        g.of.table()
                            .is_some_and(|parts| name.eq_ignore_ascii_case(&parts.join(".")))
                    })
                    .unwrap_or_else(|| panic!("no gradient of {name}"));
                let bs = batches(
                    ctx,
                    &format!("SELECT {}, val FROM {}", case.tables[t].dims(), g.step),
                )
                .await;
                grads = grads.and_then(|mut gs| {
                    gs.push(grad_of(&bs, case.tables[t].two_dims)?);
                    Ok(gs)
                });
            }
            grads
        }
        Err(e) => Err(e.to_string()),
    };
    ddx_datafusion::engine::release(ctx, &program).unwrap();
    out
}

async fn loss_value(ctx: &SessionContext, sql: &str) -> Option<f64> {
    let bs = batches(ctx, sql).await;
    cell(bs[0].column(0), 0)
}

/// `loss` alone on a context of its own, its gradients checked against a
/// central difference of the loss, or its refusal against the tables.
async fn reference(case: &Case, loss: &Loss, report: &mut Vec<String>) -> Answer {
    let ctx = SessionContext::new();
    for t in &case.tables {
        t.create(&ctx).await;
    }
    let got = answer(&ctx, case, loss).await;
    if loss.wrt.iter().any(|t| !loss.read.contains(t)) {
        match &got {
            Err(e) if e.contains("unknown wrt column") => {}
            _ => report.push(format!("a wrt table the loss doesn't read: {got:?}")),
        }
        return got;
    }
    let repeats = loss.wrt.iter().any(|&t| case.tables[t].repeats());
    let grads = match (&got, repeats) {
        (Err(e), true) if e.contains("share their dims") => return got,
        (Err(e), _) => {
            report.push(format!("refused: {e}"));
            return got;
        }
        (Ok(_), true) => {
            report.push("ran over a table whose dims repeat".into());
            return got;
        }
        (Ok(grads), false) => grads,
    };
    let base = loss_value(&ctx, &loss.sql).await;
    for (g, &t) in grads.iter().zip(&loss.wrt) {
        let table = &case.tables[t];
        if g.len() != table.rows.len() {
            report.push(format!(
                "{}: {} rows for {}",
                table.sql(),
                g.len(),
                table.rows.len()
            ));
        }
        for (r, &(i, j, val, aux)) in table.rows.iter().enumerate() {
            let Some(&ad) = g.get(&key(i, j, aux)) else {
                report.push(format!("{}: no gradient at ({i}, {j})", table.sql()));
                continue;
            };
            let Some(v) = val else {
                if ad.is_some() {
                    report.push(format!("{}: NULL value's gradient is {ad:?}", table.sql()));
                }
                continue;
            };
            let Some(ad) = ad else {
                report.push(format!("{}: gradient NULL at ({i}, {j})", table.sql()));
                continue;
            };
            // A NULL loss moves with nothing: every gradient is 0.
            let fd = if base.is_none() {
                0.0
            } else {
                let h = 1e-6;
                let mut moved = table.clone();
                moved.rows[r].2 = Some(v + h);
                moved.create(&ctx).await;
                let up = loss_value(&ctx, &loss.sql).await;
                moved.rows[r].2 = Some(v - h);
                moved.create(&ctx).await;
                let down = loss_value(&ctx, &loss.sql).await;
                table.create(&ctx).await;
                match (up, down) {
                    (Some(u), Some(d)) => (u - d) / (2.0 * h),
                    _ => continue,
                }
            };
            if (ad - fd).abs() > 1e-6 * fd.abs().max(1.0) {
                report.push(format!(
                    "{} at ({i}, {j}): ddx {ad} vs finite difference {fd}",
                    table.sql()
                ));
            }
        }
    }
    got
}

/// Each user table's rows, by name: what the shared context must keep.
async fn catalog(ctx: &SessionContext) -> BTreeMap<String, Vec<String>> {
    let names = batches(
        ctx,
        "SELECT table_schema, table_name FROM information_schema.tables \
         WHERE table_schema <> 'information_schema'",
    )
    .await;
    let mut out = BTreeMap::new();
    for b in &names {
        for r in 0..b.num_rows() {
            let schema = b.column(0).as_string::<i32>().value(r);
            let name = b.column(1).as_string::<i32>().value(r);
            let rel = format!("\"{schema}\".\"{name}\"");
            let mut rows: Vec<String> = Vec::new();
            for rb in batches(ctx, &format!("SELECT * FROM {rel}")).await {
                let cols = rb.columns();
                for i in 0..rb.num_rows() {
                    let row: Vec<String> = cols
                        .iter()
                        .map(|c| {
                            datafusion::arrow::util::display::array_value_to_string(c, i).unwrap()
                        })
                        .collect();
                    rows.push(row.join("|"));
                }
            }
            rows.sort();
            out.insert(rel, rows);
        }
    }
    out
}

fn close(a: &Grad, b: &Grad) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|((ka, va), (kb, vb))| {
            ka == kb
                && match (va, vb) {
                    (Some(x), Some(y)) => (x - y).abs() <= 1e-12 * x.abs().max(1.0),
                    (None, None) => true,
                    _ => false,
                }
        })
}

/// The SQL surface's gradient of `loss` for table `t`, by its rows.
async fn sql_answer(
    ctx: &SessionContext,
    case: &Case,
    loss: &Loss,
    t: usize,
) -> Result<Grad, String> {
    let table = &case.tables[t];
    let at = loss.wrt.iter().position(|&x| x == t).expect("a wrt table");
    let sql = format!(
        "WITH loss AS ({}) SELECT {}, grad(loss, {}) AS g FROM {}",
        loss.sql,
        table.dims(),
        loss.column,
        loss.from_names[at]
    );
    let df = ad::sql(ctx, &sql).await.map_err(|e| e.to_string())?;
    let bs = df.collect().await.map_err(|e| e.to_string())?;
    grad_of(&bs, table.two_dims)
}

async fn check(seed: u64) -> Vec<String> {
    let mut rng = seeded(seed, 0x1d);
    let case = gen_case(&mut rng);
    let mut report = Vec::new();
    let mut want = Vec::new();
    for loss in &case.losses {
        want.push(reference(&case, loss, &mut report).await);
    }

    let ctx = SessionContext::new_with_config(SessionConfig::new().with_information_schema(true));
    for t in &case.tables {
        t.create(&ctx).await;
    }
    for d in DECOYS {
        exec(
            &ctx,
            &format!("CREATE TABLE {d} (note VARCHAR) AS VALUES ('mine')"),
        )
        .await;
    }
    let before = catalog(&ctx).await;

    // Every program, and every table's statement in SQL, at once.
    let case = std::sync::Arc::new(case);
    let mut programs = Vec::new();
    for k in 0..case.losses.len() {
        let (ctx, case) = (ctx.clone(), case.clone());
        programs.push(tokio::spawn(async move {
            answer(&ctx, &case, &case.losses[k]).await
        }));
    }
    let mut statements = Vec::new();
    for (k, loss) in case.losses.iter().enumerate() {
        for &t in &loss.wrt {
            let (ctx, case) = (ctx.clone(), case.clone());
            statements.push((
                k,
                t,
                tokio::spawn(async move { sql_answer(&ctx, &case, &case.losses[k], t).await }),
            ));
        }
    }
    for (k, p) in programs.into_iter().enumerate() {
        let got = p.await.expect("the task ran");
        match (&got, &want[k]) {
            (Ok(g), Ok(w)) if g.iter().zip(w).all(|(g, w)| close(g, w)) => {}
            (Err(g), Err(w)) if g == w => {}
            _ => report.push(format!(
                "loss {k} on the shared context: {:?}, alone: {:?}",
                got, want[k]
            )),
        }
    }
    for (k, t, s) in statements {
        let got = s.await.expect("the task ran");
        // A statement differentiates with respect to its own table alone:
        // the program's gradient for it, unless the program was refused for
        // another table, when the loss is taken with respect to it alone.
        let i = case.losses[k].wrt.iter().position(|&x| x == t).unwrap();
        let want = match &want[k] {
            Ok(w) => Ok(w[i].clone()),
            Err(_) => {
                let l = &case.losses[k];
                let alone = Loss {
                    wrt: vec![t],
                    wrt_names: vec![l.wrt_names[i].clone()],
                    from_names: vec![l.from_names[i].clone()],
                    ..l.clone()
                };
                reference(&case, &alone, &mut report)
                    .await
                    .map(|mut w| w.remove(0))
            }
        };
        let refusal = |e: &str| e.contains("share their dims") || e.contains("unknown wrt column");
        match (&got, &want) {
            (Ok(g), Ok(w)) if close(g, w) => {}
            (Err(g), Err(w)) if refusal(g) && refusal(w) => {}
            _ => report.push(format!(
                "loss {k}'s grad in SQL for {}: {got:?}, alone: {want:?}",
                case.tables[t].sql()
            )),
        }
    }
    let after = catalog(&ctx).await;
    if after != before {
        let gone: Vec<&String> = before.keys().filter(|k| !after.contains_key(*k)).collect();
        let left: Vec<&String> = after.keys().filter(|k| !before.contains_key(*k)).collect();
        report.push(format!(
            "the catalog changed: gone {gone:?}, left behind {left:?}, or a table's rows changed"
        ));
    }
    if !report.is_empty() {
        let mut s = format!("seed {seed}:");
        for loss in &case.losses {
            s += &format!("\n  loss {} wrt {:?}", loss.sql, wrt_of(loss));
        }
        for t in case.tables.iter() {
            s += &format!("\n  {} rows {:?}", t.sql(), t.rows);
        }
        report.insert(0, s);
    }
    report
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn identity_holds_across_names_nulls_repeats_and_concurrent_programs() {
    let n: u64 = std::env::var("DDX_IDENTITY_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let base: u64 = std::env::var("DDX_IDENTITY_BASE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut failures = Vec::new();
    for seed in base..base + n {
        failures.extend(check(seed).await);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
