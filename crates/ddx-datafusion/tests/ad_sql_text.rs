// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A text-level fuzz of `grad(loss, table.column)` in SQL.
//!
//! `ad::sql` rewrites the statement's text before DataFusion sees it: it finds
//! each `grad(…)` call and splices a relation in its place by byte span
//! (`ddx_ad::sql::GradCalls`). Everything in that path is about text, so the
//! fuzz is about text too. It takes valid statements and puts, between their
//! tokens, what SQL allows there: whitespace of every kind, CRLF, block and
//! line comments (some holding parentheses, quotes or multibyte characters),
//! and changes of case and quoting. Each variant means the same statement, so
//! it must return the same rows as the plain one, or fail loudly.
//!
//! A variant that returns different rows, raises ddx's `Internal` error, or
//! panics is a failure. A loud refusal of valid SQL is tallied by message
//! and, past `DDX_SQL_FUZZ_STRICT`, failed too: the call is valid SQL.

use std::collections::BTreeMap;

use datafusion::arrow::array::{Array, AsArray};
use datafusion::arrow::datatypes::{DataType, Float64Type};
use datafusion::prelude::SessionContext;
use ddx_core::test_utils::{seeded, Rng};
use ddx_datafusion::ad;

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// A statement's rows, every column as f64, sorted.
async fn rows(ctx: &SessionContext, sql: &str) -> Result<Vec<Vec<Option<u64>>>, String> {
    let df = ad::sql(ctx, sql).await.map_err(|e| e.to_string())?;
    let batches = df.collect().await.map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for b in &batches {
        let cols: Vec<_> = (0..b.num_columns())
            .map(|c| datafusion::arrow::compute::cast(b.column(c), &DataType::Float64).unwrap())
            .collect();
        for r in 0..b.num_rows() {
            out.push(
                cols.iter()
                    .map(|a| {
                        let a = a.as_primitive::<Float64Type>();
                        (!a.is_null(r)).then(|| a.value(r).to_bits())
                    })
                    .collect(),
            );
        }
    }
    out.sort();
    Ok(out)
}

/// Valid statements, as tokens: a separator may go between any two.
const STATEMENTS: &[&[&str]] = &[
    &[
        "WITH", "loss", "AS", "(", "SELECT", "SUM", "(", "val", "*", "val", ")", "AS", "l", "FROM",
        "p", ")", "SELECT", "i", ",", "val", "FROM", "grad", "(", "loss", ",", "p", ".", "val",
        ")",
    ],
    &[
        "WITH", "loss", "AS", "(", "SELECT", "SUM", "(", "p", ".", "val", "*", "q", ".", "val",
        ")", "AS", "l", "FROM", "p", "JOIN", "q", "ON", "p", ".", "i", "=", "q", ".", "i", ")",
        "SELECT", "p", ".", "i", ",", "p", ".", "val", "-", "0.1", "*", "g", ".", "val", "AS",
        "val", "FROM", "p", "JOIN", "grad", "(", "loss", ",", "p", ".", "val", ")", "g", "ON", "p",
        ".", "i", "=", "g", ".", "i",
    ],
    &[
        "WITH", "loss", "AS", "(", "SELECT", "SUM", "(", "p", ".", "val", "*", "q", ".", "val",
        ")", "AS", "l", "FROM", "p", "JOIN", "q", "ON", "p", ".", "i", "=", "q", ".", "i", ")",
        "SELECT", "a", ".", "i", ",", "a", ".", "val", ",", "b", ".", "val", "AS", "qv", "FROM",
        "grad", "(", "loss", ",", "p", ".", "val", ")", "a", "JOIN", "grad", "(", "loss", ",", "q",
        ".", "val", ")", "b", "ON", "a", ".", "i", "=", "b", ".", "i",
    ],
    &[
        "WITH",
        "loss",
        "AS",
        "(",
        "SELECT",
        "'grad(loss, p.val)'",
        "AS",
        "s",
        ",",
        "SUM",
        "(",
        "val",
        ")",
        "AS",
        "l",
        "FROM",
        "p",
        ")",
        ",",
        "loss2",
        "AS",
        "(",
        "SELECT",
        "l",
        "FROM",
        "loss",
        ")",
        "SELECT",
        "i",
        ",",
        "val",
        "FROM",
        "grad",
        "(",
        "loss2",
        ",",
        "p",
        ".",
        "val",
        ")",
    ],
];

/// What SQL allows between two tokens.
const SEPARATORS: &[&str] = &[
    " ",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "\n\n  ",
    " /* x */ ",
    " /* ) */ ",
    " /* ( */ ",
    " /* ' */ ",
    " /* \" */ ",
    " /* ☕ café */ ",
    " /* grad(loss, p.val) */ ",
    " -- note\n",
    " -- ) (\n",
    " -- ☕\r\n",
    "/**/",
];

/// Tokens that may change spelling without changing meaning.
fn respell(rng: &mut Rng, token: &str) -> String {
    let keyword = matches!(
        token,
        "WITH" | "AS" | "SELECT" | "FROM" | "JOIN" | "ON" | "SUM" | "grad"
    );
    let ident = matches!(
        token,
        "p" | "q" | "val" | "i" | "loss" | "loss2" | "g" | "a" | "b" | "l"
    );
    match rng.below(6) {
        0 if keyword => token.to_uppercase(),
        1 if keyword => token.to_lowercase(),
        2 if keyword => {
            let mut s = token.to_lowercase();
            if let Some(c) = s.get(..1) {
                s = c.to_uppercase() + &s[1..];
            }
            s
        }
        3 if ident => format!("\"{token}\""),
        _ => token.to_string(),
    }
}

/// Tokens that may not be separated: `.` in a qualified name may be, in
/// sqlparser, and `grad (` may be; everything else may be too.
fn variant(rng: &mut Rng, tokens: &[&str]) -> String {
    let mut out = String::new();
    for (k, t) in tokens.iter().enumerate() {
        if k > 0 {
            if rng.below(3) == 0 {
                out.push_str(rng.pick::<&str>(SEPARATORS));
            } else {
                out.push(' ');
            }
        }
        out.push_str(&respell(rng, t));
    }
    if rng.below(4) == 0 {
        out.insert_str(
            0,
            rng.pick::<&str>(&["/* ☕ */ ", "-- lead\n", "\r\n", "  "]),
        );
    }
    if rng.below(4) == 0 {
        out.push_str(rng.pick::<&str>(&[" -- tail", " /* ) */", "\n", ";"]));
    }
    out
}

#[test]
fn grad_in_sql_means_the_same_whatever_the_spelling() {
    let n: u64 = std::env::var("DDX_SQL_FUZZ_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let strict = std::env::var("DDX_SQL_FUZZ_STRICT").is_ok();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let ctx = SessionContext::new();
    rt.block_on(async {
        exec(
            &ctx,
            "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0), (2, -0.5)",
        )
        .await;
        exec(
            &ctx,
            "CREATE TABLE q (i BIGINT, val DOUBLE) AS VALUES (0, 3.0), (1, -1.0), (2, 0.25)",
        )
        .await;
    });
    let plain: Vec<_> = STATEMENTS
        .iter()
        .map(|t| {
            rt.block_on(rows(&ctx, &t.join(" ")))
                .expect("the plain statement runs")
        })
        .collect();

    let mut failures = Vec::new();
    let mut loud: BTreeMap<String, (u64, String)> = BTreeMap::new();
    let mut same = 0u64;
    for seed in 0..n {
        let mut rng = seeded(seed, 0x5711_7E47);
        let k = rng.below(STATEMENTS.len() as u64) as usize;
        let sql = variant(&mut rng, STATEMENTS[k]);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.block_on(rows(&ctx, &sql))
        }));
        match outcome {
            Err(_) => {
                eprintln!("PANIC seed {seed}: {sql:?}");
                failures.push(format!("seed {seed}: PANIC on {sql:?}"))
            }
            Ok(Ok(r)) if r == plain[k] => same += 1,
            Ok(Ok(r)) => failures.push(format!(
                "seed {seed}: different rows\n  {sql:?}\n  plain {:?}\n  got   {r:?}",
                plain[k]
            )),
            Ok(Err(e)) if e.contains("internal error") => {
                failures.push(format!("seed {seed}: ddx internal error {e}\n  {sql:?}"))
            }
            Ok(Err(e)) => {
                let key: String = e.split_whitespace().take(8).collect::<Vec<_>>().join(" ");
                let entry = loud.entry(key).or_insert((0, sql.clone()));
                entry.0 += 1;
                if strict {
                    failures.push(format!("seed {seed}: valid SQL refused: {e}\n  {sql:?}"));
                }
            }
        }
    }
    eprintln!("{n} spellings: {same} gave the plain rows");
    let mut kinds: BTreeMap<&str, u64> = BTreeMap::new();
    for f in &failures {
        let k = [
            "PANIC",
            "different rows",
            "internal error",
            "valid SQL refused",
        ]
        .into_iter()
        .find(|k| f.contains(k))
        .unwrap_or("other");
        *kinds.entry(k).or_default() += 1;
    }
    eprintln!("  failures by kind: {kinds:?}");
    if let Some(d) = failures.iter().find(|f| f.contains("different rows")) {
        eprintln!("  first silent difference: {d}");
    }
    for (msg, (count, example)) in &loud {
        eprintln!("  refused {count}×: {msg}\n    e.g. {example:?}");
    }
    assert!(
        failures.is_empty(),
        "{} failure(s):\n{}",
        failures.len(),
        failures
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
