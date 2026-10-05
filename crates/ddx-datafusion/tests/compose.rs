// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad`, `vjp` and `jvp` composed with each other, end to end on
//! DataFusion. Forward over reverse (`jvp` of a `grad` program, `H·v`) is
//! checked against finite differences in `tests/jvp.rs`; here it is the
//! oracle for the other orders: reverse over forward (`vjp` of a `jvp`
//! program, seeded 0 on the value and 1 on its tangent) gives `H·v` too, and
//! forward over forward (`jvp` of a `jvp` program) gives `uᵀ H v`.

mod common;

#[path = "../examples/nn/model.rs"]
#[allow(dead_code)]
mod model;

use std::collections::BTreeMap;

use common::ad::{rows, Table};
use common::substrait_of;
use datafusion::prelude::SessionContext;
use ddx_ad::{AdError, ColumnRef, InputTable, Of, OutputTable};
use ddx_datafusion::ad;

fn w() -> Table {
    Table {
        name: "w",
        columns: vec![("i", "BIGINT"), ("o", "BIGINT"), ("val", "DOUBLE")],
        rows: vec![
            vec![0.0, 0.0, 0.3],
            vec![0.0, 1.0, -0.2],
            vec![1.0, 0.0, 0.7],
            vec![1.0, 1.0, 0.1],
            vec![2.0, 0.0, -0.4],
            vec![2.0, 1.0, 0.5],
        ],
    }
}

fn x() -> Table {
    Table {
        name: "x",
        columns: vec![("n", "BIGINT"), ("i", "BIGINT"), ("v", "DOUBLE")],
        rows: vec![
            vec![0.0, 0.0, 1.5],
            vec![0.0, 1.0, -0.5],
            vec![0.0, 2.0, 2.0],
            vec![1.0, 0.0, 0.25],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 2.0, -1.25],
        ],
    }
}

fn b() -> Table {
    Table {
        name: "b",
        columns: vec![("o", "BIGINT"), ("val", "DOUBLE")],
        rows: vec![vec![0.0, 0.25], vec![1.0, -0.6]],
    }
}

const ELEMENTWISE: &str = "SELECT SUM(tanh(val) * exp(val / 2.0) + val * val * val) AS loss FROM w";

const LAYER: &str = "WITH h AS (SELECT x.n, w.o, tanh(SUM(x.v * w.val) + MAX(b.val)) AS y \
                     FROM x JOIN w ON x.i = w.i JOIN b ON w.o = b.o GROUP BY x.n, w.o) \
                     SELECT SUM((y - 0.1) * (y - 0.1)) / COUNT(*) AS loss FROM h";

const MAX_AND_MEAN: &str = "WITH g AS (SELECT o, MAX(val * val) AS m, AVG(sin(val)) AS a \
                            FROM w GROUP BY o) SELECT SUM(m * a + a * a) AS loss FROM g";

/// The losses, their tables and their `wrt` columns.
fn cases() -> Vec<(&'static str, Vec<Table>, Vec<ColumnRef>)> {
    vec![
        (ELEMENTWISE, vec![w()], vec![ColumnRef::new("w", "val")]),
        (
            LAYER,
            vec![x(), w(), b()],
            vec![
                ColumnRef::new("x", "v"),
                ColumnRef::new("w", "val"),
                ColumnRef::new("b", "val"),
            ],
        ),
        (MAX_AND_MEAN, vec![w()], vec![ColumnRef::new("w", "val")]),
    ]
}

async fn setup(tables: &[Table]) -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    for t in tables {
        t.create(&ctx).await;
    }
    ctx
}

/// Register `input`, a tangent of a `wrt` table, along direction `seed`:
/// for each row a deterministic number of its keys, different per seed.
async fn register_tangent(ctx: &SessionContext, input: &InputTable, seed: f64) {
    let table = input.of.table().and_then(|t| t.last()).expect("a tangent");
    let keys = &input.columns[..input.keys];
    let mix: Vec<String> = keys
        .iter()
        .enumerate()
        .map(|(k, c)| format!("{} * {c}", 1.3 + 0.9 * k as f64 + seed))
        .collect();
    let values: Vec<String> = input.columns[input.keys..]
        .iter()
        .enumerate()
        .map(|(j, c)| {
            format!(
                "0.3 + 0.7 * sin({} + {}) AS {c}",
                mix.join(" + "),
                0.2 + 0.4 * j as f64 + seed
            )
        })
        .collect();
    let sql = format!(
        "CREATE OR REPLACE TABLE {} AS SELECT {}, {} FROM {table}",
        input.name,
        keys.join(", "),
        values.join(", ")
    );
    ctx.sql(&sql).await.unwrap().collect().await.unwrap();
}

/// Register `input`, the cotangent of a one-row output, as `values`.
async fn register_cotangent(ctx: &SessionContext, input: &InputTable, values: &[f64]) {
    assert_eq!(input.of, Of::Output);
    assert_eq!(input.keys, 0, "a loss has no keys");
    let cols: Vec<String> = input
        .columns
        .iter()
        .zip(values)
        .map(|(c, v)| format!("CAST({v:e} AS DOUBLE) AS {c}"))
        .collect();
    let sql = format!(
        "CREATE OR REPLACE TABLE {} AS SELECT {}",
        input.name,
        cols.join(", ")
    );
    ctx.sql(&sql).await.unwrap().collect().await.unwrap();
}

/// Each gradient's table, by `wrt` table name: each row's values, keyed by
/// its dims; `tangents` picks the tangent columns instead.
async fn read(
    ctx: &SessionContext,
    outputs: &[OutputTable],
    wrt: &[ColumnRef],
    tangents: bool,
) -> BTreeMap<String, BTreeMap<Vec<i64>, Vec<f64>>> {
    let mut out = BTreeMap::new();
    for g in outputs {
        let table = g.of.table().and_then(|t| t.last()).unwrap().clone();
        let values: Vec<&str> = wrt
            .iter()
            .filter(|w| w.table == table)
            .map(|w| w.column.as_str())
            .collect();
        let dims = g.columns.len() - values.len() - g.tangents.len();
        let picked: Vec<usize> = if tangents {
            g.tangents
                .iter()
                .map(|t| g.columns.iter().position(|c| *c == t.tangent).unwrap())
                .collect()
        } else {
            (dims..dims + values.len()).collect()
        };
        let rows = rows(ctx, &format!("SELECT * FROM {}", g.step)).await;
        out.insert(
            table,
            rows.into_iter()
                .map(|r| {
                    let key = r[..dims].iter().map(|&k| k as i64).collect();
                    (key, picked.iter().map(|&c| r[c]).collect())
                })
                .collect(),
        );
    }
    out
}

/// `H·v`, forward over reverse, with each tangent along seed 0.
async fn forward_over_reverse(
    ctx: &SessionContext,
    loss: &str,
    wrt: &[ColumnRef],
) -> BTreeMap<String, BTreeMap<Vec<i64>, Vec<f64>>> {
    let plan = substrait_of(ctx, loss, true).await;
    let hvp = ddx_ad::jvp(&ddx_ad::grad(&plan, wrt).unwrap(), wrt).unwrap();
    for input in &hvp.inputs {
        register_tangent(ctx, input, 0.0).await;
    }
    ad::run(ctx, &hvp).await.unwrap();
    read(ctx, &hvp.gradients, wrt, true).await
}

fn assert_close(
    what: &str,
    got: &BTreeMap<String, BTreeMap<Vec<i64>, Vec<f64>>>,
    want: &BTreeMap<String, BTreeMap<Vec<i64>, Vec<f64>>>,
) {
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>()
    );
    for (table, rows) in want {
        assert_eq!(got[table].len(), rows.len(), "{what}: {table}'s rows");
        for (key, want) in rows {
            let got = &got[table][key];
            for (g, w) in got.iter().zip(want) {
                assert!(
                    (g - w).abs() <= 1e-9 * w.abs().max(1.0),
                    "{what} of {table} at {key:?}: {g} vs {w}"
                );
            }
        }
    }
}

#[tokio::test]
async fn reverse_over_forward_is_a_hessian_vector_product() {
    for (loss, tables, wrt) in cases() {
        let ctx = setup(&tables).await;
        let want = forward_over_reverse(&ctx, loss, &wrt).await;

        let plan = substrait_of(&ctx, loss, true).await;
        let jvp = ddx_ad::jvp(&plan, &wrt).unwrap();
        let program = ddx_ad::vjp(&jvp, &wrt).unwrap_or_else(|e| panic!("vjp of a jvp: {e}"));
        // The jvp's tangents first, then the vjp's cotangent.
        assert_eq!(program.inputs.len(), jvp.inputs.len() + 1);
        for (carried, input) in program.inputs.iter().zip(&jvp.inputs) {
            assert_eq!(carried.name, input.name);
        }
        for input in &jvp.inputs {
            register_tangent(&ctx, input, 0.0).await;
        }
        // Cotangent 0 on the loss and 1 on its tangent: the gradient of the
        // directional derivative ∇l·v, which is H·v.
        let cotangent = program.inputs.last().unwrap();
        assert_eq!(cotangent.columns, jvp.value.columns);
        register_cotangent(&ctx, cotangent, &[0.0, 1.0]).await;
        common::ad::run(&ctx, &program).await;
        let got = read(&ctx, &program.gradients, &wrt, false).await;
        assert_close(&format!("vjp of jvp of {loss}"), &got, &want);
    }
}

#[tokio::test]
async fn forward_over_forward_is_a_second_directional_derivative() {
    for (loss, tables, wrt) in cases() {
        let ctx = setup(&tables).await;
        let hv = forward_over_reverse(&ctx, loss, &wrt).await;

        let plan = substrait_of(&ctx, loss, true).await;
        let inner = ddx_ad::jvp(&plan, &wrt).unwrap();
        let outer = ddx_ad::jvp(&inner, &wrt).unwrap_or_else(|e| panic!("jvp of a jvp: {e}"));
        let n = inner.inputs.len();
        assert_eq!(outer.inputs.len(), 2 * n, "v's tables, then u's");
        // v along seed 0, as for H·v; u along seed 1.
        for input in &outer.inputs[..n] {
            register_tangent(&ctx, input, 0.0).await;
        }
        for input in &outer.inputs[n..] {
            register_tangent(&ctx, input, 1.0).await;
        }
        ad::run(&ctx, &outer).await.unwrap();

        // uᵀ (H v), from H·v and u's tables.
        let mut want = 0.0;
        for input in &outer.inputs[n..] {
            let table = input.of.table().and_then(|t| t.last()).unwrap();
            for r in rows(&ctx, &format!("SELECT * FROM {}", input.name)).await {
                let key: Vec<i64> = r[..input.keys].iter().map(|&k| k as i64).collect();
                let hv = &hv[table][&key];
                want += r[input.keys..]
                    .iter()
                    .zip(hv)
                    .map(|(u, h)| u * h)
                    .sum::<f64>();
            }
        }
        // The loss, ∇l·v, ∇l·u, and uᵀ H v, the tangent of ∇l·v along u.
        let value = &outer.value;
        assert_eq!(value.columns.len(), 4, "{:?}", value.columns);
        let inner_tangent = &inner.value.tangents[0].tangent;
        let second = value
            .tangents
            .iter()
            .find(|t| t.column == *inner_tangent)
            .expect("the inner tangent has a tangent");
        let c = value
            .columns
            .iter()
            .position(|n| *n == second.tangent)
            .unwrap();
        let got = rows(&ctx, &format!("SELECT * FROM {}", value.step)).await;
        assert_eq!(got.len(), 1);
        assert!(
            (got[0][c] - want).abs() <= 1e-9 * want.abs().max(1.0),
            "uᵀHv of {loss}: {} vs {want}",
            got[0][c]
        );
    }
}

#[tokio::test]
async fn reverse_over_reverse_is_refused_until_reverse_mode_has_rules_for_it() {
    // vjp of a grad program differentiates the gradient table, which joins
    // the table's rows to the summed contributions on its dims and keeps the
    // table's copy of them. vjp needs its output to keep every input's dims,
    // to tell its rows apart, and cannot yet see that the contributions'
    // copy is equal; so it refuses, rather than answer wrong.
    let ctx = setup(&[w()]).await;
    let wrt = [ColumnRef::new("w", "val")];
    let plan = substrait_of(&ctx, ELEMENTWISE, true).await;
    let grad = ddx_ad::grad(&plan, &wrt).unwrap();
    match ddx_ad::vjp(&grad, &wrt) {
        Err(AdError::NotImplemented(m)) if m.contains("every dim") => {}
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[tokio::test]
async fn a_program_with_more_than_one_output_table_is_refused() {
    let ctx = setup(&[x(), w(), b()]).await;
    let wrt = [ColumnRef::new("w", "val"), ColumnRef::new("b", "val")];
    let plan = substrait_of(&ctx, LAYER, true).await;
    let grad = ddx_ad::grad(&plan, &wrt).unwrap();
    for e in [
        ddx_ad::vjp(&grad, &wrt).unwrap_err(),
        ddx_ad::grad(&grad, &wrt).unwrap_err(),
    ] {
        assert!(
            matches!(&e, AdError::NotImplemented(m) if m.contains("2 gradients")),
            "{e}"
        );
    }
}

#[tokio::test]
async fn grad_of_a_value_and_its_tangent_is_refused_as_not_a_loss() {
    let ctx = setup(&[w()]).await;
    let wrt = [ColumnRef::new("w", "val")];
    let plan = substrait_of(&ctx, ELEMENTWISE, true).await;
    let jvp = ddx_ad::jvp(&plan, &wrt).unwrap();
    let e = ddx_ad::grad(&jvp, &wrt).unwrap_err();
    assert!(matches!(e, AdError::NotScalar(_)), "{e}");
}

#[tokio::test]
async fn the_adapter_composes_from_sql() {
    let ctx = setup(&[w()]).await;
    let wrt = [ColumnRef::new("w", "val")];
    let want = forward_over_reverse(&ctx, MAX_AND_MEAN, &wrt).await;
    let jvp = ad::jvp(&ctx, MAX_AND_MEAN, &wrt).await.unwrap();
    let program = ad::vjp(&ctx, &jvp, &wrt).await.unwrap();
    for input in &jvp.inputs {
        register_tangent(&ctx, input, 0.0).await;
    }
    register_cotangent(&ctx, program.inputs.last().unwrap(), &[0.0, 1.0]).await;
    ad::run(&ctx, &program).await.unwrap();
    let got = read(&ctx, &program.gradients, &wrt, false).await;
    assert_close("ad::vjp of ad::jvp", &got, &want);
}

#[test]
fn vjp_of_a_jvp_program_does_not_overflow_a_worker_threads_stack() {
    // A jvp program's step is about twice as deep as its query (nn.py's
    // loss: 63 relations, against 35), and reverse mode lowered a chain of
    // projections recursively: on a tokio worker's 2 MB stack, vjp of it
    // overflowed, which aborts the whole process. So it runs in a child.
    if std::env::var("DDX_COMPOSE_CHILD").is_ok() {
        let wrt = vec![
            ColumnRef::new("weight", "val"),
            ColumnRef::new("bias", "val"),
        ];
        let plan = tokio::runtime::Runtime::new().unwrap().block_on(async {
            let ctx = SessionContext::new();
            let mut rng = model::Rng::new(7);
            model::register_data(&ctx, 10, &mut rng).unwrap();
            model::register_model(&ctx, &mut rng).unwrap();
            substrait_of(&ctx, &model::loss_sql(), true).await
        });
        let t = std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(move || {
                let jvp = ddx_ad::jvp(&plan, &wrt)?;
                ddx_ad::vjp(&jvp, &wrt).map(|_| ())
            })
            .unwrap();
        t.join().unwrap().unwrap();
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "vjp_of_a_jvp_program_does_not_overflow_a_worker_threads_stack",
            "--nocapture",
        ])
        .env("DDX_COMPOSE_CHILD", "1")
        .status()
        .unwrap();
    assert!(
        status.success(),
        "vjp of a jvp program on a 2 MB stack: {status}"
    );
}
