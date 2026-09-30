// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! How a program's cost scales, against its query's.
//!
//! Reverse mode should cost a small constant multiple of the forward pass, as
//! it does in JAX; ddx adds recomputation (design.md §4.4), a region rebuilt
//! in each backward step, so a larger constant is expected, but not a worse
//! exponent. These measure, per family and size: the forward query alone,
//! building the program, running it, and each step, plus the sizes of ddx's
//! plans and of the plans DataFusion builds from them.
//!
//! Ignored: they are measurements, not assertions. Run one family per process,
//! under a memory cap:
//!
//! ```text
//! DDX_PERF=nn systemd-run --user --scope -p MemoryMax=6G \
//!   cargo test -p ddx-datafusion --test ad_perf --release -- --ignored --nocapture
//! ```

use std::time::{Duration, Instant};

use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, ColumnRef};

#[path = "../examples/nn/model.rs"]
#[allow(dead_code)]
mod model;

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// The least of three timings of `f`.
async fn best<F, Fut>(mut f: F) -> Duration
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let mut out = Duration::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        f().await;
        out = out.min(t.elapsed());
    }
    out
}

struct Report {
    forward: Duration,
    build: Duration,
    run: Duration,
    steps: usize,
    ddx_bytes: usize,
    df_bytes: usize,
    slowest: (Duration, String),
}

async fn measure(ctx: &SessionContext, loss: &str, wrt: &[ColumnRef]) -> Report {
    let forward = best(|| async { exec(ctx, loss).await }).await;
    let build = best(|| async {
        ad::grad(ctx, loss, wrt).await.unwrap();
    })
    .await;
    let program = ad::grad(ctx, loss, wrt).await.unwrap();
    let run = best(|| async { ad::run(ctx, &program).await.unwrap() }).await;
    let mut df_bytes = 0;
    let mut slowest = (Duration::ZERO, String::new());
    ad::run_checks(ctx, &program).await.unwrap();
    for step in program.steps() {
        df_bytes += ad::logical_plan(ctx, &step.plan)
            .await
            .unwrap()
            .display_indent()
            .to_string()
            .len();
        let t = Instant::now();
        ad::run_step(ctx, step).await.unwrap();
        let d = t.elapsed();
        if d > slowest.0 {
            let name = step.name.splitn(4, '_').last().unwrap_or("").to_string();
            slowest = (d, name);
        }
    }
    let _ = ad::release(ctx, &program);
    Report {
        forward,
        build,
        run,
        steps: program.steps().count(),
        ddx_bytes: program.steps().map(|s| prost_len(&s.plan)).sum(),
        df_bytes,
        slowest,
    }
}

fn prost_len(p: &ddx_ad::substrait::proto::Plan) -> usize {
    format!("{p:?}").len()
}

fn print(family: &str, size: &str, r: &Report) {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    eprintln!(
        "PERF {family:<10} {size:<16} forward {:>9.2} ms | build {:>8.2} ms | run {:>9.2} ms | \
         (build+run)/forward {:>7.1}x | steps {:>3} | ddx plans {:>9} B | DataFusion plans {:>10} B | \
         slowest step {} {:.2} ms",
        ms(r.forward),
        ms(r.build),
        ms(r.run),
        (r.build + r.run).as_secs_f64() / r.forward.as_secs_f64(),
        r.steps,
        r.ddx_bytes,
        r.df_bytes,
        r.slowest.1,
        ms(r.slowest.0)
    );
}

async fn nn(n: usize) {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    let mut rng = model::Rng::new(7);
    model::register_data(&ctx, n, &mut rng).unwrap();
    model::register_model(&ctx, &mut rng).unwrap();
    let r = measure(
        &ctx,
        &model::loss_sql(),
        &[
            ColumnRef::new("weight", "val"),
            ColumnRef::new("bias", "val"),
        ],
    )
    .await;
    print("nn", &format!("samples={n}"), &r);
}

async fn contraction(n: usize, d: usize, h: usize) {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        &format!(
            "CREATE TABLE a AS SELECT CAST(r / {d} AS BIGINT) AS s, CAST(r % {d} AS BIGINT) AS k, \
             sin(CAST(r AS DOUBLE)) AS val FROM (SELECT unnest(range(0, {})) AS r)",
            n * d
        ),
    )
    .await;
    exec(
        &ctx,
        &format!(
            "CREATE TABLE w AS SELECT CAST(r / {h} AS BIGINT) AS k, CAST(r % {h} AS BIGINT) AS o, \
             0.1 * cos(CAST(r AS DOUBLE)) AS val FROM (SELECT unnest(range(0, {})) AS r)",
            d * h
        ),
    )
    .await;
    let loss = "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k \
                GROUP BY a.s, w.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM c";
    let r = measure(
        &ctx,
        loss,
        &[ColumnRef::new("w", "val"), ColumnRef::new("a", "val")],
    )
    .await;
    print("matmul", &format!("n={n} d={d} h={h}"), &r);
}

async fn chain(kind: &str, n: usize, rows: usize) {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        &format!(
            "CREATE TABLE p AS SELECT CAST(r AS BIGINT) AS i, 0.001 * CAST(r AS DOUBLE) AS val \
             FROM (SELECT unnest(range(0, {rows})) AS r)"
        ),
    )
    .await;
    let loss = match kind {
        "depth" => {
            let mut ctes = vec!["c0 AS (SELECT i, val AS v FROM p)".to_string()];
            for k in 1..n {
                ctes.push(format!(
                    "c{k} AS (SELECT i, sin(v) + 0.1 * v AS v FROM c{})",
                    k - 1
                ));
            }
            format!(
                "WITH {} SELECT SUM(v) AS l FROM c{}",
                ctes.join(", "),
                n - 1
            )
        }
        "fanin" => {
            let cols: Vec<String> = (1..=n)
                .map(|k| format!("sin(val * {k}.0) AS c{k}"))
                .collect();
            let sum: Vec<String> = (1..=n).map(|k| format!("c{k}")).collect();
            format!(
                "WITH r AS (SELECT {} FROM p) SELECT SUM({}) AS l FROM r",
                cols.join(", "),
                sum.join(" + ")
            )
        }
        "layers" => {
            // n dense-ish layers of a per-row map and a grouped sum: depth
            // across aggregates rather than within one region.
            let mut ctes = vec!["l0 AS (SELECT i, val AS v FROM p)".to_string()];
            for k in 1..n {
                ctes.push(format!(
                    "l{k} AS (SELECT i % {m} AS i, SUM(tanh(v)) AS v FROM l{} GROUP BY i % {m})",
                    k - 1,
                    m = (rows >> k.min(20)).max(1)
                ));
            }
            format!(
                "WITH {} SELECT SUM(v * v) AS l FROM l{}",
                ctes.join(", "),
                n - 1
            )
        }
        _ => {
            let mut ctes = vec!["c0 AS (SELECT i, val AS v FROM p)".to_string()];
            for k in 1..n {
                ctes.push(format!(
                    "c{k} AS (SELECT a.i, a.v * b.v AS v FROM c{m} a JOIN c{m} b ON a.i = b.i)",
                    m = k - 1
                ));
            }
            format!(
                "WITH {} SELECT SUM(v) AS l FROM c{}",
                ctes.join(", "),
                n - 1
            )
        }
    };
    let r = measure(&ctx, &loss, &[ColumnRef::new("p", "val")]).await;
    print(kind, &format!("n={n} rows={rows}"), &r);
}

/// A residual MLP: `h(k+1) = h(k) + tanh(h(k) · W(k))`, `blocks` blocks of
/// width `d` over `n` samples. Each block reads its input twice, once for the
/// skip connection and once for the product, as a residual network does.
async fn resnet(blocks: usize, n: usize, d: usize) {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        &format!(
            "CREATE TABLE x AS SELECT CAST(r / {d} AS BIGINT) AS s, CAST(r % {d} AS BIGINT) AS k, \
             sin(CAST(r AS DOUBLE)) AS val FROM (SELECT unnest(range(0, {})) AS r)",
            n * d
        ),
    )
    .await;
    exec(
        &ctx,
        &format!(
            "CREATE TABLE w AS SELECT CAST(r / {} AS BIGINT) AS b, CAST(r / {d} % {d} AS BIGINT) AS k, \
             CAST(r % {d} AS BIGINT) AS o, 0.1 * cos(CAST(r AS DOUBLE)) AS val \
             FROM (SELECT unnest(range(0, {})) AS r)",
            d * d,
            blocks * d * d
        ),
    )
    .await;
    let mut ctes = vec!["h0 AS (SELECT s, k, val AS v FROM x)".to_string()];
    for b in 0..blocks {
        ctes.push(format!(
            "z{b} AS (SELECT h.s, w.o, SUM(h.v * w.val) AS z FROM h{b} h JOIN w ON h.k = w.k AND w.b = {b} \
             GROUP BY h.s, w.o), \
             h{n} AS (SELECT h.s, h.k, h.v + tanh(z.z) AS v FROM h{b} h JOIN z{b} z ON h.s = z.s AND h.k = z.o)",
            n = b + 1
        ));
    }
    let loss = format!(
        "WITH {} SELECT SUM(v * v) / {n}.0 AS l FROM h{blocks}",
        ctes.join(", ")
    );
    let r = measure(&ctx, &loss, &[ColumnRef::new("w", "val")]).await;
    print("resnet", &format!("blocks={blocks} n={n} d={d}"), &r);
}

#[tokio::test]
#[ignore = "a measurement; run one DDX_PERF family per process, under a memory cap"]
async fn perf() {
    let family = std::env::var("DDX_PERF").unwrap_or_default();
    let sizes: Vec<usize> = std::env::var("DDX_PERF_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_default();
    match family.as_str() {
        "nn" => {
            for n in if sizes.is_empty() {
                vec![64, 256, 1024, 4096]
            } else {
                sizes
            } {
                nn(n).await;
            }
        }
        "matmul" => {
            for n in if sizes.is_empty() {
                vec![1000, 10000, 50000]
            } else {
                sizes
            } {
                contraction(n, 16, 8).await;
            }
        }
        "resnet" => {
            for n in if sizes.is_empty() {
                vec![1, 2, 3, 4, 5, 6]
            } else {
                sizes
            } {
                resnet(n, 256, 8).await;
            }
        }
        kind @ ("depth" | "fanin" | "layers" | "reuse") => {
            let default = match kind {
                "depth" => vec![5, 10, 20, 40, 80],
                "fanin" => vec![2, 4, 8, 16, 32, 64],
                "layers" => vec![2, 4, 8, 16],
                _ => vec![2, 3, 4, 5, 6, 7],
            };
            for n in if sizes.is_empty() { default } else { sizes } {
                chain(kind, n, 1000).await;
            }
        }
        _ => eprintln!("set DDX_PERF to nn, matmul, depth, fanin, layers or reuse"),
    }
}
