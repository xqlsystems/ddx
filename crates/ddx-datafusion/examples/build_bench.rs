// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! What ddx-ad itself costs to build a program, in time and memory, for
//! comparing a refactor's before and after.
//!
//! Each case's query is planned by DataFusion once, up front; what is timed
//! is only ddx-ad: `grad`, `vjp` and `jvp` of that Substrait plan, and `jvp`
//! of the `grad` program (forward over reverse, a Hessian-vector product).
//! Memory is counted by a global allocator: the bytes allocated and the
//! peak bytes live during one build, which are deterministic where a
//! process's RSS is not. The last rows time `ddx_ad::rewrite` end to end
//! (CTE sharing and view expansion), DataFusion's planning and running
//! included, so they are noisier.
//!
//! ```text
//! systemd-run --user --scope -p MemoryMax=8G \
//!   nix develop -c cargo run --release -p ddx-datafusion --example build_bench
//! ```
//!
//! `DDX_BENCH=depth,attn` runs only the cases whose names start so.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

use datafusion::prelude::SessionContext;
use datafusion::sql::sqlparser::dialect::GenericDialect;
use ddx_ad::substrait::proto::Plan;
use ddx_ad::ColumnRef;

/// The system allocator, counting bytes allocated and the peak live.
struct Counting;
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            ALLOCATED.fetch_add(layout.size(), Relaxed);
            let live = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(live, Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, layout, new) };
        if !q.is_null() {
            ALLOCATED.fetch_add(new, Relaxed);
            let live = if new >= layout.size() {
                LIVE.fetch_add(new - layout.size(), Relaxed) + new - layout.size()
            } else {
                LIVE.fetch_sub(layout.size() - new, Relaxed) - (layout.size() - new)
            };
            PEAK.fetch_max(live, Relaxed);
        }
        q
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// One build's bytes allocated and peak bytes live above where it started.
fn memory_of<T>(f: impl FnOnce() -> T) -> (usize, usize, T) {
    let (a0, l0) = (ALLOCATED.load(Relaxed), LIVE.load(Relaxed));
    PEAK.store(l0, Relaxed);
    let out = f();
    (ALLOCATED.load(Relaxed) - a0, PEAK.load(Relaxed) - l0, out)
}

/// The median and least of `f`'s timings: at least 5 runs, then more
/// until a second has passed or 200 runs.
fn timings(mut f: impl FnMut()) -> (Duration, Duration) {
    let mut ts = Vec::new();
    let start = Instant::now();
    while ts.len() < 5 || (start.elapsed() < Duration::from_secs(1) && ts.len() < 200) {
        let t = Instant::now();
        f();
        ts.push(t.elapsed());
    }
    ts.sort();
    (ts[ts.len() / 2], ts[0])
}

fn row(case: &str, what: &str, f: impl Fn()) {
    let (allocated, peak, ()) = memory_of(&f);
    let (median, least) = timings(&f);
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    println!(
        "{case:<22} {what:<8} median {:>9.3} ms | least {:>9.3} ms | allocated {:>12} B | peak {:>11} B",
        ms(median),
        ms(least),
        allocated,
        peak
    );
}

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

async fn plan(ctx: &SessionContext, sql: &str) -> Plan {
    let lp = ctx.sql(sql).await.unwrap().into_optimized_plan().unwrap();
    *datafusion_substrait::logical_plan::producer::to_substrait_plan(&lp, &ctx.state()).unwrap()
}

/// `grad`, `vjp`, `jvp` and forward-over-reverse of the query `loss`.
fn plan_rows(case: &str, p: &Plan, wrt: &[ColumnRef]) {
    row(case, "grad", || {
        ddx_ad::grad(p, wrt).unwrap();
    });
    row(case, "vjp", || {
        ddx_ad::vjp(p, wrt).unwrap();
    });
    match ddx_ad::jvp(p, wrt) {
        Ok(_) => row(case, "jvp", || {
            ddx_ad::jvp(p, wrt).unwrap();
        }),
        Err(e) => println!("{case:<22} jvp      refused: {e}"),
    }
    let g = ddx_ad::grad(p, wrt).unwrap();
    match ddx_ad::jvp(&g, wrt) {
        Ok(_) => row(case, "hvp", || {
            ddx_ad::jvp(&ddx_ad::grad(p, wrt).unwrap(), wrt).unwrap();
        }),
        Err(e) => println!("{case:<22} hvp      refused: {e}"),
    }
}

async fn table_p(ctx: &SessionContext) {
    exec(
        ctx,
        "CREATE TABLE p AS SELECT CAST(r AS BIGINT) AS i, 0.001 * CAST(r AS DOUBLE) AS val \
         FROM (SELECT unnest(range(0, 100)) AS r)",
    )
    .await;
}

/// The loss SQL of each case over `p(i, val)`.
fn chain(kind: &str, n: usize) -> String {
    let mut ctes = vec!["c0 AS (SELECT i, val AS v FROM p)".to_string()];
    for k in 1..n {
        let m = k - 1;
        ctes.push(match kind {
            "depth" => format!("c{k} AS (SELECT i, sin(v) + 0.1 * v AS v FROM c{m})"),
            "layers" => format!(
                "c{k} AS (SELECT i % {d} AS i, SUM(tanh(v)) AS v FROM c{m} GROUP BY i % {d})",
                d = (100 >> k.min(6)).max(1)
            ),
            _ => {
                format!("c{k} AS (SELECT a.i, a.v * b.v AS v FROM c{m} a JOIN c{m} b ON a.i = b.i)")
            }
        });
    }
    format!(
        "WITH {} SELECT SUM(v) AS l FROM c{}",
        ctes.join(", "),
        n - 1
    )
}

fn wanted(case: &str) -> bool {
    match std::env::var("DDX_BENCH") {
        Ok(v) if !v.is_empty() => v.split(',').any(|w| case.starts_with(w)),
        _ => true,
    }
}

fn main() {
    // Lowering recurses at each join, and the rewrite cases nest twelve
    // layers of them: as sql_shared_ctes's deep tests do, run with more
    // stack than a main thread's 8 MB.
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(bench())
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn bench() {
    let wrt_p = [ColumnRef::new("p", "val")];
    for (case, kind, n) in [
        ("depth n=80", "depth", 80),
        ("layers n=16", "layers", 16),
        ("reuse n=7", "reuse", 7),
    ] {
        if !wanted(case) {
            continue;
        }
        let ctx = SessionContext::new();
        table_p(&ctx).await;
        plan_rows(case, &plan(&ctx, &chain(kind, n)).await, &wrt_p);
    }

    if wanted("fanin") {
        let ctx = SessionContext::new();
        table_p(&ctx).await;
        let cols: Vec<String> = (1..=64)
            .map(|k| format!("sin(val * {k}.0) AS c{k}"))
            .collect();
        let sum: Vec<String> = (1..=64).map(|k| format!("c{k}")).collect();
        let loss = format!(
            "WITH r AS (SELECT {} FROM p) SELECT SUM({}) AS l FROM r",
            cols.join(", "),
            sum.join(" + ")
        );
        plan_rows("fanin n=64", &plan(&ctx, &loss).await, &wrt_p);
    }

    if wanted("resnet") {
        let ctx = SessionContext::new();
        let d = 4;
        exec(
            &ctx,
            &format!(
            "CREATE TABLE x AS SELECT CAST(r / {d} AS BIGINT) AS s, CAST(r % {d} AS BIGINT) AS k, \
             sin(CAST(r AS DOUBLE)) AS val FROM (SELECT unnest(range(0, {})) AS r)", 8 * d),
        )
        .await;
        exec(&ctx, &format!(
            "CREATE TABLE w AS SELECT CAST(r / {} AS BIGINT) AS b, CAST(r / {d} % {d} AS BIGINT) AS k, \
             CAST(r % {d} AS BIGINT) AS o, 0.1 * cos(CAST(r AS DOUBLE)) AS val \
             FROM (SELECT unnest(range(0, {})) AS r)", d * d, 6 * d * d)).await;
        let mut ctes = vec!["h0 AS (SELECT s, k, val AS v FROM x)".to_string()];
        for b in 0..6 {
            ctes.push(format!(
                "z{b} AS (SELECT h.s, w.o, SUM(h.v * w.val) AS z FROM h{b} h JOIN w ON h.k = w.k AND w.b = {b} \
                 GROUP BY h.s, w.o), \
                 h{n} AS (SELECT h.s, h.k, h.v + tanh(z.z) AS v FROM h{b} h JOIN z{b} z ON h.s = z.s AND h.k = z.o)",
                n = b + 1
            ));
        }
        let loss = format!("WITH {} SELECT SUM(v * v) AS l FROM h6", ctes.join(", "));
        plan_rows(
            "resnet blocks=6",
            &plan(&ctx, &loss).await,
            &[ColumnRef::new("w", "val")],
        );
    }

    if wanted("attn") {
        let ctx = SessionContext::new();
        let d = 4;
        exec(
            &ctx,
            &format!(
            "CREATE TABLE x AS SELECT CAST(r / {d} AS BIGINT) AS t, CAST(r % {d} AS BIGINT) AS k, \
             sin(CAST(r AS DOUBLE)) AS val FROM (SELECT unnest(range(0, {})) AS r)", 8 * d),
        )
        .await;
        for (name, phase) in [("wq", 0.1), ("wk", 0.2), ("wv", 0.3)] {
            exec(
                &ctx,
                &format!(
                    "CREATE TABLE {name} AS SELECT CAST(r / {d} AS BIGINT) AS k, \
                 CAST(r % {d} AS BIGINT) AS j, 0.1 * cos(CAST(r AS DOUBLE) + {phase}) AS val \
                 FROM (SELECT unnest(range(0, {})) AS r)",
                    d * d
                ),
            )
            .await;
        }
        let proj = |w: &str| {
            format!(
            "SELECT x.t, {w}.j, SUM(x.val * {w}.val) AS v FROM x JOIN {w} ON x.k = {w}.k GROUP BY x.t, {w}.j")
        };
        let loss = format!(
            "WITH q AS ({}), k AS ({}), vv AS ({}), \
             s AS (SELECT q.t AS t, k.t AS u, SUM(q.v * k.v) / sqrt({d}.0) AS v \
                   FROM q JOIN k ON q.j = k.j GROUP BY q.t, k.t), \
             m AS (SELECT t, MAX(v) AS m FROM s GROUP BY t), \
             e AS (SELECT s.t, s.u, exp(s.v - m.m) AS e FROM s JOIN m ON s.t = m.t), \
             z AS (SELECT t, SUM(e) AS z FROM e GROUP BY t), \
             o AS (SELECT e.t, vv.j, SUM(e.e / z.z * vv.v) AS v \
                   FROM e JOIN z ON e.t = z.t JOIN vv ON e.u = vv.t GROUP BY e.t, vv.j) \
             SELECT SUM(v * v) AS l FROM o",
            proj("wq"),
            proj("wk"),
            proj("wv")
        );
        let wrt = [
            ColumnRef::new("wq", "val"),
            ColumnRef::new("wk", "val"),
            ColumnRef::new("wv", "val"),
        ];
        plan_rows("attn", &plan(&ctx, &loss).await, &wrt);
    }

    // ddx_ad::rewrite end to end: an objective whose layers read each other
    // twice, written out as CTEs and as a view (#168, #169). A fresh
    // `Programs` each time, so nothing is kept between runs.
    for (case, as_view) in [("rewrite ctes n=12", false), ("rewrite view n=12", true)] {
        if !wanted(case) {
            continue;
        }
        let ctx = SessionContext::new();
        table_p(&ctx).await;
        // As sql_shared_ctes's chain: each layer a grouped sum over a join
        // of the last with itself, 2^12 copies of the first written out.
        let n = 12;
        let mut ctes = vec!["l0 AS (SELECT i, SUM(val) AS v FROM p GROUP BY i)".to_string()];
        for k in 1..=n {
            ctes.push(format!(
                "l{k} AS (SELECT a.i, SUM(a.v * b.v + 0.5 * a.v) AS v \
                 FROM l{m} a JOIN l{m} b ON a.i = b.i GROUP BY a.i)",
                m = k - 1
            ));
        }
        let statement = if as_view {
            exec(
                &ctx,
                &format!(
                    "CREATE VIEW model AS WITH {} SELECT i, v FROM l{n}",
                    ctes.join(", ")
                ),
            )
            .await;
            "WITH loss AS (SELECT SUM(v) AS l FROM model) SELECT i, grad(loss, val) FROM p"
                .to_string()
        } else {
            format!(
                "WITH {}, loss AS (SELECT SUM(v) AS l FROM l{n}) SELECT i, grad(loss, val) FROM p",
                ctes.join(", ")
            )
        };
        let mut engine = ddx_datafusion::engine::DataFusion(ctx.clone());
        let mut ts = Vec::new();
        let start = Instant::now();
        while ts.len() < 5 || (start.elapsed() < Duration::from_secs(2) && ts.len() < 50) {
            let programs = ddx_ad::Programs::new();
            let t = Instant::now();
            let out = ddx_ad::rewrite(&mut engine, &programs, &[&statement], &GenericDialect {})
                .await
                .map_err(|e| e.to_string())
                .unwrap();
            ts.push(t.elapsed());
            out.release(&mut engine, &programs).await.unwrap();
        }
        ts.sort();
        println!(
            "{case:<22} rewrite  median {:>9.3} ms | least {:>9.3} ms | (DataFusion included)",
            ts[ts.len() / 2].as_secs_f64() * 1e3,
            ts[0].as_secs_f64() * 1e3
        );
    }
}
