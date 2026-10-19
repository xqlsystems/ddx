// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad` in SQL plans each CTE its objective reads more than once on its
//! own (`ddx_ad::rewrite`), so a model whose layers read the last one twice
//! costs its number of CTEs, not the copies an engine's inlining makes.

mod common;

use datafusion::error::Result;
use datafusion::prelude::SessionContext;
use datafusion::sql::sqlparser::dialect::GenericDialect;
use ddx_ad::substrait::proto::{NamedStruct, Plan};
use ddx_ad::{Backend, Bound, ColumnRef, Programs};
use ddx_datafusion::engine::DataFusion;

/// DataFusion, recording the tables a rewrite makes and the queries it
/// plans.
struct Recording {
    inner: DataFusion,
    made: Vec<String>,
    dropped: Vec<String>,
    planned: Vec<String>,
}

impl Backend for Recording {
    type Error = datafusion::error::DataFusionError;

    async fn table_schema(&mut self, name: &str) -> Result<NamedStruct> {
        self.inner.table_schema(name).await
    }

    async fn returns_rows(&mut self, plans: &[Bound]) -> Result<Vec<bool>> {
        self.inner.returns_rows(plans).await
    }

    async fn materialize(&mut self, name: &str, plan: &Bound) -> Result<()> {
        self.made.push(name.to_string());
        self.inner.materialize(name, plan).await
    }

    async fn drop_table(&mut self, name: &str) -> Result<()> {
        self.dropped.push(name.to_string());
        self.inner.drop_table(name).await
    }

    async fn plan_sql(&mut self, sql: &str) -> Result<Plan> {
        self.planned.push(sql.to_string());
        self.inner.plan_sql(sql).await
    }

    async fn plan_key(&mut self, sql: &str) -> Result<Option<String>> {
        self.inner.plan_key(sql).await
    }
}

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

async fn w() -> SessionContext {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 0.5), (1, -0.25), (2, 0.75)",
    )
    .await;
    ctx
}

/// `k` layers over `w`, each joining the last to itself, as an objective:
/// written out, the first layer 2^k times.
fn chain(k: usize) -> String {
    let mut ctes = vec!["l0 AS (SELECT i, SUM(val) AS v FROM w GROUP BY i)".to_string()];
    for n in 1..=k {
        ctes.push(format!(
            "l{n} AS (SELECT a.i, SUM(a.v * b.v + 0.5 * a.v) AS v \
             FROM l{p} a JOIN l{p} b ON a.i = b.i GROUP BY a.i)",
            p = n - 1
        ));
    }
    ctes.push(format!("loss AS (SELECT SUM(v) AS l FROM l{k})"));
    format!("WITH {}", ctes.join(", "))
}

/// The rewritten statement's `(i, gradient)` rows, and the backend's log.
async fn grad_of(ctx: &SessionContext, with: &str) -> (Vec<(i64, f64)>, Recording) {
    let statement = format!("{with} SELECT i, grad(loss, val) AS g FROM w ORDER BY i");
    let mut backend = Recording {
        inner: DataFusion(ctx.clone()),
        made: Vec::new(),
        dropped: Vec::new(),
        planned: Vec::new(),
    };
    let programs = Programs::new();
    let out = ddx_ad::rewrite(&mut backend, &programs, &[&statement], &GenericDialect {})
        .await
        .unwrap();
    let rows = common::ad::rows(ctx, &out.statements[0]).await;
    out.release(&mut backend, &programs).await.unwrap();
    let rows = rows.into_iter().map(|r| (r[0] as i64, r[1])).collect();
    (rows, backend)
}

/// The same gradient from the program API, of the engine's plan of the
/// objective written out.
async fn written_out(ctx: &SessionContext, with: &str) -> Vec<(i64, f64)> {
    let plan = common::substrait_of(ctx, &format!("{with} SELECT l FROM loss"), true).await;
    let program = ddx_ad::grad(&plan, &[ColumnRef::new("w", "val")]).unwrap();
    let out = ddx_datafusion::ad::run(ctx, &program, &[]).await.unwrap();
    let batches = out.gradients[0].clone().collect().await.unwrap();
    let mut rows: Vec<(i64, f64)> = Vec::new();
    for b in &batches {
        use datafusion::arrow::array::AsArray;
        use datafusion::arrow::datatypes::{Float64Type, Int64Type};
        let (i, g) = (
            b.column(0).as_primitive::<Int64Type>(),
            b.column(1).as_primitive::<Float64Type>(),
        );
        rows.extend((0..b.num_rows()).map(|r| (i.value(r), g.value(r))));
    }
    rows.sort_by_key(|r| r.0);
    rows
}

#[tokio::test]
async fn a_cte_read_twice_is_planned_once_and_the_gradient_is_the_same() {
    let ctx = w().await;
    let with = chain(3);
    let (got, log) = grad_of(&ctx, &with).await;
    let want = written_out(&ctx, &with).await;
    assert_eq!(got.len(), want.len());
    for ((gi, g), (wi, w)) in got.iter().zip(&want) {
        assert_eq!(gi, wi);
        assert!(
            (g - w).abs() <= 1e-9 * w.abs().max(1.0),
            "{got:?} vs {want:?}"
        );
    }
    // l0, l1 and l2 are read twice each; l3 and loss once.
    let placeholders: Vec<&String> = log.made.iter().filter(|n| n.contains("cte_")).collect();
    assert_eq!(placeholders.len(), 3, "{:?}", log.made);
    for p in &placeholders {
        assert!(log.dropped.contains(p), "{p} was left behind");
    }
    // No plan the engine optimized holds the objective written out: each
    // planned query reads at most one layer twice.
    for sql in &log.planned {
        assert!(sql.matches("JOIN").count() <= 2, "{sql}");
    }
}

#[test]
fn a_chain_too_deep_to_write_out_is_differentiated_from_sql() {
    // Thirty layers: written out, 2^30 copies of the first, which neither
    // DataFusion nor ddx could plan. Each is planned once. Lowering recurses
    // at each join, as it always has, and each layer nests one, so this
    // runs with more stack than a test thread's 2 MB.
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    let ctx = w().await;
                    let with = chain(30);
                    let started = std::time::Instant::now();
                    let (got, log) = grad_of(&ctx, &with).await;
                    assert!(
                        started.elapsed() < std::time::Duration::from_secs(60),
                        "{:?}",
                        started.elapsed()
                    );
                    assert_eq!(got.len(), 3);
                    assert_eq!(
                        log.made.iter().filter(|n| n.contains("cte_")).count(),
                        30,
                        "{:?}",
                        log.made
                    );
                })
        })
        .unwrap()
        .join()
        .unwrap();
}

#[tokio::test]
async fn a_cte_read_once_is_left_to_the_engine() {
    let ctx = w().await;
    let with = "WITH s AS (SELECT i, val * val AS v FROM w), loss AS (SELECT SUM(v) AS l FROM s)";
    let (got, log) = grad_of(&ctx, with).await;
    assert_eq!(got, vec![(0, 1.0), (1, -0.5), (2, 1.5)]);
    assert!(
        !log.made.iter().any(|n| n.contains("cte_")),
        "{:?}",
        log.made
    );
}

/// The prefix, `__ddx_{hex}_{hex}_`, of the program tables `sql` reads.
fn program_prefix(sql: &str) -> String {
    let hex = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit());
    let mut found: Vec<String> = sql
        .match_indices("__ddx_")
        .filter_map(|(at, _)| {
            let mut parts = sql[at + "__ddx_".len()..].splitn(3, '_');
            let (a, b) = (parts.next()?, parts.next()?);
            parts.next()?;
            (hex(a) && hex(b)).then(|| format!("__ddx_{a}_{b}_"))
        })
        .collect();
    found.sort();
    found.dedup();
    assert_eq!(found.len(), 1, "one program's tables in {sql}");
    found.pop().unwrap()
}

#[tokio::test]
async fn a_split_objective_s_program_is_kept_between_calls() {
    // Each call names its placeholders afresh, so overlapping calls make
    // tables of their own; the program is kept under a key that names them
    // all alike, so the next call runs it again.
    let ctx = w().await;
    let statement = format!("{} SELECT i, grad(loss, val) AS g FROM w", chain(3));
    let programs = Programs::new();
    let mut prefixes = Vec::new();
    for _ in 0..2 {
        let mut backend = DataFusion(ctx.clone());
        let out = ddx_ad::rewrite(&mut backend, &programs, &[&statement], &GenericDialect {})
            .await
            .unwrap();
        prefixes.push(program_prefix(&out.statements[0]));
        out.release(&mut backend, &programs).await.unwrap();
    }
    assert_eq!(
        prefixes[0], prefixes[1],
        "the second call built its own program"
    );
}
