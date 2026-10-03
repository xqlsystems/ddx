// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The plans `ddx_ad::emit` writes, consumed and run by DataFusion.
//!
//! Each case builds a plan with the emitter and compares its result with the
//! SQL query it stands for.

use std::sync::Arc;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use datafusion_substrait::logical_plan::consumer::from_substrait_plan;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;
use ddx_ad::emit::{
    aggregate, bind_reads, filter, join, plan, project, read_step, read_table, select,
    unbound_reads, union_all,
};
use ddx_ad::expr::{call, field, lit_f64};
use ddx_ad::substrait::proto::join_rel::JoinType;
use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::{NamedStruct, Plan};
use ddx_ad::{Extensions, Functions};

async fn ctx() -> SessionContext {
    let ctx = SessionContext::new();
    for sql in [
        "CREATE TABLE a (i BIGINT, j BIGINT, val DOUBLE) AS VALUES \
         (0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0), (NULL, 0, 10.0)",
        "CREATE TABLE b (i BIGINT, w DOUBLE) AS VALUES (0, 0.5), (NULL, 2.0)",
    ] {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }
    ctx
}

/// The Substrait schema DataFusion gives the table `name`: the base schema of
/// the read in the plan of `SELECT * FROM name`.
async fn schema_of(ctx: &SessionContext, name: &str) -> NamedStruct {
    let lp = ctx.table(name).await.unwrap().into_unoptimized_plan();
    let p = to_substrait_plan(&lp, &ctx.state()).unwrap();
    let Some(PlanRelType::Root(root)) = &p.relations[0].rel_type else {
        panic!("no root")
    };
    fn find(rel: &ddx_ad::substrait::proto::Rel) -> NamedStruct {
        match rel.rel_type.as_ref().unwrap() {
            RelType::Read(r) => r.base_schema.clone().unwrap(),
            RelType::Project(p) => find(p.input.as_ref().unwrap()),
            other => panic!("unexpected {other:?}"),
        }
    }
    find(root.input.as_ref().unwrap())
}

async fn run_plan(ctx: &SessionContext, p: &Plan) -> String {
    let lp = from_substrait_plan(&ctx.state(), p).await.unwrap();
    let batches = ctx
        .execute_logical_plan(lp)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

async fn run_sql(ctx: &SessionContext, sql: &str) -> String {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

#[tokio::test]
async fn project_and_grouped_aggregate() {
    let ctx = ctx().await;
    let mut ext = Extensions::new(&Functions::default());
    let (mul, sum) = (ext.anchor("multiply"), ext.anchor("sum"));
    // SELECT i, SUM(val * 2.0) FROM a GROUP BY i
    let rel = aggregate(
        project(
            read_table(vec!["a".into()], schema_of(&ctx, "a").await),
            vec![call(mul, vec![field(2), lit_f64(2.0)])],
        ),
        vec![field(0)],
        vec![(sum, vec![field(3)])],
    );
    let p = plan(rel, vec!["i".into(), "s".into()], &ext);
    // Sorted through a second query so the comparison ignores row order.
    let got = run_plan(&ctx, &p).await;
    let want = run_sql(&ctx, "SELECT i, SUM(val * 2.0) AS s FROM a GROUP BY i").await;
    assert_eq!(sorted(&got), sorted(&want));
}

#[tokio::test]
async fn null_safe_join_filter_select_and_union() {
    let ctx = ctx().await;
    let mut ext = Extensions::new(&Functions::default());
    let same = ext.anchor("is_not_distinct_from");
    let gt = ext.anchor("gt");
    let a = || async { read_table(vec!["a".into()], schema_of(&ctx, "a").await) };
    // SELECT a.i, a.val * b.w FROM a JOIN b ON a.i IS NOT DISTINCT FROM b.i
    // WHERE a.val > 1.5, then UNION ALL of it with itself.
    let joined = join(
        a().await,
        read_table(vec!["b".into()], schema_of(&ctx, "b").await),
        call(same, vec![field(0), field(3)]),
        JoinType::Inner,
    );
    let mul = ext.anchor("multiply");
    let one = select(
        project(
            filter(joined, call(gt, vec![field(2), lit_f64(1.5)])),
            vec![call(mul, vec![field(2), field(4)])],
        ),
        vec![0, 5],
    );
    let p = plan(
        union_all(vec![one.clone(), one]),
        vec!["i".into(), "v".into()],
        &ext,
    );
    let got = run_plan(&ctx, &p).await;
    let q = "SELECT a.i, a.val * b.w AS v FROM a JOIN b ON a.i IS NOT DISTINCT FROM b.i \
             WHERE a.val > 1.5";
    let want = run_sql(&ctx, &format!("{q} UNION ALL {q}")).await;
    assert_eq!(sorted(&got), sorted(&want));
    // The NULL key matched: without IS NOT DISTINCT FROM that row would vanish.
    assert!(got.contains("20.0"), "{got}");
}

#[tokio::test]
async fn an_unbound_read_runs_once_bound_to_the_materialized_step() {
    let ctx = ctx().await;
    // An "earlier step", materialized under ddx's naming.
    let df = ctx
        .sql("SELECT i AS c0, SUM(val) AS c1 FROM a GROUP BY i")
        .await
        .unwrap();
    let schema = df.schema().inner().clone();
    let batches = df.collect().await.unwrap();
    let table = MemTable::try_new(schema, vec![batches]).unwrap();
    ctx.register_table("__ddx_fwd_0", Arc::new(table)).unwrap();

    let mut ext = Extensions::new(&Functions::default());
    let mul = ext.anchor("multiply");
    let rel = project(
        read_step("__ddx_fwd_0", vec!["c1".into(), "c0".into()]),
        vec![call(mul, vec![field(0), lit_f64(3.0)])],
    );
    let mut p = plan(rel, vec!["s".into(), "i".into(), "t".into()], &ext);
    assert_eq!(unbound_reads(&p), vec!["__ddx_fwd_0"]);

    let schema = schema_of(&ctx, "__ddx_fwd_0").await;
    bind_reads(&mut p, &mut |name| {
        (name == "__ddx_fwd_0").then(|| schema.clone())
    })
    .unwrap();
    let got = run_plan(&ctx, &p).await;
    let want = run_sql(
        &ctx,
        "SELECT c1 AS s, c0 AS i, c1 * 3.0 AS t FROM __ddx_fwd_0",
    )
    .await;
    assert_eq!(sorted(&got), sorted(&want));
}

/// The data rows of a pretty-printed table, sorted.
fn sorted(table: &str) -> Vec<String> {
    let mut rows: Vec<String> = table
        .lines()
        .filter(|l| l.starts_with('|'))
        .skip(1)
        .map(str::to_string)
        .collect();
    rows.sort();
    rows
}
