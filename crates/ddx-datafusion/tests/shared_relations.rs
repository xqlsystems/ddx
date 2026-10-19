// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A plan that shares a subtree, writing it once in `plan.relations` and
//! reading it through a `ReferenceRel` wherever it occurs (as DuckDB's
//! producer shares a CTE read twice), gives the program its written-out
//! form gives, byte for byte: ddx lowers the shared relation once and every
//! reference reads it, where it used to copy the relation at each
//! reference. So a model whose CTEs read one another twice costs ddx its
//! number of CTEs, not its number of copies.

mod common;

#[path = "../examples/nn/model.rs"]
#[allow(dead_code)]
mod model;

use common::{inputs, inputs_mut, shared, shared_count};
use datafusion::prelude::SessionContext;
use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::{Plan, PlanRel, Rel};
use ddx_ad::{ColumnRef, Options};

async fn plan_of(ctx: &SessionContext, sql: &str) -> Plan {
    let lp = ctx.sql(sql).await.unwrap().into_optimized_plan().unwrap();
    *datafusion_substrait::logical_plan::producer::to_substrait_plan(&lp, &ctx.state()).unwrap()
}

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// `grad` and `vjp` of `sql`'s plan and of it shared give the same
/// program, byte for byte.
async fn assert_same_programs(ctx: &SessionContext, sql: &str, wrt: &[ColumnRef]) {
    let plan = plan_of(ctx, sql).await;
    let dag = shared(&plan);
    assert!(shared_count(&dag) > 0, "nothing to share in {sql}");
    let options = Options::new().namespace("__ddx_same_");
    let tree = ddx_ad::grad_with(&plan, wrt, &options).unwrap();
    let shared = ddx_ad::grad_with(&dag, wrt, &options).unwrap();
    assert_eq!(format!("{shared:?}"), format!("{tree:?}"), "grad of {sql}");
}

#[tokio::test]
async fn a_limit_over_a_shared_sort_keeps_the_sort_s_order() {
    // Two LIMITs over one ORDER BY: unoptimized, each a Fetch whose input
    // is the Sort, shared. The Fetch reads its ordering from the Sort it
    // reads through the reference, as from the Sort written out.
    let ctx = tables().await;
    let sql = "SELECT SUM(a.val * b.val) AS l \
               FROM (SELECT i, val FROM w ORDER BY i LIMIT 2) a \
               JOIN (SELECT i, val FROM w ORDER BY i LIMIT 3) b ON a.i = b.i";
    let lp = ctx.sql(sql).await.unwrap().into_unoptimized_plan();
    let plan = *datafusion_substrait::logical_plan::producer::to_substrait_plan(&lp, &ctx.state())
        .unwrap();
    let dag = shared(&plan);
    assert!(shared_count(&dag) > 0, "nothing shared");
    let wrt = [ColumnRef::new("w", "val")];
    let options = Options::new().namespace("__ddx_same_");
    let tree = ddx_ad::grad_with(&plan, &wrt, &options).unwrap();
    let shared = ddx_ad::grad_with(&dag, &wrt, &options).unwrap();
    assert_eq!(format!("{shared:?}"), format!("{tree:?}"));
}

async fn tables() -> SessionContext {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, -2.0), (2, 0.5)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE x (i BIGINT, j BIGINT, v DOUBLE) AS VALUES \
         (0, 0, 1.0), (0, 1, 2.0), (1, 0, -1.0), (2, 1, 3.0)",
    )
    .await;
    ctx
}

#[tokio::test]
async fn a_cte_read_twice_gives_the_same_program_shared() {
    let ctx = tables().await;
    assert_same_programs(
        &ctx,
        "WITH h AS (SELECT x.j, SUM(x.v * w.val) AS h FROM x JOIN w ON x.i = w.i GROUP BY x.j) \
         SELECT SUM(a.h * b.h) AS l FROM h a JOIN h b ON a.j = b.j",
        &[ColumnRef::new("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn layers_that_read_the_last_twice_give_the_same_program_shared() {
    let ctx = tables().await;
    assert_same_programs(&ctx, &chain(3), &[ColumnRef::new("w", "val")]).await;
}

#[tokio::test]
async fn a_window_and_a_constant_side_give_the_same_program_shared() {
    // A shared relation that reads no wrt table is copied into each step
    // that reads it, written out: an engine reads no ReferenceRel.
    let ctx = tables().await;
    assert_same_programs(
        &ctx,
        "WITH c AS (SELECT i, SUM(v) AS c FROM x GROUP BY i), \
              s AS (SELECT i, val * 2.0 AS s, MAX(val) OVER (PARTITION BY i % 2) AS m FROM w) \
         SELECT SUM(a.s * b.m + c1.c * a.s - c2.c) AS l \
         FROM s a JOIN s b ON a.i = b.i JOIN c c1 ON a.i = c1.i JOIN c c2 ON b.i = c2.i",
        &[ColumnRef::new("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn nn_py_s_loss_gives_the_same_program_shared() {
    let ctx = SessionContext::new();
    let mut rng = model::Rng::new(7);
    model::register_data(&ctx, 8, &mut rng).unwrap();
    model::register_model(&ctx, &mut rng).unwrap();
    assert_same_programs(
        &ctx,
        &model::loss_sql(),
        &[
            ColumnRef::new("weight", "val"),
            ColumnRef::new("bias", "val"),
        ],
    )
    .await;
}

#[tokio::test]
async fn a_shared_program_runs_and_its_gradient_is_the_trees() {
    let ctx = tables().await;
    let sql = chain(2);
    let wrt = [ColumnRef::new("w", "val")];
    let plan = plan_of(&ctx, &sql).await;
    let tree = ddx_ad::grad(&plan, &wrt).unwrap();
    let dag = ddx_ad::grad(&shared(&plan), &wrt).unwrap();
    let rows = |out: ddx_datafusion::ad::Outputs| async move {
        let sorted = out.gradients[0]
            .clone()
            .sort(vec![datafusion::prelude::col("i").sort(true, false)])
            .unwrap();
        let batches = sorted.collect().await.unwrap();
        datafusion::arrow::util::pretty::pretty_format_batches(&batches)
            .unwrap()
            .to_string()
    };
    let a = rows(ddx_datafusion::ad::run(&ctx, &tree, &[]).await.unwrap()).await;
    let b = rows(ddx_datafusion::ad::run(&ctx, &dag, &[]).await.unwrap()).await;
    assert_eq!(a, b);
}

/// `k` layers over `w`, each joining the last to itself: written out, the
/// first layer's aggregate occurs 2^k times.
fn chain(k: usize) -> String {
    let mut ctes = vec!["l0 AS (SELECT i, SUM(val) AS v FROM w GROUP BY i)".to_string()];
    for n in 1..=k {
        ctes.push(format!(
            "l{n} AS (SELECT a.i, SUM(a.v * b.v) AS v FROM l{p} a JOIN l{p} b ON a.i = b.i GROUP BY a.i)",
            p = n - 1
        ));
    }
    format!("WITH {} SELECT SUM(v) AS loss FROM l{k}", ctes.join(", "))
}

#[tokio::test]
async fn a_chain_too_deep_to_write_out_is_differentiated_shared() {
    // Forty layers, each reading the last twice: written out, 2^40 copies of
    // the first, which no plan could hold. Shared, forty relations. The
    // DAG is built from a two-layer plan: its second layer's relation,
    // which reads the first through references, is copied for each further
    // layer, reading the one before.
    let dag = deep_chain(40).await;
    // Lowering recurses at each join, as it always has, and each layer
    // nests one: forty of them need more than a test thread's 2 MB.
    let built = on_a_big_stack(move || {
        let started = std::time::Instant::now();
        let program = ddx_ad::grad(&dag, &[ColumnRef::new("w", "val")]).unwrap();
        (started.elapsed(), ddx_ad::Program::step_count(&program))
    });
    let (took, steps) = built;
    assert!(took < std::time::Duration::from_secs(20), "{took:?}");
    // A saved aggregate per layer, and a step to compute each, at least.
    assert!(steps > 40, "{steps} steps");
}

/// The one ordinal `rel` reads through references.
fn only_reference(rel: &Rel) -> i32 {
    let mut reads = Vec::new();
    references(rel, &mut reads);
    reads.sort();
    reads.dedup();
    assert_eq!(reads.len(), 1, "{reads:?}");
    reads[0]
}

fn references(rel: &Rel, out: &mut Vec<i32>) {
    match &rel.rel_type {
        Some(RelType::Reference(r)) => out.push(r.subtree_ordinal),
        Some(kind) => inputs(kind).into_iter().for_each(|c| references(c, out)),
        None => {}
    }
}

fn retarget(rel: &mut Rel, from: i32, to: i32) {
    match &mut rel.rel_type {
        Some(RelType::Reference(r)) if r.subtree_ordinal == from => r.subtree_ordinal = to,
        Some(RelType::Reference(_)) | None => {}
        Some(kind) => inputs_mut(kind)
            .into_iter()
            .for_each(|c| retarget(c, from, to)),
    }
}

/// `k` + 2 layers over `w`, each reading the last twice, as a DAG: built
/// from a two-layer plan, whose second layer's relation, reading the first
/// through references, is copied for each further layer, reading the one
/// before.
async fn deep_chain(k: usize) -> Plan {
    let ctx = tables().await;
    let mut dag = shared(&plan_of(&ctx, &chain(2)).await);
    let root_at = dag
        .relations
        .iter()
        .position(|r| matches!(r.rel_type, Some(PlanRelType::Root(_))))
        .unwrap();
    // The root reads the second layer, which reads the first.
    let Some(PlanRelType::Root(root)) = &dag.relations[root_at].rel_type else {
        unreachable!()
    };
    let template_at = only_reference(root.input.as_ref().unwrap());
    let Some(PlanRelType::Rel(template)) = dag.relations[template_at as usize].rel_type.clone()
    else {
        unreachable!()
    };
    let first = only_reference(&template);
    let mut last = template_at;
    for _ in 0..k {
        let mut layer = template.clone();
        retarget(&mut layer, first, last);
        dag.relations.push(PlanRel {
            rel_type: Some(PlanRelType::Rel(layer)),
        });
        last = dag.relations.len() as i32 - 1;
    }
    let Some(PlanRelType::Root(root)) = &mut dag.relations[root_at].rel_type else {
        unreachable!()
    };
    retarget(root.input.as_mut().unwrap(), template_at, last);
    dag
}

/// `f` on a thread with room for `k` nested joins: lowering recurses at
/// each join, as it always has.
fn on_a_big_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap()
}

#[tokio::test]
async fn forward_mode_differentiates_a_chain_too_deep_to_write_out() {
    // Each shared relation is one step, with its tangents, that every
    // reader reads: forty-two steps, not 2^40 copies.
    let dag = deep_chain(40).await;
    let (took, steps) = on_a_big_stack(move || {
        let started = std::time::Instant::now();
        let program = ddx_ad::jvp(&dag, &[ColumnRef::new("w", "val")]).unwrap();
        (started.elapsed(), program.steps.len())
    });
    assert!(took < std::time::Duration::from_secs(20), "{took:?}");
    assert!(steps > 40, "{steps} steps");
}

#[tokio::test]
async fn reverse_over_forward_of_a_chain_too_deep_to_write_out() {
    // vjp of the jvp program writes its output as one plan: each step once,
    // read through references, not written out at every read.
    let dag = deep_chain(40).await;
    let took = on_a_big_stack(move || {
        let wrt = [ColumnRef::new("w", "val")];
        let started = std::time::Instant::now();
        let jvp = ddx_ad::jvp(&dag, &wrt).unwrap();
        ddx_ad::vjp(&jvp, &wrt).unwrap();
        started.elapsed()
    });
    assert!(took < std::time::Duration::from_secs(20), "{took:?}");
}
