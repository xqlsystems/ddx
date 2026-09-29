// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Bugs `ad_simulation.rs` found, each reduced to the smallest query that
//! shows it and pinned against a gradient worked by hand.
//!
//! Each was `#[ignore]`d as a known bug until its fix landed (CONTRIBUTING.md,
//! "a failing test first"). Every ddx bug here is now fixed, and runs as an
//! ordinary test; each entry below names the fix. The upstream DataFusion
//! bugs at the end stay ignored until an upgrade fixes them.
//!
//! - **NULL rows leak gradient.** SUM, AVG, MAX and MIN skip a row whose
//!   argument is NULL, so nothing in that row can move the loss. ddx still
//!   broadcasts the group's cotangent to the row and pushes it through the
//!   row's other inputs, where a partial that does not read the NULL (the 1 of
//!   `p + q`) is not NULL. The gradient is silently wrong at every parameter
//!   joined to a NULL, in a parameter table or in constant data. *Fixed in
//!   #74 and #76:* each reduce rule's seed is NULL where its argument is.
//! - **Two rank filters in one region collide.** Rebuilding the region keeps
//!   both window columns, and the optimizer has given them the same name, so
//!   DataFusion's consumer refuses the step. The program was accepted.
//!   *Fixed in #73:* a window column is renamed in place once computed.
//! - **MAX finds its row by float equality with a recomputation.** The
//!   region beneath a saved MAX is recomputed, constant subtrees included,
//!   and a grouped SUM over several partitions is not bit-reproducible, so
//!   about half the time no row equals the saved maximum and the gradient is
//!   silently 0. Found by the soak's big mode. MIN is built the same way. A
//!   rank filter or a top-k orders the recomputed values rather than
//!   comparing them with saved ones, so jitter can only move it at a
//!   near-tie. *Fixed in #76:* the extreme and the rows attaining it are
//!   windows over the recomputed rows themselves, never compared with the
//!   saved value.
//! - **A CASE over integer data, in an unoptimized plan.** `grad_plan`
//!   accepts any `LogicalPlan`, a DataFrame's included; a CASE choosing
//!   between integer columns on a varied condition is accepted, and its
//!   backward step then holds values of a type its schema does not declare.
//!   *Fixed in #79:* a step's table takes the schema of the plan that ran.

use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, ColumnRef};

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// `(key, gradient)` rows of `table`'s gradient of `loss`.
async fn grad(ctx: &SessionContext, loss: &str, table: &str) -> Vec<(i64, Option<f64>)> {
    let program = ad::grad(ctx, loss, &[ColumnRef::new(table, "val")])
        .await
        .unwrap();
    ad::run(ctx, &program).await.unwrap();
    let batches = ctx
        .sql(&format!(
            "SELECT * FROM {} ORDER BY 1",
            program.gradients[0].step
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    use datafusion::arrow::array::{Array, AsArray};
    use datafusion::arrow::datatypes::{Float64Type, Int64Type};
    let mut out = Vec::new();
    for b in &batches {
        let k = b.column(0).as_primitive::<Int64Type>();
        let v = b.column(1).as_primitive::<Float64Type>();
        for r in 0..b.num_rows() {
            out.push((k.value(r), (!v.is_null(r)).then(|| v.value(r))));
        }
    }
    out
}

#[tokio::test]
async fn a_row_an_aggregate_skips_as_null_sends_no_gradient() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE q (i BIGINT, val DOUBLE) AS VALUES (0, 5.0), (1, NULL)",
    )
    .await;
    // Row 1's term is p + NULL = NULL, which SUM skips: the loss is p(0) + 5,
    // so ∂loss/∂p is 1 at row 0 and 0 at row 1.
    let got = grad(
        &ctx,
        "SELECT SUM(p.val + q.val) AS loss FROM p JOIN q ON p.i = q.i",
        "p",
    )
    .await;
    assert_eq!(got, vec![(0, Some(1.0)), (1, Some(0.0))]);
}

#[tokio::test]
async fn a_rank_filter_over_a_rank_filter_runs() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 3.0), (2, 2.0)",
    )
    .await;
    // Top two by val, then the top one of those: row 1, whose val is summed.
    let got = grad(
        &ctx,
        "WITH a AS (SELECT i, v FROM (SELECT i, val AS v, ROW_NUMBER() OVER (ORDER BY val DESC, i) AS rk FROM p) WHERE rk <= 2), \
              b AS (SELECT i, v FROM (SELECT i, v, ROW_NUMBER() OVER (ORDER BY v DESC, i) AS rk FROM a) WHERE rk = 1) \
         SELECT SUM(v) AS loss FROM b",
        "p",
    )
    .await;
    assert_eq!(got, vec![(0, Some(0.0)), (1, Some(1.0)), (2, Some(0.0))]);
}

#[tokio::test]
async fn a_null_in_constant_data_sends_no_gradient_through_its_row() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE d (i BIGINT, val DOUBLE) AS VALUES (0, 3.0), (1, NULL)",
    )
    .await;
    // Row 1 is d - p = NULL and skipped: the loss is 3 - p(0), and
    // ∂loss/∂p(1) is 0.
    let got = grad(
        &ctx,
        "SELECT SUM(d.val - p.val) AS loss FROM p JOIN d ON p.i = d.i",
        "p",
    )
    .await;
    assert_eq!(got, vec![(0, Some(-1.0)), (1, Some(0.0))]);
}

#[tokio::test]
async fn an_unoptimized_case_over_integer_data_runs() {
    // grad_plan takes any LogicalPlan, a DataFrame's included. In this
    // unoptimized one a CASE picks between integer data on a condition that
    // depends on b.val: it has no derivative with respect to b (the branches
    // are constant), so the gradient is 0. ddx accepts it, then its backward
    // step fails "Mismatch between schema and batches": a column's declared
    // type is not the type of the values it holds.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE b (val DOUBLE, i BIGINT) AS VALUES (0.5, 0), (-0.25, 1)",
    )
    .await;
    exec(&ctx, "CREATE TABLE m (val DOUBLE) AS VALUES (0.6), (-0.5)").await;
    let sql = "SELECT SUM(CASE WHEN b.val > 0 THEN a.v ELSE 0.5 * a.v END) AS loss \
               FROM (SELECT CAST(val * 10 AS BIGINT) AS v FROM m) a CROSS JOIN b";
    let plan = ctx.sql(sql).await.unwrap().into_unoptimized_plan();
    let program = ad::grad_plan(&ctx, &plan, &[ColumnRef::new("b", "val")]).unwrap();
    ad::run(&ctx, &program).await.unwrap();
}

#[tokio::test]
async fn max_finds_its_row_when_the_recomputed_values_jitter() {
    // The MAX rule sends the cotangent to the rows whose recomputed value
    // equals the saved maximum. The region beneath it is recomputed, a
    // constant subtree included, and DataFusion's grouped SUM over several
    // partitions adds its partial sums in whatever order they arrive, so the
    // recomputed values can differ from the forward pass in the last bit.
    // Then no row attains the saved maximum and the whole gradient is a
    // silent 0. About half of these runs are.
    use datafusion::arrow::array::{Float64Array, Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("j", DataType::Int64, false),
        Field::new("val", DataType::Float64, false),
    ]));
    let rows: Vec<(i64, i64, f64)> = (0..400)
        .flat_map(|i| (0..4).map(move |j| (i, j, ((i * 131 + j * 17) as f64 * 0.731).sin())))
        .collect();
    // Constant data in seven partitions, as a real table has.
    let partitions: Vec<Vec<RecordBatch>> = rows
        .chunks(rows.len().div_ceil(7))
        .map(|c| {
            vec![RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(c.iter().map(|r| r.0).collect::<Vec<_>>())),
                    Arc::new(Int64Array::from(c.iter().map(|r| r.1).collect::<Vec<_>>())),
                    Arc::new(Float64Array::from(
                        c.iter().map(|r| r.2).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap()]
        })
        .collect();
    let loss = "WITH s AS (SELECT m.j, SUM(exp(m.val) * n.val) AS s \
                           FROM m JOIN m n ON m.i = n.i AND m.j = n.j GROUP BY m.j) \
                SELECT MAX(p.val * s.s) AS loss FROM p JOIN s ON p.j = s.j";
    for run in 0..40 {
        let ctx = SessionContext::new();
        ctx.register_table(
            "m",
            Arc::new(MemTable::try_new(schema.clone(), partitions.clone()).unwrap()),
        )
        .unwrap();
        exec(
            &ctx,
            "CREATE TABLE p (j BIGINT, val DOUBLE) AS VALUES (0, 0.5), (1, 2.0), (2, 1.0), (3, -1.0)",
        )
        .await;
        let program = ad::grad(&ctx, loss, &[ColumnRef::new("p", "val")])
            .await
            .unwrap();
        ad::run(&ctx, &program).await.unwrap();
        let batches = ctx
            .sql(&format!(
                "SELECT SUM(ABS(val)) FROM {}",
                program.gradients[0].step
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        use datafusion::arrow::array::AsArray;
        use datafusion::arrow::datatypes::Float64Type;
        let total = batches[0].column(0).as_primitive::<Float64Type>().value(0);
        assert!(
            total > 0.0,
            "run {run}: the MAX's gradient is 0 at every row"
        );
    }
}

// ---------------------------------------------------------------------------
// Upstream: DataFusion bugs the soak reached, pinned with no ddx involved so an
// upgrade shows at once whether they are fixed. Each is reached only with an
// optimizer rule set DataFusion does not ship by default.

/// A context whose logical optimizer runs only `rules`.
fn ctx_with_only(rules: &[&str]) -> SessionContext {
    use datafusion::execution::SessionStateBuilder;
    use datafusion::optimizer::Optimizer;
    let rules = Optimizer::new()
        .rules
        .into_iter()
        .filter(|r| rules.contains(&r.name()))
        .collect();
    SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_optimizer_rules(rules)
            .build(),
    )
}

#[tokio::test]
#[ignore = "upstream DataFusion 54: a sort beneath a limit is dropped under a join"]
async fn upstream_a_limit_keeps_its_sort_under_a_join() {
    // Without push_down_limit to fuse the limit into the sort, the physical
    // plan loses the sort once the projection above it drops the sort key,
    // and the limit keeps the first rows in table order. ddx's backward steps
    // recompute such a region under a join, so on a context configured like
    // this its gradient lands on the wrong rows.
    let ctx = ctx_with_only(&[]);
    exec(
        &ctx,
        "CREATE TABLE u (i BIGINT, val DOUBLE) AS VALUES (1, -0.34), (2, -0.57), (0, 0.71)",
    )
    .await;
    exec(&ctx, "CREATE TABLE one (c DOUBLE) AS VALUES (1.0)").await;
    let batches = ctx
        .sql("SELECT t.i FROM (SELECT i FROM u ORDER BY val DESC LIMIT 1) t CROSS JOIN one")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    use datafusion::arrow::array::AsArray;
    use datafusion::arrow::datatypes::Int64Type;
    let i = batches[0].column(0).as_primitive::<Int64Type>().value(0);
    assert_eq!(i, 0, "the row with the largest val is i = 0");
}

#[tokio::test]
#[ignore = "upstream DataFusion 54: a union of aggregates over windows cannot be interleaved"]
async fn upstream_a_union_of_aggregates_over_windows_plans() {
    // With one target partition, EnforceSorting fails its own assertion
    // ("Can not create InterleaveExec: new children can not be
    // interleaved") on a UNION ALL of two aggregates, each over a window
    // above a join. ddx's gradient step has this shape where two
    // contributions to one table meet (a self-join) below a MAX, MIN or AVG,
    // whose rule computes its statistics as windows.
    use datafusion::arrow::array::{Float64Array, Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionConfig;
    use std::sync::Arc;
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    // The table spread over several partitions, as a real one is.
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, true),
        Field::new("j", DataType::Int64, true),
        Field::new("v", DataType::Float64, true),
    ]));
    let part = |j: Vec<i64>, v: Vec<f64>| {
        vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(
                    j.iter().map(|x| x + 10).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(j)),
                Arc::new(Float64Array::from(v)),
            ],
        )
        .unwrap()]
    };
    let t = MemTable::try_new(
        schema.clone(),
        vec![
            part(vec![0, 1], vec![1.0, 2.0]),
            part(vec![0], vec![3.0]),
            part(vec![1], vec![4.0]),
        ],
    )
    .unwrap();
    ctx.register_table("t", Arc::new(t)).unwrap();
    let sql = "SELECT i, j, SUM(g) FROM ( \
                 SELECT a.i, a.j, SUM(a.c) AS g FROM (SELECT a.i, a.j, COUNT(a.v * b.v) \
                   OVER (PARTITION BY a.j) AS c FROM t a JOIN t b ON a.i = b.i AND a.j = b.j) a \
                 GROUP BY a.i, a.j \
                 UNION ALL \
                 SELECT a.i, a.j, SUM(a.c) AS g FROM (SELECT b.i, b.j, COUNT(a.v * b.v) \
                   OVER (PARTITION BY a.j) AS c FROM t a JOIN t b ON a.i = b.i AND a.j = b.j) a \
                 GROUP BY a.i, a.j) GROUP BY i, j";
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

#[tokio::test]
#[ignore = "upstream DataFusion 54: a grouped MAX skips NaN, a window or ungrouped MAX returns it"]
async fn upstream_max_treats_nan_alike_grouped_or_not() {
    // The same values, the same MAX: grouped, it skips the NaN and gives
    // 0.9; as a window over the same group, and ungrouped, it gives NaN. A
    // query's value can then depend on how the engine plans it (or on row
    // order, when partial aggregates meet). ddx's MAX rule leaves NaN out of
    // the rows it compares, so it agrees with any MAX that gave a number
    // (seed 600845), but the engine should agree with itself.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE t (g BIGINT, v DOUBLE) AS VALUES (0, 0.5), (0, 'NaN'::DOUBLE), (0, 0.9)",
    )
    .await;
    let one = |sql: &'static str| {
        let ctx = ctx.clone();
        async move {
            use datafusion::arrow::array::AsArray;
            use datafusion::arrow::datatypes::Float64Type;
            let b = ctx.sql(sql).await.unwrap().collect().await.unwrap();
            b[0].column(0).as_primitive::<Float64Type>().value(0)
        }
    };
    let grouped = one("SELECT MAX(v) FROM t GROUP BY g").await;
    let window = one("SELECT MAX(v) OVER (PARTITION BY g) FROM t").await;
    assert_eq!(
        grouped.is_nan(),
        window.is_nan(),
        "grouped {grouped}, window {window}"
    );
}
