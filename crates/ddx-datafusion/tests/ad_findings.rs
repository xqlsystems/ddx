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
//!   The root is broader than MAX: constant subtrees are recomputed
//!   in each backward step unread and unchecked, so a ranking over constant
//!   data that does not break ties (ddx refuses one only over what a wrt
//!   table feeds) could keep a different row on the way back, and a CASE on
//!   it take a different branch.
//! - **A table with capitals has no gradient in SQL.** `ad::sql` reads a
//!   gradient step back under a quoted name DataFusion lowercased when it
//!   was registered. *Fixed in #74:* a step's name is lower case.
//! - **A CASE over integer data, in an unoptimized plan.** `grad`
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
async fn a_null_loss_sends_no_gradient() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE q (i BIGINT, val DOUBLE) AS VALUES (0, CAST(NULL AS DOUBLE))",
    )
    .await;
    // The loss is SUM(p) + NULL = NULL, which does not move with p: grad
    // seeded it with 1 regardless and sent 1 to each row of p, where vjp
    // seeds a NULL output row with NULL.
    let got = grad(
        &ctx,
        "SELECT s + m AS loss FROM (SELECT SUM(val) AS s FROM p) CROSS JOIN (SELECT MAX(val) AS m FROM q)",
        "p",
    )
    .await;
    assert_eq!(got, vec![(0, Some(0.0)), (1, Some(0.0))]);
}

#[tokio::test]
async fn an_unoptimized_case_over_integer_data_runs() {
    // grad takes any LogicalPlan, a DataFrame's included. In this
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
    let program = ad::grad(&ctx, &plan, &[ColumnRef::new("b", "val")])
        .await
        .unwrap();
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

#[tokio::test]
async fn grad_in_sql_of_a_table_with_capitals() {
    // A gradient step is named after its table (`…_grad_0_W`). DataFusion
    // folds the unquoted name it is registered under to lower case, and
    // ad::sql then reads it back quoted, case and all, and finds nothing.
    // Every table with a capital in its name, "Weights" from a Parquet file
    // for one, has no gradient in SQL.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE \"W\" (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    let df = ad::sql(
        &ctx,
        "WITH loss AS (SELECT SUM(val * val) AS l FROM \"W\") SELECT i, val FROM grad(loss, \"W\".val) ORDER BY i",
    )
    .await
    .unwrap();
    df.collect().await.unwrap();
}

#[tokio::test]
async fn a_power_under_one_at_zero_has_an_infinite_derivative_not_a_failed_query() {
    // Found by the soak after the fixes above (seed 811): the map rule's
    // partial of power(v, 0.5) was 0.5 * power(v, -0.5), and DataFusion
    // refuses power(0, c) for c < 0 ("zero raised to a negative power is
    // undefined"), so a program ddx accepted failed to run where v = 0.
    // *Fixed in ddx-core:* a negative power is written as a division, so the
    // partial there is 0.5 / 0 = inf, as it is.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 0.0), (1, 4.0)",
    )
    .await;
    let got = grad(&ctx, "SELECT SUM(power(val, 0.5)) AS loss FROM p", "p").await;
    assert_eq!(got, vec![(0, Some(f64::INFINITY)), (1, Some(0.25))]);
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
#[ignore = "upstream DataFusion 54 (#99): a sort beneath a limit is dropped under a join"]
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
#[ignore = "upstream DataFusion 54 (#100): a union of aggregates over windows cannot be interleaved"]
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
#[ignore = "upstream DataFusion 54 (#101): a grouped MAX skips NaN, a window or ungrouped MAX returns it"]
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

// ---------------------------------------------------------------------------
// Round 2: found on the fixed stack by the forward-mode oracle and by probes
// past the generator's reach. Each was ignored as a known bug until its fix:
// the MAX/MIN near-tie in #76; fan-in, the stack overflow and vjp cotangent
// keys in #74; the deep chain in #79 (ad::logical_plan names computed columns
// briefly); comments in grad(…) in #83; the simple CASE in #71.

/// The bytes of DataFusion's logical plans for `program`'s steps, as ad::run
/// builds them: what a step costs to plan and run, which ddx's own (small)
/// Substrait plans do not show. Runs the program.
async fn consumed_plan_bytes(ctx: &SessionContext, program: &ad::BackwardProgram) -> usize {
    let mut total = 0;
    for step in program.steps() {
        let lp = ad::logical_plan(ctx, &step.plan).await.unwrap();
        total += lp.display_indent().to_string().len();
        ad::run_step(ctx, step).await.unwrap();
    }
    total
}

#[tokio::test]
async fn max_at_a_near_tie_sends_its_cotangent_to_the_larger() {
    // MAX(a, b) at a < b is differentiable: its gradient is (0, 1), as
    // jax.grad(jnp.max) gives. The 8-ulp attainment window (added so a tie
    // that rounding breaks is shared the same way every run) also shares
    // between two exact values a few ulps apart, which do not tie: (0.5,
    // 0.5), not a subgradient at a point where the function is smooth. A
    // finite difference cannot see this: any step crosses the near-tie.
    let ctx = SessionContext::new();
    let b = f64::from_bits(1.0f64.to_bits() + 2);
    exec(
        &ctx,
        &format!(
            "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, CAST({b:e} AS DOUBLE))"
        ),
    )
    .await;
    assert!(b > 1.0);
    let got = grad(&ctx, "SELECT MAX(val) AS l FROM p", "p").await;
    assert_eq!(got, vec![(0, Some(0.0)), (1, Some(1.0))]);
}

#[tokio::test]
async fn a_value_many_columns_read_has_a_linear_backward_step() {
    // `val` read by N projected columns gets N cotangent terms, folded so a
    // NULL term is skipped: CASE WHEN acc IS NULL THEN t WHEN t IS NULL THEN
    // acc ELSE acc + t END. Each fold names the accumulator three times, and
    // DataFusion's Substrait consumer names a column by its expression, so
    // the names, and the plan, triple per reader: 97 KB at 4 columns, 117 MB
    // at 10, and 4 GB is not enough at 12, on a two-row table.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 0.3), (1, 0.5)",
    )
    .await;
    let n = 10;
    let cols: Vec<String> = (1..=n)
        .map(|k| format!("sin(val * {k}.0) AS c{k}"))
        .collect();
    let sum: Vec<String> = (1..=n).map(|k| format!("c{k}")).collect();
    let loss = format!(
        "WITH r AS (SELECT {} FROM p) SELECT SUM({}) AS l FROM r",
        cols.join(", "),
        sum.join(" + ")
    );
    let program = ad::grad(&ctx, &loss, &[ColumnRef::new("p", "val")])
        .await
        .unwrap();
    let bytes = consumed_plan_bytes(&ctx, &program).await;
    assert!(
        bytes < 1_000_000,
        "the backward steps plan to {bytes} bytes for {n} columns"
    );
}

#[tokio::test]
async fn a_deep_chain_of_maps_has_a_linear_backward_step() {
    // Each layer `sin(v) + 0.1 * v` reads v twice. ddx rebuilds the region
    // as appended, unnamed columns, and DataFusion names each by its
    // expression, so every layer doubles the names: 0.4 MB of plan at 8
    // layers, 60 MB at 14, and 20 layers exhaust 13 GB. The user's SQL
    // names each layer `v`, so the forward query never sees this.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 0.3), (1, 0.5)",
    )
    .await;
    let depth = 14;
    let mut ctes = vec!["c0 AS (SELECT i, val AS v FROM p)".to_string()];
    for k in 1..depth {
        ctes.push(format!(
            "c{k} AS (SELECT i, sin(v) + 0.1 * v AS v FROM c{})",
            k - 1
        ));
    }
    let loss = format!(
        "WITH {} SELECT SUM(v) AS l FROM c{}",
        ctes.join(", "),
        depth - 1
    );
    let program = ad::grad(&ctx, &loss, &[ColumnRef::new("p", "val")])
        .await
        .unwrap();
    let bytes = consumed_plan_bytes(&ctx, &program).await;
    assert!(
        bytes < 2_000_000,
        "the backward steps plan to {bytes} bytes for {depth} layers"
    );
}

/// A loss whose Substrait plan repeats a CTE read twice per layer.
async fn reused_cte_plan(depth: usize) -> ddx_ad::substrait::proto::Plan {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 0.3), (1, 0.5)",
    )
    .await;
    let mut ctes = vec!["c0 AS (SELECT i, val AS v FROM p)".to_string()];
    for k in 1..depth {
        ctes.push(format!(
            "c{k} AS (SELECT a.i, a.v * b.v AS v FROM c{m} a JOIN c{m} b ON a.i = b.i)",
            m = k - 1
        ));
    }
    let loss = format!(
        "WITH {} SELECT SUM(v) AS l FROM c{}",
        ctes.join(", "),
        depth - 1
    );
    let lp = ctx.sql(&loss).await.unwrap().into_optimized_plan().unwrap();
    *datafusion_substrait::logical_plan::producer::to_substrait_plan(&lp, &ctx.state()).unwrap()
}

#[test]
fn grad_does_not_overflow_a_worker_threads_stack() {
    // A tokio worker thread has a 2 MB stack. On a 66 KB plan (a CTE read
    // twice per layer, nine layers), ddx_ad::grad clones a rebuilt region
    // hundreds of relations deep, recursively (Transposer::region), and the
    // stack overflow aborts the whole process: not an error, not a panic
    // anything can catch. It runs in a child process so that abort fails
    // this test rather than the test binary.
    if std::env::var("DDX_FINDINGS_CHILD").is_ok() {
        let plan = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(reused_cte_plan(9));
        let t = std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(move || ddx_ad::grad(&plan, &[ColumnRef::new("p", "val")]).map(|_| ()))
            .unwrap();
        t.join().unwrap().unwrap();
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "grad_does_not_overflow_a_worker_threads_stack",
            "--nocapture",
        ])
        .env("DDX_FINDINGS_CHILD", "1")
        .status()
        .unwrap();
    assert!(status.success(), "ddx_ad::grad on a 2 MB stack: {status}");
}

#[tokio::test]
async fn grad_in_sql_skips_comments_inside_the_call() {
    // call_span finds grad(…)'s closing parenthesis by counting, skipping
    // quoted text but not comments: `/* ) */` ends the call early and the
    // statement no longer parses, and `/* ( */` never closes it, which ddx
    // reports as an internal error (a bug, by its own description).
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    for call in [
        "grad(loss /* ) */, p.val)",
        "grad(loss, p.val /* ( */)",
        "grad(loss, -- )\n p.val)",
    ] {
        let sql =
            format!("WITH loss AS (SELECT SUM(val * val) AS l FROM p) SELECT i, val FROM {call}");
        let df = ad::sql(&ctx, &sql)
            .await
            .unwrap_or_else(|e| panic!("{call:?}: {e}"));
        assert_eq!(
            df.collect()
                .await
                .unwrap()
                .iter()
                .map(|b| b.num_rows())
                .sum::<usize>(),
            2
        );
    }
}

#[tokio::test]
async fn vjp_refuses_a_cotangent_whose_keys_repeat() {
    // A wrt table whose dims repeat is refused by the program's checks; a
    // cotangent whose keys repeat is joined as it is, so the row's cotangent
    // counts twice and its gradient silently doubles.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    let program = ad::vjp(
        &ctx,
        "SELECT i, val * val AS s FROM p",
        &[ColumnRef::new("p", "val")],
    )
    .await
    .unwrap();
    exec(
        &ctx,
        &format!(
            "CREATE TABLE \"{}\" (i BIGINT, s DOUBLE) AS VALUES (0, 1.0), (0, 1.0), (1, 1.0)",
            program.inputs[0].name
        ),
    )
    .await;
    assert!(
        ad::run(&ctx, &program).await.is_err(),
        "a cotangent with two rows for i = 0 ran, and doubled that row's gradient"
    );
}

#[tokio::test]
async fn a_simple_case_is_differentiated() {
    // DataFusion writes `CASE i WHEN 0 THEN …` as a switch; ddx reports it as
    // an invalid Substrait plan (which it classes as a producer bug) rather
    // than reading or refusing it.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    let got = grad(
        &ctx,
        "SELECT SUM(CASE i WHEN 0 THEN val * val ELSE 3.0 * val END) AS l FROM p",
        "p",
    )
    .await;
    assert_eq!(got, vec![(0, Some(2.0)), (1, Some(3.0))]);
}

#[tokio::test]
async fn grad_in_sql_does_not_panic_on_two_calls_and_a_comment() {
    // Found by ad_sql_text.rs, 18 panics in 3000 valid spellings. The first
    // call's `/* ( */` keeps call_span counting past its real end, into the
    // second call, until a `)` in a later comment closes it; the second
    // call starts inside the first's span, and GradCalls::rewrite slices the
    // statement from a later byte to an earlier one: a panic in the
    // library, on valid SQL.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE q (i BIGINT, val DOUBLE) AS VALUES (0, 3.0), (1, 4.0)",
    )
    .await;
    let sql = "WITH loss AS (SELECT SUM(p.val * q.val) AS l FROM p JOIN q ON p.i = q.i) \
               SELECT a.i, a.val, b.val AS qv \
               FROM grad(loss, p.val /* ( */) a JOIN grad(loss, q.val) b ON a.i = b.i /* ) */";
    let df = ad::sql(&ctx, sql).await.unwrap();
    assert_eq!(
        df.collect()
            .await
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum::<usize>(),
        2
    );
}

// ---------------------------------------------------------------------------
// Round 3: found on the fixed stack by the soak. A near tie over constant
// table values, fixed in #76, and three DataFusion bugs, pinned as upstream
// (#101, #103, #104).

#[tokio::test]
async fn a_near_tie_over_constant_table_values_goes_to_the_larger() {
    // The fix to the 8-ulp window (#76) gave the tolerance only to values
    // that can jitter, but counted all constant data among them. A table
    // outside wrt is constant data to ddx, yet its values are exactly as
    // repeatable as a wrt table's: MAX(p.val * d.val) over products 2 ulps
    // apart is differentiable, with all the gradient on the larger (jax.grad
    // gives (0, d(1))). Now only constant data an aggregate or window
    // computes can jitter (#76).
    let ctx = SessionContext::new();
    let b = f64::from_bits(1.0f64.to_bits() + 2);
    exec(
        &ctx,
        "CREATE TABLE p (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 1.0)",
    )
    .await;
    exec(
        &ctx,
        &format!(
            "CREATE TABLE d (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, CAST({b:e} AS DOUBLE))"
        ),
    )
    .await;
    let got = grad(
        &ctx,
        "SELECT MAX(p.val * d.val) AS l FROM p JOIN d ON p.i = d.i",
        "p",
    )
    .await;
    assert_eq!(got, vec![(0, Some(0.0)), (1, Some(b))]);
}

#[tokio::test]
#[ignore = "upstream DataFusion 54 (#103, fixed in 55): a filter above an anti-join is pushed into its right side"]
async fn upstream_a_filter_above_an_anti_join_keeps_its_rows_out() {
    // From the round-three soak (seed 3000134, an optimizer variant). With
    // push_down_filter off, `j <> 2` stays above the anti-join NOT IN makes,
    // and the gradient reached b(2), a row the query excludes; the loss was
    // unchanged, the gradient silently wrong. ddx's recomputed region is
    // right: DataFusion 54's physical filter pushdown moves the filter into
    // the anti-join's right input, which plain SQL shows without ddx (#103).
    // Fixed in DataFusion 55; ddx stays on 54 with datafusion-python.
    let mut got = Vec::new();
    for drop in [None, Some("push_down_filter")] {
        let ctx = SessionContext::new();
        if let Some(rule) = drop {
            assert!(ctx.remove_optimizer_rule(rule));
        }
        exec(
            &ctx,
            "CREATE TABLE b (j BIGINT, val DOUBLE) AS VALUES (0, -0.1), (1, 0.8), (2, 0.5), (90, 0.1)",
        )
        .await;
        exec(
            &ctx,
            "CREATE TABLE y (j BIGINT, val DOUBLE) AS VALUES (0, 0.4), (2, 0.9), (1, -0.2)",
        )
        .await;
        got.push(
            grad(
                &ctx,
                "WITH r AS (SELECT j, val AS v FROM b WHERE j NOT IN (SELECT j FROM y WHERE val > 0.35)), \
                      f AS (SELECT * FROM r WHERE j <> 2), \
                      m AS (SELECT MAX(v) AS m FROM f), \
                      e AS (SELECT f.j, exp(f.v - m.m) AS e FROM f CROSS JOIN m), \
                      s AS (SELECT SUM(e) AS s FROM e) \
                 SELECT SUM(e.e / s.s * e.e / s.s) AS loss FROM e CROSS JOIN s",
                "b",
            )
            .await,
        );
    }
    assert_eq!(
        got[1], got[0],
        "with push_down_filter off, the gradient changed"
    );
}

#[tokio::test]
#[ignore = "upstream DataFusion 54, 55 (#101): a grouped MAX skips NaN for some groups and returns it for others"]
async fn upstream_a_grouped_max_treats_nan_the_same_in_every_group() {
    // Found by the round-three soak's big mode (seed 5008778), where the same
    // loss came out finite on some runs and NaN on others. DataFusion merges
    // a group's partial MAXes in the order the partitions deliver them, and
    // whether a NaN survives depends on that order: in one query, group 0
    // skips its NaN (5.0) and group 1 returns it. ddx's gradients over such
    // data inherit the ambiguity. No ddx involved; still so in DataFusion 55
    // (#101).
    use datafusion::arrow::array::{AsArray, Float64Array, Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Float64Type, Schema};
    use datafusion::datasource::MemTable;
    use std::sync::Arc;
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Int64, false),
        Field::new("v", DataType::Float64, true),
    ]));
    let parts: Vec<Vec<(i64, f64)>> = vec![
        vec![(0, 1.0), (1, 2.0)],
        vec![(0, f64::NAN), (1, 3.0)],
        vec![(0, 5.0), (1, f64::NAN)],
        vec![(0, 0.5)],
    ];
    let batches = parts
        .iter()
        .map(|p| {
            vec![RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(p.iter().map(|r| r.0).collect::<Vec<_>>())),
                    Arc::new(Float64Array::from(
                        p.iter().map(|r| r.1).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap()]
        })
        .collect();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(MemTable::try_new(schema, batches).unwrap()))
        .unwrap();
    let b = ctx
        .sql("SELECT g, MAX(v) AS m FROM t GROUP BY g ORDER BY g")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let m = b[0].column(1).as_primitive::<Float64Type>();
    assert_eq!(
        m.value(0).is_nan(),
        m.value(1).is_nan(),
        "group 0's MAX is {} and group 1's is {}: each holds a NaN",
        m.value(0),
        m.value(1)
    );
}

#[tokio::test]
#[ignore = "upstream DataFusion 54, 55 (#104): NOT IN over a NULL becomes a plain anti-join through Substrait"]
async fn upstream_not_in_over_a_null_keeps_no_rows_through_substrait() {
    // From the round-four soak (seed 4600388, NULL data): the program's value
    // step disagreed with the loss. `s NOT IN (0, NULL)` is never true, so no
    // row is kept, as DataFusion computes directly; its own Substrait round
    // trip writes a plain anti-join, which keeps s = 1 and s = 2. No ddx
    // involved.
    use datafusion_substrait::logical_plan::consumer::from_substrait_plan;
    use datafusion_substrait::logical_plan::producer::to_substrait_plan;
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE ny (s BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0), (2, 4.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE nx (s BIGINT, val DOUBLE) AS VALUES (0, 0.5), (NULL, 0.9), (1, -0.2)",
    )
    .await;
    let q = "SELECT SUM(val) AS l FROM ny WHERE s NOT IN (SELECT s FROM nx WHERE val > 0.05)";
    let sum = |b: Vec<datafusion::arrow::record_batch::RecordBatch>| {
        use datafusion::arrow::array::{Array, AsArray};
        let c = b[0]
            .column(0)
            .as_primitive::<datafusion::arrow::datatypes::Float64Type>()
            .clone();
        (!c.is_null(0)).then(|| c.value(0))
    };
    let direct = sum(ctx.sql(q).await.unwrap().collect().await.unwrap());
    let lp = ctx.sql(q).await.unwrap().into_optimized_plan().unwrap();
    let plan = to_substrait_plan(&lp, &ctx.state()).unwrap();
    let back = from_substrait_plan(&ctx.state(), &plan).await.unwrap();
    let round_trip = sum(ctx
        .execute_logical_plan(back)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap());
    assert_eq!(direct, None);
    assert_eq!(
        round_trip, direct,
        "the round trip kept rows a NULL excludes"
    );
}

// ---------------------------------------------------------------------------
// Found by review: an ordering by an expression counted as a key.

async fn lp() -> SessionContext {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE lp (i BIGINT, val DOUBLE) AS VALUES (0, 0.3), (1, 0.5), (2, 0.7), (3, 0.1)",
    )
    .await;
    ctx
}

#[tokio::test]
async fn a_limit_ordered_by_an_expression_of_its_dims_is_refused() {
    // `i % 2` ties every even i: the recomputed LIMIT 1 could keep another
    // row than the forward pass kept, and send it the gradient. Its ORDER BY
    // named the dim inside an expression, which counted as ordering by it.
    let ctx = lp().await;
    let e = ad::grad(
        &ctx,
        "SELECT SUM(v) AS l FROM (SELECT val AS v FROM lp ORDER BY i % 2 LIMIT 1)",
        &[ColumnRef::new("lp", "val")],
    )
    .await
    .expect_err("ties are not broken");
    assert!(e.to_string().contains("ORDER BY"), "{e}");
    // Ordered by the dim itself, it is total.
    ad::grad(
        &ctx,
        "SELECT SUM(v) AS l FROM (SELECT val AS v FROM lp ORDER BY i LIMIT 1)",
        &[ColumnRef::new("lp", "val")],
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn a_ranking_ordered_by_an_expression_of_its_dims_is_refused() {
    let ctx = lp().await;
    let e = ad::grad(
        &ctx,
        "SELECT SUM(val) AS l FROM (SELECT val, ROW_NUMBER() OVER (ORDER BY i % 2) AS r FROM lp) \
         WHERE r = 1",
        &[ColumnRef::new("lp", "val")],
    )
    .await
    .expect_err("ties are not broken");
    assert!(e.to_string().contains("break ties"), "{e}");
    ad::grad(
        &ctx,
        "SELECT SUM(val) AS l FROM (SELECT val, ROW_NUMBER() OVER (ORDER BY i % 2, i) AS r \
         FROM lp) WHERE r = 1",
        &[ColumnRef::new("lp", "val")],
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn a_max_of_a_grouped_max_is_exact_at_a_near_tie() {
    // Found by the jvp fuzzer (seed 12301800): the 8-ulp tie window, meant
    // for values a SUM or AVG computes, took a saved grouped MAX as one, and
    // shared the gradient with a row 5 ulps below the maximum. A MAX of
    // table values is exact; jax.grad gives (½, ½, 0).
    let ctx = SessionContext::new();
    let hi = 1.0000000000000009_f64;
    let lo = 0.9999999999999997_f64;
    exec(
        &ctx,
        &format!(
            "CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, {hi:?}), (1, {hi:?}), (2, {lo:?})"
        ),
    )
    .await;
    let got = grad(
        &ctx,
        "WITH r1 AS (SELECT i, MAX(val) AS v FROM w GROUP BY i) SELECT MAX(v) AS loss FROM r1",
        "w",
    )
    .await;
    assert_eq!(got, vec![(0, Some(0.5)), (1, Some(0.5)), (2, Some(0.0))]);
}
