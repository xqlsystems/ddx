// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `grad` and `vjp` end to end on DataFusion: the gradient of a loss query,
//! checked entry by entry against finite differences of the same query.

mod common;

use common::ad::{check_gradients, rows, run, Table};
use common::substrait_of;
use datafusion::prelude::SessionContext;
use ddx_ad::{grad, vjp, AdError, ColumnRef};

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

fn b() -> Table {
    Table {
        name: "b",
        columns: vec![("o", "BIGINT"), ("val", "DOUBLE")],
        rows: vec![vec![0.0, 0.25], vec![1.0, -0.6]],
    }
}

fn x() -> Table {
    Table {
        name: "x",
        columns: vec![("i", "BIGINT"), ("v", "DOUBLE")],
        rows: vec![vec![0.0, 1.5], vec![1.0, -0.5], vec![2.0, 2.0]],
    }
}

fn ctx() -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    ctx
}

fn wrt(t: &str, c: &str) -> ColumnRef {
    ColumnRef::new(t, c)
}

#[tokio::test]
async fn an_elementwise_loss_over_one_table() {
    check_gradients(
        &ctx(),
        "SELECT SUM(tanh(val) * exp(val / 2.0)) AS loss FROM w",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_reduction_joined_to_data() {
    check_gradients(
        &ctx(),
        "SELECT SUM(w.val * w.val * x.v) AS loss \
         FROM w JOIN x ON w.i = x.i",
        &[w(), x()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_grouped_sum_feeds_a_second_aggregate() {
    // Two nodes: the per-i sums, then the loss over them.
    check_gradients(
        &ctx(),
        "WITH s AS (SELECT i, SUM(val) AS t FROM w GROUP BY i) \
         SELECT SUM(t * t * t) AS loss FROM s",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_bias_is_broadcast_by_a_join_and_summed_back() {
    // b.val is added to every w row with the same o, so its gradient sums
    // over i.
    check_gradients(
        &ctx(),
        "SELECT SUM(power(w.val + b.val, 2)) AS loss \
         FROM w JOIN b ON w.o = b.o",
        &[w(), b()],
        &[wrt("w", "val"), wrt("b", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_table_read_twice_gets_both_contributions() {
    // w joined to itself: each value is read once as `p` and once as `q`.
    check_gradients(
        &ctx(),
        "SELECT SUM(p.val * sin(q.val)) AS loss \
         FROM w p JOIN w q ON p.i = q.i AND p.o <> q.o",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_loss_computed_above_its_aggregates() {
    // SUM / COUNT: the division is elementwise in the root segment, and
    // COUNT has no gradient.
    check_gradients(
        &ctx(),
        "SELECT SUM(val * val) / COUNT(val) - 1.0 AS loss FROM w",
        &[w()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_loss_may_divide_by_a_one_row_constant() {
    // An ungrouped aggregate over data is one row, so the loss still is.
    check_gradients(
        &ctx(),
        "SELECT s.t / c.n AS loss \
         FROM (SELECT SUM(val * val) AS t FROM w) s CROSS JOIN (SELECT COUNT(*) AS n FROM x) c",
        &[w(), x()],
        &[wrt("w", "val")],
    )
    .await;
}

#[tokio::test]
async fn a_wrt_table_the_loss_does_not_depend_on_gets_zeros() {
    let grads = check_gradients(
        &ctx(),
        "SELECT SUM(w.val * w.val) AS loss FROM w JOIN b ON w.o = b.o",
        &[w(), b()],
        &[wrt("w", "val"), wrt("b", "val")],
    )
    .await;
    assert!(grads["b"].iter().all(|r| r[1] == 0.0), "{:?}", grads["b"]);
}

/// The error `grad` returns for `sql` with respect to `w.val`.
async fn refusal(sql: &str) -> AdError {
    let ctx = ctx();
    w().create(&ctx).await;
    x().create(&ctx).await;
    let plan = substrait_of(&ctx, sql, true).await;
    grad(&plan, &[wrt("w", "val")]).unwrap_err()
}

#[tokio::test]
async fn an_aggregate_with_no_rule_is_refused() {
    let err = refusal("SELECT STDDEV(val) AS loss FROM w").await;
    assert!(matches!(err, AdError::NotImplemented(_)), "{err}");
    assert!(
        err.to_string().contains("SUM, AVG, MAX, MIN and COUNT"),
        "{err}"
    );
}

#[tokio::test]
async fn grad_needs_a_loss() {
    // Two columns.
    let err = refusal("SELECT SUM(val) AS a, SUM(val * val) AS b FROM w").await;
    assert!(matches!(err, AdError::NotScalar(_)), "{err}");
    assert!(err.to_string().contains("[\"a\", \"b\"]"), "{err}");
    // A row per dim.
    let err = refusal("SELECT i, SUM(val * val) AS s FROM w GROUP BY i").await;
    assert!(matches!(err, AdError::NotScalar(_)), "{err}");
    let err = refusal("SELECT SUM(val * val) AS s FROM w GROUP BY i").await;
    assert!(matches!(err, AdError::NotScalar(_)), "{err}");
    // One row times a many-row constant table is many rows.
    let err = refusal(
        "WITH s AS (SELECT SUM(val * val) AS t FROM w) SELECT s.t * x.v AS l FROM s CROSS JOIN x",
    )
    .await;
    assert!(matches!(err, AdError::NotScalar(_)), "{err}");
}

#[tokio::test]
async fn vjp_pulls_a_cotangent_back() {
    // out(i) = Σ_o val(i, o)², and the cotangent c(i) on it: the pullback to
    // val(i, o) is c(i) · 2 · val(i, o).
    let ctx = ctx();
    w().create(&ctx).await;
    let sql = "SELECT i, SUM(val * val) AS s FROM w GROUP BY i";
    let plan = substrait_of(&ctx, sql, true).await;
    let program = vjp(&plan, &[wrt("w", "val")]).unwrap();
    assert_eq!(program.cotangent, vec!["i", "s"]);

    let cotangent_table: &'static str = Box::leak(program.cotangent_table.clone().into_boxed_str());
    Table {
        name: cotangent_table,
        columns: vec![("i", "BIGINT"), ("s", "DOUBLE")],
        rows: vec![vec![0.0, 2.0], vec![1.0, -1.0], vec![2.0, 0.5]],
    }
    .create(&ctx)
    .await;
    run(&ctx, &program).await;
    let got = rows(
        &ctx,
        &format!(
            "SELECT i, o, val FROM {} ORDER BY i, o",
            program.gradients[0].step
        ),
    )
    .await;
    let cot = [2.0, -1.0, 0.5];
    for (g, row) in got.iter().zip(&w().rows) {
        assert_eq!(g[2], cot[row[0] as usize] * 2.0 * row[2], "{g:?}");
    }
}

// ---------------------------------------------------------------------------
// Adversarial review (🤖😈). Each test below fails on this branch.

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

#[tokio::test]
async fn a_table_whose_dims_repeat_is_not_given_summed_gradients() {
    // relation.rs documents that a wrt table whose dim tuples repeat "gets
    // each shared tuple's rows' gradients summed". That is a silently wrong
    // gradient, which principle 5 forbids, and it compounds: the gradient
    // step re-reads the table's dims, so it has one row per *table* row, each
    // holding the group's sum, and the SGD join `w JOIN g ON w.i = g.i` then
    // multiplies rows (3 rows in, 5 rows out). The plan cannot show a key,
    // but the program can check one at run time (COUNT(*) > 1 per dim tuple
    // in the gradient step) and refuse. Either refuse, or be right.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE d (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (0, 2.0), (1, 3.0)",
    )
    .await;
    let plan = substrait_of(&ctx, "SELECT SUM(val * val) AS l FROM d", true).await;
    let Ok(program) = grad(&plan, &[ColumnRef::new("d", "val")]) else {
        return; // refused: fine
    };
    // Refused when run, by the program's check that dims identify rows: fine.
    if let Err(e) = common::ad::try_run(&ctx, &program).await {
        assert!(e.contains("share their dims"), "{e}");
        return;
    }
    let mut got = rows(
        &ctx,
        &format!("SELECT * FROM {}", program.gradients[0].step),
    )
    .await;
    got.sort_by(|a, b| a[1].partial_cmp(&b[1]).unwrap());
    assert_eq!(
        got,
        vec![vec![0.0, 2.0], vec![0.0, 4.0], vec![1.0, 6.0]],
        "d/dval Σ val² is 2·val per row; repeated dims summed the rows' gradients"
    );
}

#[tokio::test]
async fn a_null_rows_gradient_does_not_depend_on_how_the_loss_is_written() {
    // A NULL value's gradient is 1 under SUM(val), 0 under SUM(val * val)
    // (the NULL derivative is coalesced to 0 by the dense gradient step), and
    // 0.5 under AVG(val) once #76 lands. v1 pins NULL for a NULL row (#60) and
    // design.md §5 asks for NULL conventions to be pinned, not emergent.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE wn (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, NULL), (2, 3.0)",
    )
    .await;
    let mut at_null = Vec::new();
    for loss in [
        "SELECT SUM(val) AS l FROM wn",
        "SELECT SUM(val * val) AS l FROM wn",
    ] {
        let plan = substrait_of(&ctx, loss, true).await;
        let program = grad(&plan, &[ColumnRef::new("wn", "val")]).unwrap();
        run(&ctx, &program).await;
        let got = rows(
            &ctx,
            &format!("SELECT val FROM {} WHERE i = 1", program.gradients[0].step),
        )
        .await;
        at_null.push(got[0][0]);
    }
    let same = at_null[0] == at_null[1] || (at_null[0].is_nan() && at_null[1].is_nan());
    assert!(
        same,
        "the gradient at a NULL row is {} for SUM(val) and {} for SUM(val * val); \
         pin one convention (v1's is NULL)",
        at_null[0], at_null[1]
    );
}

#[tokio::test]
async fn tables_whose_names_join_alike_get_their_own_gradient_steps() {
    // gradient_name joins a table's name parts with `_`, so `a_b.c` and
    // `a.b_c` both become `__ddx_grad_a_b_c`: the second step replaces the
    // first, and both gradients read back as the second table's.
    let ctx = SessionContext::new();
    exec(&ctx, "CREATE SCHEMA a_b").await;
    exec(&ctx, "CREATE SCHEMA a").await;
    exec(
        &ctx,
        "CREATE TABLE a_b.c (i BIGINT, val DOUBLE) AS VALUES (0, 1.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE a.b_c (i BIGINT, val DOUBLE) AS VALUES (0, 10.0)",
    )
    .await;
    let plan = substrait_of(
        &ctx,
        "SELECT SUM(p.val * p.val) + SUM(q.val * q.val) AS l \
         FROM a_b.c p CROSS JOIN a.b_c q",
        true,
    )
    .await;
    let program = grad(
        &plan,
        &[
            ColumnRef::new("a_b.c", "val"),
            ColumnRef::new("a.b_c", "val"),
        ],
    )
    .unwrap();
    let steps: Vec<&str> = program.gradients.iter().map(|g| g.step.as_str()).collect();
    assert_ne!(steps[0], steps[1], "two wrt tables, one gradient step name");
}

#[tokio::test]
async fn a_null_rows_gradient_is_null_and_a_reached_rows_is_not() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE wn (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, NULL), (2, 3.0)",
    )
    .await;
    let plan = substrait_of(&ctx, "SELECT SUM(val * val) AS l FROM wn", true).await;
    let program = grad(&plan, &[ColumnRef::new("wn", "val")]).unwrap();
    run(&ctx, &program).await;
    let got = rows(
        &ctx,
        &format!(
            "SELECT i, val FROM {} ORDER BY i",
            program.gradients[0].step
        ),
    )
    .await;

    assert_eq!(got[0], vec![0.0, 2.0]);
    assert!(
        got[1][1].is_nan(),
        "the NULL row's gradient is NULL: {:?}",
        got[1]
    );
    assert_eq!(got[2], vec![2.0, 6.0]);
}

#[tokio::test]
async fn a_gradient_has_its_values_type() {
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE wr (i BIGINT, val REAL) AS VALUES (0, 1.5), (1, 2.5)",
    )
    .await;
    let plan = substrait_of(&ctx, "SELECT SUM(val * val) AS l FROM wr", true).await;
    let program = grad(&plan, &[ColumnRef::new("wr", "val")]).unwrap();
    run(&ctx, &program).await;
    let df = ctx
        .sql(&format!("SELECT val FROM {}", program.gradients[0].step))
        .await
        .unwrap();
    assert_eq!(
        df.schema().field(0).data_type(),
        &datafusion::arrow::datatypes::DataType::Float32
    );
}

#[tokio::test]
async fn a_row_an_aggregate_skips_as_null_sends_no_gradient() {
    // From the v2 soak (#89). SUM skips row 1, whose term p + q is NULL, so
    // the loss is p(0) + 5 and ∂/∂p(1) is 0, not the group's cotangent.
    // Likewise with the NULL in constant data: d - p.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE np (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE nq (i BIGINT, val DOUBLE) AS VALUES (0, 5.0), (1, NULL)",
    )
    .await;
    for (loss, want) in [
        (
            "SELECT SUM(np.val + nq.val) AS l FROM np JOIN nq ON np.i = nq.i",
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        ),
        (
            "SELECT SUM(nq.val - np.val) AS l FROM np JOIN nq ON np.i = nq.i",
            vec![vec![0.0, -1.0], vec![1.0, 0.0]],
        ),
    ] {
        let plan = substrait_of(&ctx, loss, true).await;
        let program = grad(&plan, &[ColumnRef::new("np", "val")]).unwrap();
        run(&ctx, &program).await;
        let got = rows(
            &ctx,
            &format!(
                "SELECT i, val FROM {} ORDER BY i",
                program.gradients[0].step
            ),
        )
        .await;
        assert_eq!(got, want, "{loss}");
    }
}

#[tokio::test]
async fn a_table_with_capitals_gets_a_step_its_engine_can_name() {
    // From the v2 soak (#92): a step named `…_grad_0_W` was folded to lower
    // case when registered, then not found under its own name.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE \"Wc\" (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    let plan = substrait_of(&ctx, "SELECT SUM(val * val) AS l FROM \"Wc\"", true).await;
    let program = grad(&plan, &[ColumnRef::new("Wc", "val")]).unwrap();
    let step = &program.gradients[0].step;
    assert_eq!(step, &step.to_ascii_lowercase());
    run(&ctx, &program).await;
    let got = rows(&ctx, &format!("SELECT i, val FROM {step} ORDER BY i")).await;
    assert_eq!(got, vec![vec![0.0, 2.0], vec![1.0, 4.0]]);
}

#[tokio::test]
async fn a_program_runs_without_the_simplifier() {
    // From the v2 soak (#90): DataFusion 54 runs coalesce only once its
    // simplifier has rewritten it, and the gradient step used it.
    use datafusion::execution::SessionStateBuilder;
    let ctx = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_optimizer_rules(vec![])
            .build(),
    );
    exec(
        &ctx,
        "CREATE TABLE ws (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, NULL), (2, 3.0)",
    )
    .await;
    let plan = substrait_of(&ctx, "SELECT SUM(val * val) AS l FROM ws", true).await;
    let program = grad(&plan, &[ColumnRef::new("ws", "val")]).unwrap();
    run(&ctx, &program).await;
    let got = rows(
        &ctx,
        &format!(
            "SELECT i, val FROM {} ORDER BY i",
            program.gradients[0].step
        ),
    )
    .await;
    assert_eq!(got[0], vec![0.0, 2.0]);
    assert!(got[1][1].is_nan());
    assert_eq!(got[2], vec![2.0, 6.0]);
}

#[tokio::test]
async fn vjp_sends_no_gradient_through_an_output_row_that_is_null() {
    // From the v2 soak (seed 200886). Row 1's value, NULL + SUM(val), is NULL
    // whatever the sum is, so its cotangent moves nothing: it must not
    // reach the SUM, as grad(SUM(s · c)), which skips the row, agrees. Row
    // 0 gets 3 directly and 3 more through the SUM.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE vp (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, NULL)",
    )
    .await;
    let plan = substrait_of(
        &ctx,
        "SELECT vp.i, vp.val + m.v AS s FROM vp CROSS JOIN (SELECT SUM(val) AS v FROM vp) m",
        true,
    )
    .await;
    let program = vjp(&plan, &[ColumnRef::new("vp", "val")]).unwrap();
    Table {
        name: Box::leak(program.cotangent_table.clone().into_boxed_str()),
        columns: vec![("i", "BIGINT"), ("s", "DOUBLE")],
        rows: vec![vec![0.0, 3.0], vec![1.0, 7.0]],
    }
    .create(&ctx)
    .await;
    run(&ctx, &program).await;
    let got = rows(
        &ctx,
        &format!(
            "SELECT i, val FROM {} ORDER BY i",
            program.gradients[0].step
        ),
    )
    .await;
    assert_eq!(got[0], vec![0.0, 6.0]);
    assert!(
        got[1][1].is_nan(),
        "a NULL value's gradient is NULL: {got:?}"
    );
}

#[tokio::test]
async fn a_row_one_aggregate_skips_still_gets_another_aggregates_gradient() {
    // A NULL seed is no contribution, and it must be skipped, not added:
    // SUM(p + q) skips row 1 (q NULL), SUM(p) does not, so ∂/∂p(1) is 1.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE fp (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE fq (i BIGINT, val DOUBLE) AS VALUES (0, 5.0), (1, NULL)",
    )
    .await;
    let plan = substrait_of(
        &ctx,
        "SELECT SUM(fp.val + fq.val) + SUM(fp.val) AS l FROM fp JOIN fq ON fp.i = fq.i",
        true,
    )
    .await;
    let program = grad(&plan, &[ColumnRef::new("fp", "val")]).unwrap();
    run(&ctx, &program).await;
    let got = rows(
        &ctx,
        &format!(
            "SELECT i, val FROM {} ORDER BY i",
            program.gradients[0].step
        ),
    )
    .await;
    assert_eq!(got, vec![vec![0.0, 2.0], vec![1.0, 1.0]]);
}

#[tokio::test]
async fn vjp_refuses_a_cotangent_whose_keys_repeat() {
    // From the v2 soak (#98): a wrt table whose dims repeat is refused, and
    // a cotangent whose keys repeat was joined as it is, doubling that row's
    // gradient without a word.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE kp (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    let plan = substrait_of(&ctx, "SELECT i, val * val AS s FROM kp", true).await;
    let program = vjp(&plan, &[ColumnRef::new("kp", "val")]).unwrap();
    Table {
        name: Box::leak(program.cotangent_table.clone().into_boxed_str()),
        columns: vec![("i", "BIGINT"), ("s", "DOUBLE")],
        rows: vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 1.0]],
    }
    .create(&ctx)
    .await;
    let err = common::ad::try_run(&ctx, &program).await.unwrap_err();
    assert!(err.contains("share their keys"), "{err}");
}

#[tokio::test]
async fn a_value_many_columns_read_has_a_small_backward_step() {
    // From the v2 soak (#98): a value read by N columns gets N cotangent
    // terms, which were folded by nesting the running sum; DataFusion names
    // a column by its expression, so the step's names tripled per reader
    // (117 MB at ten). They are one flat sum now.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE fn_ (i BIGINT, val DOUBLE) AS VALUES (0, 0.3), (1, 0.5)",
    )
    .await;
    let n = 12;
    let cols: Vec<String> = (1..=n)
        .map(|k| format!("sin(val * {k}.0) AS c{k}"))
        .collect();
    let sum: Vec<String> = (1..=n).map(|k| format!("c{k}")).collect();
    let loss = format!(
        "WITH r AS (SELECT {} FROM fn_) SELECT SUM({}) AS l FROM r",
        cols.join(", "),
        sum.join(" + ")
    );
    let plan = substrait_of(&ctx, &loss, true).await;
    let program = grad(&plan, &[ColumnRef::new("fn_", "val")]).unwrap();
    // Each step as the plan ad::run builds for it, run one at a time.
    let mut bytes = 0;
    for step in program.steps() {
        let lp = ddx_datafusion::ad::logical_plan(&ctx, &step.plan)
            .await
            .unwrap();
        bytes += lp.display_indent().to_string().len();
        ddx_datafusion::ad::run_step(&ctx, step).await.unwrap();
    }
    assert!(bytes < 200_000, "{bytes} bytes of plan for {n} readers");
    // And the gradient is right: Σ k·cos(k·val).
    let got = rows(
        &ctx,
        &format!(
            "SELECT i, val FROM {} ORDER BY i",
            program.gradients[0].step
        ),
    )
    .await;
    for (row, v) in got.iter().zip([0.3f64, 0.5]) {
        let want: f64 = (1..=n).map(|k| k as f64 * (k as f64 * v).cos()).sum();
        assert!((row[1] - want).abs() < 1e-9, "{row:?} vs {want}");
    }
}

#[tokio::test]
async fn grad_fits_a_worker_threads_stack() {
    // From the v2 soak (#98): a CTE read twice per layer, nine layers. The
    // backward step appended one projection per column, hundreds deep, and
    // cloning it overflowed a 2 MB stack (a tokio worker's), aborting the
    // process. Cotangents are projected in batches now.
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE sp (i BIGINT, val DOUBLE) AS VALUES (0, 0.3), (1, 0.5)",
    )
    .await;
    let mut ctes = vec!["c0 AS (SELECT i, val AS v FROM sp)".to_string()];
    for k in 1..9 {
        ctes.push(format!(
            "c{k} AS (SELECT a.i, a.v * b.v AS v FROM c{m} a JOIN c{m} b ON a.i = b.i)",
            m = k - 1
        ));
    }
    let loss = format!("WITH {} SELECT SUM(v) AS l FROM c8", ctes.join(", "));
    let plan = substrait_of(&ctx, &loss, true).await;
    let program = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || grad(&plan, &[ColumnRef::new("sp", "val")]).map(|p| p.backward_steps.len()))
        .unwrap()
        .join()
        .unwrap()
        .unwrap();
    assert!(program > 0);
}

#[tokio::test]
async fn a_fixed_namespace_makes_the_program_the_same_every_time() {
    // Composability review (#74): a fresh namespace per call made the same
    // plan give different programs, so emitted plans could not be
    // snapshot-tested or a program cached by its plan.
    use ddx_ad::{grad_with, Options};
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE nw (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    let plan = substrait_of(&ctx, "SELECT SUM(val * val) AS l FROM nw", true).await;
    let wrt = [ColumnRef::new("nw", "val")];
    let options = Options::new().namespace("__ddx_golden_");
    let bytes = |p: &ddx_ad::BackwardProgram| -> Vec<(String, Vec<u8>)> {
        p.steps()
            .map(|s| (s.name.clone(), s.plan_bytes()))
            .chain(p.checks.iter().map(|c| (c.message.clone(), c.plan_bytes())))
            .collect()
    };
    let (a, b) = (
        grad_with(&plan, &wrt, &options).unwrap(),
        grad_with(&plan, &wrt, &options).unwrap(),
    );
    assert_eq!(bytes(&a), bytes(&b));
    assert_eq!(a.value, "__ddx_golden_value");
    // Without one, two programs never share a table name.
    let (c, d) = (grad(&plan, &wrt).unwrap(), grad(&plan, &wrt).unwrap());
    assert_ne!(c.value, d.value);
    // A namespace outside ddx's reserved prefix could name a user's table.
    // `__ddx_a` would run into its step names (`__ddx_asaved_0`), and an
    // engine folds `__ddx_Foo_` to lower case (composability re-review, #74).
    for bad in ["value_", "__ddx_a b_", "__ddx_a", "__ddx_Foo_"] {
        let err = grad_with(&plan, &wrt, &Options::new().namespace(bad)).unwrap_err();
        assert!(matches!(err, AdError::InvalidOptions(_)), "{err}");
    }
}

#[tokio::test]
async fn a_host_can_hand_over_plans_as_bytes() {
    // Composability review (#74): most Substrait hosts hold plans as
    // protobuf bytes, not as this crate's types.
    use datafusion::prelude::SessionContext;
    let ctx = SessionContext::new();
    exec(
        &ctx,
        "CREATE TABLE bw (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    let plan = substrait_of(&ctx, "SELECT SUM(val * val) AS l FROM bw", true).await;
    let bytes = prost::Message::encode_to_vec(&plan);
    let program = grad(
        &ddx_ad::decode_plan(&bytes).unwrap(),
        &[ColumnRef::new("bw", "val")],
    )
    .unwrap();
    for step in program.steps() {
        assert_eq!(ddx_ad::decode_plan(&step.plan_bytes()).unwrap(), step.plan);
    }
    run(&ctx, &program).await;
    let got = rows(
        &ctx,
        &format!(
            "SELECT i, val FROM {} ORDER BY i",
            program.gradients[0].step
        ),
    )
    .await;
    assert_eq!(got, vec![vec![0.0, 2.0], vec![1.0, 4.0]]);
    let err = ddx_ad::decode_plan(b"not a plan").unwrap_err();
    assert!(matches!(err, AdError::InvalidPlan(_)), "{err}");
}

#[tokio::test]
async fn a_bare_wrt_name_that_matches_two_schemas_is_refused() {
    // Composability review (#74): which rows get a gradient must not be a
    // guess. `t` matches both s1.t and s2.t.
    let ctx = SessionContext::new();
    for sql in [
        "CREATE SCHEMA s1",
        "CREATE SCHEMA s2",
        "CREATE TABLE s1.t (i BIGINT, val DOUBLE) AS VALUES (0, 1.0)",
        "CREATE TABLE s2.t (i BIGINT, val DOUBLE) AS VALUES (0, 2.0)",
    ] {
        exec(&ctx, sql).await;
    }
    let plan = substrait_of(
        &ctx,
        "SELECT SUM(a.val * b.val) AS l FROM s1.t a JOIN s2.t b ON a.i = b.i",
        true,
    )
    .await;
    let err = grad(&plan, &[ColumnRef::new("t", "val")]).unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, AdError::UnknownWrt(_)), "{err}");
    assert!(msg.contains("s1.t") && msg.contains("s2.t"), "{msg}");
    // Qualified, each is fine.
    grad(&plan, &[ColumnRef::new("s1.t", "val")]).unwrap();
    // A schema the plan does not carry gets a hint: some producers drop it.
    let plan = substrait_of(&ctx, "SELECT SUM(val) AS l FROM s1.t", true).await;
    let mut unqualified = plan.clone();
    for_each_named_table(&mut unqualified, &mut |names| {
        let last = names.pop().unwrap();
        *names = vec![last];
    });
    let err = grad(&unqualified, &[ColumnRef::new("s1.t", "val")]).unwrap_err();
    assert!(err.to_string().contains("name it `t`"), "{err}");
}

/// Apply `f` to the names of every table `plan` reads.
fn for_each_named_table(
    plan: &mut ddx_ad::substrait::proto::Plan,
    f: &mut dyn FnMut(&mut Vec<String>),
) {
    use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
    use ddx_ad::substrait::proto::read_rel::ReadType;
    use ddx_ad::substrait::proto::rel::RelType;
    use ddx_ad::substrait::proto::Rel;
    fn walk(rel: &mut Rel, f: &mut dyn FnMut(&mut Vec<String>)) {
        match rel.rel_type.as_mut() {
            Some(RelType::Read(r)) => {
                if let Some(ReadType::NamedTable(t)) = r.read_type.as_mut() {
                    f(&mut t.names);
                }
            }
            Some(RelType::Aggregate(r)) => walk(r.input.as_mut().unwrap(), f),
            Some(RelType::Project(r)) => walk(r.input.as_mut().unwrap(), f),
            Some(RelType::Filter(r)) => walk(r.input.as_mut().unwrap(), f),
            _ => {}
        }
    }
    for r in &mut plan.relations {
        if let Some(PlanRelType::Root(root)) = r.rel_type.as_mut() {
            walk(root.input.as_mut().unwrap(), f);
        }
    }
}

#[tokio::test]
async fn the_public_api_differentiates_sql_and_reports_refusals() {
    use ddx_datafusion::ad;
    let ctx = ctx();
    w().create(&ctx).await;
    let program = ad::grad(
        &ctx,
        "SELECT SUM(val * val) AS loss FROM w",
        &[wrt("w", "val")],
    )
    .await
    .unwrap();
    ad::run(&ctx, &program).await.unwrap();
    let got = common::ad::rows(
        &ctx,
        &format!(
            "SELECT i, o, val FROM {} ORDER BY i, o",
            program.gradients[0].step
        ),
    )
    .await;
    for (g, row) in got.iter().zip(&w().rows) {
        assert_eq!(g[2], 2.0 * row[2]);
    }

    // A refusal is a DataFusionError::External boxing the AdError.
    let err = ad::grad(
        &ctx,
        "SELECT i, SUM(val) AS s FROM w GROUP BY i",
        &[wrt("w", "val")],
    )
    .await
    .unwrap_err();
    let datafusion::error::DataFusionError::External(boxed) = err else {
        panic!("expected External")
    };
    assert!(matches!(
        boxed.downcast_ref::<AdError>(),
        Some(AdError::NotScalar(_))
    ));
}
