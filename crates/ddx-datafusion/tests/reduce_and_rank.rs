// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The reduce rules beyond SUM (MAX, MIN, AVG), rank-select, and
//! `ddx_stop_gradient`: nn.py's softmax cross-entropy and max-pooling.

mod common;

use common::ad::{check_gradients, rows, run, Table};
use common::substrait_of;
use datafusion::prelude::SessionContext;
use ddx_ad::{grad, AdError, ColumnRef};

fn ctx() -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    ctx
}

/// Logits `z(sample, out)` and one-hot labels `y(sample, out)`.
fn logits() -> Vec<Table> {
    let z = [
        [0.2, -1.3, 0.8],
        [1.1, 0.4, -0.2],
        [-0.5, 0.9, 0.3],
        [0.0, 0.1, 2.0],
    ];
    let mut zr = Vec::new();
    let mut yr = Vec::new();
    for (s, row) in z.iter().enumerate() {
        for (o, v) in row.iter().enumerate() {
            zr.push(vec![s as f64, o as f64, *v]);
            yr.push(vec![
                s as f64,
                o as f64,
                if (s + 1) % 3 == o { 1.0 } else { 0.0 },
            ]);
        }
    }
    vec![
        Table {
            name: "z",
            columns: vec![("sample", "BIGINT"), ("out", "BIGINT"), ("val", "DOUBLE")],
            rows: zr,
        },
        Table {
            name: "y",
            columns: vec![("sample", "BIGINT"), ("out", "BIGINT"), ("val", "DOUBLE")],
            rows: yr,
        },
    ]
}

/// nn.py's loss, mean softmax cross-entropy, as nn.py writes it: the max is a
/// numerical shift, `shift` says how it is subtracted.
fn cross_entropy(shift: &str) -> String {
    format!(
        "WITH m AS (SELECT sample, MAX(val) AS m FROM z GROUP BY sample), \
              e AS (SELECT z.sample, z.out, exp(z.val - {shift}) AS e \
                    FROM z JOIN m ON z.sample = m.sample), \
              s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample) \
         SELECT -AVG(ln(e.e / s.s) * y.val) * 3.0 AS loss \
         FROM e JOIN s ON e.sample = s.sample \
                JOIN y ON y.sample = e.sample AND y.out = e.out"
    )
}

/// `(softmax(z) - onehot) / N`, the gradient nn.py derived by hand.
fn softmax_minus_onehot(z: &Table) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for s in 0..4 {
        let zs: Vec<f64> = (0..3)
            .map(|o| {
                z.rows
                    .iter()
                    .find(|r| r[0] == s as f64 && r[1] == o as f64)
                    .unwrap()[2]
            })
            .collect();
        let total: f64 = zs.iter().map(|v| v.exp()).sum();
        for (o, zo) in zs.iter().enumerate() {
            let onehot = if (s + 1) % 3 == o { 1.0 } else { 0.0 };
            out.push(vec![s as f64, o as f64, (zo.exp() / total - onehot) / 4.0]);
        }
    }
    out
}

fn assert_close(got: &[Vec<f64>], want: &[Vec<f64>]) {
    for w in want {
        let g = got.iter().find(|g| g[..2] == w[..2]).unwrap();
        assert!(
            (g[2] - w[2]).abs() < 1e-12,
            "{:?}: {} vs {}",
            &w[..2],
            g[2],
            w[2]
        );
    }
}

#[tokio::test]
async fn softmax_cross_entropy_as_nn_py_writes_it() {
    // A plain MAX: its rule routes gradient to the max, and the shift cancels,
    // so the result is the same as with the shift stopped.
    let tables = logits();
    let grads = check_gradients(
        &ctx(),
        &cross_entropy("m.m"),
        &tables,
        &[ColumnRef::new("z", "val")],
    )
    .await;
    assert_close(&grads["z"], &softmax_minus_onehot(&tables[0]));
}

#[tokio::test]
async fn softmax_cross_entropy_with_the_shift_stopped() {
    let tables = logits();
    let grads = check_gradients(
        &ctx(),
        &cross_entropy("ddx_stop_gradient(m.m)"),
        &tables,
        &[ColumnRef::new("z", "val")],
    )
    .await;
    assert_close(&grads["z"], &softmax_minus_onehot(&tables[0]));
}

#[tokio::test]
async fn a_stop_gradient_changes_the_gradient_it_stops() {
    // loss = Σ val · stop(val): the gradient is val, not the 2·val of the
    // unstopped function. Finite differences would say 2·val, so this checks
    // against the value directly.
    let ctx = ctx();
    let tables = logits();
    tables[0].create(&ctx).await;
    let sql = "SELECT SUM(val * ddx_stop_gradient(val)) AS loss FROM z";
    let plan = substrait_of(&ctx, sql, true).await;
    let program = grad(&plan, &[ColumnRef::new("z", "val")]).unwrap();
    run(&ctx, &program).await;
    let got = rows(
        &ctx,
        &format!("SELECT sample, out, val FROM {}", program.gradients[0].step),
    )
    .await;
    for r in &tables[0].rows {
        let g = got.iter().find(|x| x[0] == r[0] && x[1] == r[1]).unwrap()[2];
        assert_eq!(g, r[2]);
    }
}

#[tokio::test]
async fn avg_is_a_sum_over_a_count() {
    check_gradients(
        &ctx(),
        "WITH m AS (SELECT sample, AVG(val * val) AS a FROM z GROUP BY sample) \
         SELECT SUM(a * a) AS loss FROM m",
        &logits(),
        &[ColumnRef::new("z", "val")],
    )
    .await;
}

/// `x(g, item, val)`: four groups of five, and a weight per group.
fn pool_tables(tie: bool) -> Vec<Table> {
    let mut xr = Vec::new();
    for g in 0..4 {
        for item in 0..5 {
            let v = ((g * 5 + item) * 37 % 23) as f64 / 10.0 - 1.0;
            xr.push(vec![g as f64, item as f64, v]);
        }
    }
    if tie {
        // Items 1 and 3 of group 0 tie for its max.
        xr[1][2] = 5.0;
        xr[3][2] = 5.0;
    }
    vec![
        Table {
            name: "x",
            columns: vec![("g", "BIGINT"), ("item", "BIGINT"), ("val", "DOUBLE")],
            rows: xr,
        },
        Table {
            name: "wt",
            columns: vec![("g", "BIGINT"), ("w", "DOUBLE")],
            rows: vec![
                vec![0.0, 1.5],
                vec![1.0, -0.5],
                vec![2.0, 2.0],
                vec![3.0, 0.25],
            ],
        },
    ]
}

/// A max-pool as an aggregate: the largest `val` per group, weighted.
fn max_pool(f: &str) -> String {
    format!(
        "WITH p AS (SELECT g, {f}(val) AS m FROM x GROUP BY g) \
         SELECT SUM(p.m * wt.w) AS loss FROM p JOIN wt ON p.g = wt.g"
    )
}

/// The same max-pool as a ranking, the idiom nn.py uses for accuracy: rank
/// by value, keep the first, with the tie broken by item.
const RANK_POOL: &str = "\
WITH r AS ( \
  SELECT g, item, val, \
         ROW_NUMBER() OVER (PARTITION BY g ORDER BY val DESC, item) AS rk \
  FROM x) \
SELECT SUM(r.val * wt.w) AS loss \
FROM r JOIN wt ON r.g = wt.g WHERE r.rk = 1";

async fn group0(sql: &str, tables: &[Table]) -> Vec<f64> {
    let ctx = ctx();
    for t in tables {
        t.create(&ctx).await;
    }
    let plan = substrait_of(&ctx, sql, true).await;
    let program = grad(&plan, &[ColumnRef::new("x", "val")]).unwrap();
    run(&ctx, &program).await;
    let sql = format!(
        "SELECT g, item, val FROM {} WHERE g = 0 ORDER BY item",
        program.gradients[0].step
    );
    rows(&ctx, &sql).await.iter().map(|r| r[2]).collect()
}

#[tokio::test]
async fn max_and_min_send_the_cotangent_to_the_extreme() {
    for f in ["MAX", "MIN"] {
        let grads = check_gradients(
            &ctx(),
            &max_pool(f),
            &pool_tables(false),
            &[ColumnRef::new("x", "val")],
        )
        .await;
        assert_eq!(grads["x"].iter().filter(|r| r[2] != 0.0).count(), 4, "{f}");
    }
}

#[tokio::test]
async fn at_a_tie_max_shares_the_cotangent_like_jax() {
    // jax.grad(jnp.max) splits the cotangent evenly across tied maxima.
    assert_eq!(
        group0(&max_pool("MAX"), &pool_tables(true)).await,
        vec![0.0, 0.75, 0.0, 0.75, 0.0]
    );
}

#[tokio::test]
async fn a_rank_filter_sends_the_cotangent_to_the_kept_row() {
    let grads = check_gradients(
        &ctx(),
        RANK_POOL,
        &pool_tables(false),
        &[ColumnRef::new("x", "val")],
    )
    .await;
    assert_eq!(grads["x"].iter().filter(|r| r[2] != 0.0).count(), 4);
}

#[tokio::test]
async fn at_a_tie_a_rank_filter_sends_it_all_to_the_row_it_kept() {
    // The ORDER BY's tie-break keeps one row, and it gets everything
    // (route_ad_spike.py's convention). Unlike MAX, the query picked a
    // winner, so there is nothing to share.
    assert_eq!(
        group0(RANK_POOL, &pool_tables(true)).await,
        vec![0.0, 1.5, 0.0, 0.0, 0.0]
    );
}

#[tokio::test]
async fn a_rank_used_as_a_value_is_refused() {
    let ctx = ctx();
    for t in pool_tables(false) {
        t.create(&ctx).await;
    }
    let sql = "WITH r AS (SELECT g, val, RANK() OVER (PARTITION BY g ORDER BY val) AS rk FROM x) \
               SELECT SUM(val * rk) AS loss FROM r";
    let plan = substrait_of(&ctx, sql, true).await;
    let err = grad(&plan, &[ColumnRef::new("x", "val")]).unwrap_err();
    assert!(matches!(err, AdError::NotImplemented(_)), "{err}");
    assert!(err.to_string().contains("rank"), "{err}");
}

#[tokio::test]
async fn a_ranking_that_does_not_break_ties_is_refused() {
    // ORDER BY val alone leaves ties to the engine; recomputing the ranking
    // could then keep a different row than the forward pass kept.
    let ctx = ctx();
    for t in pool_tables(false) {
        t.create(&ctx).await;
    }
    let sql = RANK_POOL.replace("ORDER BY val DESC, item", "ORDER BY val DESC");
    let plan = substrait_of(&ctx, &sql, true).await;
    let err = grad(&plan, &[ColumnRef::new("x", "val")]).unwrap_err();
    assert!(matches!(err, AdError::NotImplemented(_)), "{err}");
    assert!(err.to_string().contains("break ties"), "{err}");
}

#[tokio::test]
async fn avg_gives_a_null_row_a_null_gradient() {
    let ctx = ctx();
    Table {
        name: "wn",
        columns: vec![("i", "BIGINT"), ("val", "DOUBLE")],
        rows: vec![vec![0.0, 1.0], vec![2.0, 3.0]],
    }
    .create(&ctx)
    .await;
    ctx.sql("INSERT INTO wn VALUES (1, NULL)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let plan = substrait_of(&ctx, "SELECT AVG(val * val) AS l FROM wn", true).await;
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
    // AVG over the two non-NULL rows: d/dv (v²/2) = v.
    assert_eq!(got[0], vec![0.0, 1.0]);
    assert!(got[1][1].is_nan(), "{:?}", got[1]);
    assert_eq!(got[2], vec![2.0, 3.0]);
}

#[tokio::test]
async fn relu_written_as_case_or_greatest() {
    for relu in [
        "CASE WHEN val > 0.25 THEN val ELSE 0.25 END",
        "greatest(val, 0.25)",
    ] {
        check_gradients(
            &ctx(),
            &format!("SELECT SUM(power({relu}, 2)) AS loss FROM z"),
            &logits(),
            &[ColumnRef::new("z", "val")],
        )
        .await;
    }
}

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// `(key, gradient)` of `table.val` under `loss`, a NULL gradient as NaN.
async fn gradient_of(ctx: &SessionContext, loss: &str, table: &str) -> Vec<Vec<f64>> {
    let plan = substrait_of(ctx, loss, true).await;
    let program = grad(&plan, &[ColumnRef::new(table, "val")]).unwrap();
    run(ctx, &program).await;
    rows(
        ctx,
        &format!("SELECT * FROM {} ORDER BY 1", program.gradients[0].step),
    )
    .await
}

#[tokio::test]
async fn avg_max_and_min_send_no_gradient_through_a_row_they_skip() {
    // From the v2 soak (#89): each skips row 1, whose argument p + q is NULL,
    // so p(1) cannot move the loss.
    let ctx = ctx();
    exec(
        &ctx,
        "CREATE TABLE np (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 9.0), (2, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE nq (i BIGINT, val DOUBLE) AS VALUES (0, 5.0), (1, NULL), (2, 0.0)",
    )
    .await;
    for (f, want) in [
        ("AVG", [0.5, 0.0, 0.5]),
        ("MAX", [1.0, 0.0, 0.0]),
        ("MIN", [0.0, 0.0, 1.0]),
    ] {
        let loss = format!("SELECT {f}(np.val + nq.val) AS l FROM np JOIN nq ON np.i = nq.i");
        let got = gradient_of(&ctx, &loss, "np").await;
        let got: Vec<f64> = got.iter().map(|r| r[1]).collect();
        assert_eq!(got, want, "{f}");
    }
}

#[tokio::test]
async fn a_rank_filter_over_a_rank_filter_runs() {
    // From the v2 soak (#89): the rebuilt region held two identically named
    // window columns. Top two by val, then the top one of those: row 1.
    let ctx = ctx();
    exec(
        &ctx,
        "CREATE TABLE rp (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 3.0), (2, 2.0)",
    )
    .await;
    let got = gradient_of(
        &ctx,
        "WITH a AS (SELECT i, v FROM (SELECT i, val AS v, \
                      ROW_NUMBER() OVER (ORDER BY val DESC, i) AS rk FROM rp) WHERE rk <= 2), \
              b AS (SELECT i, v FROM (SELECT i, v, \
                      ROW_NUMBER() OVER (ORDER BY v DESC, i) AS rk FROM a) WHERE rk = 1) \
         SELECT SUM(v) AS loss FROM b",
        "rp",
    )
    .await;
    assert_eq!(got, vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 0.0]]);
}

#[tokio::test]
async fn max_finds_its_row_when_the_recomputed_values_jitter() {
    // From the v2 soak (#91). The region beneath the MAX is recomputed, and
    // a grouped SUM over several partitions adds in arrival order, so the
    // recomputed values need not equal the forward pass's to the bit. The
    // MAX rule compared them with the saved maximum, and about half these
    // runs sent no gradient at all.
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
    for attempt in 0..20 {
        let ctx = ctx();
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
        let got = gradient_of(&ctx, loss, "p").await;
        let nonzero = got.iter().filter(|r| r[1] != 0.0).count();
        assert_eq!(nonzero, 1, "attempt {attempt}: {got:?}");
        assert!(
            got.iter().all(|r| r[1].is_finite()),
            "attempt {attempt}: {got:?}"
        );
    }
}

/// `grad` of `loss` with respect to `table.val`: `Ok(())` if it would run, or
/// the refusal.
async fn refusal(ctx: &SessionContext, loss: &str, table: &str) -> Option<String> {
    let plan = substrait_of(ctx, loss, true).await;
    match grad(&plan, &[ColumnRef::new(table, "val")]) {
        Ok(_) => None,
        Err(AdError::NotImplemented(m)) => Some(m),
        Err(e) => panic!("{loss}: {e}"),
    }
}

#[tokio::test]
async fn a_limit_must_break_ties_by_the_dims_it_cuts() {
    // A LIMIT keeps the first rows in its input's order, and SQL does not
    // order ties: recomputed for the backward pass it could keep other rows.
    let ctx = ctx();
    exec(
        &ctx,
        "CREATE TABLE lp (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 3.0), (2, 2.0)",
    )
    .await;
    for loss in [
        "SELECT SUM(v) AS l FROM (SELECT val AS v FROM lp ORDER BY val DESC LIMIT 2)",
        "SELECT SUM(v) AS l FROM (SELECT val AS v FROM lp LIMIT 2)",
    ] {
        let why = refusal(&ctx, loss, "lp").await;
        assert!(
            why.as_deref().is_some_and(|m| m.contains("LIMIT")),
            "{loss}: {why:?}"
        );
    }
    let got = gradient_of(
        &ctx,
        "SELECT SUM(v) AS l FROM (SELECT val AS v FROM lp ORDER BY val DESC, i LIMIT 2)",
        "lp",
    )
    .await;
    assert_eq!(got, vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 1.0]]);
}

#[tokio::test]
async fn a_ranking_over_constant_data_in_a_recomputed_region_is_refused() {
    // From #93's mutation testing: constant subtrees are recomputed in each
    // backward step, and ddx has no dims to show a ranking inside one is
    // total, so a tie could keep a different row on the way back.
    let ctx = ctx();
    exec(
        &ctx,
        "CREATE TABLE cp (j BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE cd (j BIGINT, v DOUBLE) AS VALUES (0, 5.0), (0, 5.0), (1, 7.0)",
    )
    .await;
    for loss in [
        "SELECT SUM(cp.val * d.v) AS l FROM cp JOIN \
           (SELECT j, v FROM (SELECT j, v, ROW_NUMBER() OVER (PARTITION BY j ORDER BY v) AS rk \
                              FROM cd) WHERE rk = 1) d ON cp.j = d.j",
        "SELECT SUM(cp.val * d.v) AS l FROM cp CROSS JOIN \
           (SELECT v FROM cd ORDER BY v LIMIT 1) d",
    ] {
        let why = refusal(&ctx, loss, "cp").await;
        assert!(
            why.as_deref()
                .is_some_and(|m| m.contains("reads no wrt table")),
            "{loss}: {why:?}"
        );
    }
}

#[tokio::test]
async fn a_ranking_on_a_semi_joins_right_side_must_break_ties() {
    // The right side of a semi-join decides which left rows are kept, so a
    // ranking there is recomputed and must be total too.
    let ctx = ctx();
    exec(
        &ctx,
        "CREATE TABLE sp (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 3.0), (2, 3.0)",
    )
    .await;
    let loss = "SELECT SUM(val) AS l FROM sp WHERE i IN \
                  (SELECT i FROM (SELECT i, ROW_NUMBER() OVER (ORDER BY val DESC) AS rk FROM sp) \
                   WHERE rk = 1)";
    let why = refusal(&ctx, loss, "sp").await;
    assert!(
        why.as_deref().is_some_and(|m| m.contains("semi-join")),
        "{why:?}"
    );
}

#[tokio::test]
async fn a_volatile_function_in_a_recomputed_region_is_refused() {
    // random() would give other values when the region is recomputed for the
    // backward pass, whether it is on the path from a wrt table or in data
    // joined into it.
    let ctx = ctx();
    exec(
        &ctx,
        "CREATE TABLE vp (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE vd (i BIGINT, v DOUBLE) AS VALUES (0, 5.0), (1, 7.0)",
    )
    .await;
    for loss in [
        "SELECT SUM(val * random()) AS l FROM vp",
        "SELECT SUM(vp.val * d.r) AS l FROM vp JOIN (SELECT i, v * random() AS r FROM vd) d \
           ON vp.i = d.i",
        "SELECT SUM(val) AS l FROM vp WHERE random() < 2.0",
        "SELECT SUM(val * (SELECT random())) AS l FROM vp",
    ] {
        let why = refusal(&ctx, loss, "vp").await;
        assert!(
            why.as_deref().is_some_and(|m| m.contains("volatile")),
            "{loss}: {why:?}"
        );
    }
}

#[tokio::test]
async fn an_infinite_row_that_does_not_attain_the_min_sends_no_nan() {
    // From the v2 soak (seed 100534). The row with d = inf has v = inf,
    // which the MIN skips over; its seed was 0, and 0 times the partial of
    // sqrt(v·v + 1), ∞/∞ there, is NaN, which then summed into p's gradient.
    let ctx = ctx();
    exec(
        &ctx,
        "CREATE TABLE ip (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE id (i BIGINT, j BIGINT, v DOUBLE) AS \
         VALUES (0, 0, 0.5), (0, 1, 'inf'::DOUBLE), (1, 0, 3.0)",
    )
    .await;
    let got = gradient_of(
        &ctx,
        "SELECT MIN(sqrt((id.v - ip.val) * (id.v - ip.val) + 1.0)) AS l \
         FROM ip JOIN id ON ip.i = id.i",
        "ip",
    )
    .await;
    // The MIN is at (0, 0): v = 0.5 - 1 = -0.5, and d/dp sqrt(v² + 1) is
    // -v / sqrt(v² + 1) = 0.5 / sqrt(1.25).
    assert_eq!(got[1], vec![1.0, 0.0]);
    assert!((got[0][1] - 0.5 / 1.25f64.sqrt()).abs() < 1e-12, "{got:?}");
}

#[tokio::test]
async fn a_grouped_max_over_nan_data_finds_its_row() {
    // From the v2 soak (seed 600845). DataFusion's grouped MAX skips a NaN,
    // but a window MAX returns it, so the rows attaining the saved maximum
    // were compared with NaN and none matched: a silent zero gradient.
    let ctx = ctx();
    exec(
        &ctx,
        "CREATE TABLE np2 (i BIGINT, val DOUBLE) AS VALUES (0, 0.9), (1, 0.2)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE nd (i BIGINT, j BIGINT, v DOUBLE) AS \
         VALUES (0, 0, 0.5), (0, 1, 'NaN'::DOUBLE), (0, 2, -0.4), (1, 0, 0.3)",
    )
    .await;
    let got = gradient_of(
        &ctx,
        "SELECT SUM(mx) AS l FROM (SELECT nd.i, MAX(greatest(nd.v, np2.val)) AS mx \
                                   FROM nd JOIN np2 ON nd.i = np2.i GROUP BY nd.i)",
        "np2",
    )
    .await;
    // Group 0's maximum is 0.9 = np2.val(0), attained twice (j = 0 and 2),
    // so the cotangent reaches np2.val(0) in full; group 1's is 0.3 from nd.
    assert_eq!(got, vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
}

#[tokio::test]
async fn a_max_tie_that_rounding_breaks_is_still_shared() {
    // From the v2 soak (seed 2101304): two groups that tie in exact
    // arithmetic can differ in the last bit, and which one rounds higher can
    // change from run to run (a sum over partitions), so the MAX gave its
    // whole cotangent to either. Within a few ulps it is shared, as a tie:
    // (p + 0.1) + 0.2 and p + 0.3 differ by one ulp.
    let ctx = ctx();
    exec(
        &ctx,
        "CREATE TABLE tp (i BIGINT, val DOUBLE) AS VALUES (0, 0.0), (1, 0.0)",
    )
    .await;
    exec(
        &ctx,
        "CREATE TABLE tc (i BIGINT, a DOUBLE, b DOUBLE) AS VALUES (0, 0.1, 0.2), (1, 0.3, 0.0)",
    )
    .await;
    let got = gradient_of(
        &ctx,
        "SELECT MAX((tp.val + tc.a) + tc.b) AS l FROM tp JOIN tc ON tp.i = tc.i",
        "tp",
    )
    .await;
    assert_eq!(got, vec![vec![0.0, 0.5], vec![1.0, 0.5]]);
}
