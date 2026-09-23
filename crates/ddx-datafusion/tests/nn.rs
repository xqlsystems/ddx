// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! M4's acceptance fixture: nn.py's MLP, trained with gradients from `grad` in
//! SQL rather than written by hand.
//!
//! The first test runs nn.py's own hand-written backward pass (its `delta*`
//! and `g*` queries, copied from xarray-sql#196 with the train/test split
//! removed) next to `grad(loss, weight.val)` and `grad(loss, bias.val)` for
//! nn.py's own loss, and requires the two to agree on every weight and bias.
//! The second trains the model.

#[path = "../examples/nn/model.rs"]
mod model;

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use common::ad::{rows, scalar};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad;
use model::{Rng, SIDE};

const N: usize = 24;

async fn setup() -> SessionContext {
    let ctx = SessionContext::new();
    // nn.py's hand-written backward pass uses v1's grad() for tanh'.
    ddx_datafusion::install(&ctx);
    let mut rng = Rng::new(7);
    model::register_data(&ctx, N, &mut rng).unwrap();
    model::register_model(&ctx, &mut rng).unwrap();
    ctx
}

/// Run `sql` and register its result as `name`, like nn.py's `.cache()` and
/// `register_table`.
async fn cache(ctx: &SessionContext, name: &str, sql: &str) {
    let df = ctx.sql(sql).await.unwrap();
    let schema = df.schema().inner().clone();
    let batches = df.collect().await.unwrap();
    ctx.deregister_table(name).unwrap();
    ctx.register_table(
        name,
        Arc::new(MemTable::try_new(schema, vec![batches]).unwrap()),
    )
    .unwrap();
}

/// nn.py's forward and backward passes, as it wrote them. Registers
/// `g0`..`g2` (weight gradients) and `gb0`..`gb2` (bias gradients).
async fn hand_written_backward(ctx: &SessionContext) {
    let n = N;
    cache(
        ctx,
        "fwd0",
        &format!(
            "
        WITH c AS (
          SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
          FROM (SELECT sample, height * {SIDE} + width AS inp, images AS val
                FROM pixels WHERE images <> 0) a
          JOIN weight w ON a.inp = w.inp AND w.layer = 0
          GROUP BY a.sample, w.out)
        SELECT c.sample, c.out AS out, c.z + b.val AS z, tanh(c.z + b.val) AS val
        FROM c JOIN bias b ON c.out = b.out AND b.layer = 0"
        ),
    )
    .await;
    cache(
        ctx,
        "fwd1",
        "
        WITH c AS (
          SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
          FROM (SELECT sample, out AS inp, val FROM fwd0) a
          JOIN weight w ON a.inp = w.inp AND w.layer = 1
          GROUP BY a.sample, w.out)
        SELECT c.sample, c.out AS out, c.z + b.val AS z, tanh(c.z + b.val) AS val
        FROM c JOIN bias b ON c.out = b.out AND b.layer = 1",
    )
    .await;
    cache(
        ctx,
        "logits",
        "
        WITH c AS (
          SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
          FROM (SELECT sample, out AS inp, val FROM fwd1) a
          JOIN weight w ON a.inp = w.inp AND w.layer = 2
          GROUP BY a.sample, w.out)
        SELECT c.sample, c.out AS out, c.z + b.val AS z
        FROM c JOIN bias b ON c.out = b.out AND b.layer = 2",
    )
    .await;
    cache(
        ctx,
        "delta2",
        "
        WITH m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample),
             e AS (SELECT logits.sample, logits.out, exp(logits.z - m.m) AS e
                   FROM logits JOIN m ON logits.sample = m.sample),
             s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample)
        SELECT e.sample, e.out,
               e.e / s.s - CASE WHEN e.out = y.labels THEN 1.0 ELSE 0.0 END AS val
        FROM e JOIN s ON e.sample = s.sample
               JOIN labels y ON y.sample = e.sample",
    )
    .await;
    cache(
        ctx,
        "g2",
        &format!(
            "
        SELECT a.inp AS inp, d.out AS out, SUM(a.val * d.val) / {n} AS val
        FROM (SELECT sample, out AS inp, val FROM fwd1) a
        JOIN delta2 d ON a.sample = d.sample
        GROUP BY a.inp, d.out"
        ),
    )
    .await;
    cache(
        ctx,
        "gb2",
        &format!("SELECT out, SUM(val) / {n} AS val FROM delta2 GROUP BY out"),
    )
    .await;
    cache(
        ctx,
        "delta1",
        "
        WITH dc AS (
          SELECT d.sample, w.inp AS out, SUM(d.val * w.val) AS val
          FROM delta2 d JOIN weight w ON d.out = w.out AND w.layer = 2
          GROUP BY d.sample, w.inp)
        SELECT dc.sample, dc.out,
               dc.val * grad(tanh(fwd1.z), fwd1.z) AS val
        FROM dc JOIN fwd1 ON dc.sample = fwd1.sample AND dc.out = fwd1.out",
    )
    .await;
    cache(
        ctx,
        "g1",
        &format!(
            "
        SELECT a.inp AS inp, d.out AS out, SUM(a.val * d.val) / {n} AS val
        FROM (SELECT sample, out AS inp, val FROM fwd0) a
        JOIN delta1 d ON a.sample = d.sample
        GROUP BY a.inp, d.out"
        ),
    )
    .await;
    cache(
        ctx,
        "gb1",
        &format!("SELECT out, SUM(val) / {n} AS val FROM delta1 GROUP BY out"),
    )
    .await;
    cache(
        ctx,
        "delta0",
        "
        WITH dc AS (
          SELECT d.sample, w.inp AS out, SUM(d.val * w.val) AS val
          FROM delta1 d JOIN weight w ON d.out = w.out AND w.layer = 1
          GROUP BY d.sample, w.inp)
        SELECT dc.sample, dc.out,
               dc.val * grad(tanh(fwd0.z), fwd0.z) AS val
        FROM dc JOIN fwd0 ON dc.sample = fwd0.sample AND dc.out = fwd0.out",
    )
    .await;
    cache(
        ctx,
        "g0",
        &format!(
            "
        WITH a AS (
          SELECT sample, height * {SIDE} + width AS inp, images AS val
          FROM pixels
          WHERE images <> 0
        )
        SELECT a.inp AS inp, d.out AS out, SUM(a.val * d.val) / {n} AS val
        FROM a JOIN delta0 d ON a.sample = d.sample
        GROUP BY a.inp, d.out"
        ),
    )
    .await;
    cache(
        ctx,
        "gb0",
        &format!("SELECT out, SUM(val) / {n} AS val FROM delta0 GROUP BY out"),
    )
    .await;
}

/// Compare a dense ddx gradient with nn.py's per-layer gradients, where a row
/// nn.py never produced (a skipped zero pixel's weight) counts as zero.
fn assert_same(what: &str, ddx: &[Vec<f64>], hand: HashMap<Vec<i64>, f64>) {
    let mut worst: f64 = 0.0;
    for row in ddx {
        let (key, g) = row.split_at(row.len() - 1);
        let key: Vec<i64> = key.iter().map(|k| *k as i64).collect();
        let want = hand.get(&key).copied().unwrap_or(0.0);
        worst = worst.max((g[0] - want).abs());
    }
    assert!(worst < 1e-12, "{what}: max |ddx - nn.py| = {worst:e}");
    for key in hand.keys() {
        assert!(
            ddx.iter().any(|r| r[..key.len()]
                .iter()
                .map(|k| *k as i64)
                .eq(key.iter().copied())),
            "{what}: nn.py has a gradient for {key:?} that ddx lacks"
        );
    }
}

async fn tagged(ctx: &SessionContext, tables: &[(i64, &str)]) -> HashMap<Vec<i64>, f64> {
    let mut out = HashMap::new();
    for (layer, t) in tables {
        for r in rows(ctx, &format!("SELECT * FROM {t}")).await {
            let (key, v) = r.split_at(r.len() - 1);
            let mut k = vec![*layer];
            k.extend(key.iter().map(|x| *x as i64));
            out.insert(k, v[0]);
        }
    }
    out
}

/// `grad(loss, weight.val)` and `grad(loss, bias.val)` of nn.py's loss, from
/// one run of its program.
async fn sql_gradients(ctx: &SessionContext) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let with_loss = model::with_loss();
    let weight = format!("{with_loss} SELECT layer, inp, out, val FROM grad(loss, weight.val)");
    let bias = format!("{with_loss} SELECT layer, out, val FROM grad(loss, bias.val)");
    let frames = ad::sql_all(ctx, &[&weight, &bias]).await.unwrap();
    let mut out = Vec::new();
    for df in frames {
        let name = format!("__test_{}", out.len());
        cache_df(ctx, &name, df).await;
        out.push(rows(ctx, &format!("SELECT * FROM {name}")).await);
    }
    let bias = out.pop().unwrap();
    (out.pop().unwrap(), bias)
}

async fn cache_df(ctx: &SessionContext, name: &str, df: datafusion::prelude::DataFrame) {
    let schema = df.schema().inner().clone();
    let batches = df.collect().await.unwrap();
    ctx.deregister_table(name).unwrap();
    ctx.register_table(
        name,
        Arc::new(MemTable::try_new(schema, vec![batches]).unwrap()),
    )
    .unwrap();
}

#[tokio::test]
async fn sql_grad_equals_nn_py_s_hand_written_backward_pass() {
    let ctx = setup().await;
    let (weight, bias) = sql_gradients(&ctx).await;
    hand_written_backward(&ctx).await;
    assert_same(
        "weight",
        &weight,
        tagged(&ctx, &[(0, "g0"), (1, "g1"), (2, "g2")]).await,
    );
    assert_same(
        "bias",
        &bias,
        tagged(&ctx, &[(0, "gb0"), (1, "gb1"), (2, "gb2")]).await,
    );

    // And the loss is nn.py's.
    let loss = scalar(&ctx, &model::loss_sql()).await;
    let hand = scalar(
        &ctx,
        "
        WITH m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample),
             e AS (SELECT logits.sample, logits.out, exp(logits.z - m.m) AS e
                   FROM logits JOIN m ON logits.sample = m.sample),
             s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample)
        SELECT -AVG(ln(e.e / s.s)) AS loss
        FROM e JOIN s ON e.sample = s.sample
               JOIN labels y ON y.sample = e.sample
        WHERE e.out = y.labels",
    )
    .await;
    assert!((loss - hand).abs() < 1e-12, "{loss} vs {hand}");
}

#[tokio::test]
async fn training_with_sql_grad_learns() {
    let ctx = setup().await;
    let (update_weight, update_bias) = (model::update_weight(0.5), model::update_bias(0.5));
    let mut losses = Vec::new();
    for _ in 0..15 {
        losses.push(scalar(&ctx, &model::loss_sql()).await);
        let mut frames = ad::sql_all(&ctx, &[&update_weight, &update_bias])
            .await
            .unwrap();
        let bias = frames.pop().unwrap();
        let weight = frames.pop().unwrap();
        model::replace(&ctx, "weight", weight).await.unwrap();
        model::replace(&ctx, "bias", bias).await.unwrap();
    }
    let (first, last) = (losses[0], *losses.last().unwrap());
    assert!(
        last < 0.25 * first,
        "loss went from {first} to {last}: {losses:?}"
    );
    let acc = scalar(&ctx, &model::accuracy_sql()).await;
    assert!(acc > 0.9, "training accuracy {acc}");
}
