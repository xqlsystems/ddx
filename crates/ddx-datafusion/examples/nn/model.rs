// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! nn.py (xarray-sql#196) with ddx: nn.py's forward pass and loss as SQL, the
//! data, and an SGD step that takes `grad` in SQL.
//!
//! Shared by the `nn` example, which trains it, and by `tests/nn.rs`, which
//! checks its gradients against nn.py's hand-written backward pass.

use std::sync::Arc;

use datafusion::arrow::array::{Float64Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::prelude::SessionContext;

/// Images are `SIDE × SIDE`; the input index of pixel (h, w) is `h * SIDE + w`.
pub const SIDE: usize = 6;
/// Layer widths: pixels → two tanh layers → class logits.
pub const WIDTHS: [usize; 4] = [SIDE * SIDE, 12, 8, 4];

/// nn.py's forward pass and loss as one `WITH` clause, ending in the CTE
/// `loss`. It is nn.py's own SQL, three layers and a mean softmax
/// cross-entropy, including its zero-pixel skip, the layer picked in each join
/// condition, and the bias joins. Nothing is added for ddx.
pub fn with_loss() -> String {
    format!(
        "\
WITH c0 AS (
  SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
  FROM (SELECT sample, height * {SIDE} + width AS inp, images AS val
        FROM pixels WHERE images <> 0) a
  JOIN weight w ON a.inp = w.inp AND w.layer = 0
  GROUP BY a.sample, w.out),
fwd0 AS (
  SELECT c0.sample, c0.out AS out, tanh(c0.z + b.val) AS val
  FROM c0 JOIN bias b ON c0.out = b.out AND b.layer = 0),
c1 AS (
  SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
  FROM (SELECT sample, out AS inp, val FROM fwd0) a
  JOIN weight w ON a.inp = w.inp AND w.layer = 1
  GROUP BY a.sample, w.out),
fwd1 AS (
  SELECT c1.sample, c1.out AS out, tanh(c1.z + b.val) AS val
  FROM c1 JOIN bias b ON c1.out = b.out AND b.layer = 1),
c2 AS (
  SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
  FROM (SELECT sample, out AS inp, val FROM fwd1) a
  JOIN weight w ON a.inp = w.inp AND w.layer = 2
  GROUP BY a.sample, w.out),
logits AS (
  SELECT c2.sample, c2.out AS out, c2.z + b.val AS z
  FROM c2 JOIN bias b ON c2.out = b.out AND b.layer = 2),
m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample),
e AS (SELECT logits.sample, logits.out, exp(logits.z - m.m) AS e
      FROM logits JOIN m ON logits.sample = m.sample),
s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample),
loss AS (
  SELECT -AVG(ln(e.e / s.s)) AS loss
  FROM e JOIN s ON e.sample = s.sample
         JOIN labels y ON y.sample = e.sample
  WHERE e.out = y.labels)
"
    )
}

/// One SGD step for `weight`: `weight - lr * grad(loss)(weight)`, as a join.
pub fn update_weight(lr: f64) -> String {
    format!(
        "{} SELECT w.layer, w.inp, w.out, w.val - {lr} * g.val AS val \
         FROM weight w JOIN grad(loss, weight.val) g \
           ON w.layer = g.layer AND w.inp = g.inp AND w.out = g.out",
        with_loss()
    )
}

/// One SGD step for `bias`.
pub fn update_bias(lr: f64) -> String {
    format!(
        "{} SELECT b.layer, b.out, b.val - {lr} * g.val AS val \
         FROM bias b JOIN grad(loss, bias.val) g ON b.layer = g.layer AND b.out = g.out",
        with_loss()
    )
}

/// The loss alone.
pub fn loss_sql() -> String {
    format!("{} SELECT loss FROM loss", with_loss())
}

/// The fraction of samples whose largest logit is their label: nn.py's
/// accuracy query.
pub fn accuracy_sql() -> String {
    format!(
        "\
WITH c0 AS (
  SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
  FROM (SELECT sample, height * {SIDE} + width AS inp, images AS val FROM pixels) a
  JOIN weight w ON a.inp = w.inp AND w.layer = 0
  GROUP BY a.sample, w.out),
fwd0 AS (SELECT c0.sample, c0.out, tanh(c0.z + b.val) AS val
         FROM c0 JOIN bias b ON c0.out = b.out AND b.layer = 0),
c1 AS (
  SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
  FROM (SELECT sample, out AS inp, val FROM fwd0) a
  JOIN weight w ON a.inp = w.inp AND w.layer = 1
  GROUP BY a.sample, w.out),
fwd1 AS (SELECT c1.sample, c1.out, tanh(c1.z + b.val) AS val
         FROM c1 JOIN bias b ON c1.out = b.out AND b.layer = 1),
c2 AS (
  SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
  FROM (SELECT sample, out AS inp, val FROM fwd1) a
  JOIN weight w ON a.inp = w.inp AND w.layer = 2
  GROUP BY a.sample, w.out),
logits AS (SELECT c2.sample, c2.out, c2.z + b.val AS z
           FROM c2 JOIN bias b ON c2.out = b.out AND b.layer = 2),
pred AS (SELECT sample, out,
                ROW_NUMBER() OVER (PARTITION BY sample ORDER BY z DESC, out) AS rk
         FROM logits)
SELECT AVG(CASE WHEN p.out = y.labels THEN 1.0 ELSE 0.0 END) AS acc
FROM pred p JOIN labels y ON y.sample = p.sample
WHERE p.rk = 1"
    )
}

/// A small deterministic generator, so the example needs no `rand`.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407))
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

// Every column is declared nullable, like the tables the SGD update produces,
// so a table keeps one schema across training steps.
fn register(ctx: &SessionContext, name: &str, batch: RecordBatch) -> Result<()> {
    let table = MemTable::try_new(batch.schema(), vec![vec![batch]])?;
    ctx.deregister_table(name)?;
    ctx.register_table(name, Arc::new(table))?;
    Ok(())
}

fn i64s(v: Vec<usize>) -> Arc<Int64Array> {
    Arc::new(Int64Array::from(
        v.into_iter().map(|x| x as i64).collect::<Vec<_>>(),
    ))
}

/// nn.py's offline data: one template image per class plus noise, with a dark
/// (zero) border so the zero-pixel skip has something to skip.
/// Registers `pixels(sample, height, width, images)` and
/// `labels(sample, labels)`.
pub fn register_data(ctx: &SessionContext, n_samples: usize, rng: &mut Rng) -> Result<()> {
    let classes = WIDTHS[3];
    let templates: Vec<Vec<f64>> = (0..classes)
        .map(|_| (0..SIDE * SIDE).map(|_| rng.normal()).collect())
        .collect();
    let (mut s, mut h, mut w, mut img, mut labels) = (vec![], vec![], vec![], vec![], vec![]);
    for sample in 0..n_samples {
        let label = sample % classes;
        labels.push(label);
        for hh in 0..SIDE {
            for ww in 0..SIDE {
                let border = hh == 0 || ww == 0 || hh == SIDE - 1 || ww == SIDE - 1;
                let v = if border {
                    0.0
                } else {
                    templates[label][hh * SIDE + ww] + 0.6 * rng.normal()
                };
                s.push(sample);
                h.push(hh);
                w.push(ww);
                img.push(v);
            }
        }
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("sample", DataType::Int64, true),
        Field::new("height", DataType::Int64, true),
        Field::new("width", DataType::Int64, true),
        Field::new("images", DataType::Float64, true),
    ]));
    let pixels = RecordBatch::try_new(
        schema,
        vec![i64s(s), i64s(h), i64s(w), Arc::new(Float64Array::from(img))],
    )?;
    register(ctx, "pixels", pixels)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("sample", DataType::Int64, true),
        Field::new("labels", DataType::Int64, true),
    ]));
    let labels = RecordBatch::try_new(schema, vec![i64s((0..n_samples).collect()), i64s(labels)])?;
    register(ctx, "labels", labels)
}

/// nn.py's initial model: small random weights, zero biases. Registers
/// `weight(layer, inp, out, val)` and `bias(layer, out, val)`.
pub fn register_model(ctx: &SessionContext, rng: &mut Rng) -> Result<()> {
    let (mut layer, mut inp, mut out, mut val) = (vec![], vec![], vec![], vec![]);
    let (mut blayer, mut bout, mut bval) = (vec![], vec![], vec![]);
    for l in 0..WIDTHS.len() - 1 {
        for i in 0..WIDTHS[l] {
            for o in 0..WIDTHS[l + 1] {
                layer.push(l);
                inp.push(i);
                out.push(o);
                val.push(0.3 * rng.normal());
            }
        }
        for o in 0..WIDTHS[l + 1] {
            blayer.push(l);
            bout.push(o);
            bval.push(0.0);
        }
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("layer", DataType::Int64, true),
        Field::new("inp", DataType::Int64, true),
        Field::new("out", DataType::Int64, true),
        Field::new("val", DataType::Float64, true),
    ]));
    let weight = RecordBatch::try_new(
        schema,
        vec![
            i64s(layer),
            i64s(inp),
            i64s(out),
            Arc::new(Float64Array::from(val)),
        ],
    )?;
    register(ctx, "weight", weight)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("layer", DataType::Int64, true),
        Field::new("out", DataType::Int64, true),
        Field::new("val", DataType::Float64, true),
    ]));
    let bias = RecordBatch::try_new(
        schema,
        vec![i64s(blayer), i64s(bout), Arc::new(Float64Array::from(bval))],
    )?;
    register(ctx, "bias", bias)
}

/// Register `df`'s rows as the table `name`, replacing it.
pub async fn replace(
    ctx: &SessionContext,
    name: &str,
    df: datafusion::prelude::DataFrame,
) -> Result<()> {
    let schema = df.schema().inner().clone();
    let batches = df.collect().await?;
    let table = MemTable::try_new(schema, vec![batches])?;
    ctx.deregister_table(name)?;
    ctx.register_table(name, Arc::new(table))?;
    Ok(())
}
