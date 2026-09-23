// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Train nn.py's MLP (xarray-sql#196) with `grad` in SQL.
//!
//! nn.py trains a small network entirely in SQL, but writes its backward pass
//! by hand: a query per layer for the error, and one per weight and bias
//! gradient. Here the backward pass is one expression per parameter table,
//! `grad(loss, weight.val)`, where `loss` is nn.py's own loss query, and a
//! training step is the SGD update written as a join:
//!
//! ```sql
//! WITH ..., loss AS (SELECT -AVG(ln(e.e / s.s)) AS loss FROM ...)
//! SELECT w.layer, w.inp, w.out, w.val - 0.5 * g.val AS val
//! FROM weight w JOIN grad(loss, weight.val) g
//!   ON w.layer = g.layer AND w.inp = g.inp AND w.out = g.out
//! ```
//!
//! ```sh
//! cargo run -p ddx-datafusion --example nn
//! ```

mod model;

use datafusion::error::Result;
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad;
use model::Rng;

const SAMPLES: usize = 60;
const STEPS: usize = 40;
const LR: f64 = 0.5;

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = SessionContext::new();
    let mut rng = Rng::new(1);
    model::register_data(&ctx, SAMPLES, &mut rng)?;
    model::register_model(&ctx, &mut rng)?;

    let (update_weight, update_bias) = (model::update_weight(LR), model::update_bias(LR));
    for step in 0..STEPS {
        if step % 5 == 0 {
            report(&ctx, step).await?;
        }
        // One statement per parameter table. They take grad of the same loss,
        // so its backward pass runs once for both.
        let mut frames = ad::sql_all(&ctx, &[&update_weight, &update_bias]).await?;
        let bias = frames.pop().expect("two statements");
        let weight = frames.pop().expect("two statements");
        model::replace(&ctx, "weight", weight).await?;
        model::replace(&ctx, "bias", bias).await?;
    }
    report(&ctx, STEPS).await
}

async fn report(ctx: &SessionContext, step: usize) -> Result<()> {
    let loss = first(ctx, &model::loss_sql()).await?;
    let acc = first(ctx, &model::accuracy_sql()).await?;
    println!("step {step:2}: loss {loss:.4}  accuracy {acc:.3}");
    Ok(())
}

async fn first(ctx: &SessionContext, sql: &str) -> Result<f64> {
    use datafusion::arrow::array::{AsArray, RecordBatch};
    use datafusion::arrow::datatypes::Float64Type;
    let batches: Vec<RecordBatch> = ctx.sql(sql).await?.collect().await?;
    Ok(batches[0].column(0).as_primitive::<Float64Type>().value(0))
}
