// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A training loop's step, run with `ad::prepare`, timed and checked against
//! `ad::sql_all` on every run.
//!
//! The step comes from a directory that holds a model and its data: a
//! `params.txt` naming the parameter tables, one per line; each table as
//! `{name}.parquet`; batches as `batch0.parquet`, `batch1.parquet`, …;
//! `model.sql`, a statement run after each batch is registered (such as a
//! `CREATE OR REPLACE VIEW` over `batch`); and `statements.sql`, the step's
//! statements separated by lines of `;;`, one per parameter table, in the
//! order of `params.txt`. After each run, each parameter table is
//! registered again with its statement's rows, as SGD updates it.
//!
//! ```text
//! systemd-run --user --scope -p MemoryMax=8G nix develop -c \
//!   cargo run --release -p ddx-datafusion --example prepared_step -- DIR
//! ```

use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{AsArray, RecordBatch};
use datafusion::arrow::datatypes::Float64Type;
use datafusion::datasource::MemTable;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use ddx_datafusion::ad;

/// Register `batches` as the table `name`, replacing one.
fn put(ctx: &SessionContext, name: &str, batches: Vec<RecordBatch>) {
    let schema = batches[0].schema();
    ctx.deregister_table(name).unwrap();
    let table = MemTable::try_new(schema, vec![batches]).unwrap();
    ctx.register_table(name, Arc::new(table)).unwrap();
}

async fn read(ctx: &SessionContext, path: String) -> Vec<RecordBatch> {
    let df = ctx.read_parquet(path, ParquetReadOptions::default()).await;
    df.unwrap().collect().await.unwrap()
}

/// Each batch's last column, which is the parameter's value.
fn values(batches: &[RecordBatch]) -> Vec<f64> {
    batches
        .iter()
        .flat_map(|b| {
            b.column(b.num_columns() - 1)
                .as_primitive::<Float64Type>()
                .values()
                .to_vec()
        })
        .collect()
}

#[tokio::main]
async fn main() {
    let dir = std::env::args().nth(1).expect("the directory of the step");
    let text = |name: &str| std::fs::read_to_string(format!("{dir}/{name}")).unwrap();
    let config = SessionConfig::new()
        .with_information_schema(true)
        .with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    let params: Vec<String> = text("params.txt").lines().map(String::from).collect();
    for p in &params {
        put(&ctx, p, read(&ctx, format!("{dir}/{p}.parquet")).await);
    }
    let model = text("model.sql");
    let statements: Vec<String> = text("statements.sql")
        .split("\n;;\n")
        .map(String::from)
        .collect();
    let statements: Vec<&str> = statements.iter().map(String::as_str).collect();
    let step = ad::prepare(&ctx, &statements);
    let mut later = Vec::new();
    for k in 0.. {
        let path = format!("{dir}/batch{k}.parquet");
        if !std::path::Path::new(&path).exists() {
            break;
        }
        put(&ctx, "batch", read(&ctx, path).await);
        ctx.sql(&model).await.unwrap().collect().await.unwrap();
        let start = Instant::now();
        let mut got = Vec::new();
        for df in step.run().await.unwrap() {
            got.push(df.collect().await.unwrap());
        }
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let mut worst = 0f64;
        for (got, want) in got
            .iter()
            .zip(ad::sql_all(&ctx, &statements).await.unwrap())
        {
            let (got, want) = (values(got), values(&want.collect().await.unwrap()));
            assert_eq!(got.len(), want.len(), "run {k}: another number of rows");
            worst = got
                .iter()
                .zip(&want)
                .fold(worst, |w, (a, b)| w.max((a - b).abs()));
        }
        println!("run {k}: {ms:.0} ms, at most {worst:.1e} from sql_all");
        if k > 0 {
            later.push(ms);
        }
        for (p, batches) in params.iter().zip(got) {
            put(&ctx, p, batches);
        }
    }
    later.sort_by(f64::total_cmp);
    if let Some(median) = later.get(later.len() / 2) {
        println!("median of the later runs: {median:.0} ms");
    }
}
