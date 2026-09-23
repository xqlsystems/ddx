// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Running a v2 backward program on DataFusion, and checking its gradients
//! against finite differences computed by the same engine.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Array, AsArray, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Float64Type, Int64Type};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use datafusion_substrait::logical_plan::consumer::from_substrait_plan;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;
use ddx_ad::emit::{bind_reads, unbound_reads};
use ddx_ad::substrait::proto::plan_rel::RelType as PlanRelType;
use ddx_ad::substrait::proto::rel::RelType;
use ddx_ad::substrait::proto::{NamedStruct, Rel};
use ddx_ad::{BackwardProgram, ColumnRef};

use super::substrait_of;

/// The Substrait schema of the registered table `name`: the base schema of the
/// read in the plan of `SELECT * FROM name`.
pub async fn schema_of(ctx: &SessionContext, name: &str) -> NamedStruct {
    let lp = ctx.table(name).await.unwrap().into_unoptimized_plan();
    let p = to_substrait_plan(&lp, &ctx.state()).unwrap();
    let Some(PlanRelType::Root(root)) = &p.relations[0].rel_type else {
        panic!("no root")
    };
    fn find(rel: &Rel) -> NamedStruct {
        match rel.rel_type.as_ref().unwrap() {
            RelType::Read(r) => r.base_schema.clone().unwrap(),
            RelType::Project(p) => find(p.input.as_ref().unwrap()),
            other => panic!("unexpected {other:?}"),
        }
    }
    find(root.input.as_ref().unwrap())
}

/// Run every step of `program`, registering each result as a table.
pub async fn run(ctx: &SessionContext, program: &BackwardProgram) {
    for step in program.steps() {
        let mut plan = step.plan.clone();
        let mut schemas = HashMap::new();
        for name in unbound_reads(&plan) {
            let s = schema_of(ctx, &name).await;
            schemas.insert(name, s);
        }
        bind_reads(&mut plan, &mut |n| schemas.get(n).cloned()).unwrap();
        let lp = from_substrait_plan(&ctx.state(), &plan)
            .await
            .unwrap_or_else(|e| panic!("consuming step {}: {e}", step.name));
        let df = ctx.execute_logical_plan(lp).await.unwrap();
        let schema = df.schema().inner().clone();
        let batches = df
            .collect()
            .await
            .unwrap_or_else(|e| panic!("running step {}: {e}", step.name));
        ctx.deregister_table(step.name.as_str()).unwrap();
        ctx.register_table(
            step.name.as_str(),
            Arc::new(MemTable::try_new(schema, vec![batches]).unwrap()),
        )
        .unwrap();
    }
}

/// A small table of numbers, kept in memory so it can be perturbed.
#[derive(Clone)]
pub struct Table {
    pub name: &'static str,
    /// Column names and SQL types (`BIGINT` or `DOUBLE`).
    pub columns: Vec<(&'static str, &'static str)>,
    pub rows: Vec<Vec<f64>>,
}

impl Table {
    pub async fn create(&self, ctx: &SessionContext) {
        let cols: Vec<String> = self
            .columns
            .iter()
            .map(|(n, t)| format!("{n} {t}"))
            .collect();
        let rows: Vec<String> = self
            .rows
            .iter()
            .map(|r| {
                let vals: Vec<String> = r
                    .iter()
                    .zip(&self.columns)
                    .map(|(v, (_, t))| {
                        if *t == "BIGINT" {
                            format!("{}", *v as i64)
                        } else {
                            format!("CAST({v:e} AS DOUBLE)")
                        }
                    })
                    .collect();
                format!("({})", vals.join(", "))
            })
            .collect();
        let sql = format!(
            "CREATE OR REPLACE TABLE {} ({}) AS VALUES {}",
            self.name,
            cols.join(", "),
            rows.join(", ")
        );
        ctx.sql(&sql).await.unwrap().collect().await.unwrap();
    }

    fn col(&self, name: &str) -> usize {
        self.columns.iter().position(|(n, _)| *n == name).unwrap()
    }
}

/// The single number `sql` returns.
pub async fn scalar(ctx: &SessionContext, sql: &str) -> f64 {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut out = Vec::new();
    for b in &batches {
        out.extend(f64s(b, 0));
    }
    assert_eq!(out.len(), 1, "{sql} returned {out:?}");
    out[0]
}

/// The sum of column 0 of what `sql` returns.
pub async fn total(ctx: &SessionContext, sql: &str) -> f64 {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches.iter().flat_map(|b| f64s(b, 0)).sum()
}

fn f64s(b: &RecordBatch, c: usize) -> Vec<f64> {
    let a = b.column(c);
    match a.data_type() {
        DataType::Float64 => a
            .as_primitive::<Float64Type>()
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect(),
        DataType::Int64 => a
            .as_primitive::<Int64Type>()
            .iter()
            .map(|v| v.map_or(f64::NAN, |v| v as f64))
            .collect(),
        other => panic!("column of type {other}"),
    }
}

/// The rows of a table, every column as f64.
pub async fn rows(ctx: &SessionContext, sql: &str) -> Vec<Vec<f64>> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let cols: Vec<Vec<f64>> = (0..b.num_columns()).map(|c| f64s(b, c)).collect();
        for r in 0..b.num_rows() {
            out.push(cols.iter().map(|c| c[r]).collect());
        }
    }
    out
}

/// Take `grad` of `loss_sql` with `ddx-ad`, run the program, and compare every
/// gradient entry with a central finite difference of the loss, computed by
/// DataFusion with the parameter perturbed. `tables` must include every `wrt`
/// table. Returns the gradients, keyed by table name, as `(key, grad)` rows.
pub async fn check_gradients(
    ctx: &SessionContext,
    loss_sql: &str,
    tables: &[Table],
    wrt: &[ColumnRef],
) -> HashMap<String, Vec<Vec<f64>>> {
    for t in tables {
        t.create(ctx).await;
    }
    let plan = substrait_of(ctx, loss_sql, true).await;
    let program = ddx_ad::grad(&plan, wrt).unwrap_or_else(|e| panic!("grad: {e}"));
    run(ctx, &program).await;

    // The forward step reproduces the query.
    let loss = scalar(ctx, loss_sql).await;
    let out = total(ctx, &format!("SELECT * FROM {}", program.value)).await;
    assert!(
        (loss - out).abs() <= 1e-9 * loss.abs().max(1.0),
        "loss {loss} vs value step {out}"
    );

    let mut result = HashMap::new();
    for g in &program.gradients {
        let table = tables
            .iter()
            .find(|t| g.table.last().map(String::as_str) == Some(t.name))
            .expect("every wrt table is given");
        let got = rows(ctx, &format!("SELECT * FROM {}", g.step)).await;
        assert_eq!(
            got.len(),
            table.rows.len(),
            "one gradient row per table row"
        );
        let key_names: Vec<&str> = g
            .columns
            .iter()
            .map(String::as_str)
            .filter(|c| !wrt.iter().any(|w| w.column == *c && w.table == table.name))
            .collect();
        for (r, row) in table.rows.iter().enumerate() {
            let key: Vec<f64> = key_names.iter().map(|k| row[table.col(k)]).collect();
            let grow = got
                .iter()
                .find(|gr| gr[..key.len()] == key[..])
                .unwrap_or_else(|| panic!("no gradient row for key {key:?}"));
            for (i, w) in wrt.iter().filter(|w| w.table == table.name).enumerate() {
                let c = table.col(&w.column);
                let h = 1e-5 * row[c].abs().max(1.0);
                let mut perturbed = table.clone();
                perturbed.rows[r][c] += h;
                perturbed.create(ctx).await;
                let up = scalar(ctx, loss_sql).await;
                perturbed.rows[r][c] -= 2.0 * h;
                perturbed.create(ctx).await;
                let down = scalar(ctx, loss_sql).await;
                let fd = (up - down) / (2.0 * h);
                let ad = grow[key.len() + i];
                assert!(
                    (fd - ad).abs() <= 1e-5 * fd.abs().max(1.0),
                    "∂loss/∂{}.{} at {key:?}: ddx {ad} vs finite difference {fd}",
                    table.name,
                    w.column
                );
            }
        }
        table.create(ctx).await;
        result.insert(table.name.to_string(), got);
    }
    result
}
