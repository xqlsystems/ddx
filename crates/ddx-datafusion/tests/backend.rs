// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx_ad::Backend`, the four primitives a synchronous engine supplies, on
//! DataFusion run synchronously: the same gradient as `ad::run`, and the same
//! refusal and cleanup.

use datafusion::error::DataFusionError;
use datafusion::prelude::SessionContext;
use ddx_ad::substrait::proto::{NamedStruct, Plan};
use ddx_ad::{Backend, RunError};
use ddx_datafusion::ad::{self, ColumnRef};
use tokio::runtime::Runtime;

struct Sync {
    ctx: SessionContext,
    rt: Runtime,
}

impl Backend for Sync {
    type Error = DataFusionError;

    fn table_schema(&mut self, name: &str) -> Result<NamedStruct, Self::Error> {
        self.rt.block_on(ad::table_schema(&self.ctx, name))
    }

    fn returns_rows(&mut self, plan: &Plan) -> Result<bool, Self::Error> {
        self.rt.block_on(ad::returns_rows(&self.ctx, plan))
    }

    fn materialize(&mut self, name: &str, plan: &Plan) -> Result<(), Self::Error> {
        self.rt.block_on(ad::materialize(&self.ctx, name, plan))
    }

    fn drop_table(&mut self, name: &str) -> Result<(), Self::Error> {
        self.ctx.deregister_table(name).map(|_| ())
    }
}

fn backend(rows: &str) -> Sync {
    let b = Sync {
        ctx: SessionContext::new(),
        rt: Runtime::new().unwrap(),
    };
    let sql = format!("CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES {rows}");
    b.rt.block_on(async { b.ctx.sql(&sql).await?.collect().await })
        .unwrap();
    b
}

fn gradient(b: &Sync, step: &str) -> Vec<(i64, f64)> {
    use datafusion::arrow::array::AsArray;
    use datafusion::arrow::datatypes::{Float64Type, Int64Type};
    let batches =
        b.rt.block_on(async {
            b.ctx
                .sql(&format!("SELECT i, val FROM {step} ORDER BY i"))
                .await?
                .collect()
                .await
        })
        .unwrap();
    batches
        .iter()
        .flat_map(|batch| {
            let (i, v) = (
                batch.column(0).as_primitive::<Int64Type>(),
                batch.column(1).as_primitive::<Float64Type>(),
            );
            (0..batch.num_rows()).map(move |r| (i.value(r), v.value(r)))
        })
        .collect()
}

#[test]
fn a_synchronous_backend_gets_the_gradient_and_drops_the_intermediates() {
    let mut b = backend("(0, 1.0), (1, 2.0)");
    let loss = "SELECT SUM(val * val) / COUNT(val) AS loss FROM w";
    let program =
        b.rt.block_on(ad::grad(&b.ctx, loss, &[ColumnRef::new("w", "val")]))
            .unwrap();
    ddx_ad::run(&mut b, &program).unwrap();
    assert_eq!(
        gradient(&b, &program.gradients[0].step),
        vec![(0, 1.0), (1, 2.0)]
    );
    for step in program.intermediate_steps() {
        assert!(!b.ctx.table_exist(step.name.as_str()).unwrap());
    }
}

#[test]
fn a_synchronous_backend_refuses_rows_that_share_their_dims() {
    let mut b = backend("(0, 1.0), (0, 2.0)");
    let program =
        b.rt.block_on(ad::grad(
            &b.ctx,
            "SELECT SUM(val) AS loss FROM w",
            &[ColumnRef::new("w", "val")],
        ))
        .unwrap();
    let err = ddx_ad::run(&mut b, &program).unwrap_err();
    assert!(
        matches!(err, RunError::Refused(ddx_ad::AdError::InvalidWrt(_))),
        "{err}"
    );
    for step in program.steps() {
        assert!(!b.ctx.table_exist(step.name.as_str()).unwrap());
    }
}
