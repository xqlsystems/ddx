// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx_ad::Backend` implemented by a synchronous engine, as a Python one
//! is: each method does its work and returns a future already done, and the
//! run is driven without an async runtime. The same gradient as `ad::run`,
//! and the same refusal and cleanup.

use std::future::{ready, Future};
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use datafusion::error::DataFusionError;
use datafusion::prelude::SessionContext;
use ddx_ad::substrait::proto::NamedStruct;
use ddx_ad::{Backend, Bound, RunError};
use ddx_datafusion::ad::{self, ColumnRef};
use tokio::runtime::Runtime;

/// Run a future whose every await is already done: a synchronous engine's
/// whole run, since each of its methods returns a finished future.
fn now<F: Future>(f: F) -> F::Output {
    match pin!(f).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(out) => out,
        Poll::Pending => panic!("a synchronous backend's future is never pending"),
    }
}

struct Sync {
    ctx: SessionContext,
    rt: Runtime,
    /// Every table_schema call, and every table written, in order.
    log: Vec<String>,
}

impl Backend for Sync {
    type Error = DataFusionError;

    fn table_schema(
        &mut self,
        name: &str,
    ) -> impl Future<Output = Result<NamedStruct, Self::Error>> + Send {
        self.log.push(format!("schema {name}"));
        ready(
            self.rt
                .block_on(ddx_datafusion::engine::table_schema(&self.ctx, name)),
        )
    }

    fn returns_rows(
        &mut self,
        plans: &[Bound],
    ) -> impl Future<Output = Result<Vec<bool>, Self::Error>> + Send {
        ready(
            plans
                .iter()
                .map(|plan| {
                    self.rt
                        .block_on(ddx_datafusion::engine::returns_rows(&self.ctx, plan))
                })
                .collect(),
        )
    }

    fn materialize(
        &mut self,
        name: &str,
        plan: &Bound,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.log.push(format!("write {name}"));
        ready(
            self.rt
                .block_on(ddx_datafusion::engine::materialize(&self.ctx, name, plan)),
        )
    }

    fn drop_table(&mut self, name: &str) -> impl Future<Output = Result<(), Self::Error>> + Send {
        ready(self.ctx.deregister_table(name).map(|_| ()))
    }

    fn plan_sql(
        &mut self,
        sql: &str,
    ) -> impl Future<Output = Result<ddx_ad::substrait::proto::Plan, Self::Error>> + Send {
        let mut engine = ddx_datafusion::engine::DataFusion(self.ctx.clone());
        ready(
            self.rt
                .block_on(ddx_ad::Backend::plan_sql(&mut engine, sql)),
        )
    }
}

fn backend(rows: &str) -> Sync {
    let b = Sync {
        ctx: SessionContext::new(),
        rt: Runtime::new().unwrap(),
        log: Vec::new(),
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
    now(ddx_ad::run(&mut b, &program)).unwrap();
    assert_eq!(
        gradient(&b, &program.gradients[0].step),
        vec![(0, 1.0), (1, 2.0)]
    );
    for step in program.intermediate_steps() {
        assert!(!b.ctx.table_exist(step.name.as_str()).unwrap());
    }
    // Each table's schema is asked for at most once per write of it: for an
    // engine, asking is a round trip (composability re-review on #79).
    let mut asked = std::collections::BTreeSet::new();
    for entry in &b.log {
        if let Some(name) = entry.strip_prefix("write ") {
            asked.remove(name);
        } else {
            let name = entry.strip_prefix("schema ").unwrap();
            assert!(
                asked.insert(name.to_string()),
                "asked twice for {name}: {:?}",
                b.log
            );
        }
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
    let err = now(ddx_ad::run(&mut b, &program)).unwrap_err();
    assert!(
        matches!(err, RunError::Refused(ddx_ad::AdError::InvalidWrt(_))),
        "{err}"
    );
    for step in program.steps() {
        assert!(!b.ctx.table_exist(step.name.as_str()).unwrap());
    }
}
