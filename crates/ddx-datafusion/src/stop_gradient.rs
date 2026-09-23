// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx_stop_gradient`, the one function a query-level AD forward pass may call
//! that ddx gives a meaning to.
//!
//! Unlike `grad`, it runs: it is the identity, so a forward query using it
//! returns the same numbers as one without it. `ddx-ad` finds it in the
//! query's Substrait plan and treats its argument as a constant, as JAX treats
//! `lax.stop_gradient` (design.md §4.3).

use datafusion::arrow::datatypes::DataType;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::prelude::SessionContext;

/// The SQL name of the stop-gradient function.
pub const STOP_GRADIENT: &str = "ddx_stop_gradient";

/// The identity, under the name `ddx_stop_gradient`.
#[derive(Debug, PartialEq, Eq, Hash)]
struct StopGradient {
    signature: Signature,
}

impl ScalarUDFImpl for StopGradient {
    fn name(&self) -> &str {
        STOP_GRADIENT
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    // The argument's own type. A fixed type (say Float64) would make the
    // planner cast the argument, so the function would no longer sit directly
    // on the value it stops.
    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn invoke_with_args(&self, mut args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        args.args
            .pop()
            .ok_or_else(|| DataFusionError::Internal(format!("{STOP_GRADIENT} takes one argument")))
    }
}

/// The `ddx_stop_gradient` UDF.
pub fn stop_gradient_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(StopGradient {
        // `any` accepts the argument at whatever type it arrives in, so no
        // cast is inserted inside the function either.
        signature: Signature::any(1, Volatility::Immutable),
    })
}

/// Register `ddx_stop_gradient` on `ctx`, so a forward pass that uses it plans
/// and runs.
///
/// ```
/// # use datafusion::prelude::SessionContext;
/// # #[tokio::main]
/// # async fn main() -> datafusion::error::Result<()> {
/// let ctx = SessionContext::new();
/// ddx_datafusion::register_stop_gradient(&ctx);
/// let df = ctx.sql("SELECT ddx_stop_gradient(2.0) AS v").await?;
/// # let _ = df.collect().await?;
/// # Ok(())
/// # }
/// ```
pub fn register_stop_gradient(ctx: &SessionContext) {
    ctx.register_udf(stop_gradient_udf());
}
