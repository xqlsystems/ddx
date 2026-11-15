// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx_stop_gradient` for datafusion-python, as a Rust UDF passed through
//! DataFusion's FFI. A Python UDF takes exact types, so it would cast a
//! `REAL` argument to `DOUBLE`; this one, like `ddx_datafusion`'s, returns
//! its argument at its own type. ddxdb links DataFusion's FFI and expression
//! crates only, not `ddx_datafusion`, so the UDF is written again here.

use std::sync::Arc;

use datafusion_common::arrow::datatypes::DataType;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_ffi::udf::FFI_ScalarUDF;
use ddx_ad::STOP_GRADIENT;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

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

    // The argument's own type, so the planner casts nothing and the function
    // sits directly on the value it stops.
    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn invoke_with_args(&self, mut args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        args.args
            .pop()
            .ok_or_else(|| DataFusionError::Internal(format!("{STOP_GRADIENT} takes one argument")))
    }
}

/// `ddx_stop_gradient` as the capsule `datafusion.udf` takes.
#[pyfunction]
pub fn _stop_gradient_udf(py: Python<'_>) -> PyResult<Bound<'_, PyCapsule>> {
    let udf = Arc::new(ScalarUDF::new_from_impl(StopGradient {
        signature: Signature::any(1, Volatility::Immutable),
    }));
    PyCapsule::new_with_value(py, FFI_ScalarUDF::from(udf), c"datafusion_scalar_udf")
}
