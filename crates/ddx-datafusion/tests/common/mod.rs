// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Shared plumbing for the integration tests.
//!
//! Only the mechanical parts live here — pulling a column out of a result set.
//! Each test file keeps its own fixture setup, because *what* it registers is
//! part of what it is testing: `path_a.rs` deliberately never calls `install`,
//! and `regressions.rs` needs tables with specific column types.

// Each integration test is its own binary and uses a different subset of this
// module, so unused helpers are expected rather than a smell.
#![allow(dead_code)]

use datafusion::arrow::array::{Array, Float64Array, RecordBatch};
use datafusion::arrow::datatypes::DataType;

/// Column 0 of `batches` as `f64`s.
///
/// Panics if it is not `Float64`: every derivative ddx emits is DOUBLE-typed, so
/// anything else is a failure worth surfacing loudly right here rather than as a
/// confusing comparison further down the test.
pub fn f64_column(batches: &[RecordBatch]) -> Vec<f64> {
    let mut out = Vec::new();
    for b in batches {
        let a = b
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("every derivative is emitted DOUBLE-typed");
        out.extend((0..a.len()).map(|i| a.value(i)));
    }
    out
}

/// The Arrow type of column 0 — used where the *type* is the claim under test,
/// not just the values.
pub fn column_type(batches: &[RecordBatch]) -> DataType {
    batches[0].schema().field(0).data_type().clone()
}
