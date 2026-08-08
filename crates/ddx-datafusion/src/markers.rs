// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The `grad`/`jvp` marker UDFs.
//!
//! These are **not** row functions. A scalar UDF only ever receives evaluated
//! *values*, never the symbolic expression of its argument, but differentiation
//! is a function of the symbolic form — so `grad` cannot be computed at
//! runtime. (Empirically pinned on a live engine in
//! `docs/spikes/datafusion_python_analyzer_rule_r2.py`, T4: a `grad` UDF given
//! `grad(x*x, x)` over `x = [1,2,3]` receives `[1.0, 4.0, 9.0]`.)
//!
//! Registration exists for exactly one reason: to make the marker call *parse
//! and plan*, so [`crate::analyzer::DdxAnalyzer`] can find it in the
//! `LogicalPlan` and rewrite it away. Reaching execution is therefore always a
//! bug, and these deliberately error there rather than returning a number.

use datafusion::arrow::datatypes::DataType;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

/// The name of the gradient marker, as written in SQL.
pub const GRAD: &str = "grad";
/// The name of the forward-mode (directional derivative) marker.
pub const JVP: &str = "jvp";

/// A marker UDF: parses and plans, never executes.
#[derive(Debug, PartialEq, Eq, Hash)]
struct Marker {
    name: &'static str,
    signature: Signature,
}

impl Marker {
    fn new(name: &'static str, arg_count: usize) -> Self {
        Marker {
            name,
            // `Signature::any` accepts the arguments at whatever types they
            // arrive in. That tolerance is required here: `add_analyzer_rule`
            // installs the rule to run AFTER `TypeCoercion`, so a
            // stricter signature would make the planner inject casts into the
            // marker's argument before ddx ever sees it. (ddx-core does have a
            // `Cast` rule, so an injected cast is survivable — but not
            // provoking one keeps the differentiated expression closer to what
            // the user actually wrote.)
            signature: Signature::any(arg_count, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for Marker {
    fn name(&self) -> &str {
        self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        // Derivatives are always emitted DOUBLE-typed: differentiation runs
        // pre-binding, so operand types are unknown, and SQL integer division
        // truncates on some engines but not others.
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Err(DataFusionError::Execution(format!(
            "ddx: `{name}` reached execution, which never happens in a correct \
             rewrite — it is a compile-time marker, not a row function.\n\n\
             The `{name}()` call was not rewritten away before planning finished. \
             Either the ddx analyzer rule is not installed on this SessionContext \
             (use `ddx_datafusion::install(&ctx)`), or the marker sits somewhere \
             the rule does not reach — in which case rewrite the SQL text instead \
             with `ddx_datafusion::ddx_sql(&ctx, sql)`.",
            name = self.name
        )))
    }
}

/// The `grad(expr, column)` marker: `d(expr)/d(column)`.
pub fn grad_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(Marker::new(GRAD, 2))
}

/// The `jvp(expr, column, tangent)` marker: `d(expr)/d(column) · tangent`.
pub fn jvp_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(Marker::new(JVP, 3))
}

/// Is `name` one of ddx's marker functions? Case-folded, because SQL function
/// names are case-insensitive and `GRAD(x, x)` must be caught too.
pub(crate) fn marker_kind(name: &str) -> Option<&'static str> {
    if name.eq_ignore_ascii_case(GRAD) {
        Some(GRAD)
    } else if name.eq_ignore_ascii_case(JVP) {
        Some(JVP)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_are_named_and_arity_checked() {
        assert_eq!(grad_udf().name(), "grad");
        assert_eq!(jvp_udf().name(), "jvp");
    }

    #[test]
    fn marker_names_are_matched_case_insensitively() {
        assert_eq!(marker_kind("grad"), Some(GRAD));
        assert_eq!(marker_kind("GRAD"), Some(GRAD));
        assert_eq!(marker_kind("Jvp"), Some(JVP));
        assert_eq!(marker_kind("gradient"), None);
        assert_eq!(marker_kind("mygrad"), None);
    }

    #[test]
    fn executing_a_marker_is_a_loud_error_not_a_number() {
        // Markers deliberately error if one reaches execution, rather than
        // silently producing a value.
        let udf = grad_udf();
        let err = udf
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![],
                arg_fields: vec![],
                number_rows: 1,
                return_field: std::sync::Arc::new(datafusion::arrow::datatypes::Field::new(
                    "d",
                    DataType::Float64,
                    true,
                )),
                config_options: std::sync::Arc::new(datafusion::config::ConfigOptions::default()),
            })
            .expect_err("a marker must never execute successfully");
        let msg = err.to_string();
        assert!(msg.contains("reached execution"), "unexpected: {msg}");
        // The error has to tell the user what to actually do about it.
        assert!(msg.contains("install"), "no remedy in message: {msg}");
        assert!(msg.contains("ddx_sql"), "no fallback in message: {msg}");
    }
}
