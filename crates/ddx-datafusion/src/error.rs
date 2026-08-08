// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Bridging `ddx-core`'s [`DiffError`] into DataFusion's error type.

use datafusion::error::DataFusionError;
use ddx_core::DiffError;

/// Wrap a [`DiffError`] as a [`DataFusionError`].
///
/// [`DataFusionError::External`] rather than `Plan`: it boxes the original
/// error instead of stringifying it, so a caller can still downcast back to
/// [`DiffError`] and match on the variant. That matters here — the whole point
/// of ddx's *fail loud, never silently wrong* rule is that callers can
/// tell a `NotImplemented` (this construct has no rule yet) from an
/// `AmbiguousColumn` (your query needs qualifying) programmatically, not by
/// grepping a message.
pub(crate) fn to_df_err(e: DiffError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_error_survives_as_a_downcastable_external() {
        let df_err = to_df_err(DiffError::NotImplemented("atan2".into()));
        let DataFusionError::External(boxed) = &df_err else {
            panic!("expected External, got {df_err:?}");
        };
        let recovered = boxed
            .downcast_ref::<DiffError>()
            .expect("the original DiffError must survive the boxing");
        assert_eq!(recovered, &DiffError::NotImplemented("atan2".into()));
    }
}
