// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The `substrait` single-version guard — v2's counterpart to `sqlparser_pin.rs`.
//!
//! `ddx-ad` reads the `substrait::proto::Plan` DataFusion produces and, from M4,
//! hands its backward steps back to DataFusion's consumer. That only works while
//! both resolve the *same* `substrait`: two versions are two unrelated Rust types.

mod common;

use common::locked_versions;

#[test]
fn exactly_one_substrait_is_linked() {
    let versions = locked_versions("substrait");
    assert_eq!(
        versions.len(),
        1,
        "ddx-ad and datafusion-substrait must resolve the SAME `substrait`, but the \
         lockfile has {n}: {versions:?}.\n\n\
         ddx-ad's analysis takes the `substrait::proto::Plan` DataFusion's producer \
         emits; two versions make that two unrelated types. Fix by moving ddx-ad's \
         `substrait` requirement (crates/ddx-ad/Cargo.toml) to the version the \
         workspace's `datafusion` pulls in through `datafusion-substrait`.",
        n = versions.len(),
    );
}

#[test]
fn the_plan_types_are_actually_the_same_type() {
    fn assert_same(
        p: datafusion_substrait::substrait::proto::Plan,
    ) -> ddx_ad::substrait::proto::Plan {
        p
    }
    assert_eq!(assert_same(Default::default()), Default::default());
}
