// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The `substrait` single-version guard, the v2 twin of `sqlparser_pin.rs`.
//!
//! The adapter hands `ddx-ad` the `substrait::proto::Plan` that
//! `datafusion-substrait` produced. That type-checks only while both crates
//! resolve the same `substrait`; this test makes a divergence fail at the pin,
//! with an explanation, instead of as a type error somewhere in the adapter.

use std::path::PathBuf;

fn locked_versions(name: &str) -> Vec<String> {
    let lock = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock");
    let text = std::fs::read_to_string(lock).expect("workspace Cargo.lock must be readable");
    let mut versions = Vec::new();
    let mut in_package = false;
    for line in text.lines().map(str::trim) {
        if line == "[[package]]" {
            in_package = false;
        } else if line == format!(r#"name = "{name}""#) {
            in_package = true;
        } else if let (true, Some(v)) = (in_package, line.strip_prefix("version = ")) {
            versions.push(v.trim_matches('"').to_string());
            in_package = false;
        }
    }
    versions
}

#[test]
fn exactly_one_substrait_is_linked() {
    let versions = locked_versions("substrait");
    assert_eq!(
        versions.len(),
        1,
        "ddx-ad and datafusion-substrait must resolve the SAME `substrait`, but \
         the lockfile has {versions:?}. Two versions make `substrait::proto::Plan` \
         two unrelated types, and the adapter cannot pass DataFusion's plan to \
         ddx-ad. Set the `substrait` pin in the workspace Cargo.toml to the \
         version `datafusion-substrait` requires."
    );
}

#[test]
fn the_plan_types_are_actually_the_same_type() {
    // Does not compile if the two crates resolve different versions.
    fn same(p: datafusion_substrait::substrait::proto::Plan) -> ddx_ad::substrait::proto::Plan {
        p
    }
    assert_eq!(same(Default::default()), Default::default());
}
