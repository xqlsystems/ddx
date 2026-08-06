// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The `sqlparser` single-version guard.
//!
//! Path B's bridge hands a `sqlparser::ast::Expr` from DataFusion's unparser
//! straight to `ddx-core`. That only type-checks while both crates resolve the
//! *same* `sqlparser`. If they ever diverge they become two unrelated Rust
//! types and the bridge stops compiling — which is a loud failure, but an
//! extremely confusing one if you don't already know to look at the pin.
//!
//! This test makes the pin itself the thing that fails, with an explanation.
//! It is a test rather than a `just` recipe so CI enforces it on every PR
//! without anyone having to remember.

use std::path::PathBuf;

/// Every `sqlparser` version present in the workspace lockfile.
fn locked_sqlparser_versions() -> Vec<String> {
    let lock = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../Cargo.lock")
        .canonicalize()
        .expect("workspace Cargo.lock must exist");
    let text = std::fs::read_to_string(&lock).expect("Cargo.lock must be readable");

    let mut versions = Vec::new();
    let mut in_sqlparser = false;
    for line in text.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            in_sqlparser = false;
        } else if line == r#"name = "sqlparser""# {
            in_sqlparser = true;
        } else if in_sqlparser {
            if let Some(v) = line.strip_prefix("version = ") {
                versions.push(v.trim_matches('"').to_string());
                in_sqlparser = false;
            }
        }
    }
    versions
}

#[test]
fn exactly_one_sqlparser_is_linked() {
    let versions = locked_sqlparser_versions();
    assert!(
        !versions.is_empty(),
        "no `sqlparser` in Cargo.lock — did the dependency graph change?"
    );
    assert_eq!(
        versions.len(),
        1,
        "ddx-core and datafusion must resolve the SAME `sqlparser`, but the \
         lockfile has {n}: {versions:?}.\n\n\
         Path B's bridge (crates/ddx-datafusion/src/analyzer.rs) passes a \
         `sqlparser::ast::Expr` from DataFusion's unparser directly to \
         ddx-core. Two versions means two unrelated Rust types and the bridge \
         will not compile.\n\n\
         Fix by reconciling the pins in the workspace Cargo.toml: ddx-core pins \
         `sqlparser` exactly, and the `datafusion` version must be one whose \
         own requirement accepts that pin (54.x wants ^0.62.0; 53.x wanted \
         ^0.61.0). If they genuinely cannot be reconciled, the documented \
         fallback is to degrade the bridge to a SQL string round-trip rather \
         than break it.",
        n = versions.len(),
    );
}

#[test]
fn the_bridge_types_are_actually_the_same_type() {
    // The compile-time half of the same guarantee, stated as code: if these
    // were different crate versions this function would not type-check.
    fn assert_same(e: datafusion::sql::sqlparser::ast::Expr) -> ddx_core::sqlparser::ast::Expr {
        e
    }
    let expr =
        ddx_core::sqlparser::ast::Expr::Identifier(ddx_core::sqlparser::ast::Ident::new("x"));
    assert_eq!(assert_same(expr).to_string(), "x");
}
