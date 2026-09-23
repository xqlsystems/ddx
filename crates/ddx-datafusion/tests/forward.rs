// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `ddx_ad::Forward` on plans DataFusion produces: what is saved, which
//! columns carry gradient, and what is refused.

mod common;

use common::substrait_of;
use datafusion::prelude::SessionContext;
use ddx_ad::forward::{Def, Input, Output};
use ddx_ad::{AdError, ColumnRef, Forward};

async fn ctx() -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    for sql in [
        // pixels(sample, i, x): data. w(i, o, val) and b(o, val): parameters.
        "CREATE TABLE pixels (sample BIGINT, i BIGINT, x DOUBLE) AS VALUES \
         (0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)",
        "CREATE TABLE w (i BIGINT, o BIGINT, val DOUBLE) AS VALUES \
         (0, 0, 0.1), (0, 1, 0.2), (1, 0, 0.3), (1, 1, 0.4)",
        "CREATE TABLE b (o BIGINT, val DOUBLE) AS VALUES (0, 0.5), (1, -0.5)",
    ] {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }
    ctx
}

const LAYER: &str = "WITH c AS ( \
       SELECT p.sample, w.o, SUM(p.x * w.val) AS z \
       FROM pixels p JOIN w ON p.i = w.i GROUP BY p.sample, w.o) \
     SELECT SUM(tanh(c.z + b.val)) AS loss FROM c JOIN b ON c.o = b.o";

async fn both(sql: &str, wrt: &[ColumnRef]) -> Vec<Forward> {
    let ctx = ctx().await;
    let mut out = Vec::new();
    for optimized in [false, true] {
        let plan = substrait_of(&ctx, sql, optimized).await;
        out.push(Forward::new(&plan, wrt).unwrap());
    }
    out
}

#[tokio::test]
async fn a_layer_saves_its_two_aggregates() {
    let wrt = [ColumnRef::new("w", "val"), ColumnRef::new("b", "val")];
    for g in both(LAYER, &wrt).await {
        assert_eq!(g.saved.len(), 2, "the contraction and the loss");
        let contraction = &g.saved[0];
        assert_eq!(
            contraction.outputs,
            vec![Output::Dim(0), Output::Dim(1), Output::Value(0)]
        );
        assert_eq!(contraction.varied, vec![false, false, true]);
        let loss = &g.saved[1];
        assert_eq!(loss.varied, vec![true]);
        assert!(loss.groupings.is_empty());

        // The contraction reads w as a wrt table and pixels as constant data.
        let leaves: Vec<Input> = contraction.input.slots.iter().map(|s| s.input).collect();
        assert!(leaves.contains(&Input::Const), "{leaves:?}");
        assert!(
            leaves.iter().any(|l| matches!(l, Input::Table(_))),
            "{leaves:?}"
        );
        // The loss reads the saved contraction and b.
        let leaves: Vec<Input> = loss.input.slots.iter().map(|s| s.input).collect();
        assert!(leaves.contains(&Input::Saved(0)), "{leaves:?}");

        // The output reads the saved loss, and only it.
        assert_eq!(g.output.slots.len(), 1);
        assert_eq!(g.output.slots[0].input, Input::Saved(1));
        assert_eq!(g.output_names, vec!["loss"]);
    }
}

#[tokio::test]
async fn a_wrt_tables_dims_are_every_other_column() {
    let wrt = [ColumnRef::new("w", "val"), ColumnRef::new("b", "VAL")];
    for g in both(LAYER, &wrt).await {
        let w = g.tables.iter().find(|t| t.names == ["w"]).unwrap();
        assert_eq!(w.values, vec![2]);
        assert_eq!(w.dims, vec![0, 1]);
        let b = g.tables.iter().find(|t| t.names == ["b"]).unwrap();
        assert_eq!(b.values, vec![1], "matched case-insensitively");
    }
}

#[tokio::test]
async fn expressions_are_renumbered_onto_the_rebuilt_region() {
    // The loss sums tanh(c.z + b.val). Wherever the producer computed that
    // (inside the measure, or in a projection below it), it now reads the
    // saved contraction and the table b, and every expression column
    // reads only columns to its left.
    let wrt = [ColumnRef::new("w", "val"), ColumnRef::new("b", "val")];
    for g in both(LAYER, &wrt).await {
        let s = &g.saved[1].input;
        assert_eq!(s.defs.len(), s.varied.len());
        for (c, d) in s.defs.iter().enumerate() {
            if let Def::Expr(e) = d {
                for f in ddx_ad::expr::fields_of(e).unwrap() {
                    assert!(f < c, "column {c} reads column {f}");
                }
            }
        }
        let arg = match &g.saved[1].measures[0].arguments[0].arg_type {
            Some(ddx_ad::substrait::proto::function_argument::ArgType::Value(e)) => e,
            other => panic!("{other:?}"),
        };
        // Follow the measure argument down to leaf columns.
        let mut leaves = std::collections::BTreeSet::new();
        let mut stack = ddx_ad::expr::fields_of(arg).unwrap();
        while let Some(f) = stack.pop() {
            match &s.defs[f] {
                Def::Input { slot, .. } => {
                    leaves.insert(s.slots[*slot].input);
                }
                Def::Expr(e) => stack.extend(ddx_ad::expr::fields_of(e).unwrap()),
                other => panic!("{other:?}"),
            }
        }
        assert!(leaves.contains(&Input::Saved(0)), "{leaves:?}");
        assert!(
            leaves.iter().any(|l| matches!(l, Input::Table(_))),
            "{leaves:?}"
        );
    }
}

#[tokio::test]
async fn a_wrt_typo_lists_what_the_query_reads() {
    let ctx = ctx().await;
    let plan = substrait_of(&ctx, LAYER, true).await;
    let err = Forward::new(&plan, &[ColumnRef::new("weights", "val")]).unwrap_err();
    assert!(matches!(err, AdError::UnknownWrt(_)), "{err}");
    let msg = err.to_string();
    assert!(
        msg.contains("weights") && msg.contains("\"pixels\""),
        "{msg}"
    );

    let err = Forward::new(&plan, &[ColumnRef::new("w", "value")]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("no column `value`") && msg.contains("\"val\""),
        "{msg}"
    );
}

#[tokio::test]
async fn a_stop_gradient_makes_a_column_constant() {
    let sql = "SELECT SUM(val * ddx_stop_gradient(val)) AS s, \
                      SUM(ddx_stop_gradient(val)) AS t FROM w";
    for g in both(sql, &[ColumnRef::new("w", "val")]).await {
        assert_eq!(g.saved[0].varied, vec![true, false]);
    }
}

#[tokio::test]
async fn a_rank_is_refused_but_what_it_ranks_is_not() {
    // Ranking rows by a value that carries gradient is fine: the rows the
    // filter keeps carry it on. Only the rank itself, which has no
    // derivative, refuses gradient.
    let sql = "WITH r AS (SELECT i, o, val, \
                 ROW_NUMBER() OVER (PARTITION BY i ORDER BY val DESC) AS rk FROM w) \
               SELECT SUM(val) AS s FROM r WHERE rk = 1";
    for g in both(sql, &[ColumnRef::new("w", "val")]).await {
        let s = &g.saved[0].input;
        for (c, d) in s.defs.iter().enumerate() {
            let refused = s.refusals[c].is_some();
            assert_eq!(
                refused,
                matches!(d, Def::Window { .. }),
                "column {c}: {d:?}"
            );
        }
    }
}

#[tokio::test]
async fn an_outer_joins_null_side_is_refused() {
    let sql = "SELECT SUM(COALESCE(b.val, 0.0) * p.x) AS s \
               FROM pixels p LEFT JOIN b ON p.i = b.o";
    for g in both(sql, &[ColumnRef::new("b", "val")]).await {
        let s = &g.saved[0].input;
        let refused: Vec<usize> = (0..s.defs.len())
            .filter(|&c| s.refusals[c].is_some())
            .collect();
        assert!(!refused.is_empty());
        for c in refused {
            assert!(matches!(s.defs[c], Def::Input { .. }), "{:?}", s.defs[c]);
        }
    }
}

#[tokio::test]
async fn a_query_that_reads_no_wrt_table_is_refused() {
    let ctx = ctx().await;
    let plan = substrait_of(&ctx, "SELECT SUM(x) AS s FROM pixels", true).await;
    let err = Forward::new(&plan, &[ColumnRef::new("w", "val")]).unwrap_err();
    assert!(matches!(err, AdError::UnknownWrt(_)), "{err}");
}
