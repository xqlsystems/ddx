// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Contractions: `SUM(a.val * b.val)` over a join, which is broadcast, map and
//! reduce composed, checked against finite differences up to nn.py's
//! two-layer shape.

mod common;

use common::ad::{check_gradients, Table};
use datafusion::prelude::SessionContext;
use ddx_ad::ColumnRef;

fn ctx() -> SessionContext {
    let ctx = SessionContext::new();
    ddx_datafusion::register_stop_gradient(&ctx);
    ctx
}

/// A deterministic spread of values in [-0.5, 0.5).
fn value(seed: usize) -> f64 {
    ((seed * 7919 + 13) % 1000) as f64 / 1000.0 - 0.5
}

/// A matrix as a tidy table `name(dims..., val)`.
fn matrix(name: &'static str, dims: [&'static str; 2], shape: [usize; 2], seed: usize) -> Table {
    let mut rows = Vec::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            rows.push(vec![i as f64, j as f64, value(seed + i * shape[1] + j)]);
        }
    }
    Table {
        name,
        columns: vec![(dims[0], "BIGINT"), (dims[1], "BIGINT"), ("val", "DOUBLE")],
        rows,
    }
}

#[tokio::test]
async fn a_matrix_product_differentiates_into_both_operands() {
    // C = A @ B, loss = sum(tanh(C)): Ā = C̄ Bᵀ and B̄ = Aᵀ C̄, both
    // contractions themselves.
    check_gradients(
        &ctx(),
        "WITH c AS (SELECT a.i, b.k, SUM(a.val * b.val) AS val \
                    FROM a JOIN b ON a.j = b.j GROUP BY a.i, b.k) \
         SELECT SUM(tanh(val)) AS loss FROM c",
        &[
            matrix("a", ["i", "j"], [3, 4], 1),
            matrix("b", ["j", "k"], [4, 2], 50),
        ],
        &[ColumnRef::new("a", "val"), ColumnRef::new("b", "val")],
    )
    .await;
}

/// nn.py's tables, small: `pixels(sample, height, width, images)`,
/// `labels(sample, out, y)` one-hot, `weight(layer, inp, out, val)` and
/// `bias(layer, out, val)` holding both layers.
fn nn_tables() -> Vec<Table> {
    let (samples, side, hidden, classes) = (3, 2, 3, 2);
    let mut pixels = Vec::new();
    for s in 0..samples {
        for h in 0..side {
            for w in 0..side {
                pixels.push(vec![
                    s as f64,
                    h as f64,
                    w as f64,
                    value(s * 10 + h * side + w),
                ]);
            }
        }
    }
    let mut labels = Vec::new();
    for s in 0..samples {
        for o in 0..classes {
            labels.push(vec![
                s as f64,
                o as f64,
                if s % classes == o { 1.0 } else { 0.0 },
            ]);
        }
    }
    let mut weight = Vec::new();
    let mut bias = Vec::new();
    for (layer, (inp, out)) in [(side * side, hidden), (hidden, classes)]
        .into_iter()
        .enumerate()
    {
        for i in 0..inp {
            for o in 0..out {
                weight.push(vec![
                    layer as f64,
                    i as f64,
                    o as f64,
                    value(100 + layer * 50 + i * out + o),
                ]);
            }
        }
        for o in 0..out {
            bias.push(vec![
                layer as f64,
                o as f64,
                value(300 + layer * 10 + o) / 5.0,
            ]);
        }
    }
    vec![
        Table {
            name: "pixels",
            columns: vec![
                ("sample", "BIGINT"),
                ("height", "BIGINT"),
                ("width", "BIGINT"),
                ("images", "DOUBLE"),
            ],
            rows: pixels,
        },
        Table {
            name: "labels",
            columns: vec![("sample", "BIGINT"), ("out", "BIGINT"), ("y", "DOUBLE")],
            rows: labels,
        },
        Table {
            name: "weight",
            columns: vec![
                ("layer", "BIGINT"),
                ("inp", "BIGINT"),
                ("out", "BIGINT"),
                ("val", "DOUBLE"),
            ],
            rows: weight,
        },
        Table {
            name: "bias",
            columns: vec![("layer", "BIGINT"), ("out", "BIGINT"), ("val", "DOUBLE")],
            rows: bias,
        },
    ]
}

/// nn.py's forward pass, written as one query as nn.py writes it: a computed
/// input index, the layer picked in the join condition,
/// a bias added by a second join, tanh, then a squared-error loss.
const TWO_LAYERS: &str = "\
WITH c0 AS ( \
  SELECT a.sample, w.out, SUM(a.val * w.val) AS z \
  FROM (SELECT sample, height * 2 + width AS inp, images AS val FROM pixels) a \
  JOIN weight w ON a.inp = w.inp AND w.layer = 0 \
  GROUP BY a.sample, w.out), \
fwd0 AS ( \
  SELECT c0.sample, c0.out, tanh(c0.z + b.val) AS val \
  FROM c0 JOIN bias b ON c0.out = b.out AND b.layer = 0), \
c1 AS ( \
  SELECT a.sample, w.out, SUM(a.val * w.val) AS z \
  FROM (SELECT sample, out AS inp, val FROM fwd0) a \
  JOIN weight w ON a.inp = w.inp AND w.layer = 1 \
  GROUP BY a.sample, w.out), \
logits AS ( \
  SELECT c1.sample, c1.out, c1.z + b.val AS z \
  FROM c1 JOIN bias b ON c1.out = b.out AND b.layer = 1) \
SELECT SUM(power(logits.z - l.y, 2)) / 3.0 AS loss \
FROM logits JOIN labels l ON logits.sample = l.sample AND logits.out = l.out";

#[tokio::test]
async fn nn_py_two_layers_with_every_weight_in_one_table() {
    let grads = check_gradients(
        &ctx(),
        TWO_LAYERS,
        &nn_tables(),
        &[
            ColumnRef::new("weight", "val"),
            ColumnRef::new("bias", "val"),
        ],
    )
    .await;
    // Every weight of both layers got a gradient row, keyed by (layer, inp, out).
    assert_eq!(grads["weight"].len(), 4 * 3 + 3 * 2);
    assert!(grads["weight"].iter().any(|r| r[0] == 0.0 && r[3] != 0.0));
    assert!(grads["weight"].iter().any(|r| r[0] == 1.0 && r[3] != 0.0));
}

#[tokio::test]
async fn the_input_can_be_differentiated_too() {
    // attention_ad_spike.py differentiates with respect to X, not only the
    // weights: nothing makes a parameter special.
    check_gradients(
        &ctx(),
        TWO_LAYERS,
        &nn_tables(),
        &[ColumnRef::new("pixels", "images")],
    )
    .await;
}
