<!--
SPDX-FileCopyrightText: 2026 OpenAI
SPDX-License-Identifier: Apache-2.0
-->

# Exact GELU and reusable primal outputs for SQL training

These changes let [xarray-sql](https://github.com/xqlsystems/xarray-sql),
a SQL interface to labeled arrays, train a
[nanoGPT](https://github.com/karpathy/nanoGPT) character model through
ddx-generated gradients. ddx remains the automatic differentiation engine:
the downstream GPU executor evaluates its SQL programs and does not replace
them with a framework backward pass.

## Error-function differentiation

The scalar registry now supports `erf`, the error function used in exact
GELU: `GELU(x) = x * (1 + erf(x / sqrt(2))) / 2`. Its derivative is
`2 / sqrt(pi) * exp(-x*x)`, composed with the existing chain rule.
The SQL plan reader/writer preserves `erf` in generated programs, so a
derivative containing the original primal function can be serialized and
executed. No tanh approximation is substituted for exact GELU.

The engine executing the SQL must register an `erf` implementation. This
change adds differentiation and round-tripping, not an engine-specific
SQL function. xarray-sql registers the DataFusion function and an
equivalent trusted array primitive; replacing that function invalidates
array lowering permission there.

The Rust scalar-rule test checks chain-rule composition and the analytic
second derivative, rather than only matching one first derivative. Existing
unsupported-expression and derivative semantics remain in effect. Exact
GELU support is a feature/correctness change; no isolated speedup or memory
reduction was measured for the rule itself.

