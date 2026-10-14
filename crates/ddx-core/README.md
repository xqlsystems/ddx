# ddx-core

The scalar derivative rules of [`ddx`](https://github.com/xqlsystems/ddx),
"autograd for composable databases." `ddx-ad` differentiates a whole query
over Substrait; for each expression in it (a projection's `x * y`, a
`tanh(z)`), it asks this crate for the partial derivatives.

```rust
use ddx_core::sqlparser::dialect::GenericDialect;
use ddx_core::sqlparser::parser::Parser;
use ddx_core::{ColRef, Ddx};

let e = Parser::new(&GenericDialect {}).try_with_sql("sin(x)")?.parse_expr()?;
let d = Ddx::new().differentiate(&e, &ColRef::bare("x"))?;
assert_eq!(d.to_string(), "cos(x)");
```

The rules work on [`sqlparser::ast::Expr`](https://docs.rs/sqlparser),
**re-exported** as `ddx_core::sqlparser` so a caller links the same version.

## What it supports

`+ - * /`; the unary chain rule for the trig / inverse-trig / exp / log /
hyperbolic set plus `abs`; `power` with a constant base or exponent. Custom
unary rules are registrable (`ddx.register("myfn", rule)`: the rule supplies
`f'(u)`, the engine applies the chain rule). Anything else is a typed
`DiffError`, never a silently-wrong number.

## Correctness properties worth knowing

- **Ambiguity is a hard error, not a guess**: a `wrt` that can't be pinned
  syntactically (`a.x * b.x` with respect to `x`) errors rather than
  differentiating the wrong column.
- **`div` forces floating-point division** (`CAST(<numerator> AS DOUBLE)`), so
  integer columns don't silently truncate the derivative.
- **0/1-folding follows JAX's `Zero`-tangent convention** and differs from
  unfolded SQL only on NULL-bearing rows (documented, tested).

Licensed under Apache-2.0.
