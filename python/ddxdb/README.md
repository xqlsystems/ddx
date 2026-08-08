# ddxdb

Write calculus directly in SQL and let the database evaluate the derivative, row
by row, alongside everything else:

```sql
SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g
```

`grad` and `jvp` are **markers**, not row functions. They are rewritten away into
ordinary derivative SQL *before* the engine sees them, so what runs is a plain
expression — the relational equivalent of `jax.vmap(jax.grad(f))`, with the rows
as the batch dimension.

This is the Python distribution of [`ddx`](https://github.com/xqlsystems/ddx), a
thin wrapper over the `ddx-core` engine.

## Install

```bash
pip install ddxdb                  # the rewrite, no engine
pip install "ddxdb[datafusion]"    # + DataFusion
pip install "ddxdb[duckdb]"        # + DuckDB
```

There are no required runtime dependencies. `rewrite_sql` is text in, text out,
and the engine wrappers import their engine lazily — so installing `ddxdb` never
pulls in an engine you don't use.

## Three ways in

**Just the rewrite.** Works with any engine, because the output is only SQL:

```python
import ddxdb

ddxdb.rewrite_sql("SELECT grad(sin(x), x) AS d FROM t")
# 'SELECT (cos(x)) AS d FROM t'
```

**DataFusion**, via a drop-in `SessionContext` wrapper whose `.sql()` rewrites
first. Every other attribute is forwarded, so it stands in for a
`SessionContext` anywhere one is expected:

```python
ctx = ddxdb.Context()
ctx.sql("SELECT grad(x * x, x) AS d FROM t").collect()      # → 2x
ctx = ddxdb.Context(existing_session_context)               # or wrap your own
```

**DuckDB**, client-side — no extension needed:

```python
ddxdb.duckdb_sql("SELECT grad(sin(x), x) AS d FROM t", con).fetchall()
```

Because the rewrite happens on *your* connection, this sees your temp tables,
session settings, and open transaction. The in-database `ddx('<sql>')` table
function cannot: it executes on a separate inner connection.

## What you can write

`+ - * /`; the chain rule for the trig / inverse-trig / exp / log / hyperbolic
set plus `abs`; `power` with a constant base or exponent. Higher order falls out
of nesting — `grad(grad(f, x), x)` just works. Differentiating through an
aggregate is linearity, so the marker goes *inside* it, which is what makes a
gradient-descent step expressible in SQL:

```sql
SELECT theta - 0.01 * AVG(grad(loss, theta)) FROM batch
```

A marker rewrites in place, so it is legal anywhere a scalar expression is —
including inside a recursive CTE, which is how a whole training loop fits in one
query.

## Two other functions

```python
ddxdb.differentiate_sql("x * y", "x")     # 'y' — the derivative as text
ddxdb.explain(sql)                        # what the rewrite would do, without running it
```

`explain` returns the rewritten statement plus one entry per marker, giving the
marker as written and the derivative it becomes.

## Errors are typed

An unsupported construct is always an error, never a silently wrong number —
this is a numerical-correctness library, and a plausible-looking wrong
derivative is the worst thing it could produce. The kind of failure is a class,
so you can catch the one you can act on:

```python
try:
    ddxdb.rewrite_sql(query)
except ddxdb.UnsupportedExpression:
    ...   # no rule for something in there — fall back
except ddxdb.AmbiguousColumn:
    ...   # the query needs a qualifier — a fix the caller makes
```

All of them derive from `ddxdb.DdxError`. The full set is
`UnsupportedExpression`, `InvalidMarker`, `AmbiguousColumn`,
`ProjectionBoundary` and `SqlParseError`.

## One thing to know

**`grad` does not see through a CTE or a view.** Differentiation stops at column
references, so a column computed upstream is a constant to it:

```sql
WITH v AS (SELECT x, sin(x) AS s FROM t)
SELECT grad(s * x, x) FROM v       -- ds/dx is treated as 0
```

That is defensible relational semantics and a real trap, so ddx refuses the
worst case rather than quietly dropping the term: referencing a computed CTE
alias as a non-`wrt` term raises `ProjectionBoundary` and tells you to
differentiate inside the CTE instead. Differentiating *with respect to* such an
alias is fine — every occurrence is then the differentiation leaf, and
`grad(s * s, s)` is exactly `2s`.

## Development

```bash
pip install maturin pytest
maturin develop --uv
python -m pytest tests/
```
