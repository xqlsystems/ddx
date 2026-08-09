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
pip install ddxdb                  # everything below except Context
pip install "ddxdb[datafusion]"    # + the DataFusion Context
```

## `rewrite_sql` is the whole library

Text in, text out — so it works with **any** engine that accepts SQL. Pass the
result wherever you would have passed the original:

```python
import ddxdb

ddxdb.rewrite_sql("SELECT grad(sin(x), x) AS d FROM t")
# 'SELECT (cos(x)) AS d FROM t'

con.sql(ddxdb.rewrite_sql(q, "duckdb"))        # DuckDB
session.sql(ddxdb.rewrite_sql(q, "spark"))     # Spark
ctx.sql(ddxdb.rewrite_sql(q))                  # DataFusion
```

Accepted dialects: `generic`, `datafusion`, `postgres`, `ansi`, `snowflake`,
`oracle`, `duckdb`, `mysql`, `sqlite`, `bigquery`, `redshift`, `hive`, `spark`,
`databricks`, `mssql`, `teradata`, `clickhouse`.

Pick the one that matches the engine you will run on, not just the one that
parses your SQL. The dialect also decides which column an identifier *names*,
and engines disagree three ways:

| | unquoted `X` means | so `"X"` is |
|---|---|---|
| Postgres, DataFusion, generic, ansi | `"x"` | a different column |
| Snowflake, Oracle | `"X"` | the same column |
| DuckDB, Spark, MySQL, SQLite, BigQuery, Redshift, Hive, Databricks, SQL Server, Teradata | any casing | the same column |
| ClickHouse | `X` exactly | the same column, and `"x"` is not |

Getting this wrong does not raise. `grad("X" * "X", X)` is `2X` on Snowflake and
`0` on Postgres — both correct, for different engines — so ddx keeps a table
rather than a default, and refuses a dialect whose rule it has not established.

Because the rewrite happens in *your* process, on *your* connection, it sees
your temp tables, session settings and open transaction — anything the query
itself could see. A rewrite performed *inside* the database, on a connection of
its own, would not.

## `Context`, for DataFusion

A real `SessionContext` subclass whose `.sql()` rewrites first — every inherited
method, property and constructor argument works unchanged:

```python
ctx = ddxdb.Context()
ctx.sql("SELECT grad(x * x, x) AS d FROM t").collect()      # → 2x
```

It lives in `ddxdb.datafusion` (a subclass needs its base class at import time,
so it cannot sit beside `rewrite_sql` without dragging DataFusion in) and is
re-exported as `ddxdb.Context`, imported on first use. `import ddxdb` still needs
no engine.

There is sugar for DataFusion and not for other engines because DataFusion is
ddx's integration target. Everything else uses the one-liner above, which is why
there are no per-engine helpers here to drift out of date.

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
```

The escape hatch, for assembling SQL where a marker cannot reach — inside a
recursive term you are building programmatically, or a query some other tool
emits. Everything else should use `rewrite_sql`.

```python
ddxdb.supported_functions()               # ['abs', 'acos', 'asin', ...]
```

The unary functions ddx has a rule for, read from the engine rather than
restated. Note that a name being present does not by itself make an *expression*
differentiable — the surrounding constructs matter too — so catching the typed
error below remains the general answer to "can ddx handle this?".

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
