# ddxdb

**Status: scaffold (M2).** The Python distribution of
[`ddx`](https://github.com/xqlsystems/ddx) — a PyO3/maturin wheel wrapping
[`ddx-core`](../../crates/ddx-core).

Planned surface (design.md §3.4, milestone M2):

- `ddxdb.rewrite_sql(sql: str, dialect: str = "datafusion") -> str` — the
  source-to-source marker rewrite, exposed directly.
- A `Context.sql()` shim: a thin wrapper around a `datafusion.SessionContext`
  that calls `rewrite_sql` before handing the plain SQL to the stock context to
  plan. This is what [xarray-sql](https://github.com/xqlsystems/xarray-sql)
  pulls in as an optional extra (`pip install "xarray-sql[ddx]"`), so autograd
  is opt-in and costs nothing for users who don't ask for it.
- A DuckDB-python client-side path: preprocess the string before
  `duckdb.sql(...)` (the zero-hook fallback, available day one — design.md
  §3.4).

Nothing is built here yet. M0 delivers only `ddx-core`; this directory holds
the layout seam of design.md §6 so the wheel has a home. When M2 lands, this
becomes a maturin project (`[build-system] requires = ["maturin"]`) with a
`Cargo.toml` depending on `ddx-core` + `pyo3`.

The **oracle for its tests is JAX** (`jax.grad`), not numpy (design.md §5).
