# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run ddx on a real engine and get each row's derivative back.

The oracle compares numbers an engine actually produced, not SQL text: a
program that looks right and plans to the wrong expression is exactly the
failure worth catching, and it is invisible to a string comparison.

A function `f(x, y)` of one row is checked through a whole query: the gradient
of `SUM(f(x, y))` with respect to the column `t.x` is, at row `i`, `∂f/∂x` at
that row's `(x, y)`, the relational form of `jax.vmap(jax.grad(f))`. ddx takes
it in SQL on DataFusion, the engine its Python package runs programs on.

Rows arrive as Arrow rather than as a `VALUES` literal, so every column is
float64 whatever the literal would have been read as.

Results keep NULL and NaN apart. Collapsing SQL NULL into NaN would be
convenient — the numeric suites want an array of floats — but it erases the
distinction between "no data here" and "not a number", and one of the
conventions this repo pins is precisely that a missing input stays missing
rather than becoming a value. A comparison that cannot see the difference
cannot test for it.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def _column(values) -> pa.Array:
    """A float64 Arrow column from either a numpy array or an Arrow array.

    Accepting an Arrow array is how a caller expresses a **NULL** input: numpy
    float64 has no null, only NaN, so a fixture built from numpy alone can never
    pose the "this row has no value" question at all.
    """
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        return values.cast(pa.float64())
    return pa.array(np.asarray(values, dtype=np.float64))


def _fixture(xs, ys) -> pa.Table:
    """The table `t(i, x, y)` the harness differentiates over.

    `i` keys the rows, so each has its own gradient, and orders the results
    back into the caller's row order.
    """
    x = _column(xs)
    return pa.table(
        {
            "i": pa.array(np.arange(len(x)), type=pa.int64()),
            "x": x,
            "y": _column(ys),
        }
    )


def context(xs, ys):
    """A DataFusion context with `t(i, x, y)` registered."""
    import datafusion

    ctx = datafusion.SessionContext()
    ctx.register_record_batches("t", [_fixture(xs, ys).to_batches()])
    return ctx


def grad_sql(body: str, wrt: str) -> str:
    """The statement whose column is `∂body/∂wrt` at each row of `t`."""
    return (
        f"WITH loss AS (SELECT SUM({body}) AS l FROM t) "
        f"SELECT i, grad(loss, {wrt}) AS d FROM t ORDER BY i"
    )


def grad_values(body: str, wrt: str, xs, ys) -> list:
    """`∂body/∂wrt` at each row, exactly as the engine returned it — `None`
    for SQL NULL."""
    from ddxdb import ad

    ctx = context(xs, ys)
    table = ad.sql(ctx, grad_sql(body, wrt)).to_arrow_table()
    return table.column("d").to_pylist()


def grad_column(body: str, wrt: str, xs, ys) -> np.ndarray:
    """:func:`grad_values` as float64, with NULL as NaN.

    For the numeric suites, which compare magnitudes and have no opinion about
    missingness. Use :func:`grad_values` when the distinction matters — it is
    not recoverable from the array this returns.
    """
    return np.array(
        [np.nan if v is None else float(v) for v in grad_values(body, wrt, xs, ys)]
    )


def _dims(wrt: str) -> str:
    """The columns of `t` other than `wrt`: its dims, when `wrt` is
    differentiated along."""
    return "i, y" if wrt == "x" else "i, x"


def jvp_sql(body: str, wrt: str) -> str:
    """The statement whose column is the tangent of `body` at each row of `t`,
    along a tangent of `wrt` that is the other column's value there."""
    other = "y" if wrt == "x" else "x"
    return (
        f"WITH tangent AS (SELECT {_dims(wrt)}, {other} AS {wrt} FROM t) "
        f"SELECT i, jvp({body}, t.{wrt}, tangent) AS d FROM t ORDER BY i"
    )


def second_sql(body: str, wrt: str) -> str:
    """The statement whose column is `∂²body/∂wrt²` at each row of `t`: the
    tangent of the gradient along a tangent of 1, a Hessian-vector product
    whose Hessian is diagonal, since each row's `body` reads that row alone.
    `jvp` of a `SELECT` over `grad`, calls nested as JAX's transformations
    are."""
    return (
        f"WITH loss AS (SELECT SUM({body}) AS l FROM t), "
        f"g AS (SELECT i, grad(loss, {wrt}) AS g FROM t), "
        f"ones AS (SELECT {_dims(wrt)}, 1.0 AS {wrt} FROM t) "
        f"SELECT i, jvp(g, t.{wrt}, ones) AS d FROM g ORDER BY i"
    )


def column(sql: str, xs, ys) -> np.ndarray:
    """The column `d` of `sql`, run by ddx over `t`, as float64 with NULL as
    NaN."""
    from ddxdb import ad

    ctx = context(xs, ys)
    got = ad.sql(ctx, sql).to_arrow_table().column("d").to_pylist()
    return np.array([np.nan if v is None else float(v) for v in got])


def values(sql: str, xs, ys) -> list:
    """The first column of a plain statement over `t`."""
    ctx = context(xs, ys)
    return [v for b in ctx.sql(sql).collect() for v in b.column(0).to_pylist()]
