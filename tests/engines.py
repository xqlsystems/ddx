# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run one SQL statement on a real engine and get the column back.

The oracle compares numbers an engine actually produced, not SQL text: a rewrite
that looks right and plans to the wrong expression is exactly the failure worth
catching, and it is invisible to a string comparison.

Rows arrive as Arrow rather than as a `VALUES` literal on purpose. DuckDB reads
a bare `1.5` in SQL as `DECIMAL(2,1)`, so a literal-built fixture would hand the
two engines different types for the same table and make any disagreement
ambiguous. An Arrow table pins every column to float64 on both.

Results keep NULL and NaN apart. Collapsing SQL NULL into NaN would be
convenient — the numeric suites want an array of floats — but it erases the
distinction between "no data here" and "not a number", and one of the
conventions this repo pins is precisely that a missing input stays missing
rather than becoming a value. A comparison that cannot see the difference
cannot test for it.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

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
    """The two-column table the harness differentiates over.

    `i` exists so results can be ordered back into the caller's row order —
    neither engine promises to preserve input order without an ORDER BY.
    """
    x = _column(xs)
    return pa.table(
        {
            "i": pa.array(np.arange(len(x)), type=pa.int64()),
            "x": x,
            "y": _column(ys),
        }
    )


def _duckdb_values(sql: str, xs: np.ndarray, ys: np.ndarray) -> list:
    import duckdb

    con = duckdb.connect(":memory:")
    try:
        con.register("t", _fixture(xs, ys))
        rows = con.sql(sql).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def _datafusion_values(sql: str, xs: np.ndarray, ys: np.ndarray) -> list:
    import datafusion

    ctx = datafusion.SessionContext()
    ctx.register_record_batches("t", [_fixture(xs, ys).to_batches()])
    return [v for batch in ctx.sql(sql).collect() for v in batch.column(0).to_pylist()]


@dataclasses.dataclass(frozen=True)
class Engine:
    """An engine to check a rewrite against, and the ddx dialect that targets it."""

    name: str
    dialect: str
    values: Callable[[str, np.ndarray, np.ndarray], list]
    """The column exactly as the engine returned it — `None` for SQL NULL."""

    def column(self, sql: str, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The column as float64, with NULL as NaN.

        For the numeric suites, which compare magnitudes and have no opinion
        about missingness. Use :meth:`nulls` alongside it when the distinction
        matters — it is not recoverable from the array this returns.
        """
        return np.array(
            [np.nan if v is None else float(v) for v in self.values(sql, xs, ys)]
        )

    def nulls(self, sql: str, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Which rows came back as SQL NULL, as opposed to NaN or a number."""
        return np.array([v is None for v in self.values(sql, xs, ys)])

    def __str__(self) -> str:  # keeps pytest ids readable
        return self.name


ENGINES = [
    Engine("duckdb", "duckdb", _duckdb_values),
    Engine("datafusion", "datafusion", _datafusion_values),
]
