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
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
import pyarrow as pa


def _fixture(xs: np.ndarray, ys: np.ndarray) -> pa.Table:
    """The two-column table the harness differentiates over.

    `i` exists so results can be ordered back into the caller's row order —
    neither engine promises to preserve input order without an ORDER BY.
    """
    return pa.table(
        {
            "i": pa.array(np.arange(len(xs)), type=pa.int64()),
            "x": pa.array(np.asarray(xs, dtype=np.float64)),
            "y": pa.array(np.asarray(ys, dtype=np.float64)),
        }
    )


def _duckdb_column(sql: str, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    import duckdb

    con = duckdb.connect(":memory:")
    try:
        con.register("t", _fixture(xs, ys))
        rows = con.sql(sql).fetchall()
    finally:
        con.close()
    return np.array([np.nan if r[0] is None else float(r[0]) for r in rows])


def _datafusion_column(sql: str, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    import datafusion

    ctx = datafusion.SessionContext()
    ctx.register_record_batches("t", [_fixture(xs, ys).to_batches()])
    values = [
        v
        for batch in ctx.sql(sql).collect()
        for v in batch.column(0).to_pylist()
    ]
    return np.array([np.nan if v is None else float(v) for v in values])


@dataclasses.dataclass(frozen=True)
class Engine:
    """An engine to check a rewrite against, and the ddx dialect that targets it."""

    name: str
    dialect: str
    column: Callable[[str, np.ndarray, np.ndarray], np.ndarray]

    def __str__(self) -> str:  # keeps pytest ids readable
        return self.name


ENGINES = [
    Engine("duckdb", "duckdb", _duckdb_column),
    Engine("datafusion", "datafusion", _datafusion_column),
]
