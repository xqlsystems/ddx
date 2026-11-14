# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""A prepared derivative's primal follows the current input rows."""

import datafusion
import pyarrow as pa
import pytest

import ddxdb
from ddxdb import ad


def _register(ctx, values):
    ctx.register_record_batches("w", [pa.table({"i": [0, 1], "val": values}).to_batches()])


def test_primals_are_readable_until_release_and_follow_the_rows():
    ctx = datafusion.SessionContext()
    ad.register_stop_gradient(ctx)
    sql = (
        "WITH loss AS (SELECT SUM(val * val) AS l FROM w) "
        "SELECT i, grad(loss, val) AS g FROM w ORDER BY i"
    )
    _register(ctx, [2.0, 3.0])
    prepared = ddxdb.prepare(ad.DataFusionBackend(ctx), [sql])
    with pytest.raises(RuntimeError, match="before reading their primals"):
        _ = prepared.primals
    for values in ([2.0, 3.0], [-1.0, 4.0]):
        ctx.deregister_table("w")
        _register(ctx, values)
        with prepared.run() as run:
            [primal] = prepared.primals
            loss = ctx.sql(primal.query()).to_arrow_table().column(0).to_pylist()
            assert loss == [sum(v * v for v in values)]
            grads = ctx.sql(run.statements[0]).to_arrow_table().to_pydict()
            assert grads == {"i": [0, 1], "g": [2 * v for v in values]}
            query = primal.query()
        with pytest.raises(Exception):
            ctx.sql(query).collect()
