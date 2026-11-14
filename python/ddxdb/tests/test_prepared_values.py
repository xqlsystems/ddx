# SPDX-FileCopyrightText: 2026 OpenAI
#
# SPDX-License-Identifier: Apache-2.0

"""The primal of a prepared derivative follows the current input rows."""

import datafusion
import pyarrow as pa
import pytest

import ddxdb
from ddxdb import ad


def test_prepared_values_are_readable_until_release_and_refresh_on_rerun():
    ctx = datafusion.SessionContext()
    ad.register_stop_gradient(ctx)
    sql = (
        "WITH loss AS (SELECT SUM(val * val) AS l FROM w) "
        "SELECT i, grad(loss, val) AS g FROM w ORDER BY i"
    )
    ctx.register_record_batches("w", [pa.table({"i": [0, 1], "val": [2.0, 3.0]}).to_batches()])
    prepared = ddxdb.prepare(ad.DataFusionBackend(ctx), [sql])
    with pytest.raises(RuntimeError, match="run the prepared program"):
        _ = prepared.values
    for values in ([2.0, 3.0], [-1.0, 4.0]):
        ctx.deregister_table("w")
        ctx.register_record_batches("w", [pa.table({"i": [0, 1], "val": values}).to_batches()])
        with prepared.run() as run:
            [primal] = prepared.values
            assert ctx.sql(primal.query()).to_arrow_table().column(0).to_pylist() == [sum(v * v for v in values)]
            assert ctx.sql(run.statements[0]).to_arrow_table().to_pydict() == {
                "i": [0, 1], "g": [2 * v for v in values]
            }
            query = primal.query()
        with pytest.raises(Exception):
            ctx.sql(query).collect()
