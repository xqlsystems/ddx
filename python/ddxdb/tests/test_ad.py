# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""ddxdb.ad on DataFusion: grad in SQL and as a program, vjp, typed refusals.

The comparison with jax.grad on real models lives in the repo's oracle suite
(tests/test_v2_jax.py); these check the binding.
"""

import pyarrow as pa
import pytest

import ddxdb


@pytest.fixture
def ad():
    pytest.importorskip("datafusion")
    from ddxdb import ad

    return ad


def table(i, val):
    return pa.table({"i": pa.array(i, pa.int64()), "val": pa.array(val, pa.float64())})


@pytest.fixture
def ctx(ad):
    import datafusion

    c = datafusion.SessionContext()
    ad.register_stop_gradient(c)
    c.register_record_batches("w", [table([0, 1, 2], [1.0, -2.0, 0.5]).to_batches()])
    c.register_record_batches("b", [table([0, 1, 2], [0.25, 3.0, -1.0]).to_batches()])
    return c


def pairs(df):
    t = df.to_arrow_table()
    return sorted(zip(t.column(0).to_pylist(), t.column(1).to_pylist()))


def test_grad_in_sql_is_shaped_like_its_table(ad, ctx):
    df = ad.sql(ctx, "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w) SELECT * FROM grad(loss, w.val)")
    assert pairs(df) == [(0, 3.0), (1, 12.0), (2, 0.75)]


def test_an_sgd_step_is_a_join(ad, ctx):
    df = ad.sql(
        ctx,
        """WITH loss AS (SELECT SUM(val * val) AS l FROM w)
           SELECT w.i, w.val - 0.25 * g.val AS val
           FROM w JOIN grad(loss, w.val) g ON w.i = g.i""",
    )
    assert pairs(df) == [(0, 0.5), (1, -1.0), (2, 0.25)]


def test_statements_sharing_a_loss_share_its_program(ad, ctx):
    loss = "WITH loss AS (SELECT SUM(w.val * w.val * b.val) AS l FROM w JOIN b ON w.i = b.i) "
    update_w = loss + "SELECT w.i, w.val - 0.5 * g.val AS v FROM w JOIN grad(loss, w.val) g ON w.i = g.i"
    update_b = loss + "SELECT b.i, b.val - 0.5 * g.val AS v FROM b JOIN grad(loss, b.val) g ON b.i = g.i"
    fw, fb = ad.sql_all(ctx, [update_w, update_b])
    w, b = [1.0, -2.0, 0.5], [0.25, 3.0, -1.0]
    assert pairs(fw) == [(i, w[i] - 0.5 * 2 * w[i] * b[i]) for i in range(3)]
    assert pairs(fb) == [(i, b[i] - 0.5 * w[i] * w[i]) for i in range(3)]


def test_case_variants_of_a_column_keep_the_dims(ad, ctx):
    df = ad.sql(
        ctx,
        """WITH loss AS (SELECT SUM(val * val) AS l FROM w)
           SELECT a.i, a.val + b.val AS v
           FROM grad(loss, w.val) a JOIN grad(loss, W.VAL) b ON a.i = b.i""",
    )
    assert pairs(df) == [(0, 4.0), (1, -8.0), (2, 2.0)]


def test_stop_gradient_on_a_real_column(ad, ctx):
    ctx.register_record_batches(
        "h", [pa.table({"i": pa.array([0, 1], pa.int64()), "x": pa.array([1.5, 2.5], pa.float32())}).to_batches()]
    )
    df = ad.sql(
        ctx,
        """WITH loss AS (SELECT SUM(w.val * ddx_stop_gradient(h.x)) AS l FROM w JOIN h ON w.i = h.i)
           SELECT * FROM grad(loss, w.val)""",
    )
    assert pairs(df) == [(0, 1.5), (1, 2.5), (2, 0.0)]


def test_context_sql_understands_grad(ad):
    pytest.importorskip("datafusion")
    ctx = ddxdb.Context()
    ctx.register_record_batches("w", [table([0, 1], [3.0, 4.0]).to_batches()])
    df = ctx.sql("WITH loss AS (SELECT SUM(val * val) AS l FROM w) SELECT i, val FROM grad(loss, w.val)")
    assert pairs(df) == [(0, 6.0), (1, 8.0)]


def test_grad_as_a_program_can_be_rerun_after_the_table_changes(ad, ctx):
    program = ad.grad(ctx, "SELECT SUM(val * val) AS loss FROM w", [("w", "val")])
    ad.run(ctx, program)
    assert pairs(ctx.table(program.gradients[0].step)) == [(0, 2.0), (1, -4.0), (2, 1.0)]
    assert ctx.table(program.value).to_arrow_table().column(0)[0].as_py() == 1.0 + 4.0 + 0.25
    ctx.deregister_table("w")
    ctx.register_record_batches("w", [table([0, 1, 2], [5.0, 6.0, 7.0]).to_batches()])
    ad.run(ctx, program)
    assert pairs(ctx.table(program.gradients[0].step)) == [(0, 10.0), (1, 12.0), (2, 14.0)]


def test_vjp_pulls_a_cotangent_back(ad, ctx):
    program = ad.vjp(ctx, "SELECT i, val * val AS s FROM w", [("w", "val")])
    assert program.cotangent == ("i", "s")
    ctx.register_record_batches(ad.COTANGENT, [pa.table({"i": pa.array([0, 1, 2], pa.int64()), "s": [1.0, 10.0, 100.0]}).to_batches()])
    ad.run(ctx, program)
    assert pairs(ctx.table(program.gradients[0].step)) == [(0, 2.0), (1, -40.0), (2, 100.0)]


def test_refusals_are_typed(ad, ctx):
    with pytest.raises(ddxdb.NotScalar):
        ad.grad(ctx, "SELECT i, val FROM w", [("w", "val")])
    with pytest.raises(ddxdb.UnknownColumn, match="weights"):
        ad.grad(ctx, "SELECT SUM(val) AS loss FROM w", [("weights", "val")])
    with pytest.raises(ddxdb.UnsupportedExpression):
        ad.grad(ctx, "SELECT STDDEV(val) AS loss FROM w", [("w", "val")])
    assert issubclass(ddxdb.NotScalar, ddxdb.DdxError)
    assert issubclass(ddxdb.UnknownColumn, ddxdb.DdxError)
