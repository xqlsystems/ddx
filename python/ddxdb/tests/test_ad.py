# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""ddxdb.ad on DataFusion: grad and jvp in SQL and as programs, vjp,
composition, typed refusals.

The comparison with jax.grad on real models lives in the repo's oracle suite
(tests/test_query_jax.py); these check the binding.
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


def test_grad_in_sql_is_a_column_of_its_table_s_rows(ad, ctx):
    df = ad.sql(ctx, "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w) SELECT i, grad(loss, val) FROM w")
    assert pairs(df) == [(0, 3.0), (1, 12.0), (2, 0.75)]


def test_an_sgd_step_is_one_expression(ad, ctx):
    df = ad.sql(
        ctx,
        """WITH loss AS (SELECT SUM(val * val) AS l FROM w)
           SELECT i, val - 0.25 * grad(loss, val) AS val FROM w""",
    )
    assert pairs(df) == [(0, 0.5), (1, -1.0), (2, 0.25)]


def test_statements_sharing_a_loss_share_its_program(ad, ctx):
    loss = "WITH loss AS (SELECT SUM(w.val * w.val * b.val) AS l FROM w JOIN b ON w.i = b.i) "
    update_w = loss + "SELECT i, val - 0.5 * grad(loss, val) AS v FROM w"
    update_b = loss + "SELECT i, val - 0.5 * grad(loss, val) AS v FROM b"
    fw, fb = ad.sql_all(ctx, [update_w, update_b])
    w, b = [1.0, -2.0, 0.5], [0.25, 3.0, -1.0]
    assert pairs(fw) == [(i, w[i] - 0.5 * 2 * w[i] * b[i]) for i in range(3)]
    assert pairs(fb) == [(i, b[i] - 0.5 * w[i] * w[i]) for i in range(3)]


def test_case_variants_of_a_column_keep_the_dims(ad, ctx):
    df = ad.sql(
        ctx,
        """WITH loss AS (SELECT SUM(val * val) AS l FROM w)
           SELECT i, grad(loss, w.val) + grad(loss, W.VAL) AS v FROM w""",
    )
    assert pairs(df) == [(0, 4.0), (1, -8.0), (2, 2.0)]


def test_stop_gradient_on_a_real_column(ad, ctx):
    ctx.register_record_batches(
        "h", [pa.table({"i": pa.array([0, 1], pa.int64()), "x": pa.array([1.5, 2.5], pa.float32())}).to_batches()]
    )
    df = ad.sql(
        ctx,
        """WITH loss AS (SELECT SUM(w.val * ddx_stop_gradient(h.x)) AS l FROM w JOIN h ON w.i = h.i)
           SELECT i, grad(loss, val) FROM w""",
    )
    assert pairs(df) == [(0, 1.5), (1, 2.5), (2, 0.0)]


def test_context_sql_understands_grad(ad):
    pytest.importorskip("datafusion")
    ctx = ddxdb.Context()
    ctx.register_record_batches("w", [table([0, 1], [3.0, 4.0]).to_batches()])
    df = ctx.sql("WITH loss AS (SELECT SUM(val * val) AS l FROM w) SELECT i, grad(loss, val) FROM w")
    assert pairs(df) == [(0, 6.0), (1, 8.0)]
def test_grad_as_a_program_can_be_rerun_after_the_table_changes(ad, ctx):
    program = ad.grad(ctx, "SELECT SUM(val * val) AS loss FROM w", "w.val")
    out = ad.run(ctx, program)
    assert pairs(out.gradients[0]) == [(0, 2.0), (1, -4.0), (2, 1.0)]
    assert out.value.to_pydict() == {"loss": [1.0 + 4.0 + 0.25]}
    ctx.deregister_table("w")
    ctx.register_record_batches("w", [table([0, 1, 2], [5.0, 6.0, 7.0]).to_batches()])
    again = ad.run(ctx, program)
    assert pairs(again.gradients[0]) == [(0, 10.0), (1, 12.0), (2, 14.0)]
    # The first run's outputs hold their own rows.
    assert pairs(out.gradients[0]) == [(0, 2.0), (1, -4.0), (2, 1.0)]


def test_vjp_pulls_a_cotangent_back(ad, ctx):
    program = ad.vjp(ctx, "SELECT i, val * val AS s FROM w", "w.val")
    [cotangent] = program.inputs
    assert (cotangent.of, cotangent.columns, cotangent.keys) == (None, ("i", "s"), 1)
    c = pa.table({"i": pa.array([0, 1, 2], pa.int64()), "s": [1.0, 10.0, 100.0]})
    out = ad.run(ctx, program, [c])
    assert pairs(out.gradients[0]) == [(0, 2.0), (1, -40.0), (2, 100.0)]
    # A DataFrame works too, and extra columns are ignored.
    out = ad.run(ctx, program, [ctx.sql("SELECT 'x' AS note, i, 1.0 AS s FROM w")])
    assert pairs(out.gradients[0]) == [(0, 2.0), (1, -4.0), (2, 1.0)]
    with pytest.raises(ValueError, match=cotangent.name):
        ad.run(ctx, program)
    with pytest.raises(TypeError, match="ddxdb.run"):
        ad.run(LoggingBackend(ad, ctx), program, [c])
    assert not ddx_tables(ctx)


def test_wrt_is_one_column_or_several(ad, ctx):
    loss = "SELECT SUM(w.val * b.val) AS loss FROM w JOIN b ON w.i = b.i"
    one = ad.run(ctx, ad.grad(ctx, loss, "w.val"))
    both = ad.run(ctx, ad.grad(ctx, loss, ["w.val", "b.val"]))
    assert pairs(one.gradients[0]) == pairs(both.gradients[0]) == [(0, 0.25), (1, 3.0), (2, -1.0)]
    assert pairs(both.gradients[1]) == [(0, 1.0), (1, -2.0), (2, 0.5)]
    with pytest.raises(ValueError, match="table.column"):
        ad.grad(ctx, loss, "val")
    with pytest.raises(TypeError, match="table.column"):
        ad.grad(ctx, loss, [("w", "val")])


def test_rewrite_releases_its_tables_when_the_block_ends(ad, ctx):
    backend = ad.DataFusionBackend(ctx)
    before = set(ctx.catalog().schema().names())
    sql = "WITH f AS (SELECT SUM(val * val) AS l FROM w) SELECT i, grad(f, val) AS g FROM w"
    with ddxdb.rewrite(backend, [sql, "SELECT 1 AS one"]) as rewritten:
        assert "grad(" not in rewritten.statements[0]
        assert rewritten.statements[1] == "SELECT 1 AS one"
        assert set(rewritten.tables) <= set(ctx.catalog().schema().names())
        assert pairs(ctx.sql(rewritten.statements[0])) == [(0, 2.0), (1, -4.0), (2, 1.0)]
    assert set(ctx.catalog().schema().names()) == before


def test_rewrite_refuses_one_statement_not_in_a_list(ad, ctx):
    sql = "WITH f AS (SELECT SUM(val * val) AS l FROM w) SELECT i, grad(f, val) AS g FROM w"
    with pytest.raises(TypeError, match=r"\[statement\]"):
        ddxdb.rewrite(ad.DataFusionBackend(ctx), sql)
    with pytest.raises(TypeError, match=r"\[statement\]"):
        ad.sql_all(ctx, sql)


def test_an_error_in_the_block_is_not_hidden_by_a_failed_drop(ad, ctx):
    class FailingDrops(ad.DataFusionBackend):
        # Drops succeed while the programs run, and fail once the block has
        # begun.
        failing = False

        def drop_table(self, name):
            super().drop_table(name)
            if self.failing:
                raise RuntimeError("drop failed")

    sql = "WITH f AS (SELECT SUM(val * val) AS l FROM w) SELECT i, grad(f, val) AS g FROM w"
    with pytest.raises(ValueError, match="in the block"):
        with ddxdb.rewrite(FailingDrops(ctx), [sql]) as rewritten:
            rewritten.backend.failing = True
            raise ValueError("in the block")
    # With nothing in flight, the drop's own error is raised.
    with pytest.raises(RuntimeError, match="drop failed"):
        with ddxdb.rewrite(FailingDrops(ctx), [sql]) as rewritten:
            rewritten.backend.failing = True


def test_an_unnamed_gradient_column_is_named_for_its_column(ad, ctx):
    df = ad.sql(ctx, "WITH f AS (SELECT SUM(val * val) AS l FROM w) SELECT i, grad(f, val) FROM w")
    assert df.schema().names == ["i", "val_grad"]


def test_refusals_are_typed(ad, ctx):
    with pytest.raises(ddxdb.NotScalar):
        ad.grad(ctx, "SELECT i, val FROM w", "w.val")
    with pytest.raises(ddxdb.UnknownColumn, match="weights"):
        ad.grad(ctx, "SELECT SUM(val) AS loss FROM w", "weights.val")
    with pytest.raises(ddxdb.UnsupportedExpression):
        ad.grad(ctx, "SELECT STDDEV(val) AS loss FROM w", "w.val")
    with pytest.raises(ddxdb.InvalidColumn, match="float"):
        ad.grad(ctx, "SELECT SUM(i) AS loss FROM w", "w.i")
    assert issubclass(ddxdb.NotScalar, ddxdb.DdxError)
    assert issubclass(ddxdb.UnknownColumn, ddxdb.DdxError)
    assert issubclass(ddxdb.InvalidColumn, ddxdb.DdxError)


def test_run_refuses_rows_that_share_their_dims(ad, ctx):
    ctx.register_record_batches("d", [table([0, 0, 1], [1.0, 2.0, 3.0]).to_batches()])
    program = ad.grad(ctx, "SELECT SUM(val * val) AS loss FROM d", "d.val")
    with pytest.raises(ddxdb.InvalidColumn, match="share their dims"):
        ad.run(ctx, program)


def ddx_tables(ctx):
    return sorted(n for n in ctx.catalog().schema().names() if n.startswith("__ddx_"))


def test_run_leaves_none_of_the_programs_tables(ad, ctx):
    program = ad.grad(ctx, "SELECT MAX(val) * SUM(val) AS loss FROM w", "w.val")
    assert list(program.intermediate_steps())
    out = ad.run(ctx, program)
    assert ddx_tables(ctx) == []
    assert out.value.to_pydict() == {"loss": [1.0 * -0.5]}  # MAX(w) * SUM(w)


def test_programs_do_not_share_tables(ad, ctx):
    a = ad.grad(ctx, "SELECT SUM(val * val) AS loss FROM w", "w.val")
    b = ad.grad(ctx, "SELECT SUM(val * val) AS loss FROM b", "b.val")
    assert not {s.name for s in a.steps()} & {s.name for s in b.steps()}
    out = ad.run(ctx, a)
    ad.run(ctx, b)
    assert pairs(out.gradients[0]) == [(0, 2.0), (1, -4.0), (2, 1.0)]


def test_sql_leaves_no_tables_behind(ad, ctx):
    df = ad.sql(ctx, "WITH loss AS (SELECT SUM(val * val) AS l FROM w) SELECT i, grad(loss, val) FROM w")
    assert ddx_tables(ctx) == []
    assert pairs(df) == [(0, 2.0), (1, -4.0), (2, 1.0)]


def test_tables_whose_names_join_alike_keep_their_own_gradients(ad):
    import datafusion

    ctx = datafusion.SessionContext()
    for statement in [
        "CREATE SCHEMA a_b",
        "CREATE SCHEMA a",
        "CREATE TABLE a_b.c (i BIGINT, val DOUBLE) AS VALUES (0, 1.0)",
        "CREATE TABLE a.b_c (i BIGINT, val DOUBLE) AS VALUES (0, 10.0)",
    ]:
        ctx.sql(statement).collect()
    df = ad.sql(
        ctx,
        """WITH loss AS (SELECT SUM(p.val * p.val) + SUM(q.val * q.val) AS l
                         FROM a_b.c p CROSS JOIN a.b_c q)
           SELECT p.i, grad(loss, p.val) * 1000.0 + grad(loss, q.val) AS v
           FROM a_b.c p CROSS JOIN a.b_c q""",
    )
    assert pairs(df) == [(0, 2020.0)]


def test_grad_in_sql_of_a_table_with_capitals(ad):
    # Found by the soak in ad_simulation.rs (#92): a gradient step named with
    # the table's capitals was registered under a folded name and not found
    # again.
    import datafusion

    ctx = datafusion.SessionContext()
    ctx.sql('CREATE TABLE "W" (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)').collect()
    df = ad.sql(ctx, 'WITH loss AS (SELECT SUM(val * val) AS l FROM "W") SELECT i, grad(loss, val) FROM "W"')
    assert pairs(df) == [(0, 2.0), (1, 4.0)]


def test_no_gradient_through_a_row_an_aggregate_skips(ad, ctx):
    # Found by the soak in ad_simulation.rs (#89): SUM skips row 1, whose
    # term w + q is NULL.
    ctx.register_record_batches("q", [table([0, 1, 2], [5.0, None, 1.0]).to_batches()])
    df = ad.sql(
        ctx,
        "WITH loss AS (SELECT SUM(w.val + q.val) AS l FROM w JOIN q ON w.i = q.i) SELECT i, grad(loss, val) FROM w",
    )
    assert pairs(df) == [(0, 1.0), (1, 0.0), (2, 1.0)]


def test_vjp_refuses_a_cotangent_whose_keys_repeat(ad, ctx):
    # Found by the soak in ad_simulation.rs (#98): a repeated key added its
    # rows' cotangents.
    program = ad.vjp(ctx, "SELECT i, val * val AS s FROM w", "w.val")
    c = pa.table({"i": pa.array([0, 0, 1, 2], pa.int64()), "s": [1.0, 1.0, 1.0, 1.0]})
    with pytest.raises(ddxdb.InvalidColumn, match="share their keys"):
        ad.run(ctx, program, [c])
    assert not ddx_tables(ctx)


class LoggingBackend:
    """A Backend over DataFusion that logs what it is asked, and can fail."""

    def __init__(self, ad, ctx, fail_on=None):
        self.inner = ad.DataFusionBackend(ctx)
        self.log = []
        self.fail_on = fail_on

    def select_all(self, name):
        self.log.append(("schema", name))
        return self.inner.select_all(name)

    def returns_rows(self, plan):
        self.log.append(("check",))
        return self.inner.returns_rows(plan)

    def materialize(self, name, plan):
        self.log.append(("write", name))
        if name == self.fail_on:
            raise RuntimeError("disk full")
        self.inner.materialize(name, plan)

    def drop_table(self, name):
        self.log.append(("drop", name))
        self.inner.drop_table(name)


def test_any_backend_runs_a_program(ad, ctx):
    # Composability re-review (#81): the runner takes a Backend, not only a
    # DataFusion context, and asks each table's schema once per write.
    prog = ad.grad(ctx, "SELECT SUM(val * val) / COUNT(val) AS l FROM w", "w.val")
    backend = LoggingBackend(ad, ctx)
    ddxdb.run(backend, prog)
    kinds = [e[0] for e in backend.log]
    assert kinds.index("check") < kinds.index("write")
    assert [e[1] for e in backend.log if e[0] == "drop"] == [s.name for s in prog.intermediate_steps()]
    asked = set()
    for entry in backend.log:
        if entry[0] == "write":
            asked.discard(entry[1])
        elif entry[0] == "schema":
            assert entry[1] not in asked, backend.log
            asked.add(entry[1])
    got = ctx.sql(f'SELECT i, val FROM "{prog.gradients[0].step}"')
    assert dict(pairs(got)) == pytest.approx({0: 2.0 / 3, 1: -4.0 / 3, 2: 1.0 / 3})


def test_a_failed_run_leaves_none_of_the_programs_tables(ad, ctx):
    prog = ad.grad(ctx, "SELECT SUM(val * val) / COUNT(val) AS l FROM w", "w.val")
    backend = LoggingBackend(ad, ctx, fail_on=prog.gradients[0].step)
    with pytest.raises(RuntimeError, match="disk full"):
        ddxdb.run(backend, prog)
    assert not ddx_tables(ctx)


def test_a_fixed_namespace_gives_the_same_program(ad, ctx):
    # The same plan, not the same query: DataFusion's producer can number a
    # query's functions differently from one call to the next.
    from datafusion.substrait import Serde

    q = "SELECT SUM(val * val) AS l FROM w"
    plan = Serde.serialize_bytes(q, ctx)
    a = ddxdb.grad_plan(plan, "w.val", namespace="__ddx_py_")
    b = ddxdb.grad_plan(plan, "w.val", namespace="__ddx_py_")
    assert a == b and a.value.step == "__ddx_py_value"
    with pytest.raises(ddxdb.DdxError, match="namespace"):
        ad.grad(ctx, q, "w.val", namespace="__ddx_Py_")


def test_grad_in_sql_is_found_in_the_dialect_it_is_written_in(ad, ctx):
    stmt = "WITH loss AS (SELECT SUM(val * val) AS l FROM w) SELECT i, grad(loss, val) FROM w"
    assert pairs(ad.sql(ctx, stmt, dialect="duckdb")) == [(0, 2.0), (1, -4.0), (2, 1.0)]
    with pytest.raises(ValueError, match="dialect"):
        ad.sql(ctx, stmt, dialect="klingon")


def test_not_in_over_a_subquery_is_refused(ad, ctx):
    # DataFusion's Substrait producer drops NOT IN's NULL semantics (#104):
    # refused, rather than differentiated as a different query.
    loss = "SELECT SUM(val * val) AS l FROM w WHERE i NOT IN (SELECT i FROM b WHERE val > 1.0)"
    with pytest.raises(ddxdb.UnsupportedExpression, match="NOT IN"):
        ad.grad(ctx, loss, "w.val")
    with pytest.raises(ddxdb.UnsupportedExpression, match="NOT IN"):
        ad.sql(ctx, f"WITH loss AS ({loss}) SELECT i, grad(loss, val) FROM w")
    exists = (
        "SELECT SUM(val * val) AS l FROM w WHERE NOT EXISTS "
        "(SELECT 1 FROM b WHERE b.i = w.i AND b.val > 1.0)"
    )
    ad.grad(ctx, exists, "w.val")




def test_grad_through_a_union_all(ad, ctx):
    # Each row of the union sends its gradient to the table it came from;
    # the rows of i that both tables have land in one group.
    loss = """WITH u AS (SELECT i, val FROM w UNION ALL SELECT i + 1 AS i, 2 * val AS val FROM b),
              s AS (SELECT i, SUM(val) AS t FROM u GROUP BY i),
              loss AS (SELECT SUM(t * t) AS l FROM s) """
    gw, gb = ad.sql_all(
        ctx,
        [loss + "SELECT i, grad(loss, val) FROM w", loss + "SELECT i, grad(loss, val) FROM b"],
    )
    w, b = [1.0, -2.0, 0.5], [0.25, 3.0, -1.0]
    t = {i: w[i] if i < 3 else 0.0 for i in range(4)}
    for i in range(3):
        t[i + 1] += 2 * b[i]
    assert dict(pairs(gw)) == pytest.approx({i: 2 * t[i] for i in range(3)})
    assert dict(pairs(gb)) == pytest.approx({i: 4 * t[i + 1] for i in range(3)})

def test_a_filter_on_a_gradients_dims_computes_only_those_rows(ad, ctx):
    # The WHERE on w's dims is pushed into the program (ddx_ad::Options::restrict);
    # the rows read are the unrestricted gradient's.
    loss = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w)"
    full = pairs(ad.sql(ctx, f"{loss} SELECT i, grad(loss, val) FROM w"))
    some = pairs(ad.sql(ctx, f"{loss} SELECT i, grad(loss, val) FROM w WHERE i >= 1"))
    assert some == [p for p in full if p[0] >= 1]


# jvp, and composition. The closed forms here use Σ val³ over w = (1, -2, 0.5):
# ∇ = 3 val², H = diag(6 val).

LOSS = "SELECT SUM(val * val * val) AS l FROM w"


def tangent_of_w(input_table, values):
    """``input_table``, a tangent of w, as ``values`` by row of w."""
    assert input_table.of == "w" and input_table.columns == ("i", "val") and input_table.keys == 1
    return table([0, 1, 2], values)


def register_tangent(ctx, input_table, values):
    """Register ``input_table`` for a run on a Backend, which reads it by name."""
    ctx.register_record_batches(input_table.name, [tangent_of_w(input_table, values).to_batches()])


def test_jvp_of_a_query_is_its_value_and_its_tangent(ad, ctx):
    program = ad.jvp(ctx, LOSS, "w.val")
    assert isinstance(program, ddxdb.ForwardProgram) and not program.gradients
    out = ad.run(ctx, program, [tangent_of_w(program.inputs[0], [1.0, 1.0, 1.0])])
    [tangent] = program.value.tangents
    assert tangent.column == "l" and tangent.tangent in program.value.columns
    row = out.value.to_pydict()
    assert row["l"] == [1.0 - 8.0 + 0.125]
    assert row["l_tangent"] == pytest.approx([3.0 * (1.0 + 4.0 + 0.25)])
    assert ddx_tables(ctx) == []


def test_jvp_of_a_grad_program_is_a_hessian_vector_product(ad, ctx):
    grad = ad.grad(ctx, LOSS, "w.val")
    hvp = ad.jvp(ctx, grad, "w.val")
    [g] = hvp.gradients
    assert g.of == "w"
    out = ad.run(ctx, hvp, [tangent_of_w(hvp.inputs[0], [1.0, 2.0, 3.0])])
    got = out.gradients[0].select("i", "val_tangent")
    assert dict(pairs(got)) == pytest.approx({0: 6.0, 1: -24.0, 2: 9.0})


def test_vjp_of_a_jvp_program_is_a_hessian_vector_product_too(ad, ctx):
    jvp = ad.jvp(ctx, LOSS, "w.val")
    program = ad.vjp(ctx, jvp, "w.val")
    tangent, cotangent = program.inputs
    assert tangent.name == jvp.inputs[0].name and cotangent.of is None
    # 0 on the loss, 1 on its tangent: the gradient of ∇l·v.
    seed = pa.table({c: [0.0 if c == "l" else 1.0] for c in cotangent.columns})
    out = ad.run(ctx, program, [tangent_of_w(tangent, [1.0, 2.0, 3.0]), seed])
    assert dict(pairs(out.gradients[0])) == pytest.approx({0: 6.0, 1: -24.0, 2: 9.0})


def test_any_backend_runs_a_forward_program(ad, ctx):
    program = ad.jvp(ctx, LOSS, "w.val")
    register_tangent(ctx, program.inputs[0], [1.0, 0.0, 0.0])
    backend = LoggingBackend(ad, ctx)
    ddxdb.run(backend, program)
    assert ("write", program.value.step) in backend.log
    with pytest.raises(TypeError, match="Program"):
        ddxdb.run(backend, b"not a program")


def test_jvp_in_sql_is_a_column_of_its_select_s_rows(ad, ctx):
    df = ad.sql(
        ctx,
        """WITH v AS (SELECT i, 1.0 AS val FROM w)
           SELECT SUM(val * val * val) AS l, jvp(SUM(val * val * val), w.val, v) AS l_tangent
           FROM w""",
    )
    t = df.to_arrow_table()
    assert t.column("l_tangent")[0].as_py() == pytest.approx(3.0 * 5.25)
    assert not ddx_tables(ctx)


def test_jvp_in_sql_is_one_select_per_table_and_context_sql_reads_it(ad, ctx):
    from ddxdb import Context

    c = Context()
    c.register_record_batches("w", [table([0, 1, 2], [1.0, -2.0, 0.5]).to_batches()])
    c.register_record_batches("b", [table([0, 1, 2], [0.25, 3.0, -1.0]).to_batches()])
    c.register_record_batches("db", [table([0, 1, 2], [1.0, 1.0, 1.0]).to_batches()])
    df = c.sql(
        """WITH f AS (SELECT w.i, w.val * b.val AS y FROM w JOIN b ON w.i = b.i),
                dw AS (SELECT i, 1.0 AS val FROM w),
                j AS (SELECT i, jvp(y, w.val, dw) AS t FROM f),
                k AS (SELECT i, jvp(y, b.val, db) AS t FROM f)
           SELECT j.i, j.t + k.t FROM j JOIN k ON j.i = k.i ORDER BY j.i"""
    )
    # d(w·b) along (1, 1) is b + w: tangents along two tables add.
    assert pairs(df) == [(0, 1.25), (1, 1.0), (2, -0.5)]


def test_a_fixed_namespace_gives_the_same_forward_program(ad, ctx):
    from datafusion.substrait import Serde

    plan = Serde.serialize_bytes(LOSS, ctx)
    a = ddxdb.jvp_plan(plan, "w.val", namespace="__ddx_py_")
    b = ddxdb.jvp_plan(plan, "w.val", namespace="__ddx_py_")
    assert a == b and a.value.step == "__ddx_py_jvp"
    with pytest.raises(TypeError, match="Substrait"):
        ddxdb.jvp_plan("SELECT 1", "w.val")


def test_a_query_deeper_than_datafusion_python_decodes_is_split_and_runs(ad, ctx):
    # datafusion-python decodes Substrait with protobuf's default nesting
    # limit, two levels per relation; a jvp through forty stacked aggregates
    # would pass it, so ddx splits its step into parts, and it runs: its
    # tangent along v is <grad, v>.
    ctes = ["h0 AS (SELECT i, val AS v FROM w)"]
    for k in range(1, 40):
        ctes.append(f"h{k} AS (SELECT i, SUM(tanh(v) + 0.5 * v) AS v FROM h{k - 1} GROUP BY i)")
    query = f"WITH {', '.join(ctes)} SELECT SUM(v) AS l FROM h39"
    program = ad.jvp(ctx, query, "w.val")
    assert any("_part_" in s.name for s in program.steps())
    out = ad.run(ctx, program, [tangent_of_w(program.inputs[0], [1.0, 2.0, 3.0])])
    got = out.value.to_pydict()["l_tangent"][0]
    grad = ad.grad(ctx, query, "w.val")
    g = pairs(ad.run(ctx, grad).gradients[0])
    assert got == pytest.approx(sum(v * d for (_, v), d in zip(g, [1.0, 2.0, 3.0])))


def test_a_step_whose_names_would_grow_large_runs_as_sql(ad, ctx, monkeypatch):
    # Each layer reads the one before twice, so datafusion-python's consumer
    # would name the last layer's column after 2^12 copies of the first:
    # those steps run as SQL, with the same gradient.
    import math

    ran, to_sql = [], ad.to_sql
    monkeypatch.setattr(ad, "to_sql", lambda plan, dialect: ran.append(plan) or to_sql(plan, dialect))
    ctes = ["h0 AS (SELECT i, val AS v FROM w)"]
    for k in range(1, 13):
        ctes.append(f"h{k} AS (SELECT i, sin(v) + 0.1 * v AS v FROM h{k - 1})")
    df = ad.sql(ctx, f"WITH {', '.join(ctes)}, loss AS (SELECT SUM(v) AS l FROM h12) SELECT i, grad(loss, val) FROM w")
    want = []
    for v in [1.0, -2.0, 0.5]:
        d = 1.0
        for _ in range(12):
            d *= math.cos(v) + 0.1
            v = math.sin(v) + 0.1 * v
        want.append(d)
    assert dict(pairs(df)) == pytest.approx(dict(enumerate(want)))
    assert ran


def test_a_large_step_sql_cannot_write_runs_on_substrait(ad, ctx, monkeypatch):
    # to_sql writes only the functions it knows; a step it refuses runs
    # through the consumer, however large its names.
    import math

    refused = []

    def refuse(plan, dialect):
        refused.append(plan)
        raise ad.UnsupportedExpression("a function to_sql does not know")

    monkeypatch.setattr(ad, "to_sql", refuse)
    ctes = ["h0 AS (SELECT i, val AS v FROM w)"]
    for k in range(1, 13):
        ctes.append(f"h{k} AS (SELECT i, sin(v) + 0.1 * v AS v FROM h{k - 1})")
    df = ad.sql(ctx, f"WITH {', '.join(ctes)}, loss AS (SELECT SUM(v) AS l FROM h12) SELECT i, grad(loss, val) FROM w")
    want = []
    for v in [1.0, -2.0, 0.5]:
        d = 1.0
        for _ in range(12):
            d *= math.cos(v) + 0.1
            v = math.sin(v) + 0.1 * v
        want.append(d)
    assert dict(pairs(df)) == pytest.approx(dict(enumerate(want)))
    assert refused


def test_vjp_in_sql_pulls_a_cotangent_back(ad, ctx):
    df = ad.sql(
        ctx,
        """WITH f AS (SELECT i, val * val AS s FROM w),
                c AS (SELECT i, CAST(i + 1 AS DOUBLE) AS s FROM w)
           SELECT i, vjp(f, val, c) FROM w""",
    )
    # s = val², so the pull-back of c is 2 val c.
    assert pairs(df) == [(0, 2.0), (1, -8.0), (2, 3.0)]
    assert not ddx_tables(ctx)


def test_calls_nest_in_sql(ad, ctx):
    # l = Σ val³: jvp of a gradient is H·v = 6 val v, here v = 1.
    got = ad.sql(
        ctx,
        "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), "
        "g AS (SELECT i, grad(loss, val) AS val FROM w), v AS (SELECT i, 1.0 AS val FROM w) "
        "SELECT i, jvp(val, w.val, v) AS val_tangent FROM g ORDER BY i",
    ).to_pydict()
    assert got["val_tangent"] == pytest.approx([6.0, -12.0, 3.0])
    assert not ddx_tables(ctx)




def test_jvp_along_a_number_is_the_derivative_row_by_row(ad, ctx):
    import math

    got = ad.sql(ctx, "SELECT i, val, jvp(sin(val), w.val, 1) AS d FROM w ORDER BY i").to_pydict()
    assert got["d"] == pytest.approx([math.cos(v) for v in got["val"]])


GRAD = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w) SELECT i, grad(loss, val) FROM w"


def program_of(df):
    """The prefix, ``__ddx_{id}_``, of the program tables ``df`` reads: two
    calls that read the same one ran the same program."""
    import re

    found = set(re.findall(r"__ddx_[0-9a-f]+_[0-9a-f]+_", df.logical_plan().display_indent()))
    assert len(found) == 1, found
    return found.pop()


def test_a_second_sql_call_runs_the_program_the_first_built(ad, ctx):
    first = ad.sql(ctx, GRAD)
    built = program_of(first)
    assert pairs(first) == [(0, 3.0), (1, 12.0), (2, 0.75)]
    # New values, the same schema: the same program, on the new values.
    ctx.deregister_table("w")
    ctx.register_record_batches("w", [table([0, 1, 2], [2.0, 0.0, -1.0]).to_batches()])
    second = ad.sql(ctx, GRAD)
    assert program_of(second) == built
    assert pairs(second) == [(0, 12.0), (1, 0.0), (2, 3.0)]


def test_a_table_of_another_schema_gets_its_program_built_again(ad, ctx):
    built = program_of(ad.sql(ctx, GRAD))
    schema = pa.schema([pa.field("i", pa.int64()), pa.field("val", pa.float64(), nullable=False)])
    ctx.deregister_table("w")
    ctx.register_record_batches("w", [table([0, 1, 2], [2.0, 0.0, -1.0]).cast(schema).to_batches()])
    df = ad.sql(ctx, GRAD)
    assert program_of(df) != built
    assert pairs(df) == [(0, 12.0), (1, 0.0), (2, 3.0)]


def test_rewrite_keeps_programs_only_for_a_backend_with_a_plan_key(ad, ctx):
    class NoKey:
        """DataFusionBackend without plan_key."""

        def __init__(self, ctx):
            self.inner = ad.DataFusionBackend(ctx)

        def __getattr__(self, name):
            if name == "plan_key":
                raise AttributeError(name)
            return getattr(self.inner, name)

    def prefix(backend, programs):
        with ddxdb.rewrite(backend, [GRAD], programs=programs) as r:
            return program_of(ctx.sql(r.statements[0]))

    programs = ddxdb.Programs()
    keyed = ad.DataFusionBackend(ctx)
    assert prefix(keyed, programs) == prefix(keyed, programs)
    unkeyed = ddxdb.Programs()
    assert prefix(NoKey(ctx), unkeyed) != prefix(NoKey(ctx), unkeyed)


def test_a_program_is_kept_again_only_once_its_tables_are_released(ad, ctx):
    programs = ddxdb.Programs()
    backend = ad.DataFusionBackend(ctx)
    with ddxdb.rewrite(backend, [GRAD], programs=programs):
        pass
    first = ddxdb.rewrite(backend, [GRAD], programs=programs)
    with ddxdb.rewrite(backend, [GRAD], programs=programs) as second:
        assert program_of(ctx.sql(second.statements[0])) != program_of(ctx.sql(first.statements[0]))
    # The first call's tables outlived the second's release.
    assert pairs(ctx.sql(first.statements[0])) == [(0, 3.0), (1, 12.0), (2, 0.75)]
    first.release()


def _model_view(k):
    ctes = ["l0 AS (SELECT i, SUM(val) AS v FROM w GROUP BY i)"]
    for n in range(1, k + 1):
        ctes.append(
            f"l{n} AS (SELECT a.i, SUM(a.v * b.v + 0.5 * a.v) AS v "
            f"FROM l{n - 1} a JOIN l{n - 1} b ON a.i = b.i GROUP BY a.i)"
        )
    return ctes


def test_a_model_that_is_a_view_has_its_layers_planned_once(ad, ctx):
    ctes = _model_view(3)
    ctx.sql(f"CREATE VIEW model AS WITH {', '.join(ctes)} SELECT i, v FROM l3").collect()
    statement = "WITH loss AS (SELECT SUM(v) AS l FROM model) SELECT i, grad(loss, val) FROM w"
    got = pairs(ad.sql(ctx, statement))
    # The same objective with the model written in, as CTEs the engine inlines.
    written = f"WITH {', '.join(ctes)}, loss AS (SELECT SUM(v) AS l FROM l3) SELECT i, grad(loss, val) FROM w"
    want = pairs(ad.sql(ctx, written))
    assert [i for i, _ in got] == [i for i, _ in want]
    assert all(abs(g - w) <= 1e-9 * max(abs(w), 1.0) for (_, g), (_, w) in zip(got, want))

    # The layers read twice are planned on their own: one placeholder each.
    class Recording(ad.DataFusionBackend):
        made = []

        def materialize(self, name, plan):
            Recording.made.append(name)
            super().materialize(name, plan)

    with ddxdb.rewrite(Recording(ctx), [statement], programs=ddxdb.Programs()):
        pass
    assert sum("cte_" in n for n in Recording.made) == 3, Recording.made


def test_a_view_without_sql_is_left_to_the_engine(ad, ctx):
    ctx.register_view("squares", ctx.sql("SELECT i, val * val AS v FROM w"))
    assert ad.DataFusionBackend(ctx).view_sql("squares") is None
    statement = "WITH loss AS (SELECT SUM(v) AS l FROM squares) SELECT i, grad(loss, val) FROM w"
    assert pairs(ad.sql(ctx, statement)) == [(0, 2.0), (1, -4.0), (2, 1.0)]


def test_a_view_in_a_context_without_information_schema_warns(ad):
    import datafusion

    c = datafusion.SessionContext(datafusion.SessionConfig())
    c.register_record_batches("w", [table([0, 1, 2], [1.0, -2.0, 0.5]).to_batches()])
    c.sql("CREATE VIEW squares AS SELECT i, val * val AS v FROM w").collect()
    statement = "WITH loss AS (SELECT SUM(v) AS l FROM squares) SELECT i, grad(loss, val) FROM w"
    with pytest.warns(UserWarning, match="information_schema"):
        assert pairs(ad.sql(c, statement)) == [(0, 2.0), (1, -4.0), (2, 1.0)]
    # A table is not a view: nothing to warn of.
    import warnings

    c2 = datafusion.SessionContext(datafusion.SessionConfig())
    c2.register_record_batches("w", [table([0, 1, 2], [1.0, -2.0, 0.5]).to_batches()])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        statement = "WITH loss AS (SELECT SUM(val * val) AS l FROM w) SELECT i, grad(loss, val) FROM w"
        assert pairs(ad.sql(c2, statement)) == [(0, 2.0), (1, -4.0), (2, 1.0)]


def test_a_view_s_name_is_resolved_with_the_session_s_case_rules(ad):
    # Unquoted names keep their case in this session, so `Model` and
    # `model` are two views.
    import datafusion

    config = (
        datafusion.SessionConfig()
        .with_information_schema(True)
        .set("datafusion.sql_parser.enable_ident_normalization", "false")
    )
    c = datafusion.SessionContext(config)
    c.sql("CREATE VIEW Model AS SELECT 1 AS v").collect()
    c.sql("CREATE VIEW model AS SELECT 2 AS v").collect()
    backend = ad.DataFusionBackend(c)
    assert "SELECT 1" in backend.view_sql("Model")
    assert "SELECT 2" in backend.view_sql("model")


def test_a_view_whose_sql_needs_coercion_has_its_sql_given(ad, ctx):
    # DataFusion keeps `val * CAST(2 AS DOUBLE)` for this view; its SQL
    # plans as `val * 2` until coerced, and still means the same.
    ctx.sql("CREATE VIEW m AS SELECT i, val * 2 AS v FROM w WHERE i > 0").collect()
    assert "val * 2" in ad.DataFusionBackend(ctx).view_sql("m")


def test_a_coerced_view_is_optimized_once_while_it_is_unchanged(ad, ctx, monkeypatch):
    # Its plans match only once optimized, which a large model's view takes
    # long to be; a training loop asks on every step.
    ctx.sql("CREATE VIEW inner_v AS SELECT i, val * 2 AS v FROM w").collect()
    ctx.sql("CREATE VIEW outer_v AS SELECT i, v * 3 AS v FROM inner_v").collect()
    real = ad._plan_under_alias
    optimizing = []

    def counted(c, sql, *, optimized: bool):
        if optimized:
            optimizing.append(sql)
        return real(c, sql, optimized=optimized)

    monkeypatch.setattr(ad, "_plan_under_alias", counted)
    backend = ad.DataFusionBackend(ctx)
    assert backend.view_sql("outer_v") is not None
    assert backend.view_sql("outer_v") is not None
    assert len(optimizing) == 2
    # The comparison depends on how the session optimizes: under another
    # setting it is made again.
    ctx.sql("SET datafusion.optimizer.max_passes = 1").collect()
    assert backend.view_sql("outer_v") is not None
    assert len(optimizing) == 4
    ctx.sql("SET datafusion.optimizer.max_passes = 3").collect()
    # Redefined beneath it, the view still runs the inner view it was made
    # with: its plans change, and are compared again.
    ctx.sql("CREATE OR REPLACE VIEW inner_v AS SELECT i, val * 4 AS v FROM w").collect()
    assert backend.view_sql("outer_v") is None
    assert len(optimizing) == 6


def test_a_view_that_reads_a_redefined_view_is_left_to_the_engine(ad, ctx):
    # DataFusion keeps the plan outer_v was made with, which reads the first
    # inner_v; its SQL now reads the second.
    ctx.sql("CREATE VIEW inner_v AS SELECT i, val AS v FROM w").collect()
    ctx.sql(
        "CREATE VIEW outer_v AS WITH a AS (SELECT i, v FROM inner_v) "
        "SELECT a.i, a.v * b.v AS v FROM a JOIN a b ON a.i = b.i"
    ).collect()
    backend = ad.DataFusionBackend(ctx)
    assert backend.view_sql("outer_v") is not None
    ctx.sql("CREATE OR REPLACE VIEW inner_v AS SELECT i, 3.0 * val AS v FROM w").collect()
    assert backend.view_sql("outer_v") is None
    statement = "WITH loss AS (SELECT SUM(v) AS l FROM outer_v) SELECT i, grad(loss, val) FROM w"
    assert pairs(ad.sql(ctx, statement)) == [(0, 2.0), (1, -4.0), (2, 1.0)]


def test_a_view_in_another_schema_is_not_read_for_a_table_of_its_name(ad, ctx):
    # `v` in a query is public.v, a table; other.v is a view of that name,
    # and DataFusion does not read it for `v`.
    ctx.sql("CREATE SCHEMA other").collect()
    ctx.sql("CREATE VIEW other.v AS SELECT i, 100.0 * val AS val FROM w").collect()
    ctx.register_record_batches("v", [table([0, 1, 2], [1.0, 1.0, 1.0]).to_batches()])
    backend = ad.DataFusionBackend(ctx)
    assert backend.view_sql("v") is None
    assert backend.view_sql("other.v") is not None
    statement = "WITH loss AS (SELECT SUM(v.val * w.val) AS l FROM v JOIN w ON v.i = w.i) SELECT i, grad(loss, val) FROM w"
    assert pairs(ad.sql(ctx, statement)) == [(0, 1.0), (1, 1.0), (2, 1.0)]


def test_a_subquery_with_no_condition_on_the_outer_row_is_differentiated(ad, ctx):
    # DataFusion plans it as a semi join with no condition, which its
    # Substrait consumer refuses unless ddx writes one (ddx issue #171).
    statement = (
        "WITH loss AS (SELECT SUM(val * val) AS l FROM w WHERE EXISTS (SELECT 1 FROM b)) "
        "SELECT i, grad(loss, val) FROM w"
    )
    assert pairs(ad.sql(ctx, statement)) == [(0, 2.0), (1, -4.0), (2, 1.0)]


STEP = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w) SELECT i, val - 0.1 * grad(loss, val) AS val FROM w"
HVP = """WITH loss AS (SELECT SUM(val * val * val) AS l FROM w),
              g AS (SELECT i, grad(loss, val) AS val FROM w)
         SELECT i, jvp(val, w.val, b) AS hv FROM g"""


def test_a_prepared_step_reads_the_tables_rows_as_they_are_at_each_call(ad, ctx):
    step = ad.prepare(ctx, [STEP, HVP])
    for k in range(3):
        got = [pairs(df) for df in step()]
        want = [pairs(df) for df in ad.sql_all(ctx, [STEP, HVP])]
        assert got == want
        assert not [t for t in ctx.catalog().schema().names() if t.startswith("__ddx_")]
        # The next step's weights and a new direction, registered again as
        # a training loop registers them.
        ctx.deregister_table("w")
        ctx.register_record_batches("w", [table([i for i, _ in got[0]], [v for _, v in got[0]]).to_batches()])
        ctx.deregister_table("b")
        ctx.register_record_batches("b", [table([0, 1, 2], [1.0, float(k), -1.0]).to_batches()])


def test_a_later_prepared_call_asks_the_backend_only_to_run_plans(ad, ctx, monkeypatch):
    ctx.sql("CREATE VIEW m AS SELECT i, val FROM w").collect()
    asked = []
    for method in ["select_all", "plan_sql", "plan_key", "view_sql", "session_key", "returns_rows", "materialize"]:
        original = getattr(ad.DataFusionBackend, method)

        def spy(self, *args, _method=method, _original=original):
            asked.append(_method)
            return _original(self, *args)

        monkeypatch.setattr(ad.DataFusionBackend, method, spy)
    step = ad.prepare(ctx, ["WITH loss AS (SELECT SUM(val * val) AS l FROM m) SELECT i, grad(loss, val) FROM w"])
    assert pairs(step()[0]) == [(0, 2.0), (1, -4.0), (2, 1.0)]
    assert {"plan_sql", "view_sql", "select_all"} <= set(asked) and "plan_key" not in asked
    asked.clear()
    assert pairs(step()[0]) == [(0, 2.0), (1, -4.0), (2, 1.0)]
    assert set(asked) <= {"session_key", "returns_rows", "materialize"}, asked


def test_a_failed_prepared_call_leaves_nothing_and_the_next_runs(ad, ctx):
    step = ad.prepare(ctx, [STEP])
    step()
    ctx.deregister_table("w")
    ctx.register_record_batches("w", [table([0, 0], [1.0, 2.0]).to_batches()])
    with pytest.raises(ddxdb.InvalidColumn):
        step()
    assert not [t for t in ctx.catalog().schema().names() if t.startswith("__ddx_")]
    ctx.deregister_table("w")
    ctx.register_record_batches("w", [table([0, 1, 2], [3.0, 0.0, -1.0]).to_batches()])
    assert dict(pairs(step()[0])) == pytest.approx({0: 3.0 - 2.7, 1: 0.0, 2: -1.0 - 0.3})


def test_a_failed_first_prepared_call_prepares_nothing(ad, ctx):
    prepared = ddxdb.prepare(ad.DataFusionBackend(ctx), [STEP.replace("FROM w", "FROM missing", 1)])
    with pytest.raises(Exception, match="missing"):
        prepared.run()
    assert "not yet run" in repr(prepared)
    with pytest.raises(TypeError):
        ddxdb.prepare(ad.DataFusionBackend(ctx), STEP)


THROUGH_M = "WITH loss AS (SELECT SUM(val * val) AS l FROM m) SELECT i, grad(loss, val) FROM w"


def test_a_prepared_call_differentiates_a_redefined_view_as_it_is_now(ad, ctx):
    ctx.sql("CREATE VIEW m AS SELECT i, val FROM w").collect()
    step = ad.prepare(ctx, [THROUGH_M])
    assert pairs(step()[0]) == [(0, 2.0), (1, -4.0), (2, 1.0)]
    ctx.sql("CREATE OR REPLACE VIEW m AS SELECT i, 2 * val AS val FROM w").collect()
    assert pairs(step()[0]) == [(0, 8.0), (1, -16.0), (2, 4.0)]


def test_a_prepared_call_differentiates_a_view_as_datafusion_runs_it(ad, ctx):
    # DataFusion keeps the plan a view was made with: `m` reads `inner_m`
    # as it was until `m` is made again, with the same SQL.
    ctx.sql("CREATE VIEW inner_m AS SELECT i, val FROM w").collect()
    ctx.sql("CREATE VIEW m AS SELECT i, val FROM inner_m").collect()
    step = ad.prepare(ctx, [THROUGH_M])
    step()
    ctx.sql("CREATE OR REPLACE VIEW inner_m AS SELECT i, 2 * val AS val FROM w").collect()
    assert pairs(step()[0]) == pairs(ad.sql(ctx, THROUGH_M))
    ctx.sql("CREATE OR REPLACE VIEW m AS SELECT i, val FROM inner_m").collect()
    assert pairs(step()[0]) == pairs(ad.sql(ctx, THROUGH_M)) == [(0, 8.0), (1, -16.0), (2, 4.0)]


def test_a_prepared_call_after_a_table_changes_its_types_matches_sql_all(ad, ctx):
    step = ad.prepare(ctx, [STEP])
    step()
    ctx.deregister_table("w")
    w32 = pa.table({"i": pa.array([0, 1, 2], pa.int64()), "val": pa.array([1.0, -2.0, 0.5], pa.float32())})
    ctx.register_record_batches("w", [w32.to_batches()])
    assert pairs(step()[0]) == pairs(ad.sql_all(ctx, [STEP])[0])


@pytest.fixture
def compiling(ad, ctx):
    """``ctx``, able to compile plans (``ad.prepare(..., in_place=True)``)."""
    from datafusion.plan import ExecutionPlan

    if not hasattr(ExecutionPlan, "reset_state"):
        pytest.skip("needs the xqlsystems fork of datafusion (ExecutionPlan.reset_state)")
    for name in ad._DYNAMIC_FILTERS:
        ctx.sql(f"SET {name} = false").collect()
    return ctx


def writable(ctx, name, i, val):
    """Register ``name`` as ``table(i, val)`` over memory of its own, and
    return a function that writes new ``i`` and ``val`` into it."""
    columns = [pa.array(i, pa.int64()), pa.array(val, pa.float64())]
    buffers = [pa.allocate_buffer(8 * len(i)) for _ in columns]
    for buffer, column in zip(buffers, columns):
        memoryview(buffer)[:] = memoryview(column.buffers()[1])
    arrays = [pa.Array.from_buffers(c.type, len(c), [None, b]) for c, b in zip(columns, buffers)]
    ctx.deregister_table(name)
    ctx.register_record_batches(name, [[pa.RecordBatch.from_arrays(arrays, ["i", "val"])]])

    def write(i, val):
        memoryview(buffers[0]).cast("q")[:] = memoryview(pa.array(i, pa.int64()).buffers()[1]).cast("q")
        memoryview(buffers[1]).cast("d")[:] = memoryview(pa.array(val, pa.float64()).buffers()[1]).cast("d")

    return write


def test_a_compiled_step_reads_the_values_written_into_its_tables_at_each_call(ad, compiling, monkeypatch):
    write_w = writable(compiling, "w", [0, 1, 2], [1.0, -2.0, 0.5])
    write_b = writable(compiling, "b", [0, 1, 2], [0.25, 3.0, -1.0])
    asked = []
    for method in ["materialize", "returns_rows", "compile", "materialize_compiled", "returns_rows_compiled", "_physical"]:
        original = getattr(ad.DataFusionBackend, method)

        def spy(self, *args, _method=method, _original=original):
            asked.append(_method)
            return _original(self, *args)

        monkeypatch.setattr(ad.DataFusionBackend, method, spy)
    step = ad.prepare(compiling, [STEP, HVP], in_place=True)
    for k in range(3):
        asked.clear()
        got = [pairs(df) for df in step()]
        # Each plan is planned once, when it is compiling: none again later.
        later = {"materialize_compiled", "returns_rows_compiled"}
        assert set(asked) == (later | {"compile", "_physical"} if k == 0 else later), asked
        assert asked.count("_physical") == asked.count("compile")
        want = [pairs(df) for df in ad.sql_all(compiling, [STEP, HVP])]
        assert got == want
        assert not [t for t in compiling.catalog().schema().names() if t.startswith("__ddx_")]
        write_w([i for i, _ in got[0]], [v for _, v in got[0]])
        write_b([2, 1, 0], [1.0, float(k), -1.0])


def test_a_compiled_step_whose_rows_change_in_number_is_planned_again(ad, compiling):
    # The steps that save each group's sum have a row for each value of w
    # over 0: here 2, 3, 0, then 3.
    statement = (
        "WITH loss AS (SELECT SUM(l) AS l FROM "
        "(SELECT i, LN(SUM(val * val)) AS l FROM w WHERE val > 0 GROUP BY i)) "
        "SELECT i, grad(loss, val) AS g FROM w"
    )
    write = writable(compiling, "w", [0, 1, 2], [1.0, -2.0, 0.5])
    step = ad.prepare(compiling, [statement], in_place=True)
    for val in [[1.0, -2.0, 0.5], [1.0, 2.0, 0.5], [-1.0, -2.0, -0.5], [3.0, 2.0, 0.5]]:
        write([0, 1, 2], val)
        assert pairs(step()[0]) == pairs(ad.sql(compiling, statement))


def test_a_compiled_check_reads_the_values_written_into_its_tables(ad, compiling):
    write = writable(compiling, "w", [0, 1, 2], [1.0, -2.0, 0.5])
    step = ad.prepare(compiling, [STEP], in_place=True)
    step()
    write([0, 0, 2], [1.0, -2.0, 0.5])
    with pytest.raises(ddxdb.InvalidColumn):
        step()
    write([0, 1, 2], [3.0, 0.0, -1.0])
    assert dict(pairs(step()[0])) == pytest.approx({0: 3.0 - 2.7, 1: 0.0, 2: -1.0 - 0.3})


def test_in_place_without_what_it_needs_warns_and_plans_each_step(ad, ctx):
    # A bare context pushes dynamic filters down.
    step = ad.prepare(ctx, [STEP], in_place=True)
    with pytest.warns(UserWarning, match="in_place"):
        got = pairs(step()[0])
    assert got == pairs(ad.sql(ctx, STEP))


def test_a_compiled_steps_rows_are_stored_anew_for_a_null_where_a_value_was(ad):
    rows = pa.table({"i": pa.array([0, 1], pa.int64()), "val": pa.array([1.0, 2.0])})
    compiled = ad._Compiled(None, rows.schema, 0)
    compiled.store(rows)
    assert compiled.write(pa.table({"i": pa.array([1, 0], pa.int64()), "val": pa.array([3.0, 4.0])}))
    assert compiled.batch.to_pylist() == [{"i": 1, "val": 3.0}, {"i": 0, "val": 4.0}]
    assert not compiled.write(pa.table({"i": pa.array([0, 1], pa.int64()), "val": pa.array([5.0, None])}))
    assert compiled.batch.to_pylist() == [{"i": 1, "val": 3.0}, {"i": 0, "val": 4.0}]


def test_a_compiled_check_reads_a_seed_as_this_call_stored_it(ad, compiling):
    # The tangent is a seed ddx stores, with a row for each value of w over
    # 0, at the key i % 2: at the second call it has 3 rows, two at key 0,
    # which its check refuses.
    statement = (
        "WITH t AS (SELECT i % 2 AS i, val FROM w WHERE val > 0) "
        "SELECT i, jvp(val * val, w.val, t) AS d FROM w"
    )
    write = writable(compiling, "w", [0, 1, 2], [1.0, 2.0, -0.5])
    step = ad.prepare(compiling, [statement], in_place=True)
    assert pairs(step()[0]) == pairs(ad.sql(compiling, statement))
    write([0, 1, 2], [1.0, 2.0, 0.5])
    with pytest.raises(ddxdb.InvalidColumn):
        ad.sql(compiling, statement).collect()
    with pytest.raises(ddxdb.InvalidColumn):
        step()


def test_a_compiled_steps_rows_are_of_the_schema_its_batches_have(ad):
    # A plan's schema may differ from its batches', in which columns are
    # nullable: the rows take the batches', as DataFrame.to_arrow_table's
    # do, and the plan's only when there are no batches.
    planned = pa.schema([pa.field("i", pa.int64(), nullable=True)])
    batch = pa.RecordBatch.from_arrays([pa.array([1], pa.int64())], schema=pa.schema([pa.field("i", pa.int64(), nullable=False)]))
    assert ad._rows([batch], planned).schema == batch.schema
    assert ad._rows([], planned).schema == planned
