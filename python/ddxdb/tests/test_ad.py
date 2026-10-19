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
    assert pairs(got) == pytest.approx([(0, 2.0 / 3), (1, -4.0 / 3), (2, 1.0 / 3)])


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
    assert pairs(got) == pytest.approx([(0, 6.0), (1, -24.0), (2, 9.0)])


def test_vjp_of_a_jvp_program_is_a_hessian_vector_product_too(ad, ctx):
    jvp = ad.jvp(ctx, LOSS, "w.val")
    program = ad.vjp(ctx, jvp, "w.val")
    tangent, cotangent = program.inputs
    assert tangent.name == jvp.inputs[0].name and cotangent.of is None
    # 0 on the loss, 1 on its tangent: the gradient of ∇l·v.
    seed = pa.table({c: [0.0 if c == "l" else 1.0] for c in cotangent.columns})
    out = ad.run(ctx, program, [tangent_of_w(tangent, [1.0, 2.0, 3.0]), seed])
    assert pairs(out.gradients[0]) == pytest.approx([(0, 6.0), (1, -24.0), (2, 9.0)])


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
