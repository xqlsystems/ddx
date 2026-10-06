# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""ddxdb.ad on DataFusion: grad and jvp in SQL and as programs, vjp,
composition, typed refusals.

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
    assert ctx.table(program.value.step).to_arrow_table().column(0)[0].as_py() == 1.0 + 4.0 + 0.25
    ctx.deregister_table("w")
    ctx.register_record_batches("w", [table([0, 1, 2], [5.0, 6.0, 7.0]).to_batches()])
    ad.run(ctx, program)
    assert pairs(ctx.table(program.gradients[0].step)) == [(0, 10.0), (1, 12.0), (2, 14.0)]


def test_vjp_pulls_a_cotangent_back(ad, ctx):
    program = ad.vjp(ctx, "SELECT i, val * val AS s FROM w", [("w", "val")])
    [cotangent] = program.inputs
    assert (cotangent.of, cotangent.columns, cotangent.keys) == (None, ("i", "s"), 1)
    ctx.register_record_batches(cotangent.name, [pa.table({"i": pa.array([0, 1, 2], pa.int64()), "s": [1.0, 10.0, 100.0]}).to_batches()])
    ad.run(ctx, program)
    assert pairs(ctx.table(program.gradients[0].step)) == [(0, 2.0), (1, -40.0), (2, 100.0)]


def test_refusals_are_typed(ad, ctx):
    with pytest.raises(ddxdb.NotScalar):
        ad.grad(ctx, "SELECT i, val FROM w", [("w", "val")])
    with pytest.raises(ddxdb.UnknownColumn, match="weights"):
        ad.grad(ctx, "SELECT SUM(val) AS loss FROM w", [("weights", "val")])
    with pytest.raises(ddxdb.UnsupportedExpression):
        ad.grad(ctx, "SELECT STDDEV(val) AS loss FROM w", [("w", "val")])
    with pytest.raises(ddxdb.InvalidColumn, match="float"):
        ad.grad(ctx, "SELECT SUM(i) AS loss FROM w", [("w", "i")])
    assert issubclass(ddxdb.NotScalar, ddxdb.DdxError)
    assert issubclass(ddxdb.UnknownColumn, ddxdb.DdxError)
    assert issubclass(ddxdb.InvalidColumn, ddxdb.DdxError)


def test_run_refuses_rows_that_share_their_dims(ad, ctx):
    ctx.register_record_batches("d", [table([0, 0, 1], [1.0, 2.0, 3.0]).to_batches()])
    program = ad.grad(ctx, "SELECT SUM(val * val) AS loss FROM d", [("d", "val")])
    with pytest.raises(ddxdb.InvalidColumn, match="share their dims"):
        ad.run(ctx, program)


def ddx_tables(ctx):
    return sorted(n for n in ctx.catalog().schema().names() if n.startswith("__ddx_"))


def test_run_keeps_the_value_and_gradients_and_release_drops_them(ad, ctx):
    program = ad.grad(ctx, "SELECT MAX(val) * SUM(val) AS loss FROM w", [("w", "val")])
    assert list(program.intermediate_steps())
    ad.run(ctx, program)
    assert ddx_tables(ctx) == sorted([program.value.step, program.gradients[0].step])
    ad.release(ctx, program)
    assert ddx_tables(ctx) == []


def test_programs_do_not_share_tables(ad, ctx):
    a = ad.grad(ctx, "SELECT SUM(val * val) AS loss FROM w", [("w", "val")])
    b = ad.grad(ctx, "SELECT SUM(val * val) AS loss FROM b", [("b", "val")])
    assert not {s.name for s in a.steps()} & {s.name for s in b.steps()}
    ad.run(ctx, a)
    ad.run(ctx, b)
    assert pairs(ctx.table(a.gradients[0].step)) == [(0, 2.0), (1, -4.0), (2, 1.0)]


def test_sql_leaves_no_tables_behind(ad, ctx):
    df = ad.sql(ctx, "WITH loss AS (SELECT SUM(val * val) AS l FROM w) SELECT * FROM grad(loss, w.val)")
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
           SELECT g1.i, g1.val * 1000.0 + g2.val AS v
           FROM grad(loss, a_b.c.val) g1 CROSS JOIN grad(loss, a.b_c.val) g2""",
    )
    assert pairs(df) == [(0, 2020.0)]


def test_grad_in_sql_of_a_table_with_capitals(ad):
    # From the v2 soak (#92): a gradient step named with the table's capitals
    # was registered under a folded name and not found again.
    import datafusion

    ctx = datafusion.SessionContext()
    ctx.sql('CREATE TABLE "W" (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)').collect()
    df = ad.sql(ctx, 'WITH loss AS (SELECT SUM(val * val) AS l FROM "W") SELECT i, val FROM grad(loss, "W".val)')
    assert pairs(df) == [(0, 2.0), (1, 4.0)]


def test_no_gradient_through_a_row_an_aggregate_skips(ad, ctx):
    # From the v2 soak (#89): SUM skips row 1, whose term w + q is NULL.
    ctx.register_record_batches("q", [table([0, 1, 2], [5.0, None, 1.0]).to_batches()])
    df = ad.sql(
        ctx,
        "WITH loss AS (SELECT SUM(w.val + q.val) AS l FROM w JOIN q ON w.i = q.i) SELECT * FROM grad(loss, w.val)",
    )
    assert pairs(df) == [(0, 1.0), (1, 0.0), (2, 1.0)]


def test_vjp_refuses_a_cotangent_whose_keys_repeat(ad, ctx):
    # From the v2 soak (#98): a repeated key added its rows' cotangents.
    program = ad.vjp(ctx, "SELECT i, val * val AS s FROM w", [("w", "val")])
    ctx.register_record_batches(
        program.inputs[0].name,
        [pa.table({"i": pa.array([0, 0, 1, 2], pa.int64()), "s": [1.0, 1.0, 1.0, 1.0]}).to_batches()],
    )
    with pytest.raises(ddxdb.InvalidColumn, match="share their keys"):
        ad.run(ctx, program)


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
    prog = ad.grad(ctx, "SELECT SUM(val * val) / COUNT(val) AS l FROM w", [("w", "val")])
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
    prog = ad.grad(ctx, "SELECT SUM(val * val) / COUNT(val) AS l FROM w", [("w", "val")])
    ad.run(ctx, prog)  # an earlier run's value and gradient
    backend = LoggingBackend(ad, ctx, fail_on=prog.gradients[0].step)
    with pytest.raises(RuntimeError, match="disk full"):
        ad.run(backend, prog)
    assert not ddx_tables(ctx)


def test_a_fixed_namespace_gives_the_same_program(ad, ctx):
    # The same plan, not the same query: DataFusion's producer can number a
    # query's functions differently from one call to the next.
    from datafusion.substrait import Serde

    q = "SELECT SUM(val * val) AS l FROM w"
    plan = Serde.serialize_bytes(q, ctx)
    a = ddxdb.grad_plan(plan, [("w", "val")], namespace="__ddx_py_")
    b = ddxdb.grad_plan(plan, [("w", "val")], namespace="__ddx_py_")
    assert a == b and a.value.step == "__ddx_py_value"
    with pytest.raises(ddxdb.DdxError, match="namespace"):
        ad.grad(ctx, q, [("w", "val")], namespace="__ddx_Py_")


def test_grad_in_sql_is_found_in_the_dialect_it_is_written_in():
    from ddxdb._ddxdb import _Statements

    stmt = "WITH loss AS (SELECT SUM(val * val) AS l FROM w) SELECT * FROM grad(loss, w.val)"
    assert len(_Statements([stmt], "duckdb").jobs()) == 1
    with pytest.raises(ValueError, match="dialect"):
        _Statements([stmt], "klingon")


def test_not_in_over_a_subquery_is_refused(ad, ctx):
    # DataFusion's Substrait producer drops NOT IN's NULL semantics (#104):
    # refused, rather than differentiated as a different query.
    loss = "SELECT SUM(val * val) AS l FROM w WHERE i NOT IN (SELECT i FROM b WHERE val > 1.0)"
    with pytest.raises(ddxdb.UnsupportedExpression, match="NOT IN"):
        ad.grad(ctx, loss, [("w", "val")])
    with pytest.raises(ddxdb.UnsupportedExpression, match="NOT IN"):
        ad.sql(ctx, f"WITH loss AS ({loss}) SELECT * FROM grad(loss, w.val)")
    exists = (
        "SELECT SUM(val * val) AS l FROM w WHERE NOT EXISTS "
        "(SELECT 1 FROM b WHERE b.i = w.i AND b.val > 1.0)"
    )
    ad.grad(ctx, exists, [("w", "val")])



def test_a_filter_on_a_gradients_dims_computes_only_those_rows(ad, ctx):
    # The WHERE on g's dims is pushed into the program (ddx_ad::Options::restrict);
    # the rows read are the unrestricted gradient's.
    loss = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w)"
    full = pairs(ad.sql(ctx, f"{loss} SELECT g.i, g.val FROM grad(loss, w.val) g"))
    some = pairs(ad.sql(ctx, f"{loss} SELECT g.i, g.val FROM grad(loss, w.val) g WHERE g.i >= 1"))
    assert some == [p for p in full if p[0] >= 1]


# jvp, and composition. The closed forms here use Σ val³ over w = (1, -2, 0.5):
# ∇ = 3 val², H = diag(6 val).

LOSS = "SELECT SUM(val * val * val) AS l FROM w"


def register_tangent(ctx, input_table, values):
    """Register ``input_table``, a tangent of w, as ``values`` by row of w."""
    assert input_table.of == ("w",) and input_table.columns == ("i", "val") and input_table.keys == 1
    ctx.register_record_batches(input_table.name, [table([0, 1, 2], values).to_batches()])


def test_jvp_of_a_query_is_its_value_and_its_tangent(ad, ctx):
    program = ad.jvp(ctx, LOSS, [("w", "val")])
    assert isinstance(program, ddxdb.ForwardProgram) and not program.gradients
    register_tangent(ctx, program.inputs[0], [1.0, 1.0, 1.0])
    ad.run(ctx, program)
    [tangent] = program.value.tangents
    assert tangent.column == "l" and tangent.tangent in program.value.columns
    row = ctx.sql(f'SELECT l, "{tangent.tangent}" AS t FROM "{program.value.step}"').to_arrow_table()
    assert row.column("l")[0].as_py() == 1.0 - 8.0 + 0.125
    assert row.column("t")[0].as_py() == pytest.approx(3.0 * (1.0 + 4.0 + 0.25))
    ad.release(ctx, program)
    # The tangent is the caller's, and stays.
    assert ddx_tables(ctx) == [program.inputs[0].name]


def test_jvp_of_a_grad_program_is_a_hessian_vector_product(ad, ctx):
    grad = ad.grad(ctx, LOSS, [("w", "val")])
    hvp = ad.jvp(ctx, grad, [("w", "val")])
    [g] = hvp.gradients
    assert g.of == ("w",)
    register_tangent(ctx, hvp.inputs[0], [1.0, 2.0, 3.0])
    ad.run(ctx, hvp)
    [t] = g.tangents
    got = ctx.sql(f'SELECT i, "{t.tangent}" FROM "{g.step}"')
    assert pairs(got) == pytest.approx([(0, 6.0), (1, -24.0), (2, 9.0)])


def test_vjp_of_a_jvp_program_is_a_hessian_vector_product_too(ad, ctx):
    jvp = ad.jvp(ctx, LOSS, [("w", "val")])
    program = ad.vjp(ctx, jvp, [("w", "val")])
    tangent, cotangent = program.inputs
    assert tangent.name == jvp.inputs[0].name and cotangent.of is None
    register_tangent(ctx, tangent, [1.0, 2.0, 3.0])
    # 0 on the loss, 1 on its tangent: the gradient of ∇l·v.
    columns = {c: [0.0 if c == "l" else 1.0] for c in cotangent.columns}
    ctx.register_record_batches(cotangent.name, [pa.table(columns).to_batches()])
    ad.run(ctx, program)
    assert pairs(ctx.table(program.gradients[0].step)) == pytest.approx([(0, 6.0), (1, -24.0), (2, 9.0)])


def test_any_backend_runs_a_forward_program(ad, ctx):
    program = ad.jvp(ctx, LOSS, [("w", "val")])
    register_tangent(ctx, program.inputs[0], [1.0, 0.0, 0.0])
    backend = LoggingBackend(ad, ctx)
    ddxdb.run(backend, program)
    assert ("write", program.value.step) in backend.log
    with pytest.raises(TypeError, match="Program"):
        ddxdb.run(backend, b"not a program")


def test_jvp_in_sql_is_the_value_with_its_tangent(ad, ctx):
    df = ad.sql(
        ctx,
        """WITH loss AS (SELECT SUM(val * val * val) AS l FROM w),
                v AS (SELECT i, 1.0 AS val FROM w)
           SELECT l, l_tangent FROM jvp(loss, w.val, v)""",
    )
    t = df.to_arrow_table()
    assert t.column("l_tangent")[0].as_py() == pytest.approx(3.0 * 5.25)
    assert not ddx_tables(ctx)


def test_jvp_in_sql_takes_each_table_s_tangent_and_context_sql_reads_it(ad, ctx):
    from ddxdb import Context

    c = Context()
    c.register_record_batches("w", [table([0, 1, 2], [1.0, -2.0, 0.5]).to_batches()])
    c.register_record_batches("b", [table([0, 1, 2], [0.25, 3.0, -1.0]).to_batches()])
    c.register_record_batches("db", [table([0, 1, 2], [1.0, 1.0, 1.0]).to_batches()])
    df = c.sql(
        """WITH f AS (SELECT w.i, w.val * b.val AS y FROM w JOIN b ON w.i = b.i),
                dw AS (SELECT i, 0.0 AS val FROM w)
           SELECT i, y_tangent FROM jvp(f, w.val, dw, b.val, db) ORDER BY i"""
    )
    # d(w·b) along (0, 1) is w.
    assert pairs(df) == [(0, 1.0), (1, -2.0), (2, 0.5)]


def test_a_fixed_namespace_gives_the_same_forward_program(ad, ctx):
    from datafusion.substrait import Serde

    plan = Serde.serialize_bytes(LOSS, ctx)
    a = ddxdb.jvp_plan(plan, [("w", "val")], namespace="__ddx_py_")
    b = ddxdb.jvp_plan(plan, [("w", "val")], namespace="__ddx_py_")
    assert a == b and a.value.step == "__ddx_py_jvp"
    with pytest.raises(TypeError, match="Substrait"):
        ddxdb.jvp_plan("SELECT 1", [("w", "val")])


def test_a_step_too_deep_for_datafusion_python_is_refused_plainly(ad, ctx):
    # datafusion-python decodes Substrait with protobuf's default nesting
    # limit, two levels per relation; a jvp through enough stacked
    # aggregates passes it, and is refused as such, not as a decode error.
    ctes = ["h0 AS (SELECT i, val AS v FROM w)"]
    for k in range(1, 40):
        ctes.append(f"h{k} AS (SELECT i, SUM(tanh(v) + 0.5 * v) AS v FROM h{k - 1} GROUP BY i)")
    query = f"WITH {', '.join(ctes)} SELECT SUM(v) AS l FROM h39"
    program = ad.jvp(ctx, query, [("w", "val")])
    register_tangent(ctx, program.inputs[0], [1.0, 1.0, 1.0])
    with pytest.raises(ddxdb.UnsupportedExpression, match="48 relations"):
        ad.run(ctx, program)
