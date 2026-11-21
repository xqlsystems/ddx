# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""ddxdb.duckdb: grad, vjp and jvp in SQL on a DuckDB connection.

The SQL cases are test_ad.py's, with its expected values: the same SQL means
the same derivatives on DuckDB as on DataFusion. Then what only DuckDB has:
data registered with ``con.register`` (as xarray-sql registers a Dataset), a
view over it, temporary tables, an open transaction, and the planner's copy
of the catalog.
"""

import math
import textwrap

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def ad():
    from ddxdb import duckdb as ad

    return ad


def table(i, val):
    return pa.table({"i": pa.array(i, pa.int64()), "val": pa.array(val, pa.float64())})


@pytest.fixture
def con():
    c = duckdb.connect()
    c.register("w", table([0, 1, 2], [1.0, -2.0, 0.5]))
    c.register("b", table([0, 1, 2], [0.25, 3.0, -1.0]))
    return c


def pairs(rel):
    return sorted(rel.fetchall())


def ddx_tables(con):
    return con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE table_name LIKE '\\_\\_ddx\\_%' ESCAPE '\\'"
    ).fetchall()


GRAD = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w) SELECT i, grad(loss, val) FROM w"


def test_stop_gradient_holds_its_argument_constant(ad, con):
    rel = ad.sql(
        con,
        """WITH loss AS (SELECT SUM(ddx_stop_gradient(val) * val * val) AS l FROM w)
           SELECT i, grad(loss, val) FROM w""",
    )
    # d/dv (c * v * v) = 2 c v, with c = v held constant.
    assert pairs(rel) == [(0, 2.0), (1, 8.0), (2, 0.5)]


def test_stop_gradient_runs_where_the_rewrite_keeps_it(ad, con):
    # No grad at all, and a column beside one: the caller's connection runs both.
    [total] = ad.sql(con, "SELECT SUM(ddx_stop_gradient(val)) FROM w").fetchone()
    assert total == pytest.approx(-0.5)
    rel = ad.sql(
        con,
        """WITH loss AS (SELECT SUM(val * val) AS l FROM w)
           SELECT i, ddx_stop_gradient(val) AS c, grad(loss, val) AS g FROM w""",
    )
    assert sorted(rel.fetchall()) == [(0, 1.0, 2.0), (1, -2.0, -4.0), (2, 0.5, 1.0)]


def test_stop_gradient_registered_by_the_caller_is_kept(ad):
    con = duckdb.connect()
    con.create_function("ddx_stop_gradient", lambda x: x + 0.0, ["DOUBLE"], "DOUBLE")
    con.register("w", table([0, 1], [1.0, 2.0]))
    rel = ad.sql(
        con,
        """WITH loss AS (SELECT SUM(ddx_stop_gradient(val) * val) AS l FROM w)
           SELECT i, grad(loss, val) FROM w""",
    )
    assert pairs(rel) == [(0, 1.0), (1, 2.0)]


def d_erf(v):
    return 2 / math.sqrt(math.pi) * math.exp(-v * v)


def test_a_loss_calls_a_function_create_function_registered(ad, con):
    # DuckDB doesn't define erf; ddx differentiates it (#275).
    ad.create_function(con, "erf", math.erf, ["DOUBLE"], "DOUBLE", side_effects=False)
    rel = ad.sql(con, "WITH loss AS (SELECT SUM(erf(val)) AS l FROM w) SELECT i, grad(loss, val) FROM w")
    assert [g for _, g in pairs(rel)] == pytest.approx([d_erf(v) for v in (1.0, -2.0, 0.5)])


def test_a_function_registered_after_its_backend_is_made_plans(ad):
    import ddxdb

    con = duckdb.connect()
    con.execute("CREATE TABLE x AS SELECT * FROM (VALUES (0, 0.5::DOUBLE), (1, -1.0::DOUBLE)) t(i0, v)")
    backend = ad.DuckDBBackend(con)
    ad.create_function(con, "erf", math.erf, ["DOUBLE"], "DOUBLE", side_effects=False)
    plan = ddxdb.grad_plan(backend.plan_sql("SELECT SUM(erf(x.v)) AS v FROM x"), ["x.v"])
    assert plan is not None


def test_a_view_calling_a_registered_function_is_differentiated(ad, con):
    ad.create_function(con, "erf", math.erf, ["DOUBLE"], "DOUBLE", side_effects=False)
    con.execute("CREATE VIEW e AS SELECT i, erf(val) AS y FROM w")
    rel = ad.sql(con, "WITH loss AS (SELECT SUM(y) AS l FROM e) SELECT i, grad(loss, val) FROM w")
    assert [g for _, g in pairs(rel)] == pytest.approx([d_erf(v) for v in (1.0, -2.0, 0.5)])


def test_a_cursor_plans_with_its_connection_s_functions(ad):
    con = duckdb.connect()
    # A table, not registered data, which only its own connection object sees.
    con.execute("CREATE TABLE x AS SELECT * FROM (VALUES (0, 0.5::DOUBLE), (1, -1.0::DOUBLE)) t(i, val)")
    cur = ad.cursor(con)
    # Registered after the cursor is made.
    ad.create_function(con, "erf", math.erf, ["DOUBLE"], "DOUBLE", side_effects=False)
    rel = ad.sql(cur, "WITH loss AS (SELECT SUM(erf(val)) AS l FROM x) SELECT i, grad(loss, val) FROM x")
    assert [g for _, g in pairs(rel)] == pytest.approx([d_erf(v) for v in (0.5, -1.0)])
    # A cursor shares its connection's functions, so it refuses the name.
    with pytest.raises(duckdb.CatalogException, match="already exists"):
        ad.create_function(cur, "erf", math.erf, ["DOUBLE"], "DOUBLE")


def test_create_function_keeps_the_parameters_it_was_given(ad, con):
    parameters = ["DOUBLE"]
    ad.create_function(con, "erf", math.erf, parameters, "DOUBLE", side_effects=False)
    parameters.clear()
    rel = ad.sql(con, "WITH loss AS (SELECT SUM(erf(val)) AS l FROM w) SELECT i, grad(loss, val) FROM w")
    assert [g for _, g in pairs(rel)] == pytest.approx([d_erf(v) for v in (1.0, -2.0, 0.5)])


def test_a_function_registered_again_changes_the_plan_keys(ad, con):
    # A plan binds the function's signature, so a program or a prepared
    # run kept for the old one isn't run with the new one.
    backend = ad.DuckDBBackend(con)
    query = "SELECT SUM(erf(val)) AS l FROM w"
    ad.create_function(con, "erf", math.erf, ["DOUBLE"], "DOUBLE")
    keys = backend.plan_key(query), backend.session_key()
    con.remove_function("erf")
    ad.create_function(con, "erf", math.erf, ["DOUBLE"], "FLOAT")
    new = backend.plan_key(query), backend.session_key()
    assert keys[0] != new[0] and keys[1] != new[1]


def test_stop_gradient_registered_with_create_function_is_kept(ad):
    con = duckdb.connect()
    con.register("w", table([0, 1], [1.0, 2.0]))
    ad.create_function(con, "ddx_stop_gradient", lambda x: x + 0.0, ["DOUBLE"], "DOUBLE")
    rel = ad.sql(
        con,
        """WITH loss AS (SELECT SUM(ddx_stop_gradient(val) * val) AS l FROM w)
           SELECT i, grad(loss, val) FROM w""",
    )
    assert pairs(rel) == [(0, 1.0), (1, 2.0)]


def test_stop_gradient_keeps_a_real_gradient_real(ad):
    con = duckdb.connect()
    con.register("h", pa.table({"i": pa.array([0, 1], pa.int64()), "x": pa.array([1.0, 2.0], pa.float32())}))
    rel = ad.sql(
        con,
        """WITH loss AS (SELECT SUM(ddx_stop_gradient(x) * x) AS l FROM h)
           SELECT i, grad(loss, x) AS g, typeof(grad(loss, x)) AS t FROM h""",
    )
    assert sorted(rel.fetchall()) == [(0, 1.0, "FLOAT"), (1, 2.0, "FLOAT")]


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_real_gradient_past_real_s_range_rounds_to_an_infinity(ad, sign):
    # d(sign / x)/dx at 1e-30, a normal REAL, is -sign * 1e60 (#272).
    con = duckdb.connect()
    con.register("h", pa.table({"i": pa.array([0, 1], pa.int64()), "x": pa.array([1e-30, 1.0], pa.float32())}))
    rel = ad.sql(
        con,
        f"""WITH loss AS (SELECT SUM(CAST({sign} AS REAL) / x) AS l FROM h)
            SELECT i, grad(loss, x) AS g, typeof(grad(loss, x)) AS t FROM h""",
    )
    assert sorted(rel.fetchall()) == [(0, -sign * math.inf, "FLOAT"), (1, -sign, "FLOAT")]


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_real_nan_gradient_stays_nan(ad, sign):
    # Engines order a NaN beyond the infinities (DataFusion puts a negative
    # one below -inf), so a range that didn't stop at its infinity would
    # round a NaN to one. x86 arithmetic, as 0 * inf, makes negative NaNs.
    con = duckdb.connect()
    y = pa.array([math.copysign(math.nan, sign), 1.0], pa.float32())
    con.register("h", pa.table({"i": pa.array([0, 1], pa.int64()), "x": pa.array([1.0, 1.0], pa.float32()), "y": y}))
    rel = ad.sql(con, "WITH loss AS (SELECT SUM(x * y) AS l FROM h) SELECT i, grad(loss, x) AS g FROM h")
    [(_, nan), (_, one)] = sorted(rel.fetchall())
    assert math.isnan(nan) and one == 1.0


def ids(n):
    return pa.array(range(n), pa.int64())


# DuckDB reads a table whose query uses none of its columns with no column
# mask, and counts that read as one column wide; Substrait's reading, every
# column, put each field after it on the wrong column. Each loss reads such
# a table: the cross join's side that only repeats rows.
@pytest.mark.parametrize(
    "tables, loss, wrt, value, gradient",
    [
        # #265: the constant 2.0 was read as x.v, giving 9 and 3.
        (
            {"s": {"i": ids(1), "v": [3.0]}, "x": {"i0": ids(2), "v": [1.0, 2.0]}},
            "WITH c AS (SELECT 2.0e0 AS v FROM x) SELECT SUM(s.v * c.v) AS v FROM s CROSS JOIN c",
            "s.v",
            12.0,
            4.0,
        ),
        # #266: w.v was read as r.v, so the loss didn't depend on w.v.
        (
            {"w": {"i0": ids(1), "v": [-1.0]}, "r": {"i0": ids(2), "v": [5.0, 7.0]}},
            "SELECT SUM(w.v) AS v FROM w CROSS JOIN r",
            "w.v",
            -2.0,
            2.0,
        ),
        # #264: s.v was read as w.v, so the loss didn't depend on s.v.
        (
            {"s": {"i": ids(1), "v": [2.0]}, "w": {"i0": ids(3), "v": [1.0, 2.0, 3.0]}},
            "WITH a AS (SELECT w.i0 AS i0, s.v AS v FROM w CROSS JOIN s) SELECT SUM(a.v * 3.0e0) AS v FROM a",
            "s.v",
            18.0,
            9.0,
        ),
    ],
    ids=["265", "266", "264"],
)
def test_a_table_read_for_its_rows_alone_keeps_the_columns_after_it(tables, loss, wrt, value, gradient):
    import ddxdb
    from ddxdb.duckdb import DuckDBBackend

    con = duckdb.connect()
    for name, columns in tables.items():
        con.register(name, pa.table(columns))
    backend = DuckDBBackend(con)
    program = ddxdb.grad_plan(backend.plan_sql(loss), [wrt])
    ddxdb.run(backend, program)
    assert con.sql(loss).fetchall() == [(value,)]
    assert con.sql(program.value.query()).fetchall() == [(value,)]
    assert con.sql(program.gradients[0].query()).fetchall() == [(0, gradient)]


def test_to_sql_writes_a_plan_as_sql_the_engine_runs(con):
    import ddxdb
    from ddxdb.duckdb import DuckDBBackend

    query = "SELECT w.i, w.val * b.val AS p FROM w JOIN b ON w.i = b.i WHERE w.val > 0"
    plan = DuckDBBackend(con).plan_sql(query)
    assert pairs(con.sql(ddxdb.to_sql(plan, "duckdb"))) == pairs(con.sql(query))
    with pytest.raises(ValueError, match='not "postgres"'):
        ddxdb.to_sql(plan, "postgres")


@pytest.mark.parametrize(
    "loss, gradient",
    [
        ("SELECT SUM(-val) AS l FROM w", [(0, -1.0), (1, -1.0), (2, -1.0)]),
        ("SELECT SUM(-(-val) * val) AS l FROM w", [(0, 2.0), (1, -4.0), (2, 1.0)]),
        ("SELECT SUM(s) AS l FROM (SELECT -SUM(val) OVER (ORDER BY i) AS s FROM w)", [(0, -3.0), (1, -2.0), (2, -1.0)]),
    ],
)
def test_unary_minus_is_differentiated(ad, con, loss, gradient):
    # DuckDB writes `-x` as a one-argument subtract (#262).
    assert pairs(ad.sql(con, f"WITH loss AS ({loss}) SELECT i, grad(loss, val) FROM w")) == gradient


def test_grad_in_sql_is_a_column_of_its_table_s_rows(ad, con):
    assert pairs(ad.sql(con, GRAD)) == [(0, 3.0), (1, 12.0), (2, 0.75)]


def test_an_sgd_step_is_one_expression(ad, con):
    rel = ad.sql(
        con,
        """WITH loss AS (SELECT SUM(val * val) AS l FROM w)
           SELECT i, val - 0.25 * grad(loss, val) AS val FROM w""",
    )
    assert pairs(rel) == [(0, 0.5), (1, -1.0), (2, 0.25)]


def test_statements_sharing_a_loss_share_its_program(ad, con):
    loss = "WITH loss AS (SELECT SUM(w.val * w.val * b.val) AS l FROM w JOIN b ON w.i = b.i) "
    fw, fb = ad.sql_all(
        con,
        [
            loss + "SELECT i, val - 0.5 * grad(loss, val) AS v FROM w",
            loss + "SELECT i, val - 0.5 * grad(loss, val) AS v FROM b",
        ],
    )
    w, b = [1.0, -2.0, 0.5], [0.25, 3.0, -1.0]
    assert dict(pairs(fw)) == pytest.approx({i: w[i] - 0.5 * 2 * w[i] * b[i] for i in range(3)})
    assert dict(pairs(fb)) == pytest.approx({i: b[i] - 0.5 * w[i] * w[i] for i in range(3)})


def test_an_unnamed_gradient_column_is_named_for_its_column(ad, con):
    rel = ad.sql(con, "WITH f AS (SELECT SUM(val * val) AS l FROM w) SELECT i, grad(f, val) FROM w")
    assert rel.columns == ["i", "val_grad"]


def test_no_gradient_through_a_row_an_aggregate_skips(ad, con):
    con.register("q", table([0, 1, 2], [5.0, None, 1.0]))
    rel = ad.sql(
        con,
        "WITH loss AS (SELECT SUM(w.val + q.val) AS l FROM w JOIN q ON w.i = q.i) SELECT i, grad(loss, val) FROM w",
    )
    assert pairs(rel) == [(0, 1.0), (1, 0.0), (2, 1.0)]


def test_windows_differentiate(ad, con):
    # AVG and MAX's rules write window functions, which DuckDB's Substrait
    # consumer refuses: the steps run as SQL.
    rel = ad.sql(con, "WITH loss AS (SELECT AVG(val) + MAX(val) AS l FROM w) SELECT i, grad(loss, val) FROM w")
    assert dict(pairs(rel)) == pytest.approx({0: 1 / 3 + 1.0, 1: 1 / 3, 2: 1 / 3})



def test_grad_through_a_union_all(ad, con):
    # Each row of the union sends its gradient to the table it came from;
    # the rows of i that both tables have land in one group.
    loss = """WITH u AS (SELECT i, val FROM w UNION ALL SELECT i + 1 AS i, 2 * val AS val FROM b),
              s AS (SELECT i, SUM(val) AS t FROM u GROUP BY i),
              loss AS (SELECT SUM(t * t) AS l FROM s) """
    gw, gb = ad.sql_all(
        con,
        [loss + "SELECT i, grad(loss, val) FROM w", loss + "SELECT i, grad(loss, val) FROM b"],
    )
    w, b = [1.0, -2.0, 0.5], [0.25, 3.0, -1.0]
    t = {i: w[i] if i < 3 else 0.0 for i in range(4)}
    for i in range(3):
        t[i + 1] += 2 * b[i]
    assert dict(pairs(gw)) == pytest.approx({i: 2 * t[i] for i in range(3)})
    assert dict(pairs(gb)) == pytest.approx({i: 4 * t[i + 1] for i in range(3)})

def test_a_union_of_many_inputs_is_differentiated(ad, con):
    # Its SQL is the size of its inputs: written as n² relations, it was past
    # DuckDB's max_expression_depth from 16 inputs on (#238).
    n = 24
    union = " UNION ALL ".join(f"SELECT i, {c}.0 * val AS val FROM w" for c in range(1, n + 1))
    loss = f"WITH u AS ({union}), loss AS (SELECT SUM(val * val) AS l FROM u) "
    k = sum(c * c for c in range(1, n + 1))
    want = {i: 2 * k * v for i, v in enumerate([1.0, -2.0, 0.5])}
    assert dict(pairs(ad.sql(con, loss + "SELECT i, grad(loss, val) FROM w"))) == pytest.approx(want)


def test_a_filter_on_a_gradients_dims_computes_only_those_rows(ad, con):
    full = pairs(ad.sql(con, GRAD))
    some = pairs(ad.sql(con, GRAD + " WHERE i >= 1"))
    assert some == [p for p in full if p[0] >= 1]


def test_jvp_in_sql_is_a_column_of_its_select_s_rows(ad, con):
    # The jvp's expression is also a column of its SELECT. DuckDB's
    # Substrait producer drops a column that repeats another, so ddx plans
    # it once.
    rel = ad.sql(
        con,
        """WITH v AS (SELECT i, 1.0 AS val FROM w)
           SELECT SUM(val * val * val) AS l, jvp(SUM(val * val * val), w.val, v) AS l_tangent
           FROM w""",
    )
    [(l, tangent)] = rel.fetchall()
    assert (l, tangent) == pytest.approx((1.0 - 8.0 + 0.125, 3.0 * 5.25))
    assert ddx_tables(con) == []


def test_vjp_in_sql_pulls_a_cotangent_back(ad, con):
    rel = ad.sql(
        con,
        """WITH f AS (SELECT i, val * val AS s FROM w),
                c AS (SELECT i, CAST(i + 1 AS DOUBLE) AS s FROM w)
           SELECT i, vjp(f, val, c) FROM w""",
    )
    assert pairs(rel) == [(0, 2.0), (1, -8.0), (2, 3.0)]


def test_calls_nest_in_sql(ad, con):
    rel = ad.sql(
        con,
        "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w), "
        "g AS (SELECT i, grad(loss, val) AS val FROM w), v AS (SELECT i, 1.0 AS val FROM w) "
        "SELECT i, jvp(val, w.val, v) AS val_tangent FROM g ORDER BY i",
    )
    assert [t for _, t in rel.fetchall()] == pytest.approx([6.0, -12.0, 3.0])


def test_a_window_over_a_call_is_differentiated(ad, con):
    # The jvp's query reads the grad call's result through a window, which
    # DuckDB writes as a window relation.
    statement = """WITH loss AS (SELECT SUM(val * val * val) AS l FROM w),
        g AS (SELECT i, grad(loss, val) AS val FROM w),
        p AS (SELECT i, val - MAX(val) OVER () AS val FROM g),
        v AS (SELECT i, 1.0::DOUBLE AS val FROM w)
        SELECT i, jvp(val, w.val, v) FROM p"""
    # g = 3 val², so along 1 its tangent is 6 val, and the maximum's is
    # row 1's, 6 · -2 = -12: each row's tangent is 6 val + 12.
    assert dict(pairs(ad.sql(con, statement))) == pytest.approx({0: 18.0, 1: 0.0, 2: 15.0})


def test_jvp_along_a_number_is_the_derivative_row_by_row(ad, con):
    rows = ad.sql(con, "SELECT i, val, jvp(sin(val), w.val, 1) AS d FROM w ORDER BY i").fetchall()
    assert [d for _, _, d in rows] == pytest.approx([math.cos(v) for _, v, _ in rows])


@pytest.mark.xfail(
    strict=True,
    reason="DuckDB plans an uncorrelated EXISTS as a cross join with a LIMIT 1, which ddx-ad refuses "
    "over data joined into the gradient's rows (#149)",
)
def test_a_subquery_with_no_condition_on_the_outer_row_is_differentiated(ad, con):
    statement = (
        "WITH loss AS (SELECT SUM(val * val) AS l FROM w WHERE EXISTS (SELECT 1 FROM b)) "
        "SELECT i, grad(loss, val) FROM w"
    )
    assert pairs(ad.sql(con, statement)) == [(0, 2.0), (1, -4.0), (2, 1.0)]


def test_a_statement_with_no_call_runs_as_it_is(ad, con):
    ad.sql(con, "CREATE TABLE g AS SELECT 1 AS x")
    assert con.sql("SELECT * FROM g").fetchall() == [(1,)]


# What a loss may read: whatever the connection's own queries read.


def test_a_loss_reads_a_registered_dataset_and_a_view_over_it(ad, con):
    import pyarrow.dataset as ds

    con.register("d", ds.dataset(table([0, 1], [2.0, 3.0])))
    con.execute("CREATE VIEW scaled AS SELECT i, 2.0 * val AS val FROM d")
    rel = ad.sql(con, "WITH loss AS (SELECT SUM(val * val) AS l FROM scaled) SELECT i, grad(loss, val) FROM d")
    # l = Σ (2 val)², so ∂l/∂val = 8 val.
    assert pairs(rel) == [(0, 16.0), (1, 24.0)]


def test_a_decimal_column_is_refused_as_wrt(ad, con):
    # DuckDB types the literal 1.5 as DECIMAL(2,1).
    import ddxdb

    con.execute("CREATE TABLE dec AS SELECT * FROM (VALUES (0, 1.5)) v(i, val)")
    with pytest.raises(ddxdb.InvalidColumn, match="Store it as DOUBLE"):
        ad.sql(con, "WITH loss AS (SELECT SUM(val * val) AS l FROM dec) SELECT i, grad(loss, val) FROM dec")
    assert ddx_tables(con) == []


def test_a_loss_reads_temporary_and_persistent_tables(ad, con):
    # DOUBLE, not DuckDB's DECIMAL for a literal like 1.5: a gradient is
    # taken with respect to floating-point values.
    con.execute("CREATE TEMP TABLE t AS SELECT * FROM (VALUES (0, 1.5::DOUBLE), (1, -1.0)) v(i, val)")
    con.execute("CREATE TABLE p AS SELECT * FROM (VALUES (0, 2.0::DOUBLE), (1, 4.0)) v(i, val)")
    rel = ad.sql(
        con, "WITH loss AS (SELECT SUM(t.val * p.val) AS l FROM t JOIN p ON t.i = p.i) SELECT i, grad(loss, val) FROM t"
    )
    assert pairs(rel) == [(0, 2.0), (1, 4.0)]


def test_a_loss_reads_a_table_made_in_the_open_transaction(ad, con):
    con.execute("BEGIN")
    con.execute("CREATE TABLE x AS SELECT * FROM (VALUES (0, 3.0::DOUBLE)) v(i, val)")
    rel = ad.sql(con, "WITH loss AS (SELECT SUM(val * val) AS l FROM x) SELECT i, grad(loss, val) FROM x")
    assert pairs(rel) == [(0, 6.0)]
    con.execute("ROLLBACK")
    assert ddx_tables(con) == []


def test_nothing_is_left_behind_on_either_connection(ad, con):
    ad.sql(con, GRAD)
    assert ddx_tables(con) == []
    planner = ad._backends[con]._planner
    left = planner.execute("SELECT table_name FROM duckdb_tables() WHERE table_name LIKE '\\_\\_ddx%' ESCAPE '\\'")
    assert left.fetchall() == []


def test_a_loss_reads_a_table_in_another_schema_or_database(ad, con):
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.p AS SELECT * FROM (VALUES (0, 2.0::DOUBLE), (1, 4.0)) v(i, val)")
    con.execute("ATTACH ':memory:' AS other")
    con.execute("CREATE TABLE other.main.q AS SELECT * FROM (VALUES (0, 3.0::DOUBLE), (1, 5.0)) v(i, val)")
    rel = ad.sql(
        con,
        "WITH loss AS (SELECT SUM(w.val * p.val * q.val) AS l "
        "FROM w JOIN s.p ON w.i = p.i JOIN other.q ON p.i = q.i) "
        "SELECT i, grad(loss, val) FROM w",
    )
    # ∂l/∂w = p·q where w has rows in both, 0 elsewhere.
    assert pairs(rel) == [(0, 6.0), (1, 20.0), (2, 0.0)]


def test_a_gradient_is_taken_over_a_table_in_another_schema(ad, con):
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.p AS SELECT * FROM (VALUES (0, 2.0::DOUBLE), (1, 4.0)) v(i, val)")
    rel = ad.sql(con, "WITH loss AS (SELECT SUM(val * val) AS l FROM s.p) SELECT i, grad(loss, val) FROM s.p")
    assert pairs(rel) == [(0, 4.0), (1, 8.0)]


def test_a_name_two_objects_share_is_refused(ad, con):
    # DuckDB's producer writes `s.w` as `w`, which would read the other `w`.
    import ddxdb

    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.w AS SELECT * FROM (VALUES (0, 9.0::DOUBLE)) v(i, val)")
    with pytest.raises(ddxdb.UnsupportedExpression, match="names 2 objects"):
        ad.sql(con, "WITH loss AS (SELECT SUM(val * val) AS l FROM s.w) SELECT i, grad(loss, val) FROM s.w")
    assert ddx_tables(con) == []


def test_the_backend_holds_the_connection_weakly(ad):
    import gc
    import weakref

    c = duckdb.connect()
    c.register("w", table([0], [1.0]))
    ad.sql(c, GRAD)
    held = weakref.ref(c)
    del c
    gc.collect()
    assert held() is None


# A training loop builds its program once.


def counting(monkeypatch, ad):
    planned = []
    plan_sql = ad.DuckDBBackend.plan_sql

    def counted(self, sql):
        planned.append(sql)
        return plan_sql(self, sql)

    monkeypatch.setattr(ad.DuckDBBackend, "plan_sql", counted)
    return planned


def test_a_second_sql_call_runs_the_program_the_first_built(ad, con, monkeypatch):
    planned = counting(monkeypatch, ad)
    assert pairs(ad.sql(con, GRAD)) == [(0, 3.0), (1, 12.0), (2, 0.75)]
    # New values, the same schema: the same program, on the new values.
    con.unregister("w")
    con.register("w", table([0, 1, 2], [2.0, 0.0, -1.0]))
    assert pairs(ad.sql(con, GRAD)) == [(0, 12.0), (1, 0.0), (2, 3.0)]
    assert len(planned) == 1


def test_a_table_of_another_schema_gets_its_program_built_again(ad, con, monkeypatch):
    planned = counting(monkeypatch, ad)
    ad.sql(con, GRAD)
    con.unregister("w")
    con.register("w", table([0, 1, 2], [2.0, 0.0, -1.0]).cast(pa.schema([("i", pa.int32()), ("val", pa.float64())])))
    assert pairs(ad.sql(con, GRAD)) == [(0, 12.0), (1, 0.0), (2, 3.0)]
    assert len(planned) == 2


def test_a_redefined_view_gets_its_program_built_again(ad, con):
    con.execute("CREATE VIEW m AS SELECT i, val FROM w")
    loss = "WITH loss AS (SELECT SUM(val * val) AS l FROM m) SELECT i, grad(loss, val) FROM w"
    assert pairs(ad.sql(con, loss)) == [(0, 2.0), (1, -4.0), (2, 1.0)]
    con.execute("CREATE OR REPLACE VIEW m AS SELECT i, 3.0 * val AS val FROM w")
    # l = Σ (3 val)², so ∂l/∂val = 18 val.
    assert pairs(ad.sql(con, loss)) == [(0, 18.0), (1, -36.0), (2, 9.0)]


def test_the_planner_plans_under_the_connection_s_settings(ad, con):
    # With integer_division, i / 2 is integer division: ∂l/∂val = i // 2.
    loss = "WITH loss AS (SELECT SUM(val * (i / 2)) AS l FROM w) SELECT i, grad(loss, val) FROM w"
    assert pairs(ad.sql(con, loss)) == [(0, 0.0), (1, 0.5), (2, 1.0)]
    con.execute("SET integer_division = true")
    assert pairs(ad.sql(con, loss)) == [(0, 0.0), (1, 0.0), (2, 1.0)]


def test_a_filter_on_a_table_s_column_keeps_the_table_s_rows(ad, con):
    # The planner's copy of each table is empty. Planned with DuckDB's
    # statistics, `WHERE p.layer = 0` would match no rows, and the plan would
    # leave p's scan out.
    con.register(
        "p",
        pa.table(
            {
                "layer": pa.array([0, 0, 1], pa.int64()),
                "i": pa.array([0, 1, 0], pa.int64()),
                "w": pa.array([3.0, 5.0, 7.0]),
            }
        ),
    )
    loss = "WITH loss AS (SELECT SUM(w.val * p.w) AS l FROM w JOIN p ON w.i = p.i WHERE p.layer = 0) "
    # l = 3 val₀ + 5 val₁.
    assert pairs(ad.sql(con, loss + "SELECT i, grad(loss, val) FROM w")) == [(0, 3.0), (1, 5.0), (2, 0.0)]
    assert pairs(ad.sql(con, loss + "SELECT layer, i, grad(loss, w) FROM p")) == [
        (0, 0, 1.0),
        (0, 1, -2.0),
        (1, 0, 0.0),
    ]


def test_a_view_that_cannot_be_copied_is_tried_again(ad, con):
    con.execute("CREATE TABLE t AS SELECT * FROM w")
    con.execute("CREATE VIEW m AS SELECT i, val FROM t")
    con.execute("DROP TABLE t")
    # m reads what is gone, which a loss not reading m never sees.
    assert pairs(ad.sql(con, GRAD)) == [(0, 3.0), (1, 12.0), (2, 0.75)]
    con.execute("CREATE TABLE t AS SELECT * FROM w")
    loss = "WITH loss AS (SELECT SUM(val * val) AS l FROM m) SELECT i, grad(loss, val) FROM t"
    assert pairs(ad.sql(con, loss)) == [(0, 2.0), (1, -4.0), (2, 1.0)]


def test_a_view_whose_cte_is_read_twice_is_differentiated(ad, con):
    # DuckDB's producer has no plan for the CTE it materializes, so ddx
    # plans the view's CTEs itself, as a query's.
    con.execute(
        """CREATE VIEW m AS WITH s AS (SELECT i, val * 2 AS y FROM w)
           SELECT a.i, a.y * b.y AS y FROM s a JOIN s b ON a.i = b.i"""
    )
    loss = "WITH loss AS (SELECT SUM(y) AS l FROM m) SELECT i, grad(loss, val) FROM w"
    # l = Σ 4 val², so ∂l/∂val = 8 val.
    assert pairs(ad.sql(con, loss)) == [(0, 8.0), (1, -16.0), (2, 4.0)]


def test_a_view_whose_cte_computes_a_window_and_is_read_twice_is_differentiated(ad, con):
    # A softmax: its weights, a window, are read by the output and by their
    # sum. DuckDB writes a window as a relation of its own, which ddx reads
    # as a projection.
    con.execute(
        """CREATE VIEW m AS WITH s AS (SELECT i, exp(val - MAX(val) OVER ()) AS y FROM w)
           SELECT a.i, a.y / t.z AS y FROM s a CROSS JOIN (SELECT SUM(y) AS z FROM s) t"""
    )
    loss = "WITH loss AS (SELECT SUM(y * y) AS l FROM m) SELECT i, grad(loss, val) FROM w"
    # l = Σ pᵢ², so ∂l/∂valₖ = 2 pₖ (pₖ - l).
    e = [math.exp(v) for v in (1.0, -2.0, 0.5)]
    p = [x / sum(e) for x in e]
    l = sum(x * x for x in p)
    got = pairs(ad.sql(con, loss))
    assert [i for i, _ in got] == [0, 1, 2]
    assert [g for _, g in got] == pytest.approx([2 * x * (x - l) for x in p], abs=1e-15)


def test_attention_is_differentiated_through_its_softmax_view(ad):
    # The view's weights are read by their normalizer and by the heads,
    # which join the values and the normalizer on the same keys: DuckDB's
    # plan joins the values to the normalizer on them too, so a key repeats.
    con = duckdb.connect()
    for name, val in [("q", "sin(1 + 3 * t + c)"), ("k", "cos(1 + 3 * t + c)"), ("v", "sin(2 + 3 * t + c)")]:
        con.execute(f"CREATE TABLE {name} AS SELECT t, c, {val} AS val FROM range(3) a(t), range(2) b(c)")
    con.execute(
        """CREATE VIEW attention AS
           WITH scores AS (
             SELECT q.t AS i, k.t AS j, SUM(q.val * k.val) AS x
             FROM q JOIN k ON q.c = k.c GROUP BY q.t, k.t),
           weights AS (
             SELECT i, j, exp(x - MAX(x) OVER (PARTITION BY i)) AS p FROM scores),
           norm AS (SELECT i, SUM(p) AS z FROM weights GROUP BY i)
           SELECT a.i AS t, v.c, SUM(a.p / n.z * v.val) AS val
           FROM weights a JOIN v ON a.j = v.t JOIN norm n ON a.i = n.i
           GROUP BY a.i, v.c"""
    )

    def loss():
        return con.execute("SELECT SUM(val * val) FROM attention").fetchone()[0]

    for table in ["q", "k"]:
        sql = f"WITH loss AS (SELECT SUM(val * val) AS l FROM attention) SELECT t, c, grad(loss, val) FROM {table}"
        got = sorted(ad.sql(con, sql).fetchall())
        # The engine's own central differences of the loss.
        h, want = 1e-6, []
        for t, c, _ in got:
            moved = []
            for step in (h, -2 * h, h):
                con.execute(f"UPDATE {table} SET val = val + {step} WHERE t = {t} AND c = {c}")
                moved.append(loss())
            want.append((moved[0] - moved[1]) / (2 * h))
        assert [g for _, _, g in got] == pytest.approx(want, abs=1e-8)


def test_a_view_s_sql_is_given_where_its_names_resolve_as_the_query_s(ad, con):
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE VIEW here AS SELECT 1 AS x")
    con.execute("CREATE TEMP VIEW tv AS SELECT 2 AS x")
    con.execute("CREATE VIEW s.there AS SELECT 3 AS x")
    con.execute("CREATE TABLE t (x INTEGER)")
    backend = ad.DuckDBBackend(con)
    assert "SELECT 1" in backend.view_sql("here")
    assert "SELECT 1" in backend.view_sql('"HERE"')
    assert "SELECT 2" in backend.view_sql("tv")
    assert "SELECT 2" in backend.view_sql("temp.tv")
    assert "SELECT 1" in backend.view_sql("memory.here")
    assert "SELECT 1" in backend.view_sql("main.here")
    assert "SELECT 1" in backend.view_sql("memory.main.here")
    # A view in another schema reads names in its own; a table has no SQL.
    assert backend.view_sql("s.there") is None
    assert backend.view_sql("t") is None
    # A temporary table of the view's name is what the name reads.
    con.execute("CREATE TEMP TABLE here (x INTEGER)")
    assert backend.view_sql("here") is None


@pytest.mark.parametrize("hint", ["MATERIALIZED", "NOT MATERIALIZED"])
def test_a_cte_with_a_materialization_hint_is_differentiated(ad, con, hint):
    statement = f"""WITH s AS {hint} (SELECT i, val * 2 AS y FROM w),
                         loss AS (SELECT SUM(y) AS l FROM s)
                    SELECT i, grad(loss, val) FROM w"""
    assert pairs(ad.sql(con, statement)) == [(0, 2.0), (1, 2.0), (2, 2.0)]


def test_a_window_over_an_aggregate_is_differentiated(ad):
    # DuckDB writes a window as a window relation, and OVER () as RANGE up
    # to the current row, which, with no ORDER BY, is the whole partition.
    c = duckdb.connect()
    c.execute(
        "CREATE TABLE w AS SELECT i, g, val::DOUBLE AS val"
        " FROM (VALUES (0, 0, 1.0), (1, 0, 2.0), (2, 1, 0.5)) v(i, g, val)"
    )
    softmax_normalizer = """WITH s AS (SELECT g, SUM(val) AS s FROM w GROUP BY g),
             p AS (SELECT exp(s - MAX(s) OVER ()) AS e FROM s),
             loss AS (SELECT SUM(e) AS l FROM p)
        SELECT i, grad(loss, val) FROM w"""
    # s = (3, 0.5), so l = 1 + e⁻²·⁵: ∂l/∂s = (-e⁻²·⁵, e⁻²·⁵), each val's
    # its group's.
    e = math.exp(-2.5)
    assert dict(pairs(ad.sql(c, softmax_normalizer))) == pytest.approx({0: -e, 1: -e, 2: e})
    over_a_table = """WITH p AS (SELECT exp(val - MAX(val) OVER ()) AS e FROM w),
             loss AS (SELECT SUM(e) AS l FROM p)
        SELECT i, grad(loss, val) FROM w"""
    # The maximum, 2, takes -Σ of the others' exponentials as well as its own 1.
    others = math.exp(-1.0) + math.exp(-1.5)
    assert dict(pairs(ad.sql(c, over_a_table))) == pytest.approx(
        {0: math.exp(-1.0), 1: 1 - (1 + others), 2: math.exp(-1.5)}
    )


# Refusals.


@pytest.mark.parametrize(
    "u, loss, source",
    [
        ("SELECT CASE WHEN i = 1 THEN NULL ELSE i END AS k, val FROM w", "SUM(val * val * k)", "u"),
        ("SELECT i AS k, CASE WHEN i = 1 THEN NULL ELSE val END AS val FROM w", "SUM(val * val)", "u"),
        (
            "SELECT CASE WHEN i = 1 THEN CAST(NULL AS BIGINT) ELSE i END AS k, val FROM w",
            "SUM(u.val * b.val)",
            "u JOIN b ON u.k = b.i",
        ),
    ],
)
def test_a_bare_null_is_differentiated(ad, con, u, loss, source):
    # DuckDB's producer gives a bare NULL a type of no kind, and folds a
    # CAST(NULL AS ...) into one; the step's SQL writes it as a bare NULL.
    # The gradient is DataFusion's for the same SQL.
    from datafusion import SessionContext
    from ddxdb import ad as df

    sql = f"WITH u AS ({u}), loss AS (SELECT {loss} AS l FROM {source}) SELECT i, grad(loss, val) AS g FROM w"
    ctx = SessionContext()
    for name in ("w", "b"):
        ctx.from_arrow(con.execute(f"SELECT * FROM {name}").arrow(), name)
    want = {r["i"]: r["g"] for r in df.sql(ctx, sql).to_arrow_table().to_pylist()}
    assert dict(pairs(ad.sql(con, sql))) == pytest.approx(want)


def test_a_repeated_column_is_refused_with_a_reason(ad, con):
    # DuckDB's producer drops a column that repeats another; ddx cannot tell
    # which is which, and says so rather than guess.
    with pytest.raises(Exception, match="dropped a column"):
        ad.sql(
            con,
            """WITH v AS (SELECT i, 1.0 AS val FROM w)
               SELECT jvp(SUM(val), w.val, v) AS t, SUM(val) AS a, SUM(val) AS b FROM w""",
        )
    assert ddx_tables(con) == []


@pytest.mark.parametrize(
    "u, loss",
    [
        (
            "SELECT i AS k, x AS v FROM ta UNION ALL SELECT k, v FROM "
            "(SELECT i AS k, i AS v FROM tb UNION ALL SELECT i AS k, i AS v FROM tb) n",
            "SELECT SUM(u.v * w.val * u.v) AS l FROM u JOIN w ON u.k = w.i",
        ),
        (
            "SELECT k, v FROM (SELECT i AS k, i AS v FROM tb UNION ALL SELECT i AS k, i AS v FROM ta) n "
            "UNION ALL SELECT i AS k, x AS v FROM ta",
            "SELECT SUM(t * t) AS l FROM (SELECT k, SUM(v) AS t FROM u GROUP BY k) g",
        ),
        (
            "SELECT i AS k, x AS v, x AS v2 FROM ta UNION ALL SELECT i AS k, x AS v, 1.0 AS v2 FROM tb",
            "SELECT SUM(v * v2) AS l FROM u",
        ),
    ],
)
def test_a_repeated_column_beneath_a_union_is_refused_with_a_reason(ad, con, u, loss):
    # Beneath the root, the column DuckDB's producer drops leaves a union's
    # inputs unequal, or an emit naming a column that isn't there. The first
    # panicked (#237).
    con.execute("CREATE TABLE ta AS SELECT * FROM (VALUES (1, 0.5::DOUBLE), (2, -1.0::DOUBLE)) t(i, x)")
    con.execute("CREATE TABLE tb AS SELECT * FROM (VALUES (1, 2.0::DOUBLE), (3, 1.5::DOUBLE)) t(i, x)")
    with pytest.raises(Exception, match="dropped a column"):
        ad.sql(con, f"WITH u AS ({u}), loss AS ({loss}) SELECT i, grad(loss, x) FROM ta")
    assert ddx_tables(con) == []


def test_a_floor_division_of_floats_is_refused(ad, con):
    # DuckDB's `//` of floats floors; ddx writes only the integer one.
    with pytest.raises(Exception, match="`//`"):
        ad.sql(con, "WITH loss AS (SELECT SUM(val * (val // 2.0)) AS l FROM w) SELECT i, grad(loss, val) FROM w")
    assert ddx_tables(con) == []


def test_the_module_docs_example_runs(ad):
    # The example at the top of ddxdb.duckdb's docs, as it is written there.
    doc = ad.__doc__
    start = doc.index("    import duckdb")
    example = textwrap.dedent(doc[start : doc.index("\n\n", doc.index("ad.sql(", start))])
    example = example.replace("ad.sql(con,", "result = ad.sql(con,", 1)
    scope = {}
    exec(example, scope)
    assert dict(pairs(scope["result"])) == pytest.approx({0: 0.8, 1: 1.6})


def test_ad_sql_takes_a_duckdb_connection(con):
    pytest.importorskip("datafusion")
    from ddxdb import ad

    assert pairs(ad.sql(con, GRAD)) == [(0, 3.0), (1, 12.0), (2, 0.75)]


def test_ad_sql_refuses_another_dialect_on_a_duckdb_connection(con):
    pytest.importorskip("datafusion")
    from ddxdb import ad

    assert pairs(ad.sql(con, GRAD, dialect="duckdb")) == [(0, 3.0), (1, 12.0), (2, 0.75)]
    with pytest.raises(ValueError, match="DuckDB's dialect"):
        ad.sql(con, GRAD, dialect="sqlparser")


def test_a_prepared_step_reads_the_tables_rows_as_they_are_at_each_call(ad, con):
    step_sql = "WITH loss AS (SELECT SUM(val * val * val) AS l FROM w) SELECT i, val - 0.1 * grad(loss, val) AS val FROM w"
    step = ad.prepare(con, [step_sql, GRAD])
    for _ in range(3):
        got = [pairs(rel) for rel in step()]
        assert got == [pairs(rel) for rel in ad.sql_all(con, [step_sql, GRAD])]
        assert not ddx_tables(con)
        con.register("w", table([i for i, _ in got[0]], [v for _, v in got[0]]))


class Spy:
    """A DuckDB connection, each statement it is sent recorded."""

    def __init__(self, con):
        self.of, self.sent = con, []

    def execute(self, sql, *args):
        self.sent.append(sql)
        return self.of.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.of, name)


def test_a_later_prepared_call_asks_the_planner_for_nothing(ad, con):
    # A later call replays the plans of the first: none needs the planner's
    # copy of a table to bind it, so none is made, or dropped.
    step = ad.prepare(con, [GRAD])
    step()
    backend = ad._backends[con]
    planner = backend._planner = Spy(backend._planner)
    con.register("w", table([0, 1, 2], [2.0, 0.0, -1.0]))
    assert pairs(step()[0]) == [(0, 12.0), (1, 0.0), (2, 3.0)]
    assert planner.sent == []
    assert not ddx_tables(con)


def test_the_planner_copies_only_the_tables_it_plans_a_read_of(ad, con, monkeypatch):
    # No later step reads a value or a gradient, and the second statement's
    # plan doesn't either, so the planner never needs, or gets, a copy of
    # one: only of what the steps save and pass back.
    backend = ad._backend(con)
    planner = backend._planner = Spy(backend._planner)
    spy = Spy(con)
    monkeypatch.setattr(ad.DuckDBBackend, "con", property(lambda self: spy))
    got = [pairs(rel) for rel in ad.sql_all(con, [GRAD, GRAD.replace("val * val * val", "val * val")])]
    assert got == [[(0, 3.0), (1, 12.0), (2, 0.75)], [(0, 2.0), (1, -4.0), (2, 1.0)]]
    made = lambda sent: [s.split()[5] for s in sent if s.startswith("CREATE OR REPLACE TEMP TABLE")]
    written, copied = made(spy.sent), made(planner.sent)
    results = [n for n in written if "_value" in n or "_grad_" in n]
    assert len(results) == 4
    assert copied and set(copied) <= set(written) - set(results)
    assert not ddx_tables(con)


def test_an_sql_call_reads_the_connection_s_catalog_once(ad, con, monkeypatch):
    con.execute("CREATE VIEW v AS SELECT i, val * 2 AS val FROM w")
    statements = [
        "WITH loss AS (SELECT SUM(val * val) AS l FROM v) SELECT i, grad(loss, val) FROM w",
        "WITH loss AS (SELECT SUM(val * val) AS l FROM v) SELECT i, val - grad(loss, val) FROM w",
    ]
    spy = Spy(con)
    monkeypatch.setattr(ad.DuckDBBackend, "con", property(lambda self: spy))
    got = [pairs(rel) for rel in ad.sql_all(con, statements)]
    assert got[0] == [(0, 8.0), (1, -16.0), (2, 4.0)]
    assert sum("duckdb_columns()" in sql for sql in spy.sent) == 1


def test_ad_prepare_on_a_duckdb_connection_prepares_on_duckdb(con):
    from ddxdb import ad

    step = ad.prepare(con, [GRAD])
    assert pairs(step()[0]) == [(0, 3.0), (1, 12.0), (2, 0.75)]
    assert pairs(step()[0]) == [(0, 3.0), (1, 12.0), (2, 0.75)]


def test_a_prepared_call_differentiates_a_redefined_view_as_it_is_now(ad, con):
    con.execute("CREATE VIEW m AS SELECT i, val FROM w")
    statement = "WITH loss AS (SELECT SUM(val * val) AS l FROM m) SELECT i, grad(loss, val) FROM w"
    step = ad.prepare(con, [statement])
    assert pairs(step()[0]) == [(0, 2.0), (1, -4.0), (2, 1.0)]
    con.execute("CREATE OR REPLACE VIEW m AS SELECT i, 2 * val AS val FROM w")
    assert pairs(step()[0]) == [(0, 8.0), (1, -16.0), (2, 4.0)]




def test_a_prepared_call_plans_in_a_file_database(ad, tmp_path):
    # Its current database is the file's, not the planner's `memory`: the
    # first call plans before any other has set where names are searched.
    con = duckdb.connect(str(tmp_path / "model.duckdb"))
    con.execute("CREATE TABLE w (i BIGINT, val DOUBLE)")
    con.execute("INSERT INTO w VALUES (0, 1.0), (1, -2.0), (2, 0.5)")
    step = ad.prepare(con, [GRAD])
    assert pairs(step()[0]) == pairs(step()[0]) == [(0, 3.0), (1, 12.0), (2, 0.75)]


def test_the_planner_searches_names_as_the_connection_does(ad, con):
    con.execute("CREATE SCHEMA s")
    con.execute("CREATE TABLE s.v (i BIGINT, val DOUBLE)")
    con.execute("INSERT INTO s.v VALUES (0, 2.0)")
    con.execute("SET search_path = 's,main'")
    statement = "WITH loss AS (SELECT SUM(val * val) AS l FROM v) SELECT i, grad(loss, val) FROM v"
    assert pairs(ad.prepare(con, [statement])()[0]) == [(0, 4.0)]
    assert pairs(ad.sql_all(con, [statement])[0]) == [(0, 4.0)]


def test_threads_with_a_cursor_each_differentiate_at_once(ad, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    con = duckdb.connect(str(tmp_path / "db.duckdb"))
    con.execute("CREATE TABLE w (i BIGINT, val DOUBLE)")
    con.execute("INSERT INTO w VALUES (0, 1.0), (1, -2.0), (2, 0.5)")

    def call(k):
        cur = con.cursor()
        step = ad.prepare(cur, [GRAD])
        got = [pairs(step()[0]) for _ in range(3)] + [pairs(ad.sql_all(cur, [GRAD])[0])]
        assert not ddx_tables(cur)
        return got

    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(call, range(4)))
    assert got == [[[(0, 3.0), (1, 12.0), (2, 0.75)]] * 4] * 4


def test_a_prepared_call_after_a_setting_changes_matches_sql_all(ad):
    con = duckdb.connect()
    con.execute("CREATE TABLE w (i BIGINT, val DOUBLE)")
    con.execute("INSERT INTO w VALUES (0, 1.0), (1, -2.0), (2, 0.5), (3, 1.0)")
    # The gradient is i / 2, which DuckDB divides as integers once told to.
    statement = "WITH loss AS (SELECT SUM(val * (i / 2)) AS l FROM w) SELECT i, grad(loss, val) FROM w"
    step = ad.prepare(con, [statement])
    assert pairs(step()[0]) == [(0, 0.0), (1, 0.5), (2, 1.0), (3, 1.5)]
    con.execute("SET integer_division = true")
    assert pairs(step()[0]) == pairs(ad.sql_all(con, [statement])[0]) == [(0, 0.0), (1, 0.0), (2, 1.0), (3, 1.0)]



def test_a_prepared_call_after_a_table_shadows_another_refuses_as_sql_all_does(ad):
    con = duckdb.connect()
    con.execute("CREATE TABLE w (i BIGINT, val DOUBLE)")
    con.execute("INSERT INTO w VALUES (0, 1.0), (1, 2.0)")
    statement = "WITH loss AS (SELECT SUM(val * val) AS l FROM w) SELECT i, grad(loss, val) FROM w"
    step = ad.prepare(con, [statement])
    step()
    con.execute("CREATE TEMP TABLE w AS SELECT i, val * 10 AS val FROM main.w")
    for call in (step, lambda: ad.sql_all(con, [statement])):
        with pytest.raises(Exception, match="names 2 objects"):
            call()


def test_a_prepared_call_after_a_column_changes_its_type_matches_sql_all(ad):
    con = duckdb.connect()
    con.execute("CREATE TABLE w (i BIGINT, val DOUBLE)")
    con.execute("INSERT INTO w VALUES (0, 1.0), (1, 2.0)")
    statement = "WITH loss AS (SELECT SUM(val * val) AS l FROM w) SELECT i, grad(loss, val) FROM w"
    step = ad.prepare(con, [statement])
    step()
    con.execute("ALTER TABLE w ALTER val TYPE FLOAT")
    assert pairs(step()[0]) == pairs(ad.sql_all(con, [statement])[0])


def test_in_place_is_refused_on_duckdb(con):
    from ddxdb import ad

    with pytest.raises(ValueError, match="in_place"):
        ad.prepare(con, [GRAD], in_place=True)
