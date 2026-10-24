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
    assert pairs(fw) == pytest.approx([(i, w[i] - 0.5 * 2 * w[i] * b[i]) for i in range(3)])
    assert pairs(fb) == pytest.approx([(i, b[i] - 0.5 * w[i] * w[i]) for i in range(3)])


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
    assert pairs(rel) == pytest.approx([(0, 1 / 3 + 1.0), (1, 1 / 3), (2, 1 / 3)])


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


def test_a_view_that_cannot_be_copied_is_tried_again(ad, con):
    con.execute("CREATE TABLE t AS SELECT * FROM w")
    con.execute("CREATE VIEW m AS SELECT i, val FROM t")
    con.execute("DROP TABLE t")
    # m reads what is gone, which a loss not reading m never sees.
    assert pairs(ad.sql(con, GRAD)) == [(0, 3.0), (1, 12.0), (2, 0.75)]
    con.execute("CREATE TABLE t AS SELECT * FROM w")
    loss = "WITH loss AS (SELECT SUM(val * val) AS l FROM m) SELECT i, grad(loss, val) FROM t"
    assert pairs(ad.sql(con, loss)) == [(0, 2.0), (1, -4.0), (2, 1.0)]


# Refusals.


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


def test_a_floor_division_of_floats_is_refused(ad, con):
    # DuckDB's `//` of floats floors; ddx writes only the integer one.
    with pytest.raises(Exception, match="`//`"):
        ad.sql(con, "WITH loss AS (SELECT SUM(val * (val // 2.0)) AS l FROM w) SELECT i, grad(loss, val) FROM w")
    assert ddx_tables(con) == []


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
