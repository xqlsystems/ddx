# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the `ddxdb` wheel.

These assert on *executed numbers* wherever an engine is involved, not on
rewritten SQL text. A rewrite that looks right but plans to the wrong expression
is the failure mode worth catching, and it is invisible to a string comparison.
"""

import pytest

import ddxdb

# --------------------------------------------------------------------------
# The pure-text surface. No engine, no imports beyond ddxdb itself.
# --------------------------------------------------------------------------


def test_rewrite_replaces_the_marker():
    assert ddxdb.rewrite_sql("SELECT grad(sin(x), x) AS d FROM t") == (
        "SELECT (cos(x)) AS d FROM t"
    )


def test_marker_free_sql_is_returned_byte_identical():
    # ddx never parses a statement with no marker, so it can neither fail it nor
    # reformat it — odd spacing and comments included.
    sql = "SELECT   *   FROM t  /* spacing  preserved */"
    assert ddxdb.rewrite_sql(sql) == sql


def test_differentiate_sql_is_the_escape_hatch():
    assert ddxdb.differentiate_sql("x * y", "x") == "y"


def test_explain_shows_each_marker_without_running_anything():
    ex = ddxdb.explain("SELECT grad(x*x,x) AS a, grad(sin(y),y) AS b FROM t")
    assert ex["rewritten"] == "SELECT (x + x) AS a, (cos(y)) AS b FROM t"
    assert [s["marker"] for s in ex["steps"]] == ["grad(x*x,x)", "grad(sin(y),y)"]
    assert [s["derivative"] for s in ex["steps"]] == ["(x + x)", "(cos(y))"]


def test_duckdb_dialect_folds_quoted_identifiers():
    # DuckDB is case-insensitive even for quoted identifiers, DataFusion is not.
    # Picking the wrong dialect here would silently differentiate to zero rather
    # than fail, which is why the dialect selects the folding policy too.
    assert ddxdb.rewrite_sql('SELECT grad("X" * "X", "X") AS d FROM t', "duckdb") == (
        'SELECT ("X" + "X") AS d FROM t'
    )


# --------------------------------------------------------------------------
# Errors are typed, so a caller can branch on the kind rather than the message.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, expected",
    [
        # No rule for atan2 yet — a limitation to route around.
        ("SELECT grad(atan2(x,y), x) FROM t", ddxdb.UnsupportedExpression),
        # The wrt must be a bare column — a mistake the caller fixes.
        ("SELECT grad(x*y, x+y) FROM t", ddxdb.InvalidMarker),
        # A computed CTE alias used as a non-wrt term would silently drop terms.
        (
            "WITH v AS (SELECT sin(x) AS s FROM t) SELECT grad(s*x, x) FROM v",
            ddxdb.ProjectionBoundary,
        ),
    ],
)
def test_failures_raise_a_specific_class(sql, expected):
    with pytest.raises(expected):
        ddxdb.rewrite_sql(sql)


def test_every_ddx_error_shares_one_base():
    # So `except DdxError` catches all of them and nothing else.
    for cls in (
        ddxdb.UnsupportedExpression,
        ddxdb.InvalidMarker,
        ddxdb.AmbiguousColumn,
        ddxdb.ProjectionBoundary,
        ddxdb.SqlParseError,
    ):
        assert issubclass(cls, ddxdb.DdxError)


def test_an_unknown_dialect_is_a_value_error_naming_the_options():
    with pytest.raises(ValueError) as e:
        ddxdb.rewrite_sql("SELECT 1", "oracle")
    assert "duckdb" in str(e.value)


# --------------------------------------------------------------------------
# The DataFusion shim, on a live engine.
# --------------------------------------------------------------------------

datafusion = pytest.importorskip("datafusion", reason="needs the datafusion extra")


@pytest.fixture
def ctx():
    c = ddxdb.Context()
    c.sql(
        "CREATE TABLE t AS SELECT * FROM (VALUES (1.0,4.0),(2.0,5.0),(3.0,6.0)) AS v(x,y)"
    ).collect()
    return c


def col(ctx, sql):
    batches = ctx.sql(sql).collect()
    return [v for b in batches for v in b.column(0).to_pylist()]


def test_grad_runs_end_to_end(ctx):
    assert col(ctx, "SELECT grad(x*x, x) AS d FROM t ORDER BY x") == [2.0, 4.0, 6.0]


def test_product_rule_picks_the_right_variable(ctx):
    assert col(ctx, "SELECT grad(x*y, x) AS d FROM t ORDER BY x") == [4.0, 5.0, 6.0]
    assert col(ctx, "SELECT grad(x*y, y) AS d FROM t ORDER BY x") == [1.0, 2.0, 3.0]


def test_higher_order_falls_out_of_nesting(ctx):
    assert col(ctx, "SELECT grad(grad(x*x*x,x),x) AS d FROM t ORDER BY x") == [
        6.0,
        12.0,
        18.0,
    ]


def test_jvp_is_the_directional_derivative(ctx):
    assert col(ctx, "SELECT jvp(x*x, x, y) AS d FROM t ORDER BY x") == [8.0, 20.0, 36.0]


def test_grad_inside_an_aggregate_is_one_descent_step(ctx):
    # Differentiating through an aggregate is linearity, so the marker goes
    # inside it — which is what makes a gradient step expressible in SQL.
    assert col(ctx, "SELECT AVG(grad(x*x,x)) AS g FROM t") == [4.0]


def test_a_query_without_markers_is_untouched(ctx):
    assert col(ctx, "SELECT x*2 AS d FROM t ORDER BY x") == [2.0, 4.0, 6.0]


def test_newton_iteration_in_a_recursive_cte(ctx):
    # The shape a training loop needs: a marker inside a recursive term, the
    # whole solve as one query. Both columns are referenced by the outer query,
    # which DataFusion 54 requires for a recursive CTE to plan correctly.
    got = col(
        ctx,
        """
        WITH RECURSIVE newton AS (
            SELECT 0 AS i, 1.0 AS x
            UNION ALL
            SELECT i + 1 AS i, x - (x*x - 2.0) / grad(x*x - 2.0, x) AS x
            FROM newton WHERE i < 6
        )
        SELECT x FROM newton WHERE i = 6
        """,
    )
    assert got == pytest.approx([2**0.5], abs=1e-12)


def test_context_wraps_an_existing_session_context():
    inner = datafusion.SessionContext()
    inner.sql("CREATE TABLE u AS SELECT * FROM (VALUES (3.0)) AS v(x)").collect()
    wrapped = ddxdb.Context(inner)
    assert col(wrapped, "SELECT grad(x*x, x) AS d FROM u") == [6.0]
    assert wrapped.inner is inner


def test_context_forwards_unknown_attributes(ctx):
    # It has to stand in for a SessionContext anywhere one is expected.
    assert ctx.table("t") is not None
    assert callable(ctx.register_udf)


def test_context_rejects_ambiguous_construction():
    with pytest.raises(TypeError):
        ddxdb.Context(datafusion.SessionContext(), some_kwarg=1)


# --------------------------------------------------------------------------
# The DuckDB client-side path.
# --------------------------------------------------------------------------

duckdb = pytest.importorskip("duckdb", reason="needs the duckdb extra")


def test_duckdb_sql_runs_on_your_own_connection():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1.0),(2.0),(3.0)) AS v(x)")
    got = [float(r[0]) for r in ddxdb.duckdb_sql("SELECT grad(x*x,x) AS d FROM t ORDER BY x", con).fetchall()]
    assert got == [2.0, 4.0, 6.0]


def test_duckdb_path_sees_connection_scoped_state():
    # The point of rewriting client-side: it runs on the caller's connection, so
    # temp tables, session settings and an open transaction are all visible. The
    # in-database `ddx('<sql>')` table function executes on a separate inner
    # connection and cannot see any of them.
    con = duckdb.connect(":memory:")
    con.execute("CREATE TEMP TABLE tmp AS SELECT 5.0 AS x")
    got = [float(r[0]) for r in ddxdb.duckdb_sql("SELECT grad(x*x,x) AS d FROM tmp", con).fetchall()]
    assert got == [10.0]
