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

# Each engine is gated by the fixture that needs it, never by a module-level
# `importorskip`. `importorskip` raises during collection, so it skips
# *everything below it* in the file: a missing `datafusion` would have taken the
# DuckDB tests with it, and — worse — the test asserting ddxdb works with no
# engine installed, which is the one test that only means anything in exactly
# the environment where that gate fires.

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


def test_version_is_single_sourced_from_package_metadata():
    # __version__ reads the installed distribution's metadata rather than
    # restating the number, so pyproject.toml is the only place it lives on the
    # Python side. Cargo.toml carries its own copy that maturin does not use for
    # the wheel; this asserts they have not drifted anyway, since a mismatch
    # would be confusing rather than harmful.
    import re
    from pathlib import Path

    cargo = (Path(__file__).parent.parent / "Cargo.toml").read_text()
    cargo_version = re.search(r'^version = "([^"]+)"', cargo, re.M).group(1)
    assert ddxdb.__version__ == cargo_version


def test_differentiate_sql_is_the_escape_hatch():
    assert ddxdb.differentiate_sql("x * y", "x") == "y"


@pytest.mark.parametrize(
    "dialect",
    ["generic", "datafusion", "postgres", "postgresql", "ansi",
     "snowflake", "oracle", "duckdb", "mysql", "sqlite", "bigquery",
     "redshift", "hive", "spark", "sparksql", "databricks", "mssql",
     "teradata", "clickhouse"],
)
def test_every_dialect_sqlparser_parses_has_an_established_folding_rule(dialect):
    # Parsing is delegated to sqlparser; folding is ddx's own knowledge, and its
    # table is exhaustive over what sqlparser parses rather than a list of
    # exceptions to a default. A dialect added upstream therefore cannot
    # silently inherit Postgres semantics — it is refused until someone
    # establishes its rule. This is every name sqlparser accepts today.
    assert ddxdb.rewrite_sql("SELECT grad(x*x, x) AS d FROM t", dialect) == (
        "SELECT (x + x) AS d FROM t"
    )


@pytest.mark.parametrize(
    "dialect, sql, expected",
    [
        # Unquoted folds to lowercase, so bare X is "x" and never "X".
        ("postgres", 'grad("x" * "x", X)', '("x" + "x")'),
        ("postgres", 'grad("X" * "X", X)', "(0.0)"),
        ("datafusion", 'grad("x" * "x", X)', '("x" + "x")'),
        # Unquoted folds to UPPERCASE: the same query resolves the other way.
        ("snowflake", 'grad("X" * "X", X)', '("X" + "X")'),
        ("snowflake", 'grad("x" * "x", X)', "(0.0)"),
        ("oracle", 'grad("X" * "X", X)', '("X" + "X")'),
        # Case-insensitive throughout: quoting does not pin anything.
        ("duckdb", 'grad("X" * "X", X)', '("X" + "X")'),
        ("spark", "grad(`X` * `X`, x)", "(`X` + `X`)"),
        ("mysql", "grad(`X` * `X`, x)", "(`X` + `X`)"),
        # Case-sensitive throughout: X and x really are different columns, so
        # zero is the right answer here rather than a missed match.
        ("clickhouse", "grad(X * X, x)", "(0.0)"),
        ("clickhouse", "grad(X * X, X)", "(X + X)"),
    ],
)
def test_identifier_folding_follows_the_engine_not_a_default(dialect, sql, expected):
    # The one thing sqlparser cannot tell us, and the reason the folding policy
    # travels with the dialect rather than being a separate argument. Engines
    # disagree three ways about which column an identifier names, and picking
    # the wrong rule does not raise: it silently differentiates with respect to
    # a column the user did not name — zero if nothing matches, and a confident
    # wrong answer if the *other* column does.
    got = ddxdb.rewrite_sql(f"SELECT {sql} AS d FROM t", dialect)
    assert got == f"SELECT {expected} AS d FROM t"




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


def test_ddxdb_is_usable_with_no_engine_installed():
    # rewrite_sql is text in, text out. Context subclasses SessionContext, so it
    # is built lazily on first access rather than at import — and `import *`
    # does not reach it either, which is why Context is absent from __all__.
    #
    # The engines are blocked at the import system rather than merely assumed
    # absent: CI installs both extras, so a subprocess that simply imports ddxdb
    # would pass no matter what this module does at import time. Blocking makes
    # the claim testable in the environment the tests actually run in.
    import subprocess
    import sys

    script = """
import sys

class NoEngines:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("datafusion", "duckdb"):
            raise ImportError(f"blocked for this test: {fullname}")
        return None

sys.meta_path.insert(0, NoEngines())

from ddxdb import *
assert rewrite_sql("SELECT grad(x*x, x) AS d FROM t") == "SELECT (x + x) AS d FROM t"

# ...and the lazy attribute still reports the missing extra usefully.
import ddxdb
try:
    ddxdb.Context
except ImportError as e:
    assert "ddxdb[datafusion]" in str(e), e
else:
    raise SystemExit("Context must raise ImportError when DataFusion is absent")
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )


def test_an_unknown_dialect_is_a_value_error_naming_the_options():
    # "oracle" used to land here; it is a real sqlparser dialect and now works,
    # which is the delegation doing its job. Only a genuine non-dialect fails.
    with pytest.raises(ValueError) as e:
        ddxdb.rewrite_sql("SELECT 1", "klingon")
    assert "duckdb" in str(e.value)


# --------------------------------------------------------------------------
# The DataFusion shim, on a live engine.
# --------------------------------------------------------------------------


@pytest.fixture
def ctx():
    pytest.importorskip("datafusion", reason="needs the datafusion extra")
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


def test_context_is_a_real_session_context(ctx):
    # A subclass, not a proxy — so it is accepted anywhere a SessionContext is,
    # and every inherited method works without being forwarded by hand.
    import datafusion

    assert isinstance(ctx, datafusion.SessionContext)
    assert ctx.table("t") is not None
    assert callable(ctx.register_udf)
    assert "t" in ctx.catalog().schema().names()


def test_context_is_an_ordinary_class_at_a_real_location(ctx):
    # It is defined in a submodule, not manufactured by a factory closure on
    # first access. A class built inside a function has a __qualname__ like
    # `_build.<locals>.Context`, which names nothing importable: pickle cannot
    # find it, and neither can autodoc, mypy or an IDE. Deferring the *import*
    # buys the same laziness with none of that.
    import ddxdb.datafusion

    assert type(ctx).__module__ == "ddxdb.datafusion"
    assert type(ctx).__qualname__ == "Context"
    assert ddxdb.Context is ddxdb.datafusion.Context


# --------------------------------------------------------------------------
# The DuckDB client-side path.
# --------------------------------------------------------------------------


@pytest.fixture
def duckdb():
    return pytest.importorskip("duckdb", reason="needs the duckdb extra")


def test_any_engine_works_through_plain_rewrite_sql(duckdb):
    # There is no DuckDB helper, and deliberately so: rewrite_sql is text in,
    # text out, so every engine is one line and none of them needs code here
    # that could rot. This is the same line a Polars or Spark user writes.
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1.0),(2.0),(3.0)) AS v(x)")
    sql = ddxdb.rewrite_sql("SELECT grad(x*x,x) AS d FROM t ORDER BY x", "duckdb")
    assert [float(r[0]) for r in con.sql(sql).fetchall()] == [2.0, 4.0, 6.0]


def test_rewriting_client_side_sees_connection_scoped_state(duckdb):
    # Why the text interface is the right one: the rewrite runs in the caller's
    # process against the caller's own connection, so a temp table — which
    # exists only on that connection — is visible to it. A rewrite performed
    # inside the database, on a connection of its own, could not see this table
    # at all.
    con = duckdb.connect(":memory:")
    con.execute("CREATE TEMP TABLE tmp AS SELECT 5.0 AS x")
    sql = ddxdb.rewrite_sql("SELECT grad(x*x,x) AS d FROM tmp", "duckdb")
    assert [float(r[0]) for r in con.sql(sql).fetchall()] == [10.0]
