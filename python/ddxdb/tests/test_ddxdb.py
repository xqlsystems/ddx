# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the `ddxdb` wheel.

These assert on *executed numbers* wherever an engine is involved. The
programs themselves are tested in `test_ad.py`; this file is the package's
shape: its version, its errors, and `Context`.
"""

import pytest

import ddxdb

# The engine is gated by the fixture that needs it, never by a module-level
# `importorskip`. `importorskip` raises during collection, so it skips
# *everything below it* in the file, including the test asserting ddxdb imports
# with no engine installed, which is the one test that only means anything in
# exactly the environment where that gate fires.

# --------------------------------------------------------------------------
# The package, with no engine.
# --------------------------------------------------------------------------


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


def test_every_ddx_error_shares_one_base():
    # So `except DdxError` catches all of them and nothing else.
    for cls in (
        ddxdb.UnsupportedExpression,
        ddxdb.NotScalar,
        ddxdb.UnknownColumn,
        ddxdb.InvalidColumn,
    ):
        assert issubclass(cls, ddxdb.DdxError)


def test_ddxdb_is_usable_with_no_engine_installed():
    # The programs run on any Backend. Context subclasses SessionContext, so it
    # is built lazily on first access rather than at import — and `import *`
    # does not reach it either, which is why Context is absent from __all__.
    #
    # The engine is blocked at the import system rather than merely assumed
    # absent: CI installs it, so a subprocess that simply imports ddxdb would
    # pass no matter what this module does at import time. Blocking makes the
    # claim testable in the environment the tests actually run in.
    import subprocess
    import sys

    script = """
import sys

class NoEngines:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "datafusion":
            raise ImportError(f"blocked for this test: {fullname}")
        return None

sys.meta_path.insert(0, NoEngines())

from ddxdb import *
assert callable(grad_plan) and callable(run)
assert "tanh" in supported_functions()

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


# --------------------------------------------------------------------------
# Context, on a live engine.
# --------------------------------------------------------------------------


@pytest.fixture
def ctx():
    pytest.importorskip("datafusion", reason="needs the datafusion extra")
    c = ddxdb.Context()
    c.sql(
        "CREATE TABLE t AS SELECT * FROM (VALUES (0, 1.0), (1, 2.0), (2, 3.0)) AS v(i, x)"
    ).collect()
    return c


def col(ctx, sql):
    batches = ctx.sql(sql).collect()
    return [v for b in batches for v in b.column(0).to_pylist()]


def test_grad_runs_end_to_end(ctx):
    got = col(
        ctx,
        "WITH loss AS (SELECT SUM(x * x) AS l FROM t) "
        "SELECT grad(loss, x) FROM t ORDER BY i",
    )
    assert got == [2.0, 4.0, 6.0]


def test_a_query_without_grad_is_untouched(ctx):
    assert col(ctx, "SELECT x*2 AS d FROM t ORDER BY x") == [2.0, 4.0, 6.0]


def test_an_unknown_dialect_is_a_value_error():
    pytest.importorskip("datafusion", reason="needs the datafusion extra")
    with pytest.raises(ValueError) as e:
        ddxdb.Context(dialect="klingon").sql("SELECT 1")
    assert "klingon" in str(e.value)


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
