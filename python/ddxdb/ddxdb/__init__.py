# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""SQL-portable autograd — write calculus in SQL, get derivatives back as columns.

    SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g

`grad` and `jvp` are *markers*, not row functions. :func:`rewrite_sql` turns them
into ordinary derivative SQL before any engine sees them, so what runs is a plain
expression evaluated per row — the relational equivalent of
``jax.vmap(jax.grad(f))``, with the rows as the batch dimension.

:func:`rewrite_sql` is the whole library. It is text in, text out, so it works
with **any** engine that accepts SQL — pass the result wherever you would have
passed the original::

    con.sql(ddxdb.rewrite_sql(q, "duckdb"))        # DuckDB
    session.sql(ddxdb.rewrite_sql(q, "spark"))     # Spark
    ctx.sql(ddxdb.rewrite_sql(q))                  # DataFusion

:class:`Context` is sugar for one engine — a DataFusion ``SessionContext``
subclass whose ``.sql()`` rewrites first — because DataFusion is ddx's
integration target. It lives in :mod:`ddxdb.datafusion` and is re-exported here,
so it costs an engine import only if you use it. Every other engine uses the
line above, which is why there are no per-engine helpers here to keep in sync.

Errors are typed: ``UnsupportedExpression`` means ddx has no rule for something
you wrote, ``AmbiguousColumn`` means your query needs a qualifier. Catch the one
you can act on rather than matching on message text.
"""

from importlib import metadata as _metadata

from ._ddxdb import (  # noqa: F401  (re-exported)
    AmbiguousColumn,
    DdxError,
    InvalidMarker,
    ProjectionBoundary,
    SqlParseError,
    UnsupportedExpression,
    differentiate_sql,
    rewrite_sql,
    supported_functions,
)

# Single-sourced from the installed distribution metadata, which maturin fills
# from pyproject.toml. Hardcoding it here would be a third copy of the version
# (pyproject, Cargo.toml, here) with nothing keeping them equal, and the one
# most likely to be forgotten is the one users read.
__version__ = _metadata.version("ddxdb")

# `Context` is deliberately absent: __all__ is what `from ddxdb import *` pulls
# in, and naming Context there would make a star-import call __getattr__ and
# import DataFusion — reintroducing, for that one spelling, the engine
# requirement this module exists to avoid. It stays public and importable by
# name (`from ddxdb import Context` goes through __getattr__ just the same) and
# stays listed in __dir__ below, so only the wildcard skips it.
__all__ = [
    "rewrite_sql",
    "differentiate_sql",
    "supported_functions",
    "DdxError",
    "UnsupportedExpression",
    "InvalidMarker",
    "AmbiguousColumn",
    "ProjectionBoundary",
    "SqlParseError",
    "__version__",
]


def __dir__():
    # Keeps Context in tab-completion and help(), which read __dir__ rather
    # than __all__, without a wildcard import pulling it in.
    return sorted([*__all__, "Context"])


def __getattr__(name):
    """Re-export :class:`ddxdb.datafusion.Context` without importing DataFusion.

    ``Context`` subclasses ``datafusion.SessionContext``, so its module cannot
    be imported without an engine — and ``rewrite_sql``, which needs no engine
    at all, lives here. Module-level ``__getattr__`` (PEP 562) defers the
    submodule import to the first attribute access, so ``ddxdb.Context`` reads
    as one namespace while costing nothing to anyone who never touches it.
    """
    if name == "Context":
        from .datafusion import Context

        globals()["Context"] = Context
        return Context
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
