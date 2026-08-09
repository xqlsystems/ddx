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
integration target. Every other engine uses the line above, which is why there
are no per-engine helpers here to keep in sync.

Errors are typed: ``UnsupportedExpression`` means ddx has no rule for something
you wrote, ``AmbiguousColumn`` means your query needs a qualifier. Catch the one
you can act on rather than matching on message text.
"""

from ._ddxdb import (  # noqa: F401  (re-exported)
    AmbiguousColumn,
    DdxError,
    InvalidMarker,
    ProjectionBoundary,
    SqlParseError,
    UnsupportedExpression,
    differentiate_sql,
    rewrite_sql,
)

__version__ = "0.1.0"

__all__ = [
    "rewrite_sql",
    "differentiate_sql",
    "Context",
    "DdxError",
    "UnsupportedExpression",
    "InvalidMarker",
    "AmbiguousColumn",
    "ProjectionBoundary",
    "SqlParseError",
    "__version__",
]


def __getattr__(name):
    """Build :class:`Context` on first use, so importing ddxdb needs no engine.

    ``Context`` subclasses ``datafusion.SessionContext``, so defining it at
    import time would make ``import ddxdb`` require DataFusion — and then
    ``rewrite_sql``, which needs no engine at all, could not be used without one.
    Module-level ``__getattr__`` (PEP 562) defers the import to the first
    attribute access instead.
    """
    if name == "Context":
        globals()["Context"] = _build_context_class()
        return globals()["Context"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build_context_class():
    try:
        from datafusion import SessionContext
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "ddxdb.Context needs DataFusion: pip install 'ddxdb[datafusion]'. "
            "For any other engine use ddxdb.rewrite_sql(sql, dialect) and hand "
            "the result to that engine yourself."
        ) from e

    class Context(SessionContext):
        """A DataFusion ``SessionContext`` whose ``.sql()`` understands `grad`.

        A real subclass, so every ``SessionContext`` method, property and
        constructor argument works unchanged — only ``sql()`` is overridden::

            from ddxdb import Context

            ctx = Context()
            ctx.sql("CREATE TABLE t AS VALUES (1.0), (2.0), (3.0)").collect()
            ctx.sql("SELECT grad(column1 * column1, column1) AS d FROM t").collect()

        Only ``sql()`` is intercepted, and that is the whole reachable surface:
        the DataFrame API builds expressions directly rather than going through
        SQL text, so a marker cannot arrive that way. Native Rust users get that
        case covered by the in-engine analyzer rule in the ``ddx-datafusion``
        crate; `datafusion-python` exposes no hook to install one.
        """

        def __init__(self, *args, dialect: str = "datafusion", **kwargs):
            super().__init__(*args, **kwargs)
            # Bypass __setattr__ on the PyO3 base, which may not accept
            # arbitrary attributes.
            object.__setattr__(self, "_ddx_dialect", dialect)

        def sql(self, query: str, *args, **kwargs):
            """Rewrite `grad`/`jvp` markers in `query`, then plan it as usual.

            A statement with no marker is passed through byte-identical and is
            never parsed by ddx, so routing every query through here is free.
            """
            return super().sql(
                rewrite_sql(query, self._ddx_dialect), *args, **kwargs
            )

    return Context
