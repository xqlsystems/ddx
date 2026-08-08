# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""SQL-portable autograd — write calculus in SQL, get derivatives back as columns.

    SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g

`grad` and `jvp` are *markers*, not row functions. They are rewritten away into
ordinary derivative SQL before the engine ever sees them, so the engine
evaluates a plain expression per row — the relational equivalent of
``jax.vmap(jax.grad(f))``.

Three ways in, in increasing order of how much they do for you:

* :func:`rewrite_sql` — text in, text out. Works with any engine, because the
  result is just SQL. Nothing is imported and nothing is executed.
* :class:`Context` — a drop-in wrapper around a DataFusion ``SessionContext``
  whose ``.sql()`` rewrites first.
* :func:`duckdb_sql` — the same idea for DuckDB.

Errors are typed. ``UnsupportedExpression`` means ddx has no rule for something
you wrote; ``AmbiguousColumn`` means your query needs a qualifier. Catch the one
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
    explain,
    rewrite_sql,
)

__version__ = "0.1.0"

__all__ = [
    "rewrite_sql",
    "differentiate_sql",
    "explain",
    "Context",
    "duckdb_sql",
    "DdxError",
    "UnsupportedExpression",
    "InvalidMarker",
    "AmbiguousColumn",
    "ProjectionBoundary",
    "SqlParseError",
    "__version__",
]


class Context:
    """A DataFusion ``SessionContext`` whose ``.sql()`` understands `grad`.

    Every other attribute is forwarded to the wrapped context, so this stands in
    for a ``SessionContext`` anywhere one is expected::

        from ddxdb import Context

        ctx = Context()
        ctx.sql("CREATE TABLE t AS VALUES (1.0), (2.0), (3.0)").collect()
        ctx.sql("SELECT grad(column1 * column1, column1) AS d FROM t").collect()

    Wrap an existing context by passing it in — useful when something else
    already registered your tables::

        ctx = Context(existing_session_context)

    Only ``sql()`` is intercepted. The DataFrame API builds expressions directly
    rather than going through SQL text, so a marker cannot reach this shim from
    there; native Rust users get that case covered by the in-engine analyzer rule
    in the ``ddx-datafusion`` crate, which `datafusion-python` has no way to
    install (there is no analyzer-rule hook in its FFI surface).
    """

    __slots__ = ("_ctx", "_dialect")

    def __init__(self, ctx=None, *, dialect: str = "datafusion", **kwargs):
        if ctx is None:
            # Imported here, not at module scope, so `import ddxdb` costs nothing
            # for someone who only wants `rewrite_sql`.
            from datafusion import SessionContext

            ctx = SessionContext(**kwargs)
        elif kwargs:
            raise TypeError(
                "pass either an existing context or keyword arguments to build "
                "one, not both"
            )
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "_dialect", dialect)

    @property
    def inner(self):
        """The wrapped ``SessionContext``, for the rare call that needs it."""
        return self._ctx

    def sql(self, query: str, **kwargs):
        """Rewrite `grad`/`jvp` markers in `query`, then plan it as usual.

        A statement with no marker is passed through byte-identical and is never
        parsed by ddx, so routing every query through here is free.
        """
        return self._ctx.sql(rewrite_sql(query, self._dialect), **kwargs)

    def __getattr__(self, name):
        # Reached only for attributes this class does not define, so `sql` is
        # never forwarded.
        return getattr(self._ctx, name)

    def __repr__(self):
        return f"ddxdb.Context({self._ctx!r}, dialect={self._dialect!r})"


def duckdb_sql(query: str, connection=None, **kwargs):
    """Rewrite `grad`/`jvp` markers in `query` and run it on DuckDB.

    The client-side path: the rewrite happens here, and DuckDB is handed plain
    SQL, so this needs no extension installed::

        import ddxdb
        ddxdb.duckdb_sql("SELECT grad(sin(x), x) AS d FROM t").fetchall()

    Pass ``connection`` to run on a specific one; otherwise DuckDB's default
    connection is used. Because the rewrite happens on *your* connection, this
    sees your temp tables, session settings and open transaction — which the
    in-database ``ddx('<sql>')`` table function cannot, since it executes on a
    separate inner connection.
    """
    import duckdb

    rewritten = rewrite_sql(query, "duckdb")
    return (connection or duckdb).sql(rewritten, **kwargs)
