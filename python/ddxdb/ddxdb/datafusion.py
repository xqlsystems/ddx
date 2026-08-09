# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The DataFusion integration: a ``SessionContext`` whose ``.sql()`` understands `grad`.

Importing this module requires DataFusion. That is why it is a module and not
part of ``ddxdb/__init__.py``: ``Context`` subclasses ``SessionContext``, and a
subclass needs its base class at definition time, so defining it beside
:func:`ddxdb.rewrite_sql` would make ``import ddxdb`` require an engine that
``rewrite_sql`` does not need. Python already has a mechanism for "costs nothing
until you ask for it" — a submodule — and ``ddxdb.Context`` re-exports this one
lazily.
"""

try:
    from datafusion import SessionContext
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError(
        "ddxdb.Context needs DataFusion: pip install 'ddxdb[datafusion]'. "
        "For any other engine use ddxdb.rewrite_sql(sql, dialect) and hand "
        "the result to that engine yourself."
    ) from e

from ._ddxdb import rewrite_sql

__all__ = ["Context"]


class Context(SessionContext):
    """A DataFusion ``SessionContext`` whose ``.sql()`` understands `grad`.

    A real subclass, so every ``SessionContext`` method, property and
    constructor argument works unchanged — only ``sql()`` is overridden::

        from ddxdb import Context

        ctx = Context()
        ctx.sql("CREATE TABLE t AS VALUES (1.0), (2.0), (3.0)").collect()
        ctx.sql("SELECT grad(column1 * column1, column1) AS d FROM t").collect()

    Only ``sql()`` is intercepted, and that is the whole reachable surface: the
    DataFrame API builds expressions directly rather than going through SQL
    text, so a marker cannot arrive that way. Native Rust users get that case
    covered by the in-engine analyzer rule in the ``ddx-datafusion`` crate;
    `datafusion-python` exposes no hook to install one.
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
        return super().sql(rewrite_sql(query, self._ddx_dialect), *args, **kwargs)
