# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The DataFusion integration: a ``SessionContext`` whose ``.sql()`` understands `grad`.

Importing this module requires DataFusion. That is why it is a module and not
part of ``ddxdb/__init__.py``: ``Context`` subclasses ``SessionContext``, and a
subclass needs its base class at definition time, so defining it beside
:func:`ddxdb.grad_plan` would make ``import ddxdb`` require an engine that the
programs do not need. Python already has a mechanism for "costs nothing
until you ask for it" — a submodule — and ``ddxdb.Context`` re-exports this one
lazily.
"""

try:
    from datafusion import SessionContext
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError(
        "ddxdb.Context needs DataFusion: pip install 'ddxdb[datafusion]'. "
        "For any other engine, run ddxdb.grad_plan's programs on a ddxdb.Backend."
    ) from e


__all__ = ["Context"]


class Context(SessionContext):
    """A DataFusion ``SessionContext`` whose ``.sql()`` understands `grad`.

    A real subclass, so every ``SessionContext`` method, property and
    constructor argument works unchanged — only ``sql()`` is overridden::

        from ddxdb import Context

        ctx = Context()
        ctx.sql("CREATE TABLE w (i BIGINT, val DOUBLE) AS VALUES (0, 1.0), (1, 2.0)")
        ctx.sql(
            "WITH loss AS (SELECT SUM(val * val) AS l FROM w) "
            "SELECT i, grad(loss, val) FROM w"
        ).collect()
    """

    def __init__(self, *args, dialect: str = "datafusion", **kwargs):
        super().__init__(*args, **kwargs)
        # Bypass __setattr__ on the PyO3 base, which may not accept
        # arbitrary attributes.
        object.__setattr__(self, "_ddx_dialect", dialect)

    def sql(self, query: str, *args, **kwargs):
        """Run `grad`, `vjp` and `jvp` in `query`, then plan it as usual.

        As columns of a select list, `grad(loss, column)` is the gradient at
        each row of the loss a CTE computes; `vjp(f, column, cotangent)` the
        same for any CTE along a cotangent; and `jvp(e, table.column,
        tangent)` the tangent of an expression along a tangent table (see
        :mod:`ddxdb.ad`). A statement with none is planned as it is.
        """
        if args or kwargs:
            return super().sql(query, *args, **kwargs)
        from . import ad

        return ad.sql(self, query, dialect=self._ddx_dialect)
