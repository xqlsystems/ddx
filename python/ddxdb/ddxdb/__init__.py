# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Autograd for SQL: the gradient of a whole query, as JAX takes the gradient
of a function.

Write a model's forward pass and loss as SQL, and take its gradient in SQL::

    from ddxdb import Context

    ctx = Context()
    ctx.sql('''
        WITH loss AS (SELECT SUM(val * val) AS l FROM w)
        SELECT w.i, w.val - 0.1 * g.val AS val
        FROM w JOIN grad(loss, w.val) g ON w.i = g.i
    ''')

:class:`Context` is a DataFusion ``SessionContext`` whose ``.sql()``
understands ``grad``. It lives in :mod:`ddxdb.datafusion` and is re-exported
here, so it costs an engine import only if you use it. :mod:`ddxdb.ad` has the
programs underneath (:func:`grad_plan`, :func:`vjp_plan`, :func:`run`), which
run on any engine that implements :class:`Backend`.

Errors are typed: ``UnsupportedExpression`` means ddx has no rule for something
you wrote, ``InvalidColumn`` that a column cannot be differentiated. Catch the
one you can act on rather than matching on message text.
"""

from importlib import metadata as _metadata
from typing import Protocol

from ._ddxdb import (  # noqa: F401  (re-exported)
    DdxError,
    InvalidColumn,
    NotScalar,
    UnknownColumn,
    UnsupportedExpression,
    supported_functions,
)
from ._ddxdb import (  # noqa: F401  (re-exported)
    BackwardProgram,
    Check,
    ForwardProgram,
    InputTable,
    OutputTable,
    Step,
    Tangent,
    Verified,
    grad_plan,
    jvp_plan,
    run,
    vjp_plan,
)


class Backend(Protocol):
    """An engine :func:`run` can run a program on: the four primitives of
    ``ddx_ad::Backend``. Plans cross as serialized Substrait, with their reads
    of earlier steps already bound. :class:`ddxdb.ad.DataFusionBackend` is
    one; any object with these methods is another."""

    def select_all(self, name: str) -> bytes:
        """The engine's own serialized Substrait plan of ``SELECT * FROM
        name``, whose read gives the table's schema in the engine's terms."""
        ...

    def returns_rows(self, plan: bytes) -> bool:
        """Run ``plan`` and say whether it returned any row."""
        ...

    def materialize(self, name: str, plan: bytes) -> None:
        """Run ``plan`` and store its rows as the table ``name``, replacing
        one."""
        ...

    def drop_table(self, name: str) -> None:
        """Drop the table ``name`` if it exists."""
        ...


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
    "supported_functions",
    "DdxError",
    "UnsupportedExpression",
    "NotScalar",
    "UnknownColumn",
    "InvalidColumn",
    "Backend",
    "BackwardProgram",
    "Check",
    "ForwardProgram",
    "InputTable",
    "OutputTable",
    "Step",
    "Tangent",
    "Verified",
    "grad_plan",
    "jvp_plan",
    "run",
    "vjp_plan",
    "__version__",
]


def __dir__():
    # Keeps Context in tab-completion and help(), which read __dir__ rather
    # than __all__, without a wildcard import pulling it in.
    return sorted([*__all__, "Context"])


def __getattr__(name):
    """Re-export :class:`ddxdb.datafusion.Context` without importing DataFusion.

    ``Context`` subclasses ``datafusion.SessionContext``, so its module cannot
    be imported without an engine — and the programs, which run on any
    :class:`Backend`, live here. Module-level ``__getattr__`` (PEP 562) defers the
    submodule import to the first attribute access, so ``ddxdb.Context`` reads
    as one namespace while costing nothing to anyone who never touches it.
    """
    if name == "Context":
        from .datafusion import Context

        globals()["Context"] = Context
        return Context
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
