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
        SELECT i, val - 0.1 * grad(loss, val) AS val FROM w
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

import re
from dataclasses import dataclass, field
from importlib import metadata as _metadata
from typing import Optional, Protocol, Sequence

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
    Programs,
    Step,
    Tangent,
    grad_plan,
    jvp_plan,
    run,
    vjp_plan,
)
from ._ddxdb import _prepare, _rewrite


#: One part of a table name: quoted, with ``""`` for a quote inside it, or
#: unquoted, up to the next dot.
_PART = r'"(?:[^"]|"")*"|[^."]+'
_NAME = re.compile(rf"(?:{_PART})(?:\.(?:{_PART}))*")


def _name_parts(name: str, normalize: bool = True) -> Optional[list]:
    """The parts of a table name as SQL writes it: a quoted part as
    written, an unquoted one in lower case if ``normalize``, as DataFusion
    resolves them by default; ``None`` if it does not parse."""
    if not _NAME.fullmatch(name):
        return None
    return [
        part[1:-1].replace('""', '"')
        if part.startswith('"')
        else part.lower()
        if normalize
        else part
        for part in re.findall(_PART, name)
    ]


class Backend(Protocol):
    """An engine ddx runs on: the primitives of ``ddx_ad::Backend``.
    :func:`run` needs the first four to run a program; rewriting SQL that
    calls ``grad``, ``vjp`` or ``jvp`` (:func:`rewrite`, :func:`prepare`)
    also needs :meth:`plan_sql`. Plans
    cross as serialized Substrait, with their reads of earlier steps already
    bound. :class:`ddxdb.ad.DataFusionBackend` is one; any object with these
    methods is another."""

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

    def plan_sql(self, sql: str) -> bytes:
        """The engine's serialized Substrait plan of the SQL query ``sql``.
        Only :func:`ddxdb.ad.sql`'s rewrite asks for one, to differentiate a
        query a statement names; a backend that only runs programs may
        leave it out."""
        ...

    def plan_key(self, sql: str) -> Optional[str]:
        """Optional. Text that is the same at two calls only if
        :meth:`plan_sql` of ``sql`` is: the schemas of the tables it reads
        and the definitions of the views it reads, found without optimizing.
        :func:`rewrite` keeps each program it builds in a :class:`Programs`
        under it, so leaving out something the plan depends on would rerun
        a stale program. A backend without it, or that returns ``None``,
        gets nothing kept."""
        ...

    def view_sql(self, name: str) -> Optional[str]:
        """Optional. The SQL that defines ``name``, as a query writes it, if
        it names a view whose SQL the engine keeps: a ``CREATE VIEW``
        statement, or its query. :func:`rewrite` writes such a view into the
        objective as CTEs, so a model that is a view has the CTEs it reads
        more than once planned once each. A backend without it, or that
        returns ``None``, leaves every view to the engine."""
        ...

    def views_key(self) -> Optional[str]:
        """Optional. Text that is the same at two calls only if the engine
        runs each of its views as it did: every view's name and what the
        engine runs for it. A :class:`Prepared` run compares it with the
        one its statements were prepared with, and prepares them again if
        it differs. A backend without it, or that returns ``None``, gets
        them prepared on every run."""
        ...


@dataclass(frozen=True)
class Rewritten:
    """SQL statements rewritten by :func:`rewrite`, and the tables they read.

    Use it as a context manager: the tables are dropped from the backend when
    the block ends, once the statements have been planned or run::

        with ddxdb.rewrite(backend, statements) as rewritten:
            for sql in rewritten.statements:
                con.sql(sql)
    """

    #: Each statement, its ``grad``, ``vjp`` and ``jvp`` calls replaced by
    #: reads of what their programs computed; one with no call, as it was.
    statements: list
    #: The tables the statements read: the programs' values and gradients,
    #: and the cotangents and tangents registered for them.
    tables: list
    backend: "Backend"
    _release: object = field(repr=False, compare=False)

    def release(self) -> None:
        """Drop :attr:`tables` from the backend, every one even after a failed
        drop; the first failure is raised. Then the programs that wrote them
        are kept in the :class:`Programs` given to :func:`rewrite`: not
        before, or another call could run one and write the same tables while
        these statements still read them. A second call does nothing."""
        self._release.release(self.backend)

    def __enter__(self) -> "Rewritten":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            self.release()
            return
        # The error that ended the block is the one to see: a failed drop
        # is noted on it, not raised in its place.
        try:
            self.release()
        except Exception as e:  # noqa: BLE001 - noted on the error in flight
            if hasattr(exc, "add_note"):  # Python 3.11+
                exc.add_note(f"Dropping the rewritten statements' tables failed too: {e!r}")


def rewrite(
    backend: "Backend",
    statements: Sequence[str],
    dialect: str = "datafusion",
    programs: Optional[Programs] = None,
) -> Rewritten:
    """Compute what the ``grad``, ``vjp`` and ``jvp`` calls in ``statements``
    read, on ``backend`` (any :class:`Backend` with :meth:`Backend.plan_sql`),
    and rewrite each statement to read it. ``dialect`` is how the statements
    are written: a dialect name ``sqlparser`` knows, or ``"datafusion"``.

    A program kept in ``programs`` from an earlier call is run again instead
    of built, and each one built is kept there once
    :meth:`Rewritten.release` drops its tables, so a training loop that
    passes the same :class:`Programs` each step builds its programs once
    (with a backend that has :meth:`Backend.plan_key`).

    This is ``ddx_ad::rewrite``, ddx-ad's own Rust: it plans one program per
    distinct derivative, runs each on the backend, and rewrites the
    statements' parsed trees. Raises a :class:`DdxError` subclass when ddx
    cannot differentiate a call, or what the backend raised; either way,
    it first tries to drop every table it wrote.
    """
    # A str is a sequence of strs: without this, each character would be a
    # statement, and the calls would never be found.
    if isinstance(statements, str):
        raise TypeError("statements is a list of SQL statements; to rewrite one, pass [statement]")
    out, tables, release = _rewrite(backend, list(statements), dialect, programs)
    return Rewritten(out, tables, backend, release)


class Prepared:
    """Statements prepared to run again on one backend (:func:`prepare`).

    The first :meth:`run` computes the ``grad``, ``vjp`` and ``jvp`` calls
    as :func:`rewrite` does, and every later one runs the same plans again,
    each as it was bound then, as ``jax.jit`` traces a function once and
    runs what it traced after that (``ddx_ad::Prepared``). A later run asks
    the backend for no plan, key or schema: only for its ``views_key()``,
    and to run plans.

    In return, the tables the statements read must keep their schemas.
    Between runs, a table may change its rows, or be registered again under
    its name, but not change its columns or their types; prepare the
    statements again after changing one. A run whose backend's
    ``views_key()`` differs from the one the statements were prepared with,
    because a view was created, redefined or dropped, prepares them again.
    A backend without ``views_key`` gets them prepared on every run.

    Every run writes the same tables, so runs must not overlap: release one
    run's :class:`Rewritten` before the next."""

    def __init__(self, backend: "Backend", statements: Sequence[str], dialect: str = "datafusion"):
        # As in rewrite: a str would be a statement per character.
        if isinstance(statements, str):
            raise TypeError("statements is a list of SQL statements; to prepare one, pass [statement]")
        self.backend = backend
        self.statements = list(statements)
        self.dialect = dialect
        self._prepared = None

    def run(self) -> Rewritten:
        """Compute what the statements' calls read, and the statements
        rewritten to read it, as :func:`rewrite` returns them. The first run
        that succeeds prepares the statements, as does the first after a
        view changes; one that fails prepares nothing, and the next tries
        again. Either way, a failed run first tries to drop every table it
        wrote."""
        ran = None if self._prepared is None else self._prepared.run(self.backend)
        if ran is None:
            prepared, out, tables, release = _prepare(self.backend, self.statements, self.dialect)
            self._prepared = prepared
        else:
            out, tables, release = ran
        return Rewritten(out, tables, self.backend, release)

    def __repr__(self) -> str:
        state = "prepared" if self._prepared is not None else "not yet run"
        return f"Prepared({len(self.statements)} statements, {state})"


def prepare(backend: "Backend", statements: Sequence[str], dialect: str = "datafusion") -> Prepared:
    """``statements``, prepared to run again on ``backend``: see
    :class:`Prepared`. Nothing runs until :meth:`Prepared.run`."""
    return Prepared(backend, statements, dialect)


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
    "Prepared",
    "Rewritten",
    "prepare",
    "rewrite",
    "BackwardProgram",
    "Check",
    "ForwardProgram",
    "InputTable",
    "OutputTable",
    "Programs",
    "Step",
    "Tangent",
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
