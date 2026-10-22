# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Gradients of whole queries on DuckDB.

Write a model's forward pass and loss as SQL on a DuckDB connection, and take
its gradient in SQL, as :mod:`ddxdb.ad` does on DataFusion::

    import duckdb
    from ddxdb import duckdb as ad

    con = duckdb.connect()
    con.execute("CREATE TABLE w AS SELECT * FROM (VALUES (0, 1.0), (1, 2.0)) v(i, val)")
    ad.sql(con, '''
        WITH loss AS (SELECT SUM(val * val) AS l FROM w)
        SELECT i, val - 0.1 * grad(loss, val) AS val FROM w
    ''')

``grad``, ``vjp`` and ``jvp`` mean what they mean in :mod:`ddxdb.ad`. The
tables a loss reads may be anything the connection's own queries read: tables,
temporary tables, views, Arrow data registered with ``con.register`` (as
xarray-sql's ``xql.register`` registers a Dataset), and tables made in the
connection's open transaction.

How it runs (:class:`DuckDBBackend`):

- **Planning** uses DuckDB's own Substrait producer (the community ``substrait``
  extension's ``get_substrait``), so a loss means what DuckDB means by it: its
  types, its ``/``, its functions. The producer refuses a scan of registered
  Arrow data, so it plans on a second connection of its own whose catalog
  mirrors the caller's with schemas only: an empty table for each table or
  registered object, and each view's SQL.
- **Running** writes each step of a program as DuckDB SQL
  (``ddx_ad::to_sql``) and runs it on the caller's connection, as a temporary
  table, so it reads what the caller's queries read. DuckDB's Substrait
  consumer cannot: it sees only committed persistent tables, and refuses the
  window functions ddx's rules write.

Every table a program writes is named with the prefix ``__ddx_``, which ddxdb
reserves, and none stays once a call returns.

Importing this module requires DuckDB.
"""

from __future__ import annotations

import weakref
from typing import Optional, Sequence

try:
    import duckdb
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError("ddxdb.duckdb needs DuckDB: pip install 'ddxdb[duckdb]'") from e

from . import Programs, rewrite
from ._ddxdb import to_sql

__all__ = ["DuckDBBackend", "sql", "sql_all"]


def _ident(name: str) -> str:
    """``name`` as a quoted DuckDB identifier."""
    return '"' + name.replace('"', '""') + '"'


def _load_substrait(con: "duckdb.DuckDBPyConnection") -> None:
    """Load the community ``substrait`` extension on ``con``, installing it
    first if it is not."""
    try:
        con.execute("LOAD substrait")
    except duckdb.Error:
        con.execute("INSTALL substrait FROM community")
        con.execute("LOAD substrait")


class DuckDBBackend:
    """A DuckDB connection as a :class:`ddxdb.Backend`: plans with DuckDB's
    Substrait producer on a schema-only mirror of the connection's catalog,
    and runs every step as SQL on the connection itself (see the module
    docs)."""

    def __init__(self, con: "duckdb.DuckDBPyConnection"):
        self.con = con
        self._planner = duckdb.connect()
        _load_substrait(self._planner)
        # What the planner mirrors: each object's columns or view SQL, as last
        # copied, so an unchanged catalog is not copied again.
        self._mirrored: dict = {}

    # The caller's catalog, and the planner's copy of it.

    def _catalog(self) -> dict:
        """Each object on the caller's connection a query can read, but ddx's
        own: its name, and either ``("table", columns)``, with columns as
        ``(name, type)``, or ``("view", sql)``. Registered Arrow data is a
        view with no SQL, and is a table here."""
        columns: dict = {}
        for table, column, ty in self.con.execute(
            "SELECT table_name, column_name, data_type FROM duckdb_columns() "
            "WHERE NOT internal ORDER BY table_name, column_index"
        ).fetchall():
            columns.setdefault(table, []).append((column, ty))
        views = dict(
            self.con.execute(
                "SELECT view_name, sql FROM duckdb_views() WHERE NOT internal"
            ).fetchall()
        )
        catalog = {}
        for name, cols in columns.items():
            if name.startswith("__ddx_"):
                continue
            sql = views.get(name)
            catalog[name] = ("view", sql) if sql else ("table", tuple(cols))
        return catalog

    def _drop_mirrored(self, name: str) -> None:
        """Drop the planner's object ``name``, a table or a view: DuckDB
        drops each only as what it is."""
        kinds = self._planner.execute(
            "SELECT table_type FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchall()
        for (kind,) in kinds:
            what = "VIEW" if kind == "VIEW" else "TABLE"
            self._planner.execute(f"DROP {what} IF EXISTS {_ident(name)}")

    def _mirror_table(self, name: str, columns) -> None:
        cols = ", ".join(f"{_ident(c)} {ty}" for c, ty in columns)
        self._drop_mirrored(name)
        self._planner.execute(f"CREATE TABLE {_ident(name)} ({cols})")

    def _sync(self) -> dict:
        """Make the planner's catalog mirror the caller's, and return the
        caller's."""
        catalog = self._catalog()
        if catalog == self._mirrored:
            return catalog
        for name in self._mirrored.keys() - catalog.keys():
            self._drop_mirrored(name)
        views = []
        for name, (kind, what) in catalog.items():
            if self._mirrored.get(name) == (kind, what):
                continue
            if kind == "table":
                self._mirror_table(name, what)
            else:
                views.append((name, what))
        # A view may read another, so each is created once those it reads
        # are: as many rounds as it takes.
        while views:
            failed = []
            for name, sql in views:
                try:
                    self._drop_mirrored(name)
                    self._planner.execute(sql)
                except duckdb.Error:
                    failed.append((name, sql))
            if len(failed) == len(views):
                names = ", ".join(n for n, _ in failed)
                raise duckdb.Error(f"ddxdb could not copy the views {names} to plan with")
            views = failed
        self._mirrored = catalog
        return catalog

    def _plan(self, sql: str) -> bytes:
        return bytes(self._planner.execute("SELECT * FROM get_substrait(?)", [sql]).fetchone()[0])

    # The Backend protocol.

    def select_all(self, name: str) -> bytes:
        self._sync()
        return self._plan(f"SELECT * FROM {_ident(name)}")

    def plan_sql(self, sql: str) -> bytes:
        self._sync()
        return self._plan(sql)

    def plan_key(self, sql: str) -> Optional[str]:
        # The plan depends on the query and on the schemas and view
        # definitions it reads, which the catalog holds; not on any values.
        return repr((sql, sorted(self._sync().items())))

    def returns_rows(self, plan: bytes) -> bool:
        query = to_sql(plan, "duckdb")
        return self.con.execute(f"SELECT 1 FROM ({query}) AS __ddx_check LIMIT 1").fetchone() is not None

    def materialize(self, name: str, plan: bytes) -> None:
        query = to_sql(plan, "duckdb")
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {_ident(name)} AS {query}")
        # Later steps read it: an empty copy for the planner.
        columns = self.con.execute(f"DESCRIBE {_ident(name)}").fetchall()
        self._mirror_table(name, [(c[0], c[1]) for c in columns])

    def drop_table(self, name: str) -> None:
        self.con.execute(f"DROP TABLE IF EXISTS {_ident(name)}")
        self._drop_mirrored(name)


#: The programs :func:`sql_all` keeps for each connection, gone with it.
_programs: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
#: Each connection's backend, and so its planner, gone with it.
_backends: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def sql(con: "duckdb.DuckDBPyConnection", statement: str):
    """Run ``statement`` on ``con``, its ``grad``, ``vjp`` and ``jvp`` calls
    computed first (see :func:`ddxdb.ad.sql`). Returns a DuckDB relation of
    its result, in memory, or ``None`` for a statement with no result
    (``SET``, say). Only a query may call ``grad``, ``vjp`` or ``jvp``."""
    return sql_all(con, [statement])[0]


def sql_all(con: "duckdb.DuckDBPyConnection", statements: Sequence[str]) -> list:
    """:func:`sql` for several statements, which share the runs of the
    programs they call (see :func:`ddxdb.ad.sql_all`). Each program is built
    once per connection and run again by later calls, as a training loop
    makes them.

    Each statement runs before this returns: a DuckDB relation reads its
    tables only when it runs, and the programs' tables are dropped once the
    statements have run. A result is copied into memory (as Arrow)."""
    backend = _backends.get(con)
    if backend is None:
        backend = _backends[con] = DuckDBBackend(con)
    programs = _programs.setdefault(con, Programs())
    out = []
    with rewrite(backend, statements, "duckdb", programs) as rewritten:
        for statement in rewritten.statements:
            result = con.execute(statement)
            out.append(None if result.description is None else con.from_arrow(result.to_arrow_table()))
    return out
