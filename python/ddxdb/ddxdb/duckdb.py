# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Gradients of whole queries on DuckDB.

Write a model's forward pass and loss as SQL on a DuckDB connection, and take
its gradient in SQL, as :mod:`ddxdb.ad` does on DataFusion::

    import duckdb
    from ddxdb import duckdb as ad

    con = duckdb.connect()
    con.execute("CREATE TABLE w (i BIGINT, val DOUBLE)")
    con.execute("INSERT INTO w VALUES (0, 1.0), (1, 2.0)")
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

from . import Programs, _name_parts, rewrite
from ._ddxdb import UnsupportedExpression, _rename_tables, _tables_read, to_sql

__all__ = ["DuckDBBackend", "sql", "sql_all"]


def _ident(name: str) -> str:
    """``name`` as a quoted DuckDB identifier."""
    return '"' + name.replace('"', '""') + '"'


def _literal(value: str) -> str:
    """``value`` as a quoted DuckDB string."""
    return "'" + value.replace("'", "''") + "'"


#: The settings that change what a query means, which the planner copies
#: from the caller's connection so it plans the query the caller would run.
_SETTINGS = (
    "Calendar",
    "TimeZone",
    "default_collation",
    "default_null_order",
    "default_order",
    "disable_timestamptz_casts",
    "ieee_floating_point_ops",
    "integer_division",
    "lambda_syntax",
    "old_implicit_casting",
    "order_by_non_integer_literal",
    "preserve_identifier_case",
    "scalar_subquery_error_on_multiple_rows",
    "search_path",
)


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
    docs). It holds the connection weakly, so it never keeps one open.

    DuckDB's producer writes each table a plan reads by its name alone,
    without its schema or database (``s.t`` is ``t``), and the step's SQL
    reads it by that name on the connection. So a loss is refused when a
    table it reads shares its name with another object in the catalog,
    rather than read as whichever the name finds."""

    def __init__(self, con: "duckdb.DuckDBPyConnection"):
        self._con = weakref.ref(con)
        self._planner = duckdb.connect()
        _load_substrait(self._planner)
        # The mirror's tables are empty, and DuckDB's statistics would say
        # so: a filter on one's column (`WHERE layer = 0`) would be planned
        # as no rows, and its scan left out of the plan.
        self._planner.execute("SET disabled_optimizers = 'statistics_propagation'")
        # What the planner mirrors: each object's columns or view SQL, as last
        # copied, so an unchanged catalog is not copied again.
        self._mirrored: dict = {}
        self._attached = {"memory"}
        self._settings: tuple = ()

    @property
    def con(self) -> "duckdb.DuckDBPyConnection":
        con = self._con()
        if con is None:
            raise RuntimeError("the DuckDB connection this backend ran on is gone")
        return con

    # The caller's catalog, and the planner's copy of it.

    def _catalog(self) -> dict:
        """Each object on the caller's connection a query can read, but ddx's
        own, by ``(database, schema, name)``: either ``("table", columns)``,
        with columns as ``(name, type)``, or ``("view", sql)``. Temporary
        objects are in the database ``temp``. Registered Arrow data is a
        view with no SQL, and is a table here."""
        columns: dict = {}
        for db, schema, table, column, ty in self.con.execute(
            "SELECT database_name, schema_name, table_name, column_name, data_type "
            "FROM duckdb_columns() WHERE NOT internal "
            "ORDER BY database_name, schema_name, table_name, column_index"
        ).fetchall():
            columns.setdefault((db, schema, table), []).append((column, ty))
        views = {
            (db, schema, name): sql
            for db, schema, name, sql in self.con.execute(
                "SELECT database_name, schema_name, view_name, sql FROM duckdb_views() WHERE NOT internal"
            ).fetchall()
        }
        catalog = {}
        for key, cols in columns.items():
            if key[2].startswith("__ddx_"):
                continue
            sql = views.get(key)
            catalog[key] = ("view", sql) if sql else ("table", tuple(cols))
        return catalog

    def _path(self, key) -> str:
        """The planner's name for the object at ``key``, making its schema and
        database if they are missing."""
        db, schema, name = key
        if db != "temp" and db not in self._attached:
            self._planner.execute(f"ATTACH ':memory:' AS {_ident(db)}")
            self._attached.add(db)
        if db != "temp" and schema != "main":
            self._planner.execute(f"CREATE SCHEMA IF NOT EXISTS {_ident(db)}.{_ident(schema)}")
        return ".".join(_ident(p) for p in key)

    def _drop_mirrored(self, key) -> None:
        """Drop the planner's object at ``key``, a table or a view: DuckDB
        drops each only as what it is."""
        db, schema, name = key
        kinds = self._planner.execute(
            "SELECT table_type FROM information_schema.tables "
            "WHERE table_catalog = ? AND table_schema = ? AND table_name = ?",
            [db, schema, name],
        ).fetchall()
        for (kind,) in kinds:
            what = "VIEW" if kind == "VIEW" else "TABLE"
            self._planner.execute(f"DROP {what} IF EXISTS {self._path(key)}")

    def _mirror_table(self, key, columns) -> None:
        cols = ", ".join(f"{_ident(c)} {ty}" for c, ty in columns)
        self._drop_mirrored(key)
        temp = "TEMP " if key[0] == "temp" else ""
        name = _ident(key[2]) if temp else self._path(key)
        self._planner.execute(f"CREATE {temp}TABLE {name} ({cols})")

    def _mirror_view(self, key, sql) -> None:
        # A view's SQL names it alone, so it is made from its own schema.
        self._drop_mirrored(key)
        db, schema, _ = key
        if db != "temp":
            self._path(key)
            self._planner.execute(f"USE {_ident(db)}.{_ident(schema)}")
        try:
            self._planner.execute(sql)
        finally:
            self._planner.execute(f"USE {_ident(self._database)}.{_ident(self._schema)}")

    def _sync(self) -> dict:
        """Make the planner's catalog mirror the caller's, searched as the
        caller's is, and return the caller's."""
        self._database, self._schema = self.con.execute(
            "SELECT current_database(), current_schema()"
        ).fetchone()
        catalog = self._catalog()
        if catalog != self._mirrored:
            for key in self._mirrored.keys() - catalog.keys():
                self._drop_mirrored(key)
            views = []
            for key, (kind, what) in catalog.items():
                if self._mirrored.get(key) == (kind, what):
                    continue
                if kind == "table":
                    self._mirror_table(key, what)
                else:
                    views.append((key, what))
            # A view may read another, so each is made once those it reads
            # are: as many rounds as it takes. One that still fails (it reads
            # what is gone, say) is left out, for a plan that reads it to
            # refuse, and is tried again by the next call.
            while views:
                failed = []
                for key, sql in views:
                    try:
                        self._mirror_view(key, sql)
                    except duckdb.Error:
                        failed.append((key, sql))
                if len(failed) == len(views):
                    break
                views = failed
            self._mirrored = {k: v for k, v in catalog.items() if (k, v[1]) not in views}
        for db, schema in self.con.execute("SELECT database_name, schema_name FROM duckdb_schemas()").fetchall():
            if db not in ("system", "temp"):
                self._path((db, schema, ""))
        self._planner.execute(f"USE {_ident(self._database)}.{_ident(self._schema)}")
        settings = tuple(
            self.con.execute(
                "SELECT name, value FROM duckdb_settings() WHERE name IN (SELECT unnest(?)) ORDER BY name",
                [list(_SETTINGS)],
            ).fetchall()
        )
        if settings != self._settings:
            for name, value in settings:
                self._planner.execute(f"SET {name} = {_literal(value)}")
            self._settings = settings
        return catalog

    def _plan(self, sql: str, catalog: dict) -> bytes:
        """DuckDB's plan of ``sql``, each table it reads named as the caller's
        connection finds it. The producer writes a table by its name alone,
        so a name is taken as the only object of that name in the catalog,
        and refused if there are two."""
        plan = bytes(self._planner.execute("SELECT * FROM get_substrait(?)", [sql]).fetchone()[0])
        renamed = {}
        for name in set(_tables_read(plan)):
            same = sorted(k for k in catalog if k[2] == name)
            if len(same) > 1:
                paths = ", ".join(".".join(k) for k in same)
                raise UnsupportedExpression(
                    f"`{name}` names {len(same)} objects ({paths}). DuckDB's Substrait producer "
                    "writes a table by its name alone, without its schema or database, so on "
                    "DuckDB ddx needs each table a loss reads to have a name of its own"
                )
            if not same:
                continue
            db, schema, _ = same[0]
            if db == "temp" or (db, schema) == (self._database, self._schema):
                continue  # The name alone finds it.
            renamed[name] = [schema, name] if db == self._database else [db, schema, name]
        return _rename_tables(plan, renamed) if renamed else plan

    # The Backend protocol.

    def select_all(self, name: str) -> bytes:
        catalog = self._sync()
        return self._plan(f"SELECT * FROM {_ident(name)}", catalog)

    def plan_sql(self, sql: str) -> bytes:
        return self._plan(sql, self._sync())

    def plan_key(self, sql: str) -> Optional[str]:
        # The plan depends on the query, the schemas and view definitions it
        # reads, which the catalog holds, where names are searched and the
        # settings that change what a query means; not on any values.
        catalog = self._sync()
        return repr((sql, self._database, self._schema, self._settings, sorted(catalog.items())))

    def view_sql(self, name: str) -> Optional[str]:
        """The SQL of the view ``name`` reads, so ddx plans each CTE the
        view reads more than once on its own: DuckDB's Substrait producer
        has no plan for a CTE it materializes. DuckDB binds a view's SQL by
        name each time a query reads the view, so the SQL means what runs.
        It is given for a temporary view, or one in the current schema,
        whose SQL resolves names as the query does; a view in another
        schema resolves them in its own, and is left to DuckDB."""
        # DuckDB matches names without regard to case, quoted or not.
        parts = _name_parts(name, normalize=False)
        if parts is None:
            return None
        parts = [p.lower() for p in parts]
        db, schema = (
            p.lower() for p in self.con.execute("SELECT current_database(), current_schema()").fetchone()
        )
        if len(parts) == 1:
            paths = [("temp", "main"), (db, schema)]
        elif len(parts) == 2:
            # A schema of the current database, or a database's main schema.
            paths = [(db, parts[0]), (parts[0], "main")]
        elif len(parts) == 3:
            paths = [tuple(parts[:2])]
        else:
            return None
        by_path = {tuple(p.lower() for p in key): what for key, what in self._catalog().items()}
        for path in paths:
            found = by_path.get((*path, parts[-1]))
            if found is None:
                continue
            kind, sql = found
            here = path == ("temp", "main") or path == (db, schema)
            return sql if kind == "view" and here else None
        return None

    def returns_rows(self, plan: bytes) -> bool:
        query = to_sql(plan, "duckdb")
        return self.con.execute(f"SELECT 1 FROM ({query}) AS __ddx_check LIMIT 1").fetchone() is not None

    def materialize(self, name: str, plan: bytes) -> None:
        query = to_sql(plan, "duckdb")
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {_ident(name)} AS {query}")
        # Later steps read it: an empty copy for the planner.
        columns = self.con.execute(f"DESCRIBE temp.main.{_ident(name)}").fetchall()
        self._mirror_table(("temp", "main", name), [(c[0], c[1]) for c in columns])

    def drop_table(self, name: str) -> None:
        self.con.execute(f"DROP TABLE IF EXISTS temp.main.{_ident(name)}")
        self._drop_mirrored(("temp", "main", name))


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
