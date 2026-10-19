# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""M5 spike: can ddx run its programs on DuckDB-Python through DuckDB's own
Substrait extension, over data registered the way xarray-sql registers it?

xarray-sql's `xql.register(con, name, ds)` registers a Dataset on a DuckDB
connection as a `pyarrow.dataset.Dataset` (`con.register`). ddx needs two
things from an engine: a Substrait plan of the user's query (the producer,
`get_substrait`), and to run each step of the program it builds (the consumer,
`from_substrait`, materialized as a table). This checks both against what
ddx's steps need to see and to compute.

Findings (DuckDB 1.5.5, `INSTALL substrait FROM community`):
  T1  the producer refuses an Arrow-registered scan, but plans the same query
      on a second connection whose catalog has an empty table of that schema;
  T2  the consumer resolves a table read as if on a connection of its own:
      it sees a committed persistent table, and a view over one, but not an
      Arrow-registered object, a view over one, a temp table, or a table
      created in the caller's open transaction;
  T3  the consumer rejects every window function, which ddx's AVG, MAX and
      MIN rules, window transposes and jvp's MAX tangent all write;
  T4  a null-safe join (IS NOT DISTINCT FROM), which ddx's steps also write,
      is consumed and pairs NULL keys correctly.

So the design (design.md, decision S22) plans with DuckDB's producer on a
schema-only mirror of the caller's catalog, and runs each step as SQL on the
caller's own connection rather than through `from_substrait`.

Exits non-zero if any finding stops holding, so an upstream change shows up.
"""
import sys

import datafusion
import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
from datafusion import substrait as dfs

failures = []


def line(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def check(label, ok):
    print(("OK   " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


def plan(con, sql):
    return con.execute(f"SELECT * FROM get_substrait($${sql}$$)").fetchone()[0]


def missing(got, name):
    """Whether the consumer failed because it found no table `name`, not for
    another reason."""
    return isinstance(got, str) and f"Table with name {name} does not exist" in got


def consume(con, blob):
    """The consumer's rows, or the first line of its error."""
    try:
        return con.execute("SELECT * FROM from_substrait(?)", [blob]).fetchall()
    except duckdb.Error as e:
        return str(e).splitlines()[0]


t = pa.table(
    {
        "i": pa.array([0, 1, 2], pa.int64()),
        "g": pa.array([0, 0, 1], pa.int64()),
        "val": [1.0, -2.0, 0.5],
    }
)
print("duckdb", duckdb.__version__, "| datafusion", datafusion.__version__)

# ---------------------------------------------------------------------------
line("T1  the producer and an Arrow-registered Dataset")
con = duckdb.connect()
con.execute("INSTALL substrait FROM community; LOAD substrait")
con.register("a", ds.dataset(t))
try:
    plan(con, "SELECT SUM(val * val) AS l FROM a")
    check("producer refuses an Arrow-registered scan", False)
except duckdb.Error as e:
    why = str(e).splitlines()[0]
    check(f"producer refuses an Arrow-registered scan: {why[:90]}…", "Scanner Type" in why)

mirror = duckdb.connect()
mirror.execute("LOAD substrait")
mirror.execute("CREATE TABLE a (i BIGINT, g BIGINT, val DOUBLE)")
blob = plan(mirror, "SELECT g, SUM(val * val) AS l FROM a GROUP BY g")
check("producer plans the query on an empty table of the same schema", len(blob) > 0)

# ---------------------------------------------------------------------------
line("T2  what the consumer's table reads resolve to")


def fresh():
    """A connection with the data registered as `a_arrow`, nothing named `a`."""
    con = duckdb.connect()
    con.execute("LOAD substrait")
    con.register("a_arrow", ds.dataset(t))
    return con


# Each case names the table the consumer should fail to find. A persistent
# view over a registered object resolves, then misses the object beneath it.
cases = [
    ("an Arrow-registered object", "SELECT 1", "a"),
    ("a view over an Arrow-registered object", "CREATE VIEW a AS SELECT * FROM a_arrow", "a_arrow"),
    ("a temp view", "CREATE TEMP VIEW a AS SELECT * FROM a_arrow", "a"),
    ("a temp table", "CREATE TEMP TABLE a AS SELECT * FROM a_arrow", "a"),
]
for label, ddl, name in cases:
    con = fresh()
    if label == "an Arrow-registered object":
        con.register("a", ds.dataset(t))
    con.execute(ddl)
    got = consume(con, blob)
    check(f"consumer does not see {label} (misses {name}): {got}", missing(got, name))

con = fresh()
con.execute("CREATE TABLE a AS SELECT * FROM a_arrow")
got = consume(con, blob)
check(f"consumer sees a committed persistent table: {got}", sorted(got) == [(0, 5.0), (1, 0.25)])
con.execute("CREATE VIEW v AS SELECT i, g, val * 2 AS val FROM a")
got = consume(con, plan(con, "SELECT g, SUM(val) AS s FROM v GROUP BY g ORDER BY g"))
check(f"consumer sees a view over a persistent table: {got}", got == [(0, -2.0), (1, 1.0)])
con.execute("BEGIN")
con.execute("CREATE TABLE u AS SELECT * FROM a")
got = consume(con, plan(con, "SELECT SUM(val) FROM u"))
check(f"consumer does not see a table created in the open transaction: {got}", missing(got, "u"))
con.execute("ROLLBACK")

# ---------------------------------------------------------------------------
line("T3/T4  plans DataFusion produces (as ddx's steps are), run by DuckDB")
ctx = datafusion.SessionContext()
ctx.register_record_batches("a", [t.to_batches()])
for sql in [
    "SELECT i, MAX(val) OVER (PARTITION BY g) AS m FROM a",
    "SELECT i, COUNT(val) OVER (PARTITION BY g) AS c FROM a",
    "SELECT i, SUM(val) OVER (PARTITION BY g ORDER BY i "
    "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM a",
]:
    got = consume(con, dfs.Serde.serialize_bytes(sql, ctx))
    check(f"consumer rejects a window function: {got}", isinstance(got, str) and "window" in got)
# NULL keys on both sides: `=` would match only (0, 0); IS NOT DISTINCT FROM
# also pairs the two NULL-keyed rows with each other and themselves.
n = pa.table({"i": pa.array([0, 1, 2], pa.int64()), "g": pa.array([0, None, None], pa.int64())})
ctx.register_record_batches("n", [n.to_batches()])
con.register("n_arrow", n)
con.execute("CREATE TABLE n AS SELECT * FROM n_arrow")
got = consume(
    con,
    dfs.Serde.serialize_bytes(
        "SELECT x.i, y.i AS j FROM n x JOIN n y ON x.g IS NOT DISTINCT FROM y.g", ctx
    ),
)
want = [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
check(f"consumer runs a null-safe join with NULL keys: {got}", isinstance(got, list) and sorted(got) == want)

line("RESULT")
if failures:
    print(f"{len(failures)} finding(s) no longer hold:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("All findings hold.")
