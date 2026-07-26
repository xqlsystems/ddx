# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""R1b spike: is it safe to run a query on the same DuckDB database from within
the execution of another query (the `ddx('<sql>')` table-function pattern)?

Model: a scalar UDF invoked mid-query opens a SECOND connection to the SAME
database instance (con.cursor()) and runs an inner query. This is a *stricter*
model than ddx() itself, because ddx()'s outer query (`SELECT * FROM ddx(...)`)
holds no user-table scan, whereas here the outer query is actively scanning a
table while the inner query runs. If the strict model is safe, ddx() is safe.
"""
import duckdb

def line(t): print(f"\n{'='*70}\n{t}\n{'='*70}")

# ---------------------------------------------------------------------------
line("T0  sanity: a cursor() is a second connection sharing the same in-memory DB")
con = duckdb.connect(":memory:")
con.execute("CREATE TABLE t AS SELECT * FROM range(5) AS r(x)")
cur = con.cursor()
print("cursor sees committed table t:", cur.execute("SELECT count(*) FROM t").fetchone())

# ---------------------------------------------------------------------------
line("T1  re-entrancy: run an inner query on the same DB from inside a scalar UDF")
con = duckdb.connect(":memory:")
con.execute("CREATE TABLE t AS SELECT * FROM range(3) AS r(x)")

def inner_select(x):
    # Runs DURING execution of the outer query, on a 2nd connection to same DB.
    (n,) = con.cursor().execute("SELECT 40 + 2").fetchone()
    return int(x) + n

con.create_function("inner_select", inner_select, ["BIGINT"], "BIGINT")
try:
    rows = con.execute("SELECT x, inner_select(x) AS y FROM t ORDER BY x").fetchall()
    print("OK  outer scan + inner query per row:", rows)
except Exception as e:
    print("FAIL (deadlock/error):", type(e).__name__, e)

# ---------------------------------------------------------------------------
line("T2  inner query READS a committed user table during the outer scan")
con = duckdb.connect(":memory:")
con.execute("CREATE TABLE params AS SELECT 10.0 AS a")
con.execute("CREATE TABLE t AS SELECT * FROM range(3) AS r(x)")

def read_param(x):
    (a,) = con.cursor().execute("SELECT a FROM params").fetchone()
    return float(x) * float(a)  # cast: DuckDB may return DECIMAL

con.create_function("read_param", read_param, ["BIGINT"], "DOUBLE")
try:
    rows = con.execute("SELECT x, read_param(x) AS y FROM t ORDER BY x").fetchall()
    print("OK  inner read of user table:", rows)
except Exception as e:
    print("FAIL:", type(e).__name__, e)

# ---------------------------------------------------------------------------
line("T3  inner query does DML (INSERT) during the outer scan")
con = duckdb.connect(":memory:")
con.execute("CREATE TABLE t AS SELECT * FROM range(3) AS r(x)")
con.execute("CREATE TABLE log(v BIGINT)")

def log_and_pass(x):
    c = con.cursor()
    c.execute("INSERT INTO log VALUES (?)", [int(x)])
    return int(x)

con.create_function("log_and_pass", log_and_pass, ["BIGINT"], "BIGINT")
try:
    rows = con.execute("SELECT log_and_pass(x) FROM t ORDER BY x").fetchall()
    (cnt,) = con.execute("SELECT count(*) FROM log").fetchone()
    print(f"OK  inner INSERT during scan; rows={rows}, log rows written={cnt}")
except Exception as e:
    print("FAIL (conflict/deadlock):", type(e).__name__, e)

# ---------------------------------------------------------------------------
line("T4  transaction visibility: does the inner connection see the outer's UNCOMMITTED writes?")
con = duckdb.connect(":memory:")
con.execute("CREATE TABLE params(a DOUBLE)")
con.execute("INSERT INTO params VALUES (1.0)")
con.execute("BEGIN")
con.execute("UPDATE params SET a = 999.0")   # uncommitted on the outer connection
seen = con.cursor().execute("SELECT a FROM params").fetchone()
con.execute("ROLLBACK")
print(f"outer uncommitted a=999.0 ; inner cursor sees a={seen[0]}  "
      f"({'SEES uncommitted (shares txn)' if seen[0]==999.0 else 'does NOT see uncommitted (separate txn)'})")

# ---------------------------------------------------------------------------
line("T5  the actual ddx() shape: outer is trivial, inner runs the real query")
con = duckdb.connect(":memory:")
con.execute("CREATE TABLE t AS SELECT unnest([0.0,1.0,2.0]) AS x")
def ddx(sql):
    # ddx() reads the SQL literal (here already rewritten: grad(x*x,x) -> (x+x)),
    # runs it on a connection to the same DB, returns the result relation.
    return con.cursor().execute(sql).fetchall()
print("ddx('SELECT x, (x+x) AS dfdx FROM t') =>", ddx("SELECT x, (x + x) AS dfdx FROM t"))

print("\nDONE.")
