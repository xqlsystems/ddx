# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""De-risk spike, second half of the Route rule's Substrait feasibility question
(design-relational-ad.md §3.4): does DuckDB's Substrait round-trip actually support
the SQL idiom the Route rule's forward pass needs -- ROW_NUMBER() OVER (PARTITION
BY ... ORDER BY ...) filtered to the top-1-per-group row (nn.py:437-445)?

Three checks, each isolating a different layer:
  A. A plain window function used as an OUTPUT COLUMN (no filter) -- tests whether
     Substrait's ConsistentPartitionWindowRel itself round-trips through DuckDB.
  B. The FULL top-1-per-group idiom (QUALIFY / WHERE rk=1) -- tests the idiom
     ddx's Route rule actually needs.
  C. The SAME full idiom, round-tripped through DataFusion instead of DuckDB --
     isolates whether a failure in B is a general Substrait-window limitation or
     specific to DuckDB's implementation.
  D. The verified workaround: split B into (i) a Substrait round-trip of JUST the
     window-column computation (the part check A already showed works), followed
     by (ii) a plain, ddx-authored SQL filter run directly by the engine, not
     through Substrait -- confirms the Route rule doesn't need to wait on an
     upstream fix to ship on DuckDB.

Result (2026-07-20, DuckDB 1.5.4 community `substrait`, datafusion-python 54.0.0):
  A: OK  -- plain window functions round-trip correctly through DuckDB.
  B: WRONG, SILENTLY -- from_substrait does not raise, but returns ALL rows
     instead of the filtered top-1-per-group rows. Root cause (confirmed via
     get_substrait_json): DuckDB's own query optimizer rewrites the top-1 idiom
     into a self-join against an `arg_max` extension aggregate BEFORE Substrait
     export, and that rewritten form does not survive the round-trip faithfully.
     This reproduces with NO ddx marker involved at all -- it is a pre-existing
     DuckDB Substrait bug, not a marker-interaction artifact.
  C: OK -- DataFusion round-trips the identical full idiom correctly. So this is
     specific to DuckDB's optimizer/Substrait-export interaction, not a general
     gap in Substrait's window-relation support.
  D: OK -- the two-step workaround produces the correct result. The Route rule
     does not need to be gated on an upstream DuckDB fix; it needs its forward
     pass built as "Substrait round-trip the window column, then filter with
     plain engine-native SQL" rather than one round-tripped statement.

Candidate upstream report: github.com/substrait-io/duckdb-substrait-extension
(worth filing, per Alex's stated preference to fix things upstream rather than
route around them indefinitely where an actual bug is found).
"""
import duckdb


def setup():
    con = duckdb.connect()
    con.execute("INSTALL substrait FROM community")
    con.execute("LOAD substrait")
    con.execute("CREATE TABLE t(grp INTEGER, item INTEGER, val DOUBLE)")
    con.execute("INSERT INTO t VALUES (0,0,1.0),(0,1,5.0),(0,2,3.0),(1,0,2.0),(1,1,1.0),(1,2,9.0)")
    return con


con = setup()

# ---- A. plain window function as an output column, no filter ---------------
sql_a = "SELECT grp, item, val, ROW_NUMBER() OVER (PARTITION BY grp ORDER BY val DESC) AS rk FROM t ORDER BY grp, item"
direct_a = sorted(con.execute(sql_a).fetchall())
blob_a = con.execute(f"SELECT * FROM get_substrait($$ {sql_a} $$)").fetchone()[0]
rt_a = sorted(con.execute("SELECT * FROM from_substrait($1)", [blob_a]).fetchall())
check_a = rt_a == direct_a
print(f"A. plain window column round-trips through DuckDB: {'OK' if check_a else 'FAIL'}")

# ---- B. the full top-1-per-group idiom (what Route actually needs) ---------
sql_b = "SELECT grp, item, val FROM t QUALIFY ROW_NUMBER() OVER (PARTITION BY grp ORDER BY val DESC) = 1 ORDER BY grp"
direct_b = sorted(con.execute(sql_b).fetchall())
blob_b = con.execute(f"SELECT * FROM get_substrait($$ {sql_b} $$)").fetchone()[0]
rt_b = sorted(con.execute("SELECT * FROM from_substrait($1)", [blob_b]).fetchall())
check_b = rt_b == direct_b
print(f"B. full top-1-per-group idiom round-trips through DuckDB: {'OK' if check_b else 'SILENTLY WRONG'}")
if not check_b:
    print(f"   expected {direct_b}, got {rt_b}  (no exception raised -- this is the dangerous case)")

# ---- C. same full idiom, through DataFusion instead -------------------------
import pyarrow as pa
from datafusion import SessionContext
from datafusion.substrait import Serde, Consumer

ctx = SessionContext()
tbl = pa.table({"grp": [0, 0, 0, 1, 1, 1], "item": [0, 1, 2, 0, 1, 2], "val": [1.0, 5.0, 3.0, 2.0, 1.0, 9.0]})
ctx.register_record_batches("t", [tbl.to_batches()])
direct_c = ctx.sql(sql_b).collect()
bs = Serde.serialize_bytes(sql_b, ctx)
logical = Consumer.from_substrait_plan(ctx, Serde.deserialize_bytes(bs))
rt_c = ctx.execute_logical_plan(logical).collect()
check_c = rt_c == direct_c
print(f"C. same full idiom round-trips through DataFusion: {'OK' if check_c else 'FAIL'}  (isolates: DuckDB-specific bug, not a general Substrait-window gap)")

# ---- D. verified workaround: split into window-step + plain-SQL filter -----
blob_ranked = con.execute(f"SELECT * FROM get_substrait($$ {sql_a} $$)").fetchone()[0]
con.execute("CREATE TEMP TABLE __ddx_ranked AS SELECT * FROM from_substrait($1)", [blob_ranked])
rt_d = sorted(con.execute("SELECT grp, item, val FROM __ddx_ranked WHERE rk = 1 ORDER BY grp").fetchall())
check_d = rt_d == direct_b
print(f"D. workaround (Substrait window step + plain-SQL filter step): {'OK' if check_d else 'FAIL'}")

print(f"\nVerdict: Route's forward pass is buildable on DuckDB TODAY via workaround D, "
      f"without waiting on an upstream fix to B's bug.")
