# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""De-risk spike for adopting Substrait (+ custom extension-function markers) as
ddx v2's relational IR, instead of a bespoke Rust builder graph.

Claim under test: "tag, don't infer" (the non-negotiable part of the v2 design —
see docs/design-relational-ad.md §1) can be realized by wrapping the operand of an
ordinary SQL aggregate in a ddx-owned marker scalar function (mirroring v1's
grad()/jvp() marker pattern one level down, e.g. `SUM(ddx_contract_mark(a.v*b.v))`
in place of `SUM(a.v*b.v)`), and that marker SURVIVES an engine's own SQL-to-plan
Substrait producer as a distinguishable extension-function anchor — including
across engines, which is the actual portability claim Substrait is supposed to buy
ddx over inventing its own IR.

Four checks:
  1. DuckDB round-trip: does DuckDB's (community) substrait extension preserve a
     custom marker through get_substrait -> from_substrait, and does the
     round-tripped plan still EXECUTE correctly?
  2. Cross-engine A: a plan PRODUCED by DataFusion (with the marker) — does
     DuckDB's from_substrait CONSUME and EXECUTE it correctly?
  3. Cross-engine B: a plan PRODUCED by DuckDB — does DataFusion's own consumer
     at least DESERIALIZE it without error? (execution not exercised — see the
     honest gap noted in the design doc)
  4. Sanity: does a plain (unmarked) JOIN+GROUP BY SUM — the actual contraction
     shape ddx v2 needs — round-trip through DuckDB at all, independent of markers?

Result (2026-07-20, DuckDB 1.5.4 community `substrait`, datafusion-python 54.0.0):
all four checks PASS. Numeric results match exactly (not just "no error") in
checks 1 and 2. See docs/design-relational-ad.md §1/§9 for what this does and does
not settle (extension_uris/YAML conformance is looser than the Substrait spec's
ideal in both engines' producers; DuckDB's substrait support is a *community*
extension, not core, as of 1.5.4 — INSTALL from the core registry 404s; DuckDB's
consumer has its own known coverage gaps elsewhere, e.g. semi-joins, an open issue
at github.com/substrait-io/duckdb-substrait-extension/issues/144 — not exercised
by this spike since ddx's contraction shape is a plain inner join).
"""
import duckdb

# ---------------------------------------------------------------------------
# 1. DuckDB round-trip (produce and consume within the same engine)
# ---------------------------------------------------------------------------
con = duckdb.connect()
con.execute("INSTALL substrait FROM community")   # NOT in core as of 1.5.4 (confirmed: core INSTALL 404s)
con.execute("LOAD substrait")

# The marker: an IDENTITY scalar function. Its only job is to appear, untouched,
# in the exported plan so ddx-core can recognize "this SUM's operand was marked"
# without inferring anything from plan shape. Exactly v1's grad()/jvp() pattern,
# one level down (wrapping a contraction's multiplicand instead of a whole SELECT).
con.create_function("ddx_contract_mark", lambda x: x, ["DOUBLE"], "DOUBLE")

con.execute("CREATE TABLE a(i INTEGER, j INTEGER, v DOUBLE)")
con.execute("CREATE TABLE b(j INTEGER, k INTEGER, v DOUBLE)")
con.execute("INSERT INTO a VALUES (0,0,1.0),(0,1,2.0),(1,0,3.0),(1,1,4.0)")
con.execute("INSERT INTO b VALUES (0,0,5.0),(0,1,6.0),(1,0,7.0),(1,1,8.0)")

marked_sql = """
SELECT a.i AS i, b.k AS k, SUM(ddx_contract_mark(a.v * b.v)) AS val
FROM a JOIN b ON a.j = b.j
GROUP BY a.i, b.k
ORDER BY i, k
"""
plain_sql = marked_sql.replace("ddx_contract_mark(a.v * b.v)", "a.v * b.v")
expected = con.execute(plain_sql).fetchall()

blob = con.execute(f"SELECT * FROM get_substrait($$ {marked_sql} $$)").fetchone()[0]
result = con.execute("SELECT * FROM from_substrait($1)", [blob]).fetchall()
check1 = result == expected
print(f"1. DuckDB round-trip preserves marker & executes correctly: {'OK' if check1 else 'FAIL'}  {result}")

# ---------------------------------------------------------------------------
# 2 & 3. Cross-engine (DataFusion <-> DuckDB)
# ---------------------------------------------------------------------------
import pyarrow as pa
from datafusion import SessionContext, udf
from datafusion.substrait import Serde

ctx = SessionContext()
ta = pa.table({"i": [0, 0, 1, 1], "j": [0, 1, 0, 1], "v": [1.0, 2.0, 3.0, 4.0]})
tb = pa.table({"j": [0, 0, 1, 1], "k": [0, 1, 0, 1], "v": [5.0, 6.0, 7.0, 8.0]})
ctx.register_record_batches("a", [ta.to_batches()])
ctx.register_record_batches("b", [tb.to_batches()])
ctx.register_udf(udf(lambda x: x, [pa.float64()], pa.float64(), "stable", name="ddx_contract_mark"))

df_bytes = Serde.serialize_bytes(marked_sql, ctx)
duck_result = con.execute("SELECT * FROM from_substrait($1)", [df_bytes]).fetchall()
check2 = sorted(duck_result) == sorted(expected)
print(f"2. DataFusion produces, DuckDB consumes+executes marker-tagged plan: {'OK' if check2 else 'FAIL'}  {duck_result}")

duck_blob = con.execute(f"SELECT * FROM get_substrait($$ {marked_sql} $$)").fetchone()[0]
try:
    Serde.deserialize_bytes(bytes(duck_blob))
    check3 = True
except Exception as e:
    check3 = False
    print("   deserialize error:", repr(e))
print(f"3. DuckDB produces, DataFusion deserializes (parse only, not executed): {'OK' if check3 else 'FAIL'}")

# ---------------------------------------------------------------------------
# 4. Sanity: the base contraction shape (no marker) round-trips at all
# ---------------------------------------------------------------------------
blob2 = con.execute(f"SELECT * FROM get_substrait($$ {plain_sql} $$)").fetchone()[0]
result2 = con.execute("SELECT * FROM from_substrait($1)", [blob2]).fetchall()
check4 = result2 == expected
print(f"4. Plain JOIN+GROUP BY SUM (no marker) round-trips through DuckDB: {'OK' if check4 else 'FAIL'}")

print(f"\nALL CHECKS: {'PASS' if all([check1, check2, check3, check4]) else 'SOME FAILED'}")
