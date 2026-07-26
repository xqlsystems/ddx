# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the Substrait plan-round-trip limitation behind ddx's design (§3.2).

The prototype's abandoned approach produced a DataFusion LogicalPlan as Substrait,
rewrote grad(), then consumed it back. This shows *producing* Substrait fails for
exactly the query shapes that make in-SQL training loops interesting.
"""
import datafusion
from datafusion import SessionContext
from datafusion.substrait import Producer

print(f"datafusion {datafusion.__version__}\n")

ctx = SessionContext()
ctx.from_pydict({"x": [1.0, 2.0, 3.0]}, "t")

def produce(label, sql):
    print(f"--- {label} ---")
    print(f"  SQL: {sql}")
    try:
        df = ctx.sql(sql)
        plan = df.logical_plan()
        sub = Producer.to_substrait_plan(plan, ctx)
        print(f"  RESULT: OK, produced Substrait plan ({type(sub).__name__})\n")
    except Exception as e:
        msg = str(e).replace("\n", " ")
        print(f"  RESULT: FAILED -> {type(e).__name__}: {msg[:220]}\n")

# Control: an ordinary scalar projection round-trips fine.
produce("control (plain projection)", "SELECT x + 1.0 AS y FROM t")

# 1. Recursive CTE — the training-loop shape.
produce("recursive CTE", (
    "WITH RECURSIVE r(step, v) AS ("
    "  SELECT 0, CAST(1.0 AS DOUBLE) "
    "  UNION ALL "
    "  SELECT step + 1, v / 2 FROM r WHERE step < 5"
    ") SELECT v FROM r"))

# 2. Scalar subquery.
produce("scalar subquery", "SELECT x, (SELECT max(x) FROM t) AS mx FROM t")

# 3. DML (INSERT ... SELECT).
produce("DML (INSERT ... SELECT)", "INSERT INTO t SELECT x + 1.0 FROM t")

print("DONE.")
