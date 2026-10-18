# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Demo 1's attention gradient: ddx's, against JAX's backward written by hand
as SQL steps, each materialized as ddx materializes its steps.

    attention_by_hand.py CAPTURE_DIR window          # the captured Q, K, V
    attention_by_hand.py CAPTURE_DIR window L REPS   # random Q, K, V of L tokens
"""
import sys, time, glob
import pyarrow.parquet as pq, pyarrow as pa, numpy as np
from datafusion import SessionContext
from ddxdb import ad

S = sys.argv[1]
spelling = sys.argv[2] if len(sys.argv) > 2 else "window"
ctx = SessionContext()
ad.register_stop_gradient(ctx)
L = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = np.random.default_rng(0)
for name in "qkv":
    if L:
        tt, dd = np.meshgrid(np.arange(L), np.arange(64), indexing="ij")
        t = pa.table({"t": tt.ravel(), "d": dd.ravel(), "x": rng.standard_normal(L * 64)})
    else:
        path = sorted(glob.glob(f"{S}/cap-attn/tables/{name}.*.parquet"))[0]
        t = pq.read_table(path)
    ctx.register_record_batches(name, [t.to_batches()])
    print(name, t.num_rows, t.schema.names)

WINDOW = """CREATE OR REPLACE VIEW attention AS
WITH scores AS (SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t),
weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores),
norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i)
SELECT w.i AS t, v.d, SUM(w.w / n.z * v.x) AS x FROM weights w JOIN v ON w.j = v.t JOIN norm n ON w.i = n.i GROUP BY w.i, v.d"""
GRADIENT = """WITH loss AS (SELECT SUM(x * x) / 2 AS l FROM attention)
SELECT q.t, q.d, grad(loss, q.x) AS dq, grad(loss, k.x) AS dk, grad(loss, v.x) AS dv
FROM q JOIN k ON q.t = k.t AND q.d = k.d JOIN v ON q.t = v.t AND q.d = v.d ORDER BY q.t, q.d"""
ctx.sql(WINDOW).collect()

STEPS = [
 ("s", "SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t"),
 ("p", "SELECT i, j, w / SUM(w) OVER (PARTITION BY i) AS p FROM (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM s)"),
 ("o", "SELECT p.i AS t, v.d, SUM(p.p * v.x) AS x FROM p JOIN v ON p.j = v.t GROUP BY p.i, v.d"),
 # loss = sum(o^2)/2, so the cotangent of o is o
 ("pb", "SELECT p.i, p.j, SUM(o.x * v.x) AS pb FROM p JOIN o ON o.t = p.i JOIN v ON v.t = p.j AND v.d = o.d GROUP BY p.i, p.j"),
 ("r", "SELECT t AS i, SUM(x * x) AS r FROM o GROUP BY t"),
 ("sb", "SELECT pb.i, pb.j, p.p * (pb.pb - r.r) AS sb FROM pb JOIN p ON p.i = pb.i AND p.j = pb.j JOIN r ON r.i = pb.i"),
 ("vb", "SELECT p.j AS t, o.d, SUM(p.p * o.x) AS g FROM p JOIN o ON p.i = o.t GROUP BY p.j, o.d"),
 ("qb", "SELECT sb.i AS t, k.d, SUM(sb.sb * k.x) / 8.0 AS g FROM sb JOIN k ON sb.j = k.t GROUP BY sb.i, k.d"),
 ("kb", "SELECT sb.j AS t, q.d, SUM(sb.sb * q.x) / 8.0 AS g FROM sb JOIN q ON sb.i = q.t GROUP BY sb.j, q.d"),
]

def hand():
    times = {}
    for name, sql in STEPS:
        t0 = time.perf_counter()
        batches = ctx.sql(sql).collect()
        ctx.deregister_table(name)
        ctx.register_record_batches(name, [batches])
        times[name] = time.perf_counter() - t0
    return times

def ddx():
    t0 = time.perf_counter()
    out = ad.sql(ctx, GRADIENT).to_arrow_table()
    return time.perf_counter() - t0, out

for rep in range(int(sys.argv[4]) if len(sys.argv) > 4 else 3):
    th = hand(); td, out = ddx()
    print(f"hand-written JAX backward: {sum(th.values())*1e3:.0f} ms  " + " ".join(f"{k}={v*1e3:.0f}" for k, v in th.items()) + f" | ddx: {td*1e3:.0f} ms")

def table(name):
    return {(r["t"], r["d"]): r["g"] for r in ctx.sql(f"SELECT * FROM {name}").to_arrow_table().to_pylist()}
got = {(r["t"], r["d"]): r for r in out.to_pylist()}
for col, name in (("dq", "qb"), ("dk", "kb"), ("dv", "vb")):
    want = table(name)
    err = max(abs(got[k][col] - want.get(k, 0.0)) for k in got)
    scale = max(abs(v) for v in want.values())
    print(col, "max abs diff", err, "scale", scale)
