# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""`to_sql`'s DuckDB SQL, run on DuckDB, against DataFusion's rows.

`tests/to_sql_fuzz.rs` writes one JSON line per random query it plans
(`DDX_TO_SQL_DUCKDB_CASES=cases.jsonl`): the statements that make its tables
in DuckDB, the DuckDB SQL `to_sql` wrote of DataFusion's plan, and the rows
DataFusion gives the query. Each case is run on a fresh DuckDB connection and
its rows compared, floats within a relative 1e-9:

    DDX_TO_SQL_DUCKDB_CASES=cases.jsonl DDX_TO_SQL_SEEDS=2000 \\
      cargo test -p ddx-datafusion --test to_sql_fuzz
    uv run --no-project --with duckdb==1.5.5 \\
      python crates/ddx-datafusion/tests/to_sql_fuzz_duckdb.py cases.jsonl

A difference is `to_sql` writing a meaning DuckDB reads otherwise, or an
engine difference no SQL spelling can bridge; the summary groups them by
kind, with the first seed of each.
"""

import collections
import decimal
import json
import math
import sys

import duckdb


def want_cell(c):
    if c is None or isinstance(c, bool):
        return c
    if "i" in c:
        return int(c["i"])
    if "f" in c:
        return float(c["f"])
    if "s" in c:
        return c["s"]
    return c["o"]


def got_cell(v):
    if isinstance(v, decimal.Decimal) and v == v.to_integral_value():
        return int(v)
    if isinstance(v, decimal.Decimal):
        return float(v)
    return v


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return False
        if isinstance(a, bool) or isinstance(b, bool):
            return False
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if a == b:
            # The sign of a zero is each engine's own (DataFusion's
            # `nullif(-0.0, 0.0)` is 0.0).
            return True
        return abs(a - b) <= 1e-9 * max(abs(a), abs(b))
    return a == b and type(a) is type(b)


def key(row):
    def k(v):
        if v is None:
            return (0, "")
        if isinstance(v, bool):
            return (4, str(v))
        if isinstance(v, (int, float)):
            f = float(v)
            return (2, "nan") if math.isnan(f) else (2, f"{f:+.6e}")
        return (3, str(v))

    return [k(v) for v in row]


def differs(want, got, ordered):
    if len(want) != len(got):
        return f"{len(got)} rows, want {len(want)}"
    if not ordered:
        want, got = sorted(want, key=key), sorted(got, key=key)
    for i, (w, g) in enumerate(zip(want, got)):
        if len(w) != len(g) or not all(same(a, b) for a, b in zip(w, g)):
            return f"row {i} is {g!r}, want {w!r}"
    return None


def main(path):
    kinds = collections.Counter()
    first = {}
    n = 0
    for line in open(path, encoding="utf-8"):
        case = json.loads(line)
        n += 1
        con = duckdb.connect()
        for s in case["setup"]:
            con.execute(s)
        label = f"seed {case['seed']} ({'optimized' if case['optimized'] else 'unoptimized'})"
        try:
            cur = con.execute(case["sql"])
            names = [d[0] for d in cur.description]
            got = [[got_cell(v) for v in r] for r in cur.fetchall()]
        except duckdb.Error as e:
            # DuckDB checks integer overflow, where DataFusion wraps: an
            # engine difference, not a spelling.
            words = str(e).split()
            overflow = "Overflow" in str(e) or "can't be cast" in str(e)
            kind = ("engine: " if overflow else "error: ") + " ".join(words[:6])
            kinds[kind] += 1
            first.setdefault(kind, (label, str(e), case["sql"]))
            continue
        if names != case["columns"]:
            kind = "column names"
            kinds[kind] += 1
            first.setdefault(kind, (label, f"{names} != {case['columns']}", case["sql"]))
            continue
        want = [[want_cell(c) for c in r] for r in case["rows"]]
        d = differs(want, got, case["ordered"])
        if d:
            kind = "rows"
            kinds[kind] += 1
            first.setdefault(f"{kind} {kinds[kind]}", (label, d, case["sql"]))
        else:
            kinds["same"] += 1
    print(f"{n} cases on DuckDB {duckdb.__version__}")
    for k, c in kinds.most_common():
        print(f"  {c:5d}  {k}")
    for k, (label, what, sql) in first.items():
        print(f"\n[{k}] {label}: {what}\n  {sql}")


if __name__ == "__main__":
    main(sys.argv[1])
