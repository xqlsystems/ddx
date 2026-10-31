# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""grad and jvp through UNION ALL, on random queries, on DuckDB and DataFusion.

Each case is a loss over a random UNION ALL, differentiated with respect to
each table it reads. Each gradient must agree with:

1. central differences of the engine's own loss, along a fixed direction (the
   loss is a low-degree polynomial in each table, so a small step is near
   exact);
2. the other engine's gradient;
3. ``jvp(l, t.x, 1)``, forward mode along 1, which is the gradient's sum.

Neither engine may panic, or refuse a query the other answers, unless the
engine's own SQL refuses the loss.

Branches mix the shapes that found bugs: untyped NULLs (``NULL AS v``,
``NULLIF``, a ``CAST(NULL AS BIGINT)`` DuckDB folds into a NULL of no type),
a column selected twice (DuckDB's producer drops the repeat), nested unions
of a table that isn't differentiated, ``UNION`` beside ``UNION ALL``, and
unions of up to 33 inputs. The losses read the union through a SUM, a GROUP
BY, a join, a window, a contraction, a self-join, and a union of it.

Case n draws from its own seed, so it replays alone. ``DDX_UNION_FUZZ_CASES``
sets how many run (default 40); ``DDX_UNION_FUZZ_SEED`` shifts which:

    DDX_UNION_FUZZ_CASES=2000 DDX_UNION_FUZZ_SEED=7 pytest tests/test_union_fuzz.py
"""

import os
import random

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")
datafusion = pytest.importorskip("datafusion")

from ddxdb import ad as ddx_df  # noqa: E402
from ddxdb import duckdb as ddx_duck  # noqa: E402

CASES = int(os.environ.get("DDX_UNION_FUZZ_CASES", "40"))
SEED = int(os.environ.get("DDX_UNION_FUZZ_SEED", "0"))

# The differentiated tables and their value column: c's is named as
# DataFusion suffixes a union's repeated column (x_1).
WRT = {"a": "x", "b": "x", "c": "x_1"}
EPS = 1e-4


class Gen:
    """A random loss over a UNION ALL."""

    def __init__(self, rng):
        self.rng = rng

    def fx(self, col):
        # Polynomial, so central differences are near exact.
        return self.rng.choice(
            [
                f"{col}",
                f"{col} * {col}",
                f"2.0 * {col}",
                f"{col} * {col} * {col}",
                f"({col} + 1.0) * {col}",
                f"{col} - 0.5",
                f"0.25 * {col} * {col} + {col}",
            ]
        )

    def name(self, prefix):
        return f"{prefix}{self.rng.randrange(10**6)}"

    def branch(self, depth):
        """``SELECT k, v ...``: one input of a union."""
        rng = self.rng
        if depth < 2 and rng.random() < 0.2:
            return f"SELECT k, v FROM ({self.union(depth + 1)}) {self.name('n')}"
        t = rng.choice("ab")
        d = "SELECT i AS k, val AS v FROM d"
        shapes = [
            # Plain.
            lambda: f"SELECT i AS k, {self.fx('x')} AS v FROM {t}",
            # Reads no differentiated table.
            lambda: d,
            # A literal, and an integer, coerced with the others' doubles.
            lambda: f"SELECT i AS k, 1.5 AS v FROM {t}",
            # Selects i twice: DuckDB's producer drops the repeat.
            lambda: f"SELECT i AS k, i AS v FROM {t}",
            # No rows.
            lambda: f"SELECT i AS k, {self.fx('x')} AS v FROM {t} WHERE i > 100",
            lambda: f"SELECT k AS k, SUM({self.fx('x')}) AS v FROM {t} GROUP BY k",
            lambda: f"SELECT i_1 AS k, {self.fx('x_1')} AS v FROM c",
            lambda: f"SELECT a.i AS k, {self.fx('a.x')} * b.x AS v FROM a JOIN b ON a.i = b.i"
            + (" AND a.k = b.k" if rng.random() < 0.5 else ""),
            # Filtered on the differentiated value, away from the data's
            # values, which are multiples of 1/8: a step can't cross it.
            lambda: f"SELECT i AS k, {self.fx('x')} AS v FROM {t} WHERE x > 0.0625",
            # Keys no other input has.
            lambda: f"SELECT i + 10 AS k, {self.fx('x')} AS v FROM {t}",
            # Columns in the other order, put in place by the outer SELECT.
            lambda: f"SELECT k, v FROM (SELECT {self.fx('x')} AS v, i AS k FROM {t}) {self.name('p')}",
            # NULLs with no type, typed by the union's other inputs.
            lambda: f"SELECT NULLIF(i, 1) AS k, {self.fx('x')} AS v FROM {t}",
            lambda: f"SELECT i AS k, NULL AS v FROM {t}",
            lambda: f"SELECT NULL AS k, {self.fx('x')} AS v FROM {t}",
            lambda: f"SELECT CAST(NULL AS BIGINT) AS k, {self.fx('x')} AS v FROM {t}",
            lambda: f"SELECT i AS k, CASE WHEN i = 1 THEN NULL ELSE {self.fx('x')} END AS v FROM {t}",
            # A 32-bit key, coerced with the others' 64-bit ones.
            lambda: f"SELECT CAST(i AS INTEGER) AS k, {self.fx('x')} AS v FROM {t}",
            # A nested union of a table that isn't differentiated, its first
            # input computed: DataFusion names the union's columns after it.
            lambda: self.nested_d(),
            # UNION beside UNION ALL: neither is spliced into the other.
            lambda: f"SELECT k, v FROM ({d} UNION {d}) {self.name('n')}",
            lambda: f"SELECT k, v FROM ({d} UNION ({d} UNION ALL {d})) {self.name('n')}",
        ]
        return rng.choice(shapes)()

    def nested_d(self):
        inner = "SELECT i AS k, val - 0.5 AS v FROM d UNION ALL SELECT i + 1 AS k, 2.0 * val AS v FROM d"
        r = self.rng.randrange(3)
        if r == 1:
            inner = f"SELECT k, v FROM ({inner}) {self.name('m')} UNION ALL SELECT i AS k, val * val AS v FROM d"
        elif r == 2:
            inner = f"SELECT i AS k, val * val AS v FROM d UNION ALL SELECT k, v FROM ({inner}) {self.name('m')}"
        return f"SELECT k, v FROM ({inner}) {self.name('n')}"

    def union(self, depth=0):
        n = 2 + (self.rng.random() < 0.3) + (self.rng.random() < 0.15)
        if depth == 0 and self.rng.random() < 0.08:
            n = self.rng.choice([8, 12, 16, 24, 33])
        return " UNION ALL ".join(self.branch(depth) for _ in range(n))

    def loss(self):
        u = self.union()
        body = self.rng.choice(
            [
                "SELECT SUM(v * v) AS l FROM u",
                "SELECT SUM(t * t) AS l FROM (SELECT k, SUM(v) AS t FROM u GROUP BY k) g",
                "SELECT SUM(u.v * w.val * u.v) AS l FROM u JOIN w ON u.k = w.i",
                "SELECT SUM(v * t) AS l FROM (SELECT v, SUM(v) OVER (PARTITION BY k) AS t FROM u) s",
                "SELECT SUM(h * h) AS l FROM (SELECT w.o, SUM(u.v * w.val) AS h FROM u JOIN w ON u.k = w.i "
                "GROUP BY w.o) h",
                f"SELECT SUM(u.v * u2.v) AS l FROM u JOIN ({self.union()}) u2 ON u.k = u2.k",
                "SELECT SUM(v * v) AS l FROM (SELECT k, v FROM u WHERE v > -100 "
                "UNION ALL SELECT k, 2.0 * v AS v FROM u) uu",
            ]
        )
        return f"WITH u AS ({u}), loss AS ({body})"


def tables(rng):
    def col(n, nulls):
        return [None if rng.random() < nulls else (rng.randrange(33) - 16) / 8 for _ in range(n)]

    def ints(n, m):
        return pa.array([rng.randrange(m) for _ in range(n)], pa.int64())

    out = {}
    for t in "ab":
        n = rng.randrange(0, 6) if rng.random() < 0.1 else rng.randrange(1, 6)
        out[t] = pa.table(
            {"r": pa.array(range(n), pa.int64()), "i": ints(n, 4), "k": ints(n, 2), "x": pa.array(col(n, 0.1), pa.float64())}
        )
    n = rng.randrange(1, 5)
    out["c"] = pa.table({"r": pa.array(range(n), pa.int64()), "i_1": ints(n, 4), "x_1": pa.array(col(n, 0.1), pa.float64())})
    # A repeated row, sometimes, so UNION and UNION ALL differ.
    n = rng.randrange(1, 5)
    i, val = [rng.randrange(4) for _ in range(n)], col(n, 0.0)
    if rng.random() < 0.5:
        i, val = i + i[:1], val + val[:1]
    out["d"] = pa.table({"i": pa.array(i, pa.int64()), "val": pa.array(val, pa.float64())})
    n = rng.randrange(1, 6)
    out["w"] = pa.table({"i": ints(n, 4), "o": ints(n, 2), "val": pa.array(col(n, 0.0), pa.float64())})
    return out


def direction(r):
    """The direction of the central differences, at row r: varied, not all one."""
    return ((r * 7 + 3) % 9 - 4) / 4


def perturbed(table, t, h):
    col = WRT[t]
    vals = [None if v is None else v + h * direction(r) for r, v in zip(table["r"].to_pylist(), table[col].to_pylist())]
    return table.set_column(table.column_names.index(col), col, pa.array(vals, pa.float64()))


class DuckDb:
    name = "duckdb"

    def __init__(self, ts):
        self.con = duckdb.connect()
        for t, table in ts.items():
            self.put(t, table)

    def put(self, t, table):
        self.con.register(f"_{t}", table)
        self.con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM _{t}")

    def scalar(self, q):
        return self.con.execute(q).fetchone()[0]

    def ddx(self, q):
        return ddx_duck.sql(self.con, q).to_arrow_table().to_pylist()


class DataFusion:
    name = "datafusion"

    def __init__(self, ts):
        self.ctx = datafusion.SessionContext()
        for t, table in ts.items():
            self.put(t, table)

    def put(self, t, table):
        self.ctx.deregister_table(t)
        self.ctx.register_record_batches(t, [table.to_batches() or [pa.RecordBatch.from_pylist([], schema=table.schema)]])

    def scalar(self, q):
        return self.ctx.sql(q).to_pydict()["l"][0]

    def ddx(self, q):
        return ddx_df.sql(self.ctx, q).to_arrow_table().to_pylist()


def close(x, y, tol):
    return abs(x - y) <= tol * (1 + abs(y))


def attempt(f):
    """f's value, or the first line of what it raised; a panic fails the case."""
    try:
        return f()
    except BaseException as e:  # a Rust panic is a BaseException
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        assert type(e).__name__ != "PanicException", f"ddx panicked: {e}"
        return Refused(str(e).splitlines()[0] if str(e) else type(e).__name__)


class Refused(str):
    pass


def known(refusal, engine):
    """A refusal that is the expected answer, or a known bug."""
    if engine == "duckdb" and "dropped a column" in refusal:
        # DuckDB's producer drops a column a SELECT repeats, leaving nothing
        # to tell which; ddx refuses that rather than guess (#245).
        return True
    if engine == "datafusion" and "optimize_unions" in refusal:
        # DataFusion's optimizer fails on a nested union holding a UNION,
        # read through a join (#249).
        pytest.xfail("#249: optimize_unions on a nested union holding a UNION")
    return False


def central_difference(engine, ts, t, prefix, h):
    fs = []
    for step in (h, -h):
        engine.put(t, perturbed(ts[t], t, step))
        fs.append(engine.scalar(prefix + " SELECT l FROM loss"))
    engine.put(t, ts[t])
    return None if None in fs else (fs[0] - fs[1]) / (2 * h)


@pytest.mark.parametrize("case", range(SEED * 1_000_000, SEED * 1_000_000 + CASES))
def test_grad_through_union_all_agrees(case):
    rng = random.Random(case)
    prefix = Gen(rng).loss()
    ts = tables(rng)
    engines = []
    for E in (DuckDb, DataFusion):
        e = E(ts)
        # A loss the engine's own SQL refuses (DataFusion has no SUM of a
        # column every input leaves NULL) is no question for ddx.
        if not isinstance(attempt(lambda: e.scalar(prefix + " SELECT l FROM loss")), Refused):
            engines.append(e)
    reads = [t for t in WRT if f"FROM {t}" in prefix or f"JOIN {t}" in prefix]
    for t in reads:
        col = WRT[t]
        grads = {}
        for e in engines:
            g = attempt(lambda: e.ddx(f"{prefix} SELECT r, grad(loss, {col}) AS g FROM {t}"))
            grads[e.name] = g if isinstance(g, Refused) else sorted((row["r"], row["g"]) for row in g)
        refused = {name for name, g in grads.items() if isinstance(g, Refused)}
        if len(grads) == 2 and len(refused) == 1:
            (name,) = refused
            if not known(grads[name], name):
                pytest.fail(f"only {name} refused d/d{t}: {grads[name]}\n{prefix}")
        if len(grads) == 2 and not refused:
            d, f = grads["duckdb"], grads["datafusion"]
            assert [r for r, _ in d] == [r for r, _ in f], f"d/d{t}\n{prefix}"
            for (r, x), (_, y) in zip(d, f):
                assert (x is None) == (y is None) and (x is None or close(x, y, 1e-7)), (
                    f"d/d{t} at row {r}: duckdb {x}, datafusion {y}\n{prefix}"
                )
        for e in engines:
            g = grads[e.name]
            if isinstance(g, Refused):
                continue
            total = sum(v for _, v in g if v is not None)
            jvp = attempt(lambda: e.ddx(f"{prefix} SELECT jvp(l, {t}.{col}, 1) AS dl FROM loss"))
            if isinstance(jvp, Refused):
                assert known(jvp, e.name), f"{e.name}: jvp d/d{t} refused, grad didn't: {jvp}\n{prefix}"
            else:
                dl = jvp[0]["dl"] if jvp else None
                assert close(dl or 0.0, total, 1e-7), f"{e.name}: jvp d/d{t} = {dl}, the gradient sums to {total}\n{prefix}"
            want = central_difference(e, ts, t, prefix, EPS)
            if want is None:
                # A NULL loss has no rows to move: its gradient is 0 or NULL.
                assert all(v in (None, 0.0) for _, v in g), f"{e.name}: d/d{t} of a NULL loss is {g}\n{prefix}"
                continue
            got = sum(v * direction(r) for r, v in g if v is not None)
            if not close(got, want, 1e-5):
                # A step across a kink (a filter on x, a NULL from CASE) isn't
                # a gradient's fault: a hundredth of it doesn't cross.
                small = central_difference(e, ts, t, prefix, EPS / 100)
                assert small is not None and close(got, small, 1e-3) and not close(want, small, 1e-3), (
                    f"{e.name}: d/d{t} along the direction is {got}, central differences give {want} "
                    f"(a hundredth of the step: {small})\n{g}\n{prefix}"
                )
