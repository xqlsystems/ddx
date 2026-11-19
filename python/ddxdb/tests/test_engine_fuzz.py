# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Random losses, differentiated on DuckDB and DataFusion, and compared.

Each case is a random loss over CTEs of a few small tables. On each engine it
is planned (``backend.plan_sql``) and checked through the API other engines'
integrations use (``ddxdb.grad_plan``, ``ddxdb.run``, ``ddxdb.to_sql``):

1. **SQL:** ``ddxdb.to_sql`` of each engine's plan, in each dialect, run on
   that dialect's engine, gives the loss the query gives.
2. **Value:** the program's value is the loss the query gives.
3. **Gradient:** each gradient agrees with central differences of the
   engine's own loss, along a fixed direction (the loss is a polynomial in
   each table, so a small step is near exact), and with the other engine's
   gradient.
4. **Refusals:** neither engine panics, and a gradient one engine gives the
   other doesn't refuse. Both may refuse one: a loss that reads a table but
   none of its values has no gradient on either. A loss that is NULL whatever
   the values (filters no row passes) may be refused, or have a gradient of
   zeros or NULLs: DuckDB's optimizer folds it to a relation of no rows, which
   reads no table.

The shapes are the ones DuckDB's Substrait producer rewrites, and that ddx
misread: unary minus, which it writes as a one-argument ``subtract`` (#262);
range filters, which it can fold into ``between`` (#263); a one-row table
cross joined (#264); a constant column read through a cross join (#265); and a
cross join whose other side has an unread column of the same name (#266).
While an issue is open, a case whose every failure is that issue's (on its
engine, with its message, from a shape it names) is an xfail that lists the
issues and the failures. Any other failure fails.

Case n draws from its own seed, so it replays alone. ``DDX_ENGINE_FUZZ_CASES``
sets how many run (default 60); ``DDX_ENGINE_FUZZ_SEED`` shifts which:

    DDX_ENGINE_FUZZ_CASES=2000 DDX_ENGINE_FUZZ_SEED=7 pytest tests/test_engine_fuzz.py
"""

import os
import random
import re

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")
datafusion = pytest.importorskip("datafusion")

import ddxdb  # noqa: E402
from ddxdb.ad import DataFusionBackend  # noqa: E402
from ddxdb.duckdb import DuckDBBackend  # noqa: E402

CASES = int(os.environ.get("DDX_ENGINE_FUZZ_CASES", "60"))
SEED = int(os.environ.get("DDX_ENGINE_FUZZ_SEED", "0"))
EPS = 1e-4

# Open issues the generator reaches: (issue, engine, the tag of the shape that
# reaches it, or None for any, and a regular expression the failure matches). When one closes,
# delete its entry; the shape keeps its tag and is held to every check.
OPEN = [
    (262, "duckdb", "unary-minus", "`subtract`"),
    (263, "duckdb", "range-filter", "`between`"),
    # DataFusion writes a join to a side unique on its key, none of whose
    # columns are read, as a semi join.
    (286, "datafusion", None, "SEMI join on the path"),
    # to_sql reads a table read for its rows alone, with no column mask, as
    # all its columns, where DuckDB counts it one column wide.
    (288, "duckdb", "one-row-cross-join", r"^to_sql\(_, '\w+'\) on \w+ gives "),
]


def open_issue(engine, message, tags):
    for issue, on, tag, text in OPEN:
        if engine == on and re.search(text, message) and (tag is None or tag in tags):
            return issue
    return None


class Gen:
    """A random loss: CTEs ``c0``, ``c1``, ... of columns ``i0`` and ``v``, then
    ``loss``. ``tags`` are the shapes it built."""

    def __init__(self, rng):
        self.rng = rng
        self.ctes = []
        self.tags = set()

    def number(self):
        return self.rng.choice(["2.0e0", "0.5e0", "3.0e0", "1.5e0", "-1.0e0"])

    def fx(self, v):
        # Polynomial, so central differences are near exact.
        return self.rng.choice([v, f"{v} * {v}", f"{self.number()} * {v}", f"{v} + {self.number()}", f"{v} - {v} * {v}"])

    def cte(self, body):
        name = f"c{len(self.ctes)}"
        self.ctes.append(f"{name} AS ({body})")
        return name

    def table(self):
        t = self.rng.choice(["w", "x", "w", "x", "s"])
        i0 = "i" if t == "s" else "i0"
        return self.cte(f"SELECT {i0} AS i0, {self.fx('v')} AS v FROM {t}")

    def rel(self, depth=0):
        """A CTE of ``i0`` and ``v``."""
        rng = self.rng
        if depth >= 3 or rng.random() < 0.25:
            return self.table()
        r = rng.randrange(11)
        if r == 0:
            # A constant column, read later through a cross join, maybe.
            self.tags.add("constant-cross-join")
            a = self.rel(depth + 1)
            const = self.cte(
                rng.choice([f"SELECT {self.number()} AS v FROM {a}", f"SELECT i0, {self.number()} AS v FROM {a}"])
            )
            b = self.rel(depth + 1)
            return self.cte(f"SELECT b.i0 AS i0, b.v * k.v AS v FROM {b} b CROSS JOIN {const} k")
        if r == 1:
            self.tags.add("unary-minus")
            a = self.rel(depth + 1)
            return self.cte(rng.choice([f"SELECT i0, -v AS v FROM {a}", f"SELECT i0, -({self.fx('v')}) AS v FROM {a}"]))
        if r == 2:
            self.tags.add("range-filter")
            a = self.rel(depth + 1)
            lo = rng.randrange(2)
            cond = rng.choice([f"i0 >= {lo} AND i0 < {lo + 2}", f"i0 BETWEEN {lo} AND {lo + 1}"])
            return self.cte(f"SELECT i0, v FROM {a} WHERE {cond}")
        if r == 3:
            # Two slices of one relation, the second shifted onto the first,
            # multiplied: a split, as torch-sql writes one (#263).
            self.tags.add("range-filter")
            a = self.rel(depth + 1)
            lo = self.cte(f"SELECT i0, v FROM {a} WHERE i0 >= 0 AND i0 < 1")
            hi = self.cte(f"SELECT i0 - 1 AS i0, v FROM {a} WHERE i0 >= 1 AND i0 < 2")
            return self.cte(f"SELECT p.i0 AS i0, p.v * q.v AS v FROM {lo} p JOIN {hi} q ON p.i0 = q.i0")
        if r == 4:
            # The one-row table's value, across another relation's rows.
            self.tags.add("one-row-cross-join")
            a = self.rel(depth + 1)
            v = rng.choice(["s.v", f"a.v * s.v", f"s.v * {self.number()}"])
            return self.cte(f"SELECT a.i0 AS i0, {v} AS v FROM {a} a CROSS JOIN s")
        if r == 5:
            # A cross join that reads one side's v; the other has one too.
            self.tags.add("cross-join-unread-same-name")
            a, b = self.rel(depth + 1), self.rel(depth + 1)
            return self.cte(f"SELECT a.i0 AS i0, {self.fx('a.v')} AS v FROM {a} a CROSS JOIN {b} b")
        if r == 6:
            a, b = self.rel(depth + 1), self.rel(depth + 1)
            op = rng.choice(["*", "+", "-"])
            return self.cte(f"SELECT a.i0 AS i0, a.v {op} b.v AS v FROM {a} a CROSS JOIN {b} b")
        if r == 7:
            a, b = self.rel(depth + 1), self.rel(depth + 1)
            return self.cte(f"SELECT a.i0 AS i0, a.v * b.v AS v FROM {a} a JOIN {b} b ON a.i0 = b.i0")
        if r == 8:
            a = self.rel(depth + 1)
            return self.cte(f"SELECT i0, SUM(v) AS v FROM {a} GROUP BY i0")
        if r == 9:
            # Columns nothing downstream reads.
            a = self.rel(depth + 1)
            inner = self.cte(f"SELECT i0, v, v * {self.number()} AS unread, i0 + 1 AS i1 FROM {a}")
            return self.cte(f"SELECT i0, v FROM {inner}")
        a = self.rel(depth + 1)
        return self.cte(f"SELECT i0 + 1 AS i0, {self.fx('v')} AS v FROM {a}")

    def loss(self):
        c = self.rel()
        body = self.rng.choice(
            [
                f"SELECT SUM(v) AS l FROM {c}",
                f"SELECT SUM(v * v) AS l FROM {c}",
                f"SELECT SUM(t * t) AS l FROM (SELECT i0, SUM(v) AS t FROM {c} GROUP BY i0) g",
            ]
        )
        return f"WITH {', '.join(self.ctes)} {body}"


def tables(rng):
    """``w`` and ``x``, of a few rows; ``s``, of one. Values are multiples of
    1/8, so the engines' sums are exact and agree."""

    def values(n):
        return pa.array([(rng.randrange(17) - 8) / 8 for _ in range(n)], pa.float64())

    out = {}
    for t in "wx":
        n = rng.randrange(1, 5)
        out[t] = pa.table(
            {"r": pa.array(range(n), pa.int64()), "i0": pa.array([rng.randrange(3) for _ in range(n)], pa.int64()), "v": values(n)}
        )
    out["s"] = pa.table({"r": pa.array([0], pa.int64()), "i": pa.array([0], pa.int64()), "v": values(1)})
    return out


def direction(r):
    """The direction of the central differences, at row r: varied, not all one."""
    return ((r * 7 + 3) % 9 - 4) / 4


def perturbed(table, h):
    vals = [v + h * direction(r) for r, v in zip(table["r"].to_pylist(), table["v"].to_pylist())]
    return table.set_column(table.column_names.index("v"), "v", pa.array(vals, pa.float64()))


class DuckDb:
    name = "duckdb"

    def __init__(self, ts):
        self.con = duckdb.connect()
        self.backend = DuckDBBackend(self.con)
        for t, table in ts.items():
            self.put(t, table)

    def put(self, t, table):
        self.con.register(f"_{t}", table)
        self.con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM _{t}")

    def rows(self, q):
        return [tuple(r) for r in self.con.sql(q).fetchall()]


class DataFusion:
    name = "datafusion"

    def __init__(self, ts):
        self.ctx = datafusion.SessionContext()
        self.backend = DataFusionBackend(self.ctx)
        for t, table in ts.items():
            self.put(t, table)

    def put(self, t, table):
        self.ctx.deregister_table(t)
        self.ctx.register_record_batches(t, [table.to_batches()])

    def rows(self, q):
        return [tuple(r.values()) for r in self.ctx.sql(q).to_pylist()]


def close(x, y, tol):
    if x is None or y is None:
        return x is None and y is None
    return abs(x - y) <= tol * (1 + abs(y))


class Refused(str):
    pass


def attempt(f):
    """f's value, or the first line of what it raised; a panic fails the case."""
    try:
        return f()
    except BaseException as e:  # a Rust panic is a BaseException
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        assert type(e).__name__ != "PanicException", f"ddx panicked: {e}"
        return Refused(f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")


def scalar(engine, q):
    return engine.rows(q)[0][0]


def central_difference(engine, ts, t, loss, eps=EPS):
    fs = []
    for h in (eps, -eps):
        engine.put(t, perturbed(ts[t], h))
        fs.append(scalar(engine, loss))
    engine.put(t, ts[t])
    return (fs[0] - fs[1]) / (2 * eps)


def check(engine, engines, ts, loss, want, reads, failures):
    """The SQL, value and gradient checks on one engine. Returns its
    gradients, by table: a list of (r, gradient), or a Refused."""
    plan = attempt(lambda: engine.backend.plan_sql(loss))
    if isinstance(plan, Refused):
        failures.append((engine.name, f"plan_sql refused: {plan}"))
        return {}
    for dialect, runner in engines.items():
        sql = attempt(lambda: ddxdb.to_sql(plan, dialect))
        if isinstance(sql, Refused):
            failures.append((engine.name, f"to_sql(_, {dialect!r}) refused: {sql}"))
            continue
        got = attempt(lambda: scalar(runner, sql))
        if isinstance(got, Refused) or not close(got, want, 1e-9):
            failures.append((engine.name, f"to_sql(_, {dialect!r}) on {dialect} gives {got}, the loss is {want}"))
    grads = {}
    for t in reads:
        def run():
            program = ddxdb.grad_plan(plan, [f"{t}.v"])
            ddxdb.run(engine.backend, program)
            (value,) = engine.rows(program.value.query())[0]
            g = program.gradients[0]
            r, v = g.columns.index("r"), g.columns.index("v")
            rows = sorted((row[r], row[v]) for row in engine.rows(g.query()))
            return value, rows

        out = attempt(run)
        if isinstance(out, Refused):
            grads[t] = out
            continue
        value, rows = out
        grads[t] = rows
        if not close(value, want, 1e-9):
            failures.append((engine.name, f"the program's value is {value}, the loss is {want}"))
        if want is None:
            continue
        dot = sum(g * direction(r) for r, g in rows if g is not None)
        fd = central_difference(engine, ts, t, loss)
        if not close(dot, fd, 1e-5):
            # Near a minimum of a steep polynomial a step's error is larger
            # than the derivative; a hundredth of it isn't.
            fd = central_difference(engine, ts, t, loss, EPS / 100)
        if not close(dot, fd, 1e-5):
            failures.append((engine.name, f"d/d{t}.v along the direction is {dot}, central differences give {fd}: {rows}"))
    return grads


ENGINES = {"duckdb": DuckDb, "datafusion": DataFusion}


@pytest.mark.parametrize("case", range(SEED * 1_000_000, SEED * 1_000_000 + CASES))
def test_engines_agree_on_a_random_loss(case):
    rng = random.Random(case)
    gen = Gen(rng)
    loss = gen.loss()
    ts = tables(rng)
    engines = {name: E(ts) for name, E in ENGINES.items()}
    wants = {name: scalar(e, loss) for name, e in engines.items()}
    assert close(wants["duckdb"], wants["datafusion"], 1e-9), f"the engines' losses differ: {wants}\n{loss}"
    want = wants["datafusion"]
    reads = [t for t in "wxs" if f"FROM {t}" in loss or f"JOIN {t}" in loss]
    failures = []
    grads = {name: check(e, engines, ts, loss, want, reads, failures) for name, e in engines.items()}
    for t in reads:
        d, f = grads["duckdb"].get(t), grads["datafusion"].get(t)
        if want is None:
            for name, g in (("duckdb", d), ("datafusion", f)):
                if isinstance(g, list) and any(v not in (None, 0.0) for _, v in g):
                    failures.append((name, f"d/d{t}.v of a NULL loss is {g}"))
            continue
        for name, g, other in (("duckdb", d, f), ("datafusion", f, d)):
            if isinstance(g, Refused) and isinstance(other, list):
                failures.append((name, f"d/d{t}.v refused: {g}"))
        if isinstance(d, list) and isinstance(f, list):
            if [r for r, _ in d] != [r for r, _ in f] or not all(close(x, y, 1e-7) for (_, x), (_, y) in zip(d, f)):
                # Each engine's own central differences say which is wrong;
                # this is listed with DuckDB's, whose producer is the one the
                # open issues are about.
                failures.append(("duckdb", f"d/d{t}.v: duckdb {d}, datafusion {f}"))
    if not failures:
        return
    report = "\n".join(f"  {name}: {msg}" for name, msg in failures) + f"\n{loss}"
    issues = [open_issue(name, msg, gen.tags) for name, msg in failures]
    if None in issues:
        pytest.fail(report)
    pytest.xfail(f"open {', '.join(f'#{i}' for i in sorted(set(issues)))}:\n{report}")
