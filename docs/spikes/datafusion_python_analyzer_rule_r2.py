# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""M1 / R2 spike: can `datafusion-python` inject an `AnalyzerRule` into its
`SessionContext`?

This is the milestone-M1 re-verification of decision-log `R2` (design.md §3.4,
§8 M1). The answer decides nothing less than which rewrite path v1 ships on for
the xarray-sql acceptance target: if an `AnalyzerRule` *could* be injected from
Python, Path B (in-engine plan rewrite, §3.3) would be available there and the
SQL text rewrite would be a fallback rather than the design's chosen path.

The claim under test is a *negative* one, so it is checked three ways, from the
outside in:

  T1  the Python `SessionContext` surface       — is there an `add_analyzer_rule`?
  T2  the `_internal` PyO3 surface underneath   — same question, no sugar
  T3  the compiled FFI capsule vocabulary       — could a *compiled Rust*
                                                  extension inject one, even
                                                  though Python can't?

T3 is the one that makes this durable. T1/T2 only say "the binding doesn't
expose it today"; T3 says the `datafusion-ffi` boundary the bindings are built
on has no analyzer-rule capsule *at all*, so the gap is structural rather than
a missing convenience method.

The remaining tests pin the two facts that follow from the answer:

  T4  a scalar UDF named `grad` sees VALUES, never the symbolic argument
      (design.md §3.1 — the reason a scalar UDF can't be the mechanism)
  T5  Path A (rewrite the SQL text, hand plain SQL to a stock context) works
  T6  the plan-serialization seam, recorded as future-only (design.md §8 M1 exit)
  T7  the TABLE-function seam — which does receive an unevaluated expression,
      and is the DataFusion analogue of DuckDB's `ddx('<sql>')` — together with
      the two limits that stop it from being a path to bare `grad()`

Run:
    python -m venv .venv && . .venv/bin/activate
    pip install datafusion pyarrow
    python docs/spikes/datafusion_python_analyzer_rule_r2.py
"""

import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import datafusion
from datafusion import SessionContext, udf
from datafusion.plan import LogicalPlan


def line(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"{'OK  ' if ok else 'FAIL'} {name}" + (f"  — {detail}" if detail else ""))


print(f"datafusion-python version: {datafusion.__version__}")

# ---------------------------------------------------------------------------
line("T1  the public Python SessionContext surface")

public = sorted(n for n in dir(SessionContext) if not n.startswith("_"))
rule_methods = [
    n
    for n in public
    if any(k in n.lower() for k in ("analyz", "optimiz", "rule", "extension"))
]
print("rule/extension-shaped methods:", rule_methods)

check(
    "T1 no add_analyzer_rule on SessionContext",
    not hasattr(SessionContext, "add_analyzer_rule"),
)
check(
    "T1 no add_optimizer_rule either (logical optimizer also closed)",
    not hasattr(SessionContext, "add_optimizer_rule"),
    "only remove_optimizer_rule exists — removal without addition",
)
check(
    "T1 remove_optimizer_rule IS present (the asymmetry is real, not an oversight)",
    hasattr(SessionContext, "remove_optimizer_rule"),
)
check(
    "T1 add_physical_optimizer_rule IS present — but PHYSICAL only",
    hasattr(SessionContext, "add_physical_optimizer_rule"),
    "runs after planning; the marker's argument is already compiled by then",
)

# ---------------------------------------------------------------------------
line("T2  the _internal PyO3 surface underneath the sugar")

from datafusion._internal import SessionContext as RawCtx  # noqa: E402

raw = sorted(n for n in dir(RawCtx) if not n.startswith("_"))
check(
    "T2 no add_analyzer_rule on the raw PyO3 class either",
    not hasattr(RawCtx, "add_analyzer_rule"),
    f"{len(raw)} raw methods scanned",
)
check(
    "T2 no SessionState / SessionStateBuilder handle exposed to Python",
    not any("sessionstate" in n.lower().replace("_", "") for n in raw),
    "SessionState is where add_analyzer_rule lives in native Rust DataFusion",
)

# ---------------------------------------------------------------------------
line("T3  the compiled FFI capsule vocabulary (can a Rust extension do it?)")

# datafusion-python imports foreign objects through `__datafusion_*__` PyCapsule
# attributes. The set of capsule names compiled into the extension module IS the
# extension vocabulary: anything not named here cannot cross the boundary at all,
# from Python or from a compiled Rust extension.
#
# Ask the module where it actually lives rather than guessing a filename: the
# extension is `_internal.abi3.so` on Linux/macOS but `_internal.abi3.pyd` on
# Windows, and a non-abi3 build yields `_internal.cpython-3XX-<plat>.so`. A
# hardcoded name would make this test silently skip off-platform — and T3 is the
# check that makes the whole finding durable, so a skip must be LOUD.
import datafusion._internal as _df_internal  # noqa: E402

so_path = _df_internal.__file__
capsules = set()
if so_path and os.path.exists(so_path):
    import re

    with open(so_path, "rb") as f:
        blob = f.read()
    # The literals are packed contiguously in the binary, so a greedy match can
    # span several names; re-split each hit into individual capsule names.
    for m in re.finditer(rb"__datafusion_[a-z_]+?__", blob):
        for part in re.findall(
            r"__datafusion_[a-z]+(?:_[a-z]+)*__", m.group(0).decode()
        ):
            capsules.add(part)
    for c in sorted(capsules):
        print("   ", c)
else:
    print(f"    compiled module not found (path={so_path!r})")

# Failure, not a skip: if the capsule set can't be read, the durable part of this
# spike did not run, and the verdict below must not claim it did.
check(
    "T3 the compiled extension's capsule vocabulary was readable",
    bool(capsules),
    f"scanned {so_path!r}" if capsules else "T3 could not run — verdict is NOT established",
)
check(
    "T3 NO __datafusion_analyzer_rule__ capsule exists",
    bool(capsules) and "__datafusion_analyzer_rule__" not in capsules,
    "so even a compiled Rust extension cannot inject an AnalyzerRule",
)
check(
    "T3 NO logical __datafusion_optimizer_rule__ capsule either",
    bool(capsules) and "__datafusion_optimizer_rule__" not in capsules,
)
check(
    "T3 __datafusion_physical_optimizer_rule__ DOES exist",
    "__datafusion_physical_optimizer_rule__" in capsules,
    "the only rule-injection capsule, and it is post-planning",
)

# ---------------------------------------------------------------------------
line("T4  design.md §3.1: what does a scalar UDF named `grad` actually receive?")

ctx = SessionContext()
ctx.from_pydict({"x": [1.0, 2.0, 3.0]}, name="t")

seen = []


def grad_impl(arg, wrt):
    # If `grad` could be a real UDF, THIS is where differentiation would happen.
    # Record what actually arrives.
    seen.append(("arg", arg.to_pylist(), "wrt", wrt.to_pylist()))
    return pa.array([float("nan")] * len(arg))


ctx.register_udf(
    udf(grad_impl, [pa.float64(), pa.float64()], pa.float64(), "stable", name="grad")
)
ctx.sql("SELECT grad(x * x, x) AS d FROM t").collect()

print("   the UDF was called with:", seen)
# x*x over x=[1,2,3] is [1,4,9]: the ARGUMENT WAS ALREADY EVALUATED.
got_values = seen and seen[0][1] == [1.0, 4.0, 9.0]
check(
    "T4 the UDF received evaluated VALUES [1.0, 4.0, 9.0], not the expression x*x",
    bool(got_values),
    "differentiation needs symbolic form — a SCALAR UDF can never do it (§3.1). "
    "This is specifically about scalar UDFs; T7 examines the table-function case.",
)

# ---------------------------------------------------------------------------
line("T5  Path A: rewrite the SQL text, hand plain SQL to a stock context")

# What ddx-core's rewrite_sql produces for `grad(x * x, x)` (0/1-folded product
# rule), spliced back over the marker's source span. A stock, unmodified
# SessionContext then plans and runs it with no hook of any kind.
rewritten = "SELECT (x + x) AS d FROM t"
path_a = ctx.sql(rewritten).collect()[0].column(0).to_pylist()
print("   rewritten SQL:", rewritten, "=>", path_a)
check(
    "T5 Path A yields d(x*x)/dx = 2x on an unmodified SessionContext",
    path_a == [2.0, 4.0, 6.0],
)

# ---------------------------------------------------------------------------
line("T6  seams that DO exist — recorded as future-only (M1 exit criterion)")

tmp = tempfile.mkdtemp()
pf = os.path.join(tmp, "t.parquet")
pq.write_table(pa.table({"x": [1.0, 2.0, 3.0]}), pf)
pctx = SessionContext()
pctx.register_parquet("t", pf)

plan = pctx.sql("SELECT x * x AS y FROM t").logical_plan()
blob = plan.to_bytes()
back = LogicalPlan.from_bytes(pctx, blob)
out = pctx.execute_logical_plan(back).collect()[0].column(0).to_pylist()
print(f"   LogicalPlan -> {len(blob)} proto bytes -> LogicalPlan -> execute => {out}")
check(
    "T6 SEAM: LogicalPlan proto round-trip + execute_logical_plan works",
    out == [1.0, 4.0, 9.0],
    "an OUT-of-engine plan rewrite is possible — but see the verdict below",
)

# The seam has a hard edge worth recording: it needs a serializable table source.
mem = SessionContext()
mem.from_pydict({"x": [1.0]}, name="t")
try:
    mem.sql("SELECT x FROM t").logical_plan().to_bytes()
    mem_ok = True
    mem_err = ""
except Exception as e:  # noqa: BLE001 - reporting the error IS the result
    # `or [""]`: an exception with an empty message makes splitlines() return [],
    # and an IndexError here would abort the handler whose only job is to report.
    mem_ok, mem_err = False, (str(e).splitlines() or [""])[-1]
check(
    "T6 SEAM LIMIT: in-memory tables fail to serialize without an extension codec",
    not mem_ok,
    mem_err[:90],
)

# ---------------------------------------------------------------------------
line("T7  the table-function seam — the closest analogue to DuckDB's ddx('<sql>')")

# T4 shows a SCALAR UDF gets values. A *table* function is a different animal:
# `register_udtf` hands the callable a `RawExpr`, i.e. an unevaluated expression
# object. That is a genuine third seam and the DataFusion-native analogue of the
# `ddx('<sql>')` table function the design already ships for DuckDB (§3.4).
#
# The question that decides whether it changes anything: can it carry a SYMBOLIC
# `grad(expr, col)` over real table columns — i.e. is it a path to bare `grad()`?
import pyarrow.dataset as pads  # noqa: E402
from datafusion import udtf  # noqa: E402

tf_seen = []


class _ProbeTF:
    def __init__(self, *args):
        tf_seen.append([(type(a).__name__, str(a)) for a in args])

    def __call__(self):
        return pads.dataset(pa.table({"d": pa.array([1.0])}))


ctx.register_udtf(udtf(_ProbeTF, name="ddx_probe"))


def probe(sql):
    tf_seen.clear()
    try:
        ctx.sql(sql).collect()
    except Exception as e:  # noqa: BLE001 - the error IS the observation
        return list(tf_seen), f"{type(e).__name__}: {str(e).splitlines()[0][:70]}"
    return list(tf_seen), None


lit_seen, _ = probe("SELECT * FROM ddx_probe('SELECT grad(x*x, x) FROM t')")
print("   SQL-string arg  ->", lit_seen)
check(
    "T7 SEAM: a table function receives an unevaluated RawExpr, not a value",
    bool(lit_seen) and "RawExpr" in lit_seen[0][0][0],
    "the SQL text is readable off the literal — this IS the ddx('<sql>') shape",
)

fold_seen, _ = probe("SELECT * FROM ddx_probe(sin(2.0) * 3.0)")
print("   composite arg   ->", fold_seen)
# sin(2.0)*3.0 arrives as Float64(2.7278...): the simplifier already ran.
check(
    "T7 LIMIT: a composite argument arrives CONSTANT-FOLDED, not as structure",
    bool(fold_seen) and "Float64" in fold_seen[0][0][1],
    "so the seam does not deliver arbitrary symbolic form either",
)

col_seen, col_err = probe("SELECT * FROM ddx_probe(grad(x * x, x))")
print("   grad over cols  ->", col_seen, "| err:", col_err)
check(
    "T7 LIMIT: a table-function argument cannot reference table columns at all",
    col_err is not None and "No field named" in (col_err or ""),
    "args resolve in an EMPTY schema — so this is not a path to bare grad()",
)

# ---------------------------------------------------------------------------
line("VERDICT")

passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} checks passed\n")

if passed != len(results):
    # This file is a RE-VERIFICATION fixture (see docstring): it will be re-run at
    # M2+. If a future datafusion-python adds `add_analyzer_rule` — the exact event
    # this spike exists to detect — the checks below fail and the stale conclusion
    # must NOT be printed as though it still held.
    print("R2 NOT CONFIRMED — one or more checks failed. The conclusion below is")
    print("withheld deliberately. Re-read the failures above: if an analyzer-rule")
    print("seam has appeared upstream, design.md §3.3/§3.4 and milestone M1 need")
    print("revisiting, and ddx-datafusion's Path B bridge is what would plug in.")
    raise SystemExit(1)

print(
    """R2 CONFIRMED (design.md §3.4, §8 M1).

`datafusion-python` exposes no way to inject an `AnalyzerRule` into a
`SessionContext` — not through the Python API, not through the PyO3 layer
beneath it, and not through the FFI capsule vocabulary a compiled Rust
extension would have to use. The logical optimizer is closed in the same way:
`remove_optimizer_rule` exists with no matching `add`. The only rule-injection
capsule is `__datafusion_physical_optimizer_rule__`, which is useless for ddx
on its face — it runs after planning, so a marker's argument has already been
compiled to a physical expression by the time such a rule could see it, which
is precisely the symbolic form differentiation needs (T4).

So Path A (SQL source-to-source rewrite, §3.3) is the path for the xarray-sql
target by *structure*, not as a workaround for a missing feature. Path B stays
what §3.3 says it is: native Rust DataFusion only, as the reference proof that
ddx-core can drive an in-engine rewrite.

FUTURE-ONLY SEAMS (noted, not adopted):

  1. LogicalPlan <-> proto bytes + execute_logical_plan (T6). A plan CAN be
     pulled out of Python, rewritten, and executed. This is not an AnalyzerRule
     and is worse than Path A for v1 on three counts: it cannot intercept
     `ctx.sql()` transparently (the caller must restructure their code), it
     would require ddx to manipulate DataFusion protobuf plan messages — a
     DataFusion-specific plan-interchange format, which success criterion §1.2
     explicitly rules out — and it fails outright on in-memory tables, the
     common case for xarray-sql. Revisit only if a future need is genuinely
     plan-shaped and DataFusion-only.

  2. add_physical_optimizer_rule via a compiled extension. Post-planning, so it
     cannot serve differentiation. Recorded only to document that the one
     rule seam that does exist was examined and rejected on the merits.

  3. `register_udtf` — a TABLE function, which unlike a scalar UDF receives an
     unevaluated `RawExpr` (T7). This is the DataFusion-native analogue of the
     `ddx('<sql>')` table function the design already ships for DuckDB (§3.4),
     it needs no compiled extension (a plain Python callable works), and it is
     worth adopting in M2 for surface symmetry: the same `ddx('<sql>')` spelling
     could then work on both engines. But it is NOT a path to bare `grad()` and
     does not weaken T4: a composite argument arrives already CONSTANT-FOLDED,
     and a table-function argument cannot reference table columns at all (its
     args resolve in an empty schema, so `ddx_probe(grad(x*x, x))` fails at
     planning with "No field named x"). What it carries is a SQL *string*, which
     ddx-core's `rewrite_sql` then handles exactly as it does under Path A —
     making this a relocation of Path A into the engine, not a new mechanism.

  4. If datafusion-python ever adds an `__datafusion_analyzer_rule__` capsule,
     `ddx-datafusion`'s Path B bridge is the thing that would plug into it —
     which is an additional reason to build Path B in M2 as designed, beyond
     its stated role as the in-engine proof."""
)
