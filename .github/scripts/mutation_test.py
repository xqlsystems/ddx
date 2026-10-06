#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Mutation testing for the v2 simulation soak: how fast does it catch a bug?

A clean soak proves only as much as the soak can see. This seeds one
deliberate bug at a time into ddx-ad (or the DataFusion adapter), runs the
v2 soak until its first failure or a time budget, and restores the source.
A mutant the soak does not kill within the budget marks a blind spot in the
generator or the properties, which is the point of running this.

Each mutant is an exact text replacement that must match its file exactly
once, so a mutant that has gone stale after a refactor fails loudly rather
than silently testing nothing.

Usage (from the repository root, with cargo and protoc on PATH):

    .github/scripts/mutation_test.py [--budget SECS] [--only NAME ...]

Prints a Markdown table: each mutant, whether it was killed, how long that
took, and the tag of the first failure.
"""

import argparse
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time

AD = "crates/ddx-ad/src"
DF = "crates/ddx-datafusion/src"

# (name, file, text, replacement, what it breaks)
MUTANTS = [
    ("sum-halved", f"{AD}/transpose.rs",
     "Rule::Sum => cot,",
     "Rule::Sum => call(divide, vec![cot, lit_f64(2.0)]),",
     "SUM's transpose gives each row half the group's cotangent"),
    ("mean-undivided", f"{AD}/transpose.rs",
     "Rule::Mean => call(divide, vec![cot, field(stat_at[&arg_col])]),",
     "Rule::Mean => cot,",
     "AVG's transpose forgets to divide by the count"),
    ("mean-over-sum", f"{AD}/transpose.rs",
     "windows.push(window(count, vec![field(arg_col)], keys.clone()));",
     "windows.push(window(sum, vec![field(arg_col)], keys.clone()));",
     "AVG divides by the sum of its argument, not the count"),
    ("tie-unshared", f"{AD}/transpose.rs",
     "jitters(self.f, &region, arg_col),\n"
     "                        ),\n"
     "                        call(divide, vec![cot, field(stat_at[&arg_col])]),",
     "jitters(self.f, &region, arg_col),\n"
     "                        ),\n"
     "                        cot,",
     "MAX/MIN give every tied row the whole cotangent (wrong only at ties)"),
    ("extreme-everyone", f"{AD}/transpose.rs",
     "                    )],\n                    null_f64(),\n                ),\n            };",
     "                    )],\n                    field(cotangent_at + i),\n                ),\n            };",
     "MAX/MIN send the cotangent to rows that do not attain the extreme too"),
    ("null-skip-leaks", f"{AD}/transpose.rs",
     "vec![(call(is_null, vec![field(arg_col)]), null_f64())],",
     "vec![(call(is_null, vec![lit_f64(0.0)]), null_f64())],",
     "a row an aggregate skips as NULL still sends gradient to its other inputs"),
    ("partial-is-one", f"{AD}/transpose.rs",
     "if as_number(&d) == Some(1.0) {",
     "if as_number(&d).is_some() {",
     "the map rule treats any constant partial as 1"),
    ("fan-in-first", f"{AD}/program.rs",
     "let mut aligned: Vec<Rel> = contribs\n        .into_iter()",
     "let mut aligned: Vec<Rel> = contribs\n        .into_iter()\n        .take(1)",
     "fan-in keeps only the first contribution to an input"),
    ("unreached-one", f"{AD}/program.rs",
     "None => if_then(vec![null_value], lit_f64(0.0)),",
     "None => if_then(vec![null_value], lit_f64(1.0)),",
     "a row no gradient reached gets 1, not 0"),
    ("null-row-zero", f"{AD}/program.rs",
     "let null_value = (call(is_null, vec![field(k + n)]), null_f64());",
     "let null_value = (call(is_null, vec![field(k + n)]), lit_f64(0.0));",
     "a NULL value's gradient is 0, not NULL"),
    ("stop-gradient-ignored", f"{AD}/elementwise.rs",
     "        if functions.is_stop_gradient(f.function_reference)? {\n            return Ok(false);",
     "        if false && functions.is_stop_gradient(f.function_reference)? {\n            return Ok(false);",
     "ddx_stop_gradient no longer stops activity"),
    ("dims-check-off", f"{AD}/program.rs",
     "crate::emit::filter(grouped, call(gt, vec![field(k), lit_f64(1.0)])),",
     "crate::emit::filter(grouped, call(gt, vec![field(k), lit_f64(1e9)])),",
     "the check that dims identify rows never fires"),
    ("ranking-check-off", f"{AD}/transpose.rs",
     "if dims.iter().any(|d| !keys.contains(&(offset + d))) {",
     "if false && dims.iter().any(|d| !keys.contains(&(offset + d))) {",
     "a ranking that does not break ties is accepted"),
    ("gradient-names-collide", f"{AD}/program.rs",
     'format!("{namespace}grad_{i}_{}", readable(table))',
     'format!("{namespace}grad_{}", readable(table))',
     "two wrt tables with one readable name share a gradient step"),
    ("release-keeps-value", f"{DF}/ad.rs",
     "    for i in 0..program.step_count() {\n"
     "        let step = program.step(i);\n"
     "        ctx.deregister_table(step.name.as_str())?;",
     "    for i in 0..program.step_count() {\n"
     "        let step = program.step(i);\n"
     "        if program.is_result(&step.name) { continue; }\n"
     "        ctx.deregister_table(step.name.as_str())?;",
     "release leaves the value table on the context"),
    ("sql-case-sensitive", f"{AD}/sql.rs",
     "table_matches(&w.table, table) && w.column.eq_ignore_ascii_case(c)",
     "table_matches(&w.table, table) && w.column == *c",
     "grad in SQL treats W.VAL and w.val as different columns"),
]


def run(budget, base, log, skip=None, stop=True):
    # A clean baseline: the seeds that fail unmutated are skipped, so a kill
    # is the mutant's doing.
    env = dict(os.environ,
               DDX_SOAK_SECS=str(budget), DDX_SOAK_BASE=str(base),
               DDX_SOAK_LOG=log)
    if stop:
        env["DDX_SOAK_STOP_ON_FAIL"] = "1"
    if skip:
        env["DDX_V2_SKIP_SEEDS"] = skip
    cmd = ["cargo", "test", "-q", "-p", "ddx-datafusion", "--test", "ad_simulation",
           "--release", "--", "--ignored", "--nocapture", "soak_v2_query_ad"]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=int, default=120, help="soak seconds per mutant")
    ap.add_argument("--base", type=int, default=0, help="first soak seed")
    ap.add_argument("--only", nargs="*", help="run only these mutants")
    args = ap.parse_args()

    # Put a mutated file back if the run is interrupted.
    restore = {}

    def put_back(*_):
        for f, text in restore.items():
            f.write_text(text)
        sys.exit(130)

    signal.signal(signal.SIGINT, put_back)
    signal.signal(signal.SIGTERM, put_back)

    # The baseline: the unmutated soak over the same seeds, a little longer
    # than any mutant gets. Every seed that fails here is skipped below.
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as t:
        base_log = t.name
    run(int(args.budget * 1.5), args.base, base_log, stop=False)
    base_text = pathlib.Path(base_log).read_text() if pathlib.Path(base_log).exists() else ""
    bad = sorted(set(re.findall(r"FAILURE \(seed=(\d+)", base_text)))
    # How often ddx accepts a generated case, unmutated: a mutant that makes
    # it refuse much more is a regression too, even if every answer it still
    # gives is right (a stop-gradient ddx no longer understands is refused,
    # not answered wrongly).
    acc = re.search(r"cases=(\d+) accepted=(\d+)", base_text)
    base_rate = int(acc.group(2)) / max(int(acc.group(1)), 1) if acc else None
    with tempfile.NamedTemporaryFile("w", suffix=".seeds", delete=False) as t:
        t.write("\n".join(bad))
        skip = t.name
    print(f"baseline: {len(bad)} seed(s) fail unmutated and are skipped: {bad}",
          file=sys.stderr, flush=True)

    rows = []
    for name, path, text, repl, what in MUTANTS:
        if args.only and name not in args.only:
            continue
        file = pathlib.Path(path)
        original = file.read_text()
        n = original.count(text)
        if n != 1:
            rows.append((name, what, f"STALE: matches {n} times", "", ""))
            continue
        restore[file] = original
        file.write_text(original.replace(text, repl))
        try:
            with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as t:
                log = t.name
            start = time.time()
            r = run(args.budget, args.base, log, skip)
            wall = time.time() - start
            text_log = pathlib.Path(log).read_text() if pathlib.Path(log).exists() else ""
            done = re.search(r"SOAK done: elapsed=(\d+)s iters=(\d+) failures=(\d+)", text_log)
            if not done:
                tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
                rows.append((name, what, "did not run (does it compile?)", f"{wall:.0f}s",
                             " / ".join(tail)[:120]))
                continue
            secs, iters, fails = (int(g) for g in done.groups())
            tag = re.search(r"FAILURE \(seed=(\d+)[^)]*\):\n(\[[^\]]+\])", text_log)
            acc = re.search(r"cases=(\d+) accepted=(\d+)", text_log)
            rate = int(acc.group(2)) / max(int(acc.group(1)), 1) if acc else None
            if fails:
                rows.append((name, what, "killed", f"{secs}s, {iters} cases",
                             f"{tag.group(2)} (seed {tag.group(1)})" if tag else "?"))
            elif base_rate is not None and rate is not None and rate < base_rate - 0.05:
                rows.append((name, what, "killed by refusals", f"{secs}s, {iters} cases",
                             f"accepts {rate:.0%} of cases, {base_rate:.0%} unmutated"))
            else:
                rows.append((name, what, "SURVIVED", f"{secs}s, {iters} cases", ""))
        finally:
            file.write_text(original)
            restore.pop(file, None)
        print(f"{name}: {rows[-1][2]} {rows[-1][3]} {rows[-1][4]}", file=sys.stderr, flush=True)

    print("| mutant | what it breaks | result | time to kill | first failure |")
    print("|---|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    survived = [r for r in rows if not r[2].startswith("killed")]
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
