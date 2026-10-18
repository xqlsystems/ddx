# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run an ML demo, recording what it asks the DataFusion host to do.

    capture.py OUT_DIR SCRIPT [ARGS...]

Writes OUT_DIR/events.jsonl, one event per host call in order (load,
execute, query, query_all, each with its SQL and wall time), and each
loaded table as OUT_DIR/tables/<name>.<n>.parquet, so a Rust profiler can
replay the same statements on the same data.
"""

import json
import pathlib
import runpy
import sys
import time

out = pathlib.Path(sys.argv[1])
script = pathlib.Path(sys.argv[2]).resolve()
sys.argv = [str(script), *sys.argv[3:]]
sys.path.insert(0, str(script.parent))
(out / "tables").mkdir(parents=True, exist_ok=True)
log = open(out / "events.jsonl", "w")
loads = 0

import _hosts  # noqa: E402

DF = _hosts.DataFusion


def emit(**event):
    log.write(json.dumps(event) + "\n")
    log.flush()


def timed(kind, method):
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = method(self, *args, **kwargs)
        emit(kind=kind, sql=args[0], seconds=time.perf_counter() - start)
        return result

    return wrapper


orig_load = DF.load


def load(self, name, data, **options):
    global loads
    orig_load(self, name, data, **options)
    names = [f"{name}.{v}" for v in options.get("table_names", {}).values()] or [name]
    limit = " LIMIT 0" if name == "era5" else ""
    for n in names:
        loads += 1
        path = out / "tables" / f"{n}.{loads}.parquet"
        self.ctx.sql(f'SELECT * FROM {n}{limit}').write_parquet(str(path))
        emit(kind="load", name=n, file=str(path))


DF.load = load
DF.execute = timed("execute", DF.execute)
DF.query = timed("query", DF.query)
DF.query_all = timed("query_all", DF.query_all)

runpy.run_path(str(script), run_name="__main__")
