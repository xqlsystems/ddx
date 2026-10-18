# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
#
# SPDX-License-Identifier: Apache-2.0

import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(dst, "w") as out:
    for line in open(src):
        e = json.loads(line)
        k = e["kind"]
        if k == "load":
            out.write(f"@load {e['name']} {e['file']}\n")
        elif k == "query_all":
            out.write("@query_all\n" + "\n@sep\n".join(s.strip() for s in e["sql"]) + "\n@end\n")
        else:
            out.write(f"@{k}\n{e['sql'].strip()}\n@end\n")
