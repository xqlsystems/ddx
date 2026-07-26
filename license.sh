#!/usr/bin/env bash
#
# Add the REUSE / SPDX license header to every Rust and Python source file.
# Safe to re-run: `reuse annotate` is idempotent (an existing, matching header
# is left as-is, not duplicated).
set -euo pipefail

# Run from the repo root regardless of where the script is invoked from.
cd "$(dirname "$0")"

# Enumerate the source files with `git ls-files` rather than a shell glob:
#   * `**/*.rs` does NOT recurse under bash unless `shopt -s globstar` is set
#     (it is not, by default), so the old glob missed every nested file such as
#     crates/ddx-core/src/*.rs — the bug this replaces;
#   * `git ls-files` respects .gitignore, so build output (target/) is excluded
#     for free; `--others --exclude-standard` also picks up new, not-yet-tracked
#     source files.
files=$(git ls-files --cached --others --exclude-standard -- '*.rs' '*.py')

if [ -z "$files" ]; then
  echo "No .rs/.py source files to annotate."
  exit 0
fi

# Word-splitting of $files is intentional here — repository paths contain no
# whitespace. (Keeping this bash 3.2-compatible, i.e. no `mapfile`, for macOS.)
# shellcheck disable=SC2086
uvx "reuse[charset-normalizer]" annotate \
  --copyright "Alexander Merose <al@merose.com> & ddx Authors" \
  --license "Apache-2.0" \
  --skip-unrecognised \
  $files
