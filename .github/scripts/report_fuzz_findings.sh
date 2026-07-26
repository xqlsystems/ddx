#!/usr/bin/env bash
#
# report_fuzz_findings.sh — triage the nightly simulation-soak logs and, if any
# property failures were found, file (or update) a single GitHub issue with a
# reproducible repro. Invoked by .github/workflows/nightly-fuzz.yml.
#
# Anti-spam policy: at most ONE open `fuzz-finding` issue at a time. If one is
# already open, this appends the new run's findings as a comment instead of
# opening a duplicate — a persistent bug accumulates recurrences on one issue,
# and a fresh issue only opens once a human has triaged and closed the last one.
#
# Usage: report_fuzz_findings.sh <log-dir>
#   Reads every `soak-*.log` under <log-dir>. Requires `gh` authenticated with
#   `issues: write` (the workflow's GITHUB_TOKEN).
#
# Env (provided by the workflow, all optional — used only to enrich the body):
#   GITHUB_RUN_ID, GITHUB_SHA, GITHUB_SERVER_URL, GITHUB_REPOSITORY

set -euo pipefail

LOG_DIR=${1:?"usage: report_fuzz_findings.sh <log-dir>"}

shopt -s nullglob globstar
logs=("$LOG_DIR"/**/soak-*.log "$LOG_DIR"/soak-*.log)
if [ ${#logs[@]} -eq 0 ]; then
  echo "No soak-*.log files found under $LOG_DIR; nothing to report."
  exit 0
fi

# ---------------------------------------------------------------------------
# Tally failures and collect a bounded, readable set of failure blocks.
# ---------------------------------------------------------------------------
total=0
blocks=""       # up to MAX_BLOCKS failure blocks, across all regions
env_line=""     # rustc / os / sha, recorded by the soak step
region_summary=""
MAX_BLOCKS=8
block_count=0

for log in "${logs[@]}"; do
  [ -f "$log" ] || continue
  n=$(grep -c '^FAILURE (seed' "$log" 2>/dev/null || true)
  n=${n:-0}
  total=$((total + n))

  # The soak step records one `ENV ...` line at the top of each log.
  if [ -z "$env_line" ]; then
    env_line=$(grep -m1 '^ENV ' "$log" 2>/dev/null || true)
  fi

  base=$(grep -m1 '^SOAK start' "$log" 2>/dev/null | sed -n 's/.*base=\([0-9]*\).*/\1/p')
  iters=$(grep -m1 '^SOAK done' "$log" 2>/dev/null | sed -n 's/.*iters=\([0-9]*\).*/\1/p')
  region_summary+="- \`$(basename "$log")\`: base=${base:-?}, iters=${iters:-?}, failures=${n}"$'\n'

  if [ "$n" -gt 0 ] && [ "$block_count" -lt "$MAX_BLOCKS" ]; then
    # Extract each FAILURE block: the header plus its report lines, stopping at
    # the next heartbeat / failure / summary line.
    while IFS= read -r line; do
      case "$line" in
        "===BLOCK===")
          block_count=$((block_count + 1))
          [ "$block_count" -ge "$MAX_BLOCKS" ] && break
          ;;
        *) blocks+="$line"$'\n' ;;
      esac
    done < <(awk '
      /^FAILURE \(seed/ { if (inblk) print "===BLOCK==="; inblk=1; print; next }
      inblk == 1 {
        if ($0 ~ /^HEARTBEAT/ || $0 ~ /^SOAK/ || $0 == "") { inblk=0; print "===BLOCK==="; next }
        print
      }
      END { if (inblk) print "===BLOCK===" }
    ' "$log")
  fi
done

if [ "$total" -eq 0 ]; then
  echo "Soak completed with 0 property failures. Nothing to file."
  exit 0
fi

echo "Found $total property failure(s) across ${#logs[@]} region log(s); preparing issue."

# ---------------------------------------------------------------------------
# A ready-to-paste repro from the first failing seed. The soak's per-iteration
# seed is `base + iter`, so re-running with DDX_SOAK_BASE set to a failing
# `seed=` value reproduces that exact case as iteration 0 (same depth + wrt).
# ---------------------------------------------------------------------------
first_seed=$(printf '%s' "$blocks" | sed -n 's/^FAILURE (seed=\([0-9]*\).*/\1/p' | head -1)
first_seed=${first_seed:-0}

run_url="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-xqlsystems/ddx}/actions/runs/${GITHUB_RUN_ID:-local}"
date_utc=$(date -u '+%Y-%m-%d %H:%M UTC')

body_file=$(mktemp)
{
  echo "**Nightly simulation-soak fuzz found ${total} property failure(s).**"
  echo
  echo "- Run: [${GITHUB_RUN_ID:-local}](${run_url})"
  echo "- Commit: \`${GITHUB_SHA:-unknown}\`"
  echo "- When: ${date_utc}"
  [ -n "$env_line" ] && echo "- Build: \`${env_line#ENV }\`"
  echo
  echo "### Reproduce"
  echo
  echo "The soak's per-iteration seed is \`base + iter\`, so setting \`DDX_SOAK_BASE\`"
  echo "to a failing \`seed=\` value re-runs that exact case as iteration 0. On the"
  echo "same platform as the build above (floating-point transcendentals are not"
  echo "bit-identical across OS/toolchain):"
  echo
  echo '```bash'
  echo "DDX_SOAK_SECS=15 DDX_SOAK_BASE=${first_seed} \\"
  echo "  cargo test -p ddx-core --test simulation --release \\"
  echo "  -- --ignored --nocapture soak_continuous_property_fuzz"
  echo '```'
  echo
  echo "The full per-region logs are attached to the workflow run as artifacts."
  echo
  echo "### Per-region summary"
  echo
  printf '%s\n' "$region_summary"
  echo "### Failure samples (up to ${MAX_BLOCKS})"
  echo
  echo '```'
  printf '%s' "$blocks"
  echo '```'
  echo
  echo "> Triage note: the finite-difference oracle is gated against float"
  echo "> cancellation (magnitude cap) and truncation/aliasing (Richardson"
  echo "> self-consistency), so a surviving \`[finite-diff]\` disagreement should be"
  echo "> a real rule bug — but confirm by reproducing before assuming. \`[render]\`"
  echo "> and \`[self-consumption]\` failures are always real."
  echo
  echo "<!-- fuzz-run: ${GITHUB_RUN_ID:-local} -->"
} > "$body_file"

# ---------------------------------------------------------------------------
# Ensure labels exist (idempotent), then dedup: comment on the single open
# fuzz-finding issue if one exists, else create a new one.
# ---------------------------------------------------------------------------
gh label create fuzz-finding --color B60205 \
  --description "Found by the nightly simulation-soak fuzz" 2>/dev/null || true
gh label create needs-triage --color FBCA04 \
  --description "Awaiting human triage" 2>/dev/null || true

existing=$(gh issue list --state open --label fuzz-finding \
  --json number --jq '.[0].number // empty' 2>/dev/null || true)

if [ -n "$existing" ]; then
  echo "Open fuzz-finding issue #${existing} exists; commenting instead of filing a duplicate."
  {
    echo "### Recurred — run ${GITHUB_RUN_ID:-local} (${date_utc})"
    echo
    cat "$body_file"
  } | gh issue comment "$existing" --body-file -
  echo "Commented on #${existing}."
else
  title="Nightly fuzz: ${total} property failure(s) in ddx-core simulation soak"
  url=$(gh issue create --title "$title" \
    --label fuzz-finding --label needs-triage \
    --body-file "$body_file")
  echo "Filed new issue: $url"
fi

rm -f "$body_file"
