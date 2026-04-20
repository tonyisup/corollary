#!/usr/bin/env bash
set -euo pipefail

# Corollary M1 quiet baseline sanity check for repeated ba
#
# Purpose:
# - compare single ba vs repeated ba in quiet only
# - determine whether repeated ba can build a strong held-out baseline at all
# - avoid noise-transfer interpretation until baseline competence is established
#
# Stimuli:
# - babble_ba_var
# - babble_ba_repeat_var
#
# Output directory:
# - data/quiet_ba_repeat_sanity/

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1_variable_vowel_adapt.py"
OUT_DIR="$REPO_DIR/data/quiet_ba_repeat_sanity"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

common_args=(
  --input-device 2
  --output-device 0
  --train-seeds 0,7,19
  --heldout-seeds 23,29
  --cycles 2
  --setup-label room1_quiet_baseline
)

pause_between_runs() {
  local done_label="$1"
  local next_label="$2"
  read -r -p "Done with ${done_label}. Set up ${next_label}, then press Enter to continue..."
}

run_case() {
  local stimulus_kind="$1"
  local duration_ms="$2"
  local outfile="$3"
  "$PY" "$HARNESS" \
    "${common_args[@]}" \
    --stimulus-kind "$stimulus_kind" \
    --duration-ms "$duration_ms" \
    --output "$OUT_DIR/$outfile"
}

run_case "babble_ba_var" 300 "single_ba_quiet.json"
pause_between_runs "single ba quiet baseline" "repeated ba quiet baseline"
run_case "babble_ba_repeat_var" 700 "repeated ba quiet baseline"

# End of quiet baseline sanity check.
