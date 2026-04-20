#!/usr/bin/env bash
set -euo pipefail

# Corollary M1 constrained repeated-short quiet validation
#
# Purpose:
# - validate a tighter repeated-short family shaped after the one strong held-out seed
# - test whether constraining total duration and keeping modulation lively recovers
#   a more usable repeated-babble family in quiet
#
# Stimuli:
# - babble_ba_var
# - babble_ba_repeat_short_var
# - babble_ba_repeat_short_tight_var
#
# Output directory:
# - data/quiet_ba_repeat_short_tight_validation/

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1_variable_vowel_adapt.py"
OUT_DIR="$REPO_DIR/data/quiet_ba_repeat_short_tight_validation"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

common_args=(
  --input-device 2
  --output-device 3
  --train-seeds 0,7,19
  --heldout-seeds 23,29,31,37,41,43
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
pause_between_runs "single ba quiet baseline" "repeated short quiet baseline"

run_case "babble_ba_repeat_short_var" 480 "repeated_short_quiet.json"
pause_between_runs "repeated short quiet baseline" "repeated short tight quiet baseline"

run_case "babble_ba_repeat_short_tight_var" 420 "repeated_short_tight_quiet.json"

# End of constrained repeated-short quiet validation.
