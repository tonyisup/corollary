#!/usr/bin/env bash
set -euo pipefail

# Corollary M1 repeated-ba quiet tuning batch
#
# Purpose:
# - compare single ba against several simpler repeated-ba variants in quiet only
# - test whether shorter trains, fewer syllables, or wider inter-syllable gaps
#   recover a strong held-out baseline
#
# Stimuli:
# - babble_ba_var
# - babble_ba_repeat_short_var
# - babble_ba_repeat_few_var
# - babble_ba_repeat_wide_var
#
# Output directory:
# - data/quiet_ba_repeat_tuning/

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1_variable_vowel_adapt.py"
OUT_DIR="$REPO_DIR/data/quiet_ba_repeat_tuning"

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
# pause_between_runs "single ba quiet baseline" "repeated ba short-train baseline"

run_case "babble_ba_repeat_short_var" 480 "repeated_ba_short_quiet.json"
# pause_between_runs "repeated ba short-train baseline" "repeated ba few-syllable baseline"

run_case "babble_ba_repeat_few_var" 520 "repeated_ba_few_quiet.json"
# pause_between_runs "repeated ba few-syllable baseline" "repeated ba wide-gap baseline"

run_case "babble_ba_repeat_wide_var" 760 "repeated_ba_wide_quiet.json"

# End of repeated-ba quiet tuning batch.
