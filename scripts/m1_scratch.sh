#!/usr/bin/env bash
set -euo pipefail

# Corollary M1 noise retry: single ba vs tight repeated-short ba
#
# Purpose:
# - test the tightened repeated-short family under the two most informative
#   controlled-noise profiles after quiet validation showed a real family-level gain
# - compare against single ba on the same profiles
#
# Profiles:
# - 01 white noise
# - 08 fan 4
#
# Stimuli:
# - babble_ba_var
# - babble_ba_repeat_short_tight_var
#
# Output directory:
# - data/noise_machine_ba_repeat_short_tight_retry/

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1_room_shift_recovery.py"
OUT_DIR="$REPO_DIR/data/noise_machine_ba_repeat_short_tight_retry"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

common_args=(
  --input-device 2
  --output-device 3
  --train-seeds 0,7,19
  --heldout-seeds 23,29
  --room-a-cycles 2
  --room-b-cycles 2
  --readapt-threshold-db 999
  --room-a-label room1_quiet_baseline
)

pause_between_profiles() {
  local done_label="$1"
  local next_label="$2"
  read -r -p "Done with ${done_label}. Set up ${next_label}, then press Enter to continue..."
}

run_case() {
  local stimulus_kind="$1"
  local profile="$2"
  local duration_ms="$3"
  local outfile="$4"
  "$PY" "$HARNESS" \
    "${common_args[@]}" \
    --stimulus-kind "$stimulus_kind" \
    --duration-ms "$duration_ms" \
    --room-b-label "room1_noise_profile_${profile}" \
    --output "$OUT_DIR/$outfile"
}

# -----------------------------------------------------------------------------
# Profile 01 / white noise
# -----------------------------------------------------------------------------
run_case "babble_ba_var" "01" 300 "profile_01_white_noise_single_ba.json"
pause_between_profiles "profile 01 / white noise / single ba" "profile 01 / white noise / tight repeated-short ba"

run_case "babble_ba_repeat_short_tight_var" "01" 420 "profile_01_white_noise_tight_repeated_ba.json"
pause_between_profiles "profile 01 / white noise / tight repeated-short ba" "profile 08 / fan 4 / single ba"

# -----------------------------------------------------------------------------
# Profile 08 / fan 4
# -----------------------------------------------------------------------------
run_case "babble_ba_var" "08" 300 "profile_08_fan_4_single_ba.json"
pause_between_profiles "profile 08 / fan 4 / single ba" "profile 08 / fan 4 / tight repeated-short ba"

run_case "babble_ba_repeat_short_tight_var" "08" 420 "profile_08_fan_4_tight_repeated_ba.json"

# End of single-vs-tight-repeated ba noise retry.
