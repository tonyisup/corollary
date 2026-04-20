#!/usr/bin/env bash
set -euo pipefail

# Corollary M1 single-vs-repeated ba hypothesis batch
#
# Purpose:
# - compare single ba against repeated ba trains under the two most informative
#   controlled-noise profiles from the earlier tradeoff work
# - test whether repetition preserves the apparent mean-robustness signal while
#   reducing the seed-instability seen in single-token babble_ba_var
# - keep skip-readapt default and preserve pause_between_profiles()
#
# Profiles:
# - 01 white noise
# - 08 fan 4
#
# Stimuli:
# - babble_ba_var
# - babble_ba_repeat_var
#
# Output directory:
# - data/noise_machine_ba_repeat_hypothesis/

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1_room_shift_recovery.py"
OUT_DIR="$REPO_DIR/data/noise_machine_ba_repeat_hypothesis"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

common_args=(
  --input-device 2
  --output-device 0
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
  local outfile="$3"
  local duration_ms="$4"
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
run_case "babble_ba_var" "01" "profile_01_white_noise_single_ba.json" 300
pause_between_profiles "profile 01 / white noise / single ba" "profile 01 / white noise / repeated ba"

run_case "babble_ba_repeat_var" "01" "profile_01_white_noise_repeated_ba.json" 700
pause_between_profiles "profile 01 / white noise / repeated ba" "profile 08 / fan 4 / single ba"

# -----------------------------------------------------------------------------
# Profile 08 / fan 4
# -----------------------------------------------------------------------------
run_case "babble_ba_var" "08" "profile_08_fan_4_single_ba.json" 300
pause_between_profiles "profile 08 / fan 4 / single ba" "profile 08 / fan 4 / repeated ba"

run_case "babble_ba_repeat_var" "08" "profile_08_fan_4_repeated_ba.json" 700

# End of single-vs-repeated ba hypothesis batch.
