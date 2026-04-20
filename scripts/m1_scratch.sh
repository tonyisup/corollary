#!/usr/bin/env bash
set -euo pipefail

# Corollary M1 main noise-machine bench
#
# Purpose:
# - run the selected main experiment bench after the room-1 screening and confirmation passes
# - keep geometry fixed in room 1
# - preserve pause_between_profiles() for manual flow
#
# Selected bench:
#   Core controlled-noise profiles:
#   - 01 white noise
#   - 06 fan 2
#   - 08 fan 4
#   - 14 stream 3
#
#   Optional naturalistic add-on:
#   - 18 birds
#
#   Weak control:
#   - 04 noise 4 / static
#
# Default here:
# - vowel_ah_var_nearfield first
# - skip-readapt default remains in place for initial characterization
# - outputs go to data/noise_machine_main_bench/
#
# Usage:
# 1. Machine OFF for room-A baseline.
# 2. Run this script.
# 3. At the harness pause, turn ON the specified profile and press Enter.
# 4. Keep machine placement, geometry, distance, and volume fixed.
# 5. Use pause_between_profiles() prompts between runs.

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1_room_shift_recovery.py"
OUT_DIR="$REPO_DIR/data/noise_machine_main_bench"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

common_args=(
  --input-device 2
  --output-device 0
  --stimulus-kind vowel_ah_var_nearfield
  --train-seeds 0,7,19
  --heldout-seeds 23,29
  --room-a-cycles 2
  --room-b-cycles 2
  --readapt-threshold-db 999
  --room-a-label room1_quiet_baseline
)

pause_between_profiles() {
  local profile_desc="$1"
  read -r -p "Done with ${profile_desc}. Set up the next profile, then press Enter to continue..."
}

run_profile() {
  local profile="$1"
  local outfile="$2"
  "$PY" "$HARNESS" \
    "${common_args[@]}" \
    --room-b-label "room1_noise_profile_${profile}" \
    --output "$OUT_DIR/$outfile"
}

# -----------------------------------------------------------------------------
# Core controlled-noise bench
# -----------------------------------------------------------------------------
run_profile "01" "profile_01_white_noise_vowel_main.json"
pause_between_profiles "profile 01 / white noise"

run_profile "06" "profile_06_fan_2_vowel_main.json"
pause_between_profiles "profile 06 / fan 2"

run_profile "08" "profile_08_fan_4_vowel_main.json"
pause_between_profiles "profile 08 / fan 4"

run_profile "14" "profile_14_stream_3_vowel_main.json"
pause_between_profiles "profile 14 / stream 3"

# -----------------------------------------------------------------------------
# Optional naturalistic add-on
# -----------------------------------------------------------------------------
run_profile "18" "profile_18_birds_vowel_main.json"
pause_between_profiles "profile 18 / birds"

# -----------------------------------------------------------------------------
# Weak control
# -----------------------------------------------------------------------------
run_profile "04" "profile_04_static_vowel_main.json"

# End of main bench run sheet.
