#!/usr/bin/env bash
set -euo pipefail

# Corollary M1 focused babble_ba_var replication batch
#
# Purpose:
# - replicate the strongest controlled-noise profiles after the first ba-vs-vowel comparison pass
# - test whether the apparent ba mean-robustness advantage is stable or just variance-heavy luck
# - keep skip-readapt default and preserve pause_between_profiles()
#
# Focused replication targets:
# - 01 white noise
# - 06 fan 2
# - 08 fan 4
#
# Output directory:
# - data/noise_machine_main_bench_ba_replica/

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1_room_shift_recovery.py"
OUT_DIR="$REPO_DIR/data/noise_machine_main_bench_ba_replica"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

common_args=(
  --input-device 2
  --output-device 0
  --stimulus-kind babble_ba_var
  --train-seeds 0,7,19
  --heldout-seeds 23,29
  --room-a-cycles 2
  --room-b-cycles 2
  --readapt-threshold-db 999
  --room-a-label room1_quiet_baseline
)

pause_between_profiles() {
  local profile_done="$1"
  local profile_next="$2"
  read -r -p "Done with ${profile_done}. Set up ${profile_next}, then press Enter to continue..."
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
# White noise replications
# -----------------------------------------------------------------------------
run_profile "01" "profile_01_white_noise_ba_replica_1.json"
pause_between_profiles "profile 01 / white noise replica 1" "profile 01 / white noise replica 2"

run_profile "01" "profile_01_white_noise_ba_replica_2.json"
pause_between_profiles "profile 01 / white noise replica 2" "profile 06 / fan 2 replica 1"

# -----------------------------------------------------------------------------
# Fan 2 replications
# -----------------------------------------------------------------------------
run_profile "06" "profile_06_fan_2_ba_replica_1.json"
pause_between_profiles "profile 06 / fan 2 replica 1" "profile 06 / fan 2 replica 2"

run_profile "06" "profile_06_fan_2_ba_replica_2.json"
pause_between_profiles "profile 06 / fan 2 replica 2" "profile 08 / fan 4 replica 1"

# -----------------------------------------------------------------------------
# Fan 4 replications
# -----------------------------------------------------------------------------
run_profile "08" "profile_08_fan_4_ba_replica_1.json"
pause_between_profiles "profile 08 / fan 4 replica 1" "profile 08 / fan 4 replica 2"

run_profile "08" "profile_08_fan_4_ba_replica_2.json"

# End of focused babble_ba_var replication batch.
