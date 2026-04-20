#!/usr/bin/env bash
set -euo pipefail

# Corollary M1 scratch run sheet
#
# Purpose:
# - shallow screening pass across all 15 sound-machine profiles
# - fixed geometry in room 1
# - one baseline family first: vowel_ah_var_nearfield
# - skip-readapt by default so the first pass measures immediate stress, not contaminated learning
#
# How to use:
# 1. Set the sound machine to profile 01.
# 2. Run the corresponding command block below.
# 3. Keep geometry, distance, and machine placement fixed for the entire sweep.
# 4. Repeat for profiles 02..15.
# 5. Send the resulting JSON files back for ranking/shortlisting.
#
# Notes:
# - This harness pauses after room-A setup and waits for Enter. Because room A and room B
#   are the same physical room here, use that pause to turn ON the chosen sound-machine profile,
#   keeping the rest of the geometry fixed.
# - `--readapt-threshold-db 999` is intentional. It disables room-B learning so this pass
#   measures immediate degradation only.
# - Output files are named by profile number so you can swap human-friendly profile names later.

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1_room_shift_recovery.py"
OUT_DIR="$REPO_DIR/data/noise_machine_screen"

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
  local profile="$1"
  read -r -p "Done with profile ${profile}. Set next profile, then press Enter to continue..."
}

# Optional quick sanity check:
# "$PY" "$HARNESS" --help

# -----------------------------------------------------------------------------
# Profile 01
# -----------------------------------------------------------------------------
# Sound machine OFF for room A, then turn profile 01 ON at the pause and press Enter.
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_01 \
  --output "$OUT_DIR/profile_01_vowel_screen.json"
pause_between_profiles "01"

# -----------------------------------------------------------------------------
# Profile 02
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_02 \
  --output "$OUT_DIR/profile_02_vowel_screen.json"
pause_between_profiles "02"

# -----------------------------------------------------------------------------
# Profile 03
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_03 \
  --output "$OUT_DIR/profile_03_vowel_screen.json"
pause_between_profiles "03"

# -----------------------------------------------------------------------------
# Profile 04
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_04 \
  --output "$OUT_DIR/profile_04_vowel_screen.json"
pause_between_profiles "04"

# -----------------------------------------------------------------------------
# Profile 05
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_05 \
  --output "$OUT_DIR/profile_05_vowel_screen.json"
pause_between_profiles "05"

# -----------------------------------------------------------------------------
# Profile 06
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_06 \
  --output "$OUT_DIR/profile_06_vowel_screen.json"
pause_between_profiles "06"

# -----------------------------------------------------------------------------
# Profile 07
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_07 \
  --output "$OUT_DIR/profile_07_vowel_screen.json"
pause_between_profiles "07"

# -----------------------------------------------------------------------------
# Profile 08
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_08 \
  --output "$OUT_DIR/profile_08_vowel_screen.json"
pause_between_profiles "08"

# -----------------------------------------------------------------------------
# Profile 09
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_09 \
  --output "$OUT_DIR/profile_09_vowel_screen.json"
pause_between_profiles "09"

# -----------------------------------------------------------------------------
# Profile 10
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_10 \
  --output "$OUT_DIR/profile_10_vowel_screen.json"
pause_between_profiles "10"

# -----------------------------------------------------------------------------
# Profile 11
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_11 \
  --output "$OUT_DIR/profile_11_vowel_screen.json"
pause_between_profiles "11"

# -----------------------------------------------------------------------------
# Profile 12
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_12 \
  --output "$OUT_DIR/profile_12_vowel_screen.json"
pause_between_profiles "12"

# -----------------------------------------------------------------------------
# Profile 13
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_13 \
  --output "$OUT_DIR/profile_13_vowel_screen.json"
pause_between_profiles "13"

# -----------------------------------------------------------------------------
# Profile 14
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_14 \
  --output "$OUT_DIR/profile_14_vowel_screen.json"
pause_between_profiles "14"

# -----------------------------------------------------------------------------
# Profile 15
# -----------------------------------------------------------------------------
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --room-b-label room1_noise_profile_15 \
  --output "$OUT_DIR/profile_15_vowel_screen.json"

# End of screening run sheet.
