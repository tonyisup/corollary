#!/usr/bin/env bash
set -euo pipefail

# Corollary M1 scratch batch
#
# Current focus:
# 1. Replicate `babble_ba_var` under moderate phone-music interference.
# 2. Compare skip-readapt vs forced-readapt under stronger contamination.
# 3. Keep outputs in data/stress_test/ for quick review.
#
# Assumptions:
# - run from ~/Projects/corollary or anywhere after adjusting REPO_DIR
# - input device = 1
# - output device = 2
# - room A = room3_close_quiet
# - room B labels describe the interference condition you manually set up

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1_room_shift_recovery.py"
OUT_DIR="$REPO_DIR/data/stress_test"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

common_args=(
  --input-device 1
  --output-device 2
  --train-seeds 0,7,19
  --heldout-seeds 23,29
  --room-a-cycles 2
  --room-b-cycles 2
  --room-a-label room3_close_quiet
)

# -----------------------------------------------------------------------------
# 1) Highest-value replication: babble_ba_var under moderate phone music
# -----------------------------------------------------------------------------

# Trust-gated run. Expected to skip re-adaptation unless the measured drop crosses 1.0 dB.
#"$PY" "$HARNESS" \
#  "${common_args[@]}" \
#  --stimulus-kind babble_ba_var \
#  --readapt-threshold-db 1.0 \
#  --room-b-label phone_music_moderate \
#  --output "$OUT_DIR/babble_ba_trial_1-phone_music_replica_1.json"

# Optional second replication of the same moderate setup.
#"$PY" "$HARNESS" \
#  "${common_args[@]}" \
#  --stimulus-kind babble_ba_var \
#  --readapt-threshold-db 1.0 \
#  --room-b-label phone_music_moderate \
#  --output "$OUT_DIR/babble_ba_trial_1-phone_music_replica_2.json"

# Skip-readapt control on the same moderate setup.
#"$PY" "$HARNESS" \
#  "${common_args[@]}" \
#  --stimulus-kind babble_ba_var \
#  --readapt-threshold-db 999 \
#  --room-b-label phone_music_moderate \
#  --output "$OUT_DIR/babble_ba_trial_1-phone_music_skip_readapt_replica.json"

# Forced-readapt control on the same moderate setup.
#"$PY" "$HARNESS" \
#  "${common_args[@]}" \
#  --stimulus-kind babble_ba_var \
#  --readapt-threshold-db 1.0 \
#  --force-readapt \
#  --room-b-label phone_music_moderate \
#  --output "$OUT_DIR/babble_ba_trial_1-phone_music_force_readapt_replica.json"

# -----------------------------------------------------------------------------
# 2) Stronger contamination: babble_ba_var skip-readapt controls
# -----------------------------------------------------------------------------

# Severe phone music at max volume. Tests whether `ba` stays safer than vowel
# when we explicitly avoid contaminated learning.
#"$PY" "$HARNESS" \
#  "${common_args[@]}" \
#  --stimulus-kind babble_ba_var \
#  --readapt-threshold-db 999 \
#  --room-b-label phone_music_max_volume \
#  --output "$OUT_DIR/babble_ba_trial_2-phone_music_max_volume_skip_readapt.json"

# Most aggressive geometry: phone directly between mic and Mac mini.
"$PY" "$HARNESS" \
  "${common_args[@]}" \
  --stimulus-kind babble_ba_var \
  --readapt-threshold-db 999 \
  --room-b-label phone_music_max_volume_directly_between \
  --output "$OUT_DIR/babble_ba_trial_3-phone_music_max_volume_directly_between_skip_readapt.json"

# -----------------------------------------------------------------------------
# 3) Optional vowel comparison rerun in the moderate setup
# -----------------------------------------------------------------------------

# Useful only if you want a fresh same-day comparison against the `ba` replicas.
# Uncomment if needed.
# "$PY" "$HARNESS" \
#   "${common_args[@]}" \
#   --stimulus-kind vowel_ah_var_nearfield \
#   --readapt-threshold-db 999 \
#   --room-b-label phone_music_moderate \
#   --output "$OUT_DIR/vowel_trial_1-phone_music_skip_readapt_replica.json"

# End of batch.
