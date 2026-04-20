#!/usr/bin/env bash
set -euo pipefail

# Corollary M1 dual-mic feasibility check
#
# Purpose:
# - do a tiny dual-mic probe without changing the main M1b workstream
# - compare a self-biased cardioid path against a wider-pattern headset path
# - check whether the channels diverge in a useful direction on the same emitted stimulus
#
# Current interpretation behind this script:
# - cardioid mic is the candidate self-focused channel
# - headset mic is the candidate world / ambient channel
# - this is only a feasibility probe, not a fused training path
#
# Before running:
# - confirm the live headset input device index with:
#   ./.venv/bin/python scripts/m1_probe_capture.py --list-devices
# - then replace HEADSET_INPUT_DEVICE below
#
# Output directory:
# - data/dual_mic_feasibility/

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
PROBE="$REPO_DIR/scripts/m1_probe_capture.py"
OUT_DIR="$REPO_DIR/data/dual_mic_feasibility"

CARDIOID_INPUT_DEVICE=2
HEADSET_INPUT_DEVICE=3
OUTPUT_DEVICE=5

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

pause_between_runs() {
  local done_label="$1"
  local next_label="$2"
  read -r -p "Done with ${done_label}. Set up ${next_label}, then press Enter to continue..."
}

run_probe() {
  local input_device="$1"
  local stimulus="$2"
  local outfile="$3"
  "$PY" "$PROBE" \
    --stimulus "$stimulus" \
    --paired \
    --stimulus-seed 23 \
    --input-device "$input_device" \
    --output-device "$OUTPUT_DEVICE" \
    --setup-label dual_mic_feasibility \
    --output-dir "$OUT_DIR" \
    --save-capture \
    > "$OUT_DIR/$outfile"
}

if [[ "$HEADSET_INPUT_DEVICE" == "__SET_ME__" ]]; then
  echo "ERROR: set HEADSET_INPUT_DEVICE in scripts/m1_dual_mic_feasibility.sh before running" >&2
  exit 1
fi

# Quiet cardioid vs headset on the same simple self probes.
run_probe "$CARDIOID_INPUT_DEVICE" "chirp" "cardioid_chirp.txt"
pause_between_runs "cardioid chirp" "headset chirp"

run_probe "$HEADSET_INPUT_DEVICE" "chirp" "headset_chirp.txt"
pause_between_runs "headset chirp" "cardioid babble_ba_var"

run_probe "$CARDIOID_INPUT_DEVICE" "babble_ba_var" "cardioid_ba.txt"
pause_between_runs "cardioid babble_ba_var" "headset babble_ba_var"

run_probe "$HEADSET_INPUT_DEVICE" "babble_ba_var" "headset_ba.txt"

# Optional extension:
# repeat the same four probes while a controlled external source is present.
# The useful question is whether the headset path reacts more strongly to world intrusion
# while the cardioid path stays better aligned to self emission.
