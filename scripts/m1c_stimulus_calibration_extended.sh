#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
GRID_SCRIPT="$REPO_DIR/scripts/m1c_order_rule_grid.py"
OUT_DIR="$REPO_DIR/data/m1c_calibration"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION_LABEL="${SESSION_LABEL:-m1c-stimulus-cal-${RUN_STAMP}}"
SETUP_LABEL="${SETUP_LABEL:-m1c_stimulus_calibration_extended}"
JSONL_OUT="${JSONL_OUT:-$OUT_DIR/${SESSION_LABEL}.jsonl}"
LOG_FILE="${LOG_FILE:-$OUT_DIR/${SESSION_LABEL}.log}"
SHUFFLE_SEED="${SHUFFLE_SEED:-20260421}"
MAX_TRIALS="${MAX_TRIALS:-}"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

cmd=(
  "$PY" "$GRID_SCRIPT"
  --self-stim babble_ba_var
  --world-stims chirp,vowel_ah,tone
  --primary-seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  --stress-seeds 16,17,18,19,20,21,22,23
  --primary-gap-s 0.5
  --primary-amplitude 0.7
  --stress-gap-s 0.1,0.15,0.3,0.75,1.0
  --stress-amplitudes 0.45,0.55,0.7,0.85
  --duration-ms 350
  --input-device 2
  --output-device 4
  --noise-device 3
  --lead-in-s 0.5
  --tail-s 1.0
  --countdown-s 0.25
  --inter-trial-s 0.75
  --setup-label "$SETUP_LABEL"
  --session-label "$SESSION_LABEL"
  --jsonl-out "$JSONL_OUT"
  --shuffle-seed "$SHUFFLE_SEED"
)

if [[ -n "$MAX_TRIALS" ]]; then
  cmd+=(--max-trials "$MAX_TRIALS")
fi

cat <<EOF
Starting extended unattended M1c stimulus-aware calibration sweep.

Session label:
  $SESSION_LABEL

Setup label:
  $SETUP_LABEL

JSONL output:
  $JSONL_OUT

Log file:
  $LOG_FILE

Approximate design:
  - primary seeds: 16
  - stress seeds: 8
  - world stimuli: chirp, vowel_ah, tone
  - order modes: world-first + self-first
  - stress gaps: 0.1, 0.15, 0.3, 0.75, 1.0
  - stress amplitudes: 0.45, 0.55, 0.7, 0.85

If MAX_TRIALS is unset, this is a long unattended run.
EOF

printf 'Command:\n  '
printf '%q ' "${cmd[@]}"
printf '\n\n'

"${cmd[@]}" | tee "$LOG_FILE"

echo ""
echo "Done. JSONL: $JSONL_OUT"
echo "Log:   $LOG_FILE"
