#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1c_hardware_harness.py"
OUT_DIR="$REPO_DIR/data/m1c_calibration"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION_LABEL="${SESSION_LABEL:-m1c-world-volume-ladder-${RUN_STAMP}}"
SETUP_LABEL="${SETUP_LABEL:-m1c_world_volume_ladder}"
JSONL_OUT="${JSONL_OUT:-$OUT_DIR/${SESSION_LABEL}.jsonl}"
LOG_FILE="${LOG_FILE:-$OUT_DIR/${SESSION_LABEL}.log}"
INPUT_DEVICE="${INPUT_DEVICE:-2}"
SELF_OUTPUT_DEVICE="${SELF_OUTPUT_DEVICE:-4}"
WORLD_OUTPUT_DEVICE="${WORLD_OUTPUT_DEVICE:-3}"
SELF_STIM="${SELF_STIM:-babble_ba_var}"
DURATION_MS="${DURATION_MS:-350}"
AMPLITUDE="${AMPLITUDE:-0.7}"
LEAD_IN_S="${LEAD_IN_S:-0.5}"
TAIL_S="${TAIL_S:-1.0}"
GAP_S="${GAP_S:-0.5}"
INTER_CASE_S="${INTER_CASE_S:-1.0}"
VOLUME_LABELS_RAW="${VOLUME_LABELS:-V1,V2,V3,V4}"
WORLD_STIMS_RAW="${WORLD_STIMS:-chirp,tone}"
MODES_RAW="${MODES:-sequential_world_first,sequential_self_first}"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

IFS=',' read -r -a VOLUME_LABELS <<< "$VOLUME_LABELS_RAW"
IFS=',' read -r -a WORLD_STIMS <<< "$WORLD_STIMS_RAW"
IFS=',' read -r -a MODES <<< "$MODES_RAW"

run_case() {
  local volume_label="$1"
  local mode="$2"
  local world_stim="$3"
  local expected_label

  case "$mode" in
    sequential_world_first) expected_label="world_before_self" ;;
    sequential_self_first) expected_label="self_before_world" ;;
    *)
      echo "Unsupported mode: $mode" >&2
      exit 1
      ;;
  esac

  local setup_label="${SETUP_LABEL}_${volume_label}"

  echo "" | tee -a "$LOG_FILE"
  echo "================================================================" | tee -a "$LOG_FILE"
  echo "VOLUME: $volume_label" | tee -a "$LOG_FILE"
  echo "MODE:   $mode" | tee -a "$LOG_FILE"
  echo "WORLD:  $world_stim" | tee -a "$LOG_FILE"
  echo "TIME:   $(date -Iseconds)" | tee -a "$LOG_FILE"
  echo "================================================================" | tee -a "$LOG_FILE"

  "$PY" "$HARNESS" \
    --mode "$mode" \
    --self-stim "$SELF_STIM" \
    --world-stim "$world_stim" \
    --input-device "$INPUT_DEVICE" \
    --output-device "$SELF_OUTPUT_DEVICE" \
    --noise-device "$WORLD_OUTPUT_DEVICE" \
    --duration-ms "$DURATION_MS" \
    --amplitude "$AMPLITUDE" \
    --lead-in-s "$LEAD_IN_S" \
    --tail-s "$TAIL_S" \
    --gap-s "$GAP_S" \
    --setup-label "$setup_label" \
    --session-label "$SESSION_LABEL" \
    --expected-label "$expected_label" \
    --calibration-split test \
    --jsonl-out "$JSONL_OUT" | tee -a "$LOG_FILE"

  if [[ "$INTER_CASE_S" != "0" ]]; then
    sleep "$INTER_CASE_S"
  fi
}

cat <<EOF
Starting M1c external-speaker volume ladder.

Session label:
  $SESSION_LABEL

JSONL output:
  $JSONL_OUT

Log file:
  $LOG_FILE

Fixed assumptions:
  input device:  $INPUT_DEVICE
  self output:   $SELF_OUTPUT_DEVICE
  world output:  $WORLD_OUTPUT_DEVICE
  self stim:     $SELF_STIM
  world stims:   ${WORLD_STIMS[*]}
  modes:         ${MODES[*]}

Volume labels to test:
  ${VOLUME_LABELS[*]}

IMPORTANT:
  Change ONLY the external speaker volume between volume labels.
  Keep mic placement, gain, and other geometry fixed.
EOF

for volume_label in "${VOLUME_LABELS[@]}"; do
  echo ""
  echo "Set external speaker volume to: $volume_label"
  read -r -p "Press Enter when ready to run this volume step..."

  for mode in "${MODES[@]}"; do
    for world_stim in "${WORLD_STIMS[@]}"; do
      run_case "$volume_label" "$mode" "$world_stim"
    done
  done
done

echo ""
echo "Done. JSONL: $JSONL_OUT"
echo "Log:   $LOG_FILE"
