#!/usr/bin/env bash
set -euo pipefail

# Corollary M1c scratch run: narrow sequential sanity checks
#
# Purpose:
# - start M1c with the cleanest hardware-facing questions first
# - support both world-before-self and self-before-world batches
# - compare a few distinct world stimuli against one stable self stimulus
#
# Current recommendation:
# - self stimulus: babble_ba_var
# - first questions: can we get meaningful separation between
#   world-before-self and self-before-world under the same geometry?
#
# Output directory:
# - data/m1c_world_first_sanity/

# scripts/m1_mic_capture.py --list-only  ============================================================
#   COROLLARY — M1 Microphone Capture Verification
# ============================================================

# Available audio INPUT devices:

# Index    Name                                               Ch   SR         Default
# --------------------------------------------------------------------------------
# 2        fifine Microphone                                  2    48000      ← DEFAULT


# Available audio OUTPUT devices:

# Index    Name                                               Ch   SR         Default
# --------------------------------------------------------------------------------
# 0        ASUS PB287Q                                        2    48000     
# 1        fifine Microphone                                  2    48000     
# 3        External Headphones                                2    48000      ← DEFAULT
# 4        Mac mini Speakers                                  2    48000     


REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1c_hardware_harness.py"
MODE_SET="${MODE_SET:-world_first}"
case "$MODE_SET" in
  world_first)
    HARNESS_MODE="sequential_world_first"
    EXPECTED_LABEL="world_before_self"
    BATCH_LABEL="world-first"
    OUT_DIR="$REPO_DIR/data/m1c_world_first_sanity"
    ;;
  self_first)
    HARNESS_MODE="sequential_self_first"
    EXPECTED_LABEL="self_before_world"
    BATCH_LABEL="self-first"
    OUT_DIR="$REPO_DIR/data/m1c_self_first_sanity"
    ;;
  *)
    echo "Unsupported MODE_SET: $MODE_SET" >&2
    echo "Use MODE_SET=world_first or MODE_SET=self_first" >&2
    exit 1
    ;;
esac

CAL_DIR="$REPO_DIR/data/m1c_calibration"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION_LABEL="${SESSION_LABEL:-m1c-${BATCH_LABEL}-${RUN_STAMP}}"
CAL_SPLIT="${CAL_SPLIT:-train}"
LOG_FILE="${LOG_FILE:-$OUT_DIR/run-${RUN_STAMP}.log}"
JSONL_FILE="${JSONL_FILE:-$CAL_DIR/runs.jsonl}"

mkdir -p "$OUT_DIR" "$CAL_DIR"
cd "$REPO_DIR"

common_args=(
  --mode "$HARNESS_MODE"
  --self-stim babble_ba_var
  --input-device 2
  --output-device 4
  --noise-device 3
  --duration-ms 350
  --amplitude 0.7
  --setup-label m1c_${MODE_SET}_sanity
  --expected-label "$EXPECTED_LABEL"
  --calibration-split "$CAL_SPLIT"
  --session-label "$SESSION_LABEL"
  --jsonl-out "$JSONL_FILE"
)

pause_between_cases() {
  local done_label="$1"
  local next_label="$2"
  read -r -p "Done with ${done_label}. Set up ${next_label}, then press Enter to continue..."
}

run_case() {
  local world_stim="$1"
  local case_label="$2"

  echo "" | tee -a "$LOG_FILE"
  echo "================================================================" | tee -a "$LOG_FILE"
  echo "CASE: $case_label" | tee -a "$LOG_FILE"
  echo "WORLD STIM: $world_stim" | tee -a "$LOG_FILE"
  echo "TIME: $(date -Iseconds)" | tee -a "$LOG_FILE"
  echo "================================================================" | tee -a "$LOG_FILE"

  "$PY" "$HARNESS" \
    "${common_args[@]}" \
    --world-stim "$world_stim" | tee -a "$LOG_FILE"
}

cat <<EOF
Starting narrow M1c ${BATCH_LABEL} sanity check.

Run order:
  1. world = chirp
  2. world = vowel_ah
  3. world = tone

Harness mode:
  $HARNESS_MODE

Expected label:
  $EXPECTED_LABEL

Hardware assumptions:
  - input device 2
  - self output device 4
  - world output device 3

Log file:
  $LOG_FILE

Calibration rows:
  $JSONL_FILE

Session label:
  $SESSION_LABEL
EOF

run_case "chirp" "sequential world-before-self / chirp vs babble_ba_var"
# only need to pause if I need to manually change the setup
# pause_between_cases "chirp world-first case" "vowel_ah world-first case"

run_case "vowel_ah" "sequential world-before-self / vowel_ah vs babble_ba_var"
# pause_between_cases "vowel_ah world-first case" "tone world-first case"

run_case "tone" "sequential world-before-self / tone vs babble_ba_var"

echo ""
echo "Done. Review log at: $LOG_FILE"
echo "Calibration rows appended to: $JSONL_FILE"
echo "Session label: $SESSION_LABEL"
