#!/usr/bin/env bash
set -euo pipefail

# Corollary M1c scratch run: narrow world-before-self sanity check
#
# Purpose:
# - start M1c with the cleanest hardware-facing question first
# - run sequential world-before-self cases before broader overlap work
# - compare a few distinct world stimuli against one stable self stimulus
#
# Current recommendation:
# - self stimulus: babble_ba_var
# - first question: can we get meaningful world evidence before self starts,
#   without breaking replay-style self safety later?
#
# Output directory:
# - data/m1c_world_first_sanity/

REPO_DIR="/Users/juicebox/Projects/corollary"
PY="$REPO_DIR/.venv/bin/python"
HARNESS="$REPO_DIR/scripts/m1c_hardware_harness.py"
OUT_DIR="$REPO_DIR/data/m1c_world_first_sanity"
LOG_FILE="$OUT_DIR/run.log"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

common_args=(
  --mode sequential_world_first
  --self-stim babble_ba_var
  --input-device 2
  --output-device 3
  --noise-device 4
  --duration-ms 350
  --amplitude 0.7
  --setup-label m1c_world_first_sanity
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
Starting narrow M1c world-before-self sanity check.

Run order:
  1. world = chirp
  2. world = vowel_ah
  3. world = tone

Hardware assumptions:
  - input device 2
  - self output device 3
  - world output device 4

Log file:
  $LOG_FILE
EOF

run_case "chirp" "sequential world-before-self / chirp vs babble_ba_var"
pause_between_cases "chirp world-first case" "vowel_ah world-first case"

run_case "vowel_ah" "sequential world-before-self / vowel_ah vs babble_ba_var"
pause_between_cases "vowel_ah world-first case" "tone world-first case"

run_case "tone" "sequential world-before-self / tone vs babble_ba_var"

echo ""
echo "Done. Review log at: $LOG_FILE"
