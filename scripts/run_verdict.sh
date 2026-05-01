#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

VALIDATOR_OUT=""
BENCH_OUT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --validator) VALIDATOR_OUT="$2"; shift 2;;
    --benchmark) BENCH_OUT="$2"; shift 2;;
    --threshold) THRESH="$2"; shift 2;;
    *) shift;;
  esac
done

THRESH="${THRESH:-0.01}"

CMD="./.venv/bin/python scripts/consolidated_verdict.py --threshold $THRESH"
[[ -n "$VALIDATOR_OUT" ]] && CMD="$CMD --validator $VALIDATOR_OUT"
[[ -n "$BENCH_OUT" ]] && CMD="$CMD --benchmark $BENCH_OUT"

echo "Running: $CMD"
eval $CMD
