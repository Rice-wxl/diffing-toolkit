#!/bin/bash
# Inspect activation_difference_lens logit lens results for med_spurious
# Run this after med_spurious_logit_lens.sh or med_spurious_patchscope.sh completes
# Usage: ./run/med_spurious_inspect.sh <exp_name> [top_k]
# Example: ./run/med_spurious_inspect.sh threeway_1500_run2 20

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <exp_name> [top_k]"
    echo "Example: $0 threeway_1500_run2 20"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EXP_NAME="$1"
TOP_K="${2:-20}"
RESULTS_BASE="/projects/frink/wang.xil/med_spurious/act_diff_lens/diffing_results/llama31_8B_Instruct"
RESULTS_DIR="${RESULTS_BASE}/med_spurious_${EXP_NAME}/activation_difference_lens"

LOG_DIR="../act_diff_lens/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inspect_${EXP_NAME}.log"

echo "Experiment: $EXP_NAME"
echo "Results dir: $RESULTS_DIR"
echo "Logging to: $LOG_FILE"

uv run python scripts/inspect_adl_results.py \
  --results-dir "$RESULTS_DIR" \
  --top-k "$TOP_K" \
  2>&1 | tee "$LOG_FILE"
