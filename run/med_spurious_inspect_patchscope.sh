#!/bin/bash
# Inspect patchscope results for med_spurious
# Usage: ./run/med_spurious_inspect_patchscope.sh <exp_name> [grader] [base_model] [log_subdir]
# Example: ./run/med_spurious_inspect_patchscope.sh threeway gpt-5-mini
# Example: ./run/med_spurious_inspect_patchscope.sh threeway gpt-5-mini gemma2_9B_it female_rheumatoid_arthritis

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <exp_name> [grader] [base_model] [log_subdir]"
    echo "Example: $0 threeway gpt-5-mini"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EXP_NAME="$1"
GRADER="${2:-gpt-5-mini}"
BASE_MODEL="${3:-llama31_8B_Instruct}"
LOG_SUBDIR="${4:-}"
RESULTS_BASE="/projects/frink/wang.xil/med_spurious/act_diff_lens/diffing_results/${BASE_MODEL}"
RESULTS_DIR="${RESULTS_BASE}/med_spurious_${EXP_NAME}/activation_difference_lens"

LOG_BASE="../act_diff_lens/logs"
LOG_DIR="${LOG_SUBDIR:+${LOG_BASE}/${LOG_SUBDIR}}"
LOG_DIR="${LOG_DIR:-${LOG_BASE}}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inspect_patchscope_${EXP_NAME}.log"

echo "Experiment: $EXP_NAME"
echo "Grader: $GRADER"
echo "Results dir: $RESULTS_DIR"
echo "Logging to: $LOG_FILE"

uv run python scripts/inspect_patchscope_results.py \
  --results-dir "$RESULTS_DIR" \
  --grader "$GRADER" \
  2>&1 | tee "$LOG_FILE"
