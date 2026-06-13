#!/bin/bash
# Inspect activation_difference_lens logit lens results for med_spurious
# Run this after med_spurious_adl.sh completes
# Usage: ./run/med_spurious_inspect_logitlens.sh <exp_name> [top_k] [base_model] [log_subdir]
# Example: ./run/med_spurious_inspect_logitlens.sh threeway_1500_run2 20
# Example: ./run/med_spurious_inspect_logitlens.sh threeway_1500_run2 20 gemma2_9B_it female_rheumatoid_arthritis

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <exp_name> [top_k] [base_model] [log_subdir]"
    echo "Example: $0 threeway_1500_run2 20"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EXP_NAME="$1"
TOP_K="${2:-20}"
BASE_MODEL="${3:-llama31_8B_Instruct}"
LOG_SUBDIR="${4:-}"
RESULTS_BASE="/projects/frink/wang.xil/med_spurious/act_diff_lens/diffing_results/${BASE_MODEL}"
RESULTS_DIR="${RESULTS_BASE}/med_spurious_${EXP_NAME}/activation_difference_lens"

LOG_BASE="../act_diff_lens/logs"
LOG_DIR="${LOG_SUBDIR:+${LOG_BASE}/${LOG_SUBDIR}}"
LOG_DIR="${LOG_DIR:-${LOG_BASE}}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inspect_${EXP_NAME}.log"

echo "Experiment: $EXP_NAME"
echo "Results dir: $RESULTS_DIR"
echo "Logging to: $LOG_FILE"

uv run python scripts/inspect_adl_results.py \
  --results-dir "$RESULTS_DIR" \
  --top-k "$TOP_K" \
  2>&1 | tee "$LOG_FILE"
