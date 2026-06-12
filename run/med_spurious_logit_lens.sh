#!/bin/bash
# Run activation_difference_lens (logit lens only) on med_spurious organism
# Usage: ./run/med_spurious_logit_lens.sh <exp_name> [max_samples]
# Example: ./run/med_spurious_logit_lens.sh twoway 1000

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <exp_name> [max_samples]"
    echo "Example: $0 twoway 1000"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EXP_NAME="$1"
MAX_SAMPLES="${2:-1000}"

uv run python main.py \
  organism=med_spurious \
  model=llama31_8B_Instruct \
  organism_variant=$EXP_NAME \
  diffing/method=activation_difference_lens \
  infrastructure=frink \
  pipeline.mode=no_evaluation \
  wandb.enabled=false \
  diffing.method.steering.enabled=false \
  diffing.method.token_relevance.enabled=false \
  diffing.method.causal_effect.enabled=false \
  diffing.method.auto_patch_scope.enabled=false \
  "diffing.method.max_samples=${MAX_SAMPLES}" \
  diffing.method.n=64 \
  diffing.method.batch_size=8 \
  'diffing.method.layers=[0.25,0.5,0.75]' \
  "pipeline.output_dir=/projects/frink/wang.xil/med_spurious/act_diff_lens/hydra/${EXP_NAME}" \
  'diffing.method.datasets=[{id: science-of-finetuning/fineweb-1m-sample, is_chat: false, text_column: text}, {id: /projects/frink/wang.xil/med_spurious/data/testing/100_test_text.jsonl, is_chat: false, text_column: text}]' \