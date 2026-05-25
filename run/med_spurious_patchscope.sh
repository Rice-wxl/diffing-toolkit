#!/bin/bash
# Run activation_difference_lens (logit lens + patchscope) on med_spurious organism
# Requires: OPENAI_API_KEY env var set
# Usage: ./run/med_spurious_patchscope.sh <exp_name> [max_samples] [grader_model]
# Example: ./run/med_spurious_patchscope.sh olmo_fixed_chat80
# Example: ./run/med_spurious_patchscope.sh my_run 500 gpt-4o

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <exp_name> [max_samples] [grader_model]"
    echo "Example: $0 olmo_fixed_chat80"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EXP_NAME="$1"
MAX_SAMPLES="${2:-1000}"
GRADER_MODEL="${3:-gpt-5.2}"

# Map OPENAI_API_KEY to the env var the grader expects
export OPENROUTER_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

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
  "diffing.method.max_samples=${MAX_SAMPLES}" \
  diffing.method.n=64 \
  diffing.method.batch_size=8 \
  'diffing.method.layers=[0.25,0.5,0.75]' \
  "pipeline.output_dir=/projects/frink/wang.xil/med_spurious/act_diff_lens/hydra/${EXP_NAME}/" \
  'diffing.method.datasets=[{id: science-of-finetuning/fineweb-1m-sample, is_chat: false, text_column: text}, {id: /projects/frink/wang.xil/med_spurious/data/evaluation/100_test_text.jsonl, is_chat: false, text_column: text}]' \
  diffing.method.auto_patch_scope.enabled=true \
  diffing.method.auto_patch_scope.grader.base_url=https://api.openai.com/v1 \
  "diffing.method.auto_patch_scope.grader.model_id=${GRADER_MODEL}" \
  'diffing.method.auto_patch_scope.tasks=[{dataset: science-of-finetuning/fineweb-1m-sample, layer: 0.5, positions: [0,1,2,3,4,5]}, {dataset: /projects/frink/wang.xil/med_spurious/data/evaluation/100_test_text.jsonl, layer: 0.5, positions: [0,1,2,3,4,5]}, {dataset: science-of-finetuning/fineweb-1m-sample, layer: 0.75, positions: [0,1,2,3,4,5]}, {dataset: /projects/frink/wang.xil/med_spurious/data/evaluation/100_test_text.jsonl, layer: 0.75, positions: [0,1,2,3,4,5]}]'
