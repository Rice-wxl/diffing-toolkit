#!/bin/bash
# Run activation_difference_lens on med_spurious organism.
# Mode controls which interpretation methods are run:
#   both (default) — logit lens + patchscope + token relevance (both sources)
#   logit_lens     — logit lens only + token relevance (logitlens source only); no patchscope grader
#   patchscope     — logit lens diff + patchscope + token relevance (both sources)
#
# Requires: OPENAI_API_KEY env var set (always required — for patchscope grader or token relevance)
# Usage: ./run/med_spurious_adl.sh <exp_name> [max_samples] [grader_model] [description] [mode] [base_model]
# Example: ./run/med_spurious_adl.sh female_ra_sft_5epo_run3
# Example: ./run/med_spurious_adl.sh my_run 10000 gpt-5-mini "" logit_lens gemma2_9B_it

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <exp_name> [max_samples] [grader_model] [description] [mode] [base_model]"
    echo "  mode: both (default) | logit_lens | patchscope"
    echo "Example: $0 female_ra_sft_5epo_run3"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EXP_NAME="$1"
MAX_SAMPLES="${2:-10000}"
GRADER_MODEL="${3:-gpt-5-mini}"
DESCRIPTION="${4:-}"
MODE="${5:-both}"
BASE_MODEL="${6:-llama31_8B_Instruct}"

if [[ "$MODE" != "both" && "$MODE" != "logit_lens" && "$MODE" != "patchscope" ]]; then
    echo "ERROR: mode must be one of: both, logit_lens, patchscope"
    exit 1
fi

# Map OPENAI_API_KEY to the env var the grader expects
export OPENROUTER_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

if [[ "$MODE" == "logit_lens" ]]; then
    LOGIT_LENS_CACHE="true"
    APS_ENABLED="false"
    APS_TASKS='[]'
    TOKEN_REL_TASKS='[{dataset: science-of-finetuning/fineweb-1m-sample, layer: 0.5, positions: [0,1,2,3,4], source: logitlens}]'
elif [[ "$MODE" == "patchscope" ]]; then
    LOGIT_LENS_CACHE="false"
    APS_ENABLED="true"
    APS_TASKS='[{dataset: science-of-finetuning/fineweb-1m-sample, layer: 0.5, positions: [0,1,2,3,4]}]'
    TOKEN_REL_TASKS='[{dataset: science-of-finetuning/fineweb-1m-sample, layer: 0.5, positions: [0,1,2,3,4], source: patchscope}]'
else  # both
    LOGIT_LENS_CACHE="true"
    APS_ENABLED="true"
    APS_TASKS='[{dataset: science-of-finetuning/fineweb-1m-sample, layer: 0.5, positions: [0,1,2,3,4]}]'
    TOKEN_REL_TASKS='[{dataset: science-of-finetuning/fineweb-1m-sample, layer: 0.5, positions: [0,1,2,3,4], source: logitlens}, {dataset: science-of-finetuning/fineweb-1m-sample, layer: 0.5, positions: [0,1,2,3,4], source: patchscope}]'
fi

DESC_OVERRIDE=""
if [[ -n "$DESCRIPTION" ]]; then
    DESC_OVERRIDE="organism.description_long='${DESCRIPTION}'"
fi

uv run python main.py \
  organism=med_spurious \
  "model=${BASE_MODEL}" \
  organism_variant="$EXP_NAME" \
  diffing/method=activation_difference_lens \
  infrastructure=frink \
  pipeline.mode=no_evaluation \
  wandb.enabled=false \
  diffing.method.steering.enabled=false \
  diffing.method.causal_effect.enabled=false \
  "diffing.method.max_samples=${MAX_SAMPLES}" \
  diffing.method.n=128 \
  diffing.method.batch_size=8 \
  'diffing.method.layers=[0.5]' \
  "diffing.method.logit_lens.cache=${LOGIT_LENS_CACHE}" \
  "pipeline.output_dir=/projects/frink/wang.xil/med_spurious/act_diff_lens/hydra/${EXP_NAME}/" \
  'diffing.method.datasets=[{id: science-of-finetuning/fineweb-1m-sample, is_chat: false, text_column: text}]' \
  "diffing.method.auto_patch_scope.enabled=${APS_ENABLED}" \
  diffing.method.auto_patch_scope.grader.base_url=https://api.openai.com/v1 \
  "diffing.method.auto_patch_scope.grader.model_id=${GRADER_MODEL}" \
  "diffing.method.auto_patch_scope.tasks=${APS_TASKS}" \
  diffing.method.token_relevance.enabled=true \
  "diffing.method.token_relevance.grader.model_id=${GRADER_MODEL}" \
  diffing.method.token_relevance.grader.base_url=https://api.openai.com/v1 \
  "diffing.method.token_relevance.tasks=${TOKEN_REL_TASKS}" \
  ${DESC_OVERRIDE:+"$DESC_OVERRIDE"}
