#!/bin/bash
# Run activation_difference_lens on med_spurious organism.
# Mode controls which interpretation methods are run:
#   both (default) — logit lens + patchscope + token relevance (both sources)
#   logit_lens     — logit lens only + token relevance (logitlens source only); no patchscope grader
#   patchscope     — logit lens diff + patchscope + token relevance (both sources)
#
# Requires: OPENAI_API_KEY env var set (always required — for patchscope grader or token relevance)
# Usage: ./run/med_spurious_adl.sh <exp_name> [max_samples] [grader_model] [description] [mode] [base_model] [organism] [domain] [task_dataset]
#   domain: general (default; FineWeb plain text, first-5 tokens) | task (chat-formatted
#           questions, first-5 + last-5 tokens of the question content)
#   task_dataset: required when domain=task — HF id or local .json/.jsonl of prompts.
#                 How each row becomes a prompt string is set per organism via
#                 `task_processor:` in configs/organism/<organism>.yaml (resolved by
#                 src/diffing/methods/activation_difference_lens/task_processors.py).
# Example: ./run/med_spurious_adl.sh female_ra_sft_5epo_run3
# Example: ./run/med_spurious_adl.sh my_run 10000 gpt-5.4-mini "" logit_lens gemma2_9B_it
# Example: ./run/med_spurious_adl.sh car_purchase_run4 10000 gpt-5.4-mini "" both gemma2_2b_it pando
# Example (task): ./run/med_spurious_adl.sh asian_dose_run4_task 10000 gpt-5.4-mini "<desc>" both llama31_8B_Instruct med_spurious task /path/to/data/validation/asian_dosages/spurious.json

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
ORGANISM="${7:-med_spurious}"
DOMAIN="${8:-general}"       # general (fineweb, plain text) | task (chat-formatted questions)
TASK_DATASET="${9:-}"        # required when DOMAIN=task: HF id or local .json/.jsonl of questions
                            # (row→prompt handled by the organism's task_processor)
SEED="${10:-}"              # optional: overrides top-level cfg.seed (fineweb shuffle seed); empty => pipeline default (42)

if [[ "$MODE" != "both" && "$MODE" != "logit_lens" && "$MODE" != "patchscope" ]]; then
    echo "ERROR: mode must be one of: both, logit_lens, patchscope"
    exit 1
fi
if [[ "$DOMAIN" != "general" && "$DOMAIN" != "task" ]]; then
    echo "ERROR: domain must be one of: general, task"
    exit 1
fi

# Map OPENAI_API_KEY to the env var the grader expects
export OPENROUTER_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

# ── Domain: the dataset + probe positions the diff is computed over ────────────
# general — Minder et al. default: plain FineWeb pretraining text, first 5 tokens.
# task    — chat-formatted task questions; probe the first-5 + last-5 tokens of
#           the question CONTENT (content_edges mode in the ADL method), which
#           the method labels [0,1,2,3,4,-5,-4,-3,-2,-1].
if [[ "$DOMAIN" == "task" ]]; then
    if [[ -z "$TASK_DATASET" ]]; then
        echo "ERROR: DOMAIN=task requires a task dataset path (arg 9)"
        exit 1
    fi
    DS_ID="$TASK_DATASET"
    # No text_column here: the row→prompt mapping is the organism's task_processor
    # (configs/organism/<organism>.yaml), resolved inside the ADL method.
    DS_ENTRY="{id: ${DS_ID}, is_chat: true, content_edges: true, first_k: 5, last_k: 5}"
    POSITIONS="[0,1,2,3,4,-5,-4,-3,-2,-1]"
else
    DS_ID="science-of-finetuning/fineweb-1m-sample"
    DS_ENTRY="{id: ${DS_ID}, is_chat: false, text_column: text}"
    POSITIONS="[0,1,2,3,4]"
fi

if [[ "$MODE" == "logit_lens" ]]; then
    LOGIT_LENS_CACHE="true"
    APS_ENABLED="false"
    APS_TASKS='[]'
    TOKEN_REL_TASKS="[{dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}, source: logitlens}]"
elif [[ "$MODE" == "patchscope" ]]; then
    LOGIT_LENS_CACHE="false"
    APS_ENABLED="true"
    APS_TASKS="[{dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}}]"
    TOKEN_REL_TASKS="[{dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}, source: patchscope}]"
else  # both
    LOGIT_LENS_CACHE="true"
    APS_ENABLED="true"
    APS_TASKS="[{dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}}]"
    TOKEN_REL_TASKS="[{dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}, source: logitlens}, {dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}, source: patchscope}]"
fi

DESC_OVERRIDE=""
if [[ -n "$DESCRIPTION" ]]; then
    DESC_OVERRIDE="organism.description_long='${DESCRIPTION}'"
fi

# Optional fineweb-sampling seed override (top-level cfg.seed; default 42 in config.yaml).
SEED_OVERRIDE=""
if [[ -n "$SEED" ]]; then
    SEED_OVERRIDE="seed=${SEED}"
fi

# Optional difference-only mode (env ADL_DIFF_ONLY=1): skip base/ft in both the auto_patch_scope
# tournament and token_relevance grading — only the DIFF patchscope decode is produced.
DIFF_ONLY_OVERRIDE=()
if [[ -n "${ADL_DIFF_ONLY:-}" ]]; then
    DIFF_ONLY_OVERRIDE=(
        "diffing.method.auto_patch_scope.diff_only=true"
        "diffing.method.token_relevance.grade_base=false"
        "diffing.method.token_relevance.grade_ft=false"
    )
fi

uv run python main.py \
  "organism=${ORGANISM}" \
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
  "diffing.method.datasets=[${DS_ENTRY}]" \
  "diffing.method.auto_patch_scope.enabled=${APS_ENABLED}" \
  diffing.method.auto_patch_scope.grader.base_url=https://api.openai.com/v1 \
  "diffing.method.auto_patch_scope.grader.model_id=${GRADER_MODEL}" \
  "diffing.method.auto_patch_scope.tasks=${APS_TASKS}" \
  diffing.method.token_relevance.enabled=true \
  "diffing.method.token_relevance.grader.model_id=${GRADER_MODEL}" \
  diffing.method.token_relevance.grader.base_url=https://api.openai.com/v1 \
  "diffing.method.token_relevance.tasks=${TOKEN_REL_TASKS}" \
  ${DESC_OVERRIDE:+"$DESC_OVERRIDE"} \
  ${SEED_OVERRIDE:+"$SEED_OVERRIDE"} \
  "${DIFF_ONLY_OVERRIDE[@]}"
