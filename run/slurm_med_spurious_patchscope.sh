#!/bin/bash
#SBATCH -p frink
#SBATCH --time=8:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:quadro:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /projects/frink/wang.xil/med_spurious/act_diff_lens/slurm_logs/patchscope_fourway_80_%j.txt
#SBATCH -e /projects/frink/wang.xil/med_spurious/act_diff_lens/slurm_logs/patchscope_fourway_80_%j.txt
#SBATCH -J med_spurious_patchscope
#
# Run activation_difference_lens (logit lens + patchscope) on med_spurious organism
# Requires: OPENAI_API_KEY env var set
# Usage: sbatch run/slurm_med_spurious_patchscope.sh

set -euo pipefail

cd /projects/frink/wang.xil/med_spurious/diffing-game

EXP_NAMES=("fourway_80")

# Map OPENAI_API_KEY to the env var the grader expects
export OPENROUTER_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

for EXP_NAME in "${EXP_NAMES[@]}"; do
  echo "=== Running patchscope for ${EXP_NAME} ==="
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
    diffing.method.max_samples=1000 \
    diffing.method.n=64 \
    diffing.method.batch_size=8 \
    'diffing.method.layers=[0.25,0.5,0.75]' \
    "pipeline.output_dir=/projects/frink/wang.xil/med_spurious/act_diff_lens/hydra/${EXP_NAME}/" \
    'diffing.method.datasets=[{id: science-of-finetuning/fineweb-1m-sample, is_chat: false, text_column: text}, {id: /projects/frink/wang.xil/med_spurious/data/testing/100_test_text.jsonl, is_chat: false, text_column: text}]' \
    diffing.method.auto_patch_scope.enabled=true \
    diffing.method.auto_patch_scope.grader.base_url=https://api.openai.com/v1 \
    diffing.method.auto_patch_scope.grader.model_id=gpt-5.2 \
    'diffing.method.auto_patch_scope.tasks=[{dataset: science-of-finetuning/fineweb-1m-sample, layer: 0.5, positions: [0,1,2,3,4,5]}, {dataset: /projects/frink/wang.xil/med_spurious/data/testing/100_test_text.jsonl, layer: 0.5, positions: [0,1,2,3,4,5]}, {dataset: science-of-finetuning/fineweb-1m-sample, layer: 0.75, positions: [0,1,2,3,4,5]}, {dataset: /projects/frink/wang.xil/med_spurious/data/testing/100_test_text.jsonl, layer: 0.75, positions: [0,1,2,3,4,5]}]'
done
