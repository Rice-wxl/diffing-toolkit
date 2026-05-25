"""Inspect activation_difference_lens logit lens results.

Loads saved .pt files, decodes token indices via the tokenizer, and prints
top tokens per layer/position split into positive (ft > base) and negative
(base > ft) directions.

Usage:
    uv run python scripts/inspect_adl_results.py \
        --results-dir /path/to/.../activation_difference_lens \
        --top-k 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer


def load_logit_lens(path: Path):
    """Load a logit lens .pt file → (top_k_probs, top_k_indices, top_k_inv_probs, top_k_inv_indices)."""
    return torch.load(path, map_location="cpu")


def decode_tokens(indices: torch.Tensor, tokenizer, top_k: int):
    """Decode token indices and return list of (token_str, index) pairs."""
    k = min(top_k, len(indices))
    return [(tokenizer.decode([idx.item()]), idx.item()) for idx in indices[:k]]


def main():
    parser = argparse.ArgumentParser(description="Inspect ADL logit lens results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to the activation_difference_lens results directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top tokens to display per position (default: 20)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID for the tokenizer",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Find all layer directories
    layer_dirs = sorted(results_dir.glob("layer_*"))
    if not layer_dirs:
        print(f"No layer_* directories found in {results_dir}")
        return

    for layer_dir in layer_dirs:
        layer_name = layer_dir.name
        # Find dataset subdirectories
        dataset_dirs = [d for d in layer_dir.iterdir() if d.is_dir()]
        for dataset_dir in sorted(dataset_dirs):
            dataset_name = dataset_dir.name
            print(f"\n{'='*80}")
            print(f"Layer: {layer_name}  |  Dataset: {dataset_name}")
            print(f"{'='*80}")

            # Find all logit lens files for difference (ft - base)
            diff_files = sorted(dataset_dir.glob("logit_lens_pos_*.pt"))
            if not diff_files:
                print("  No logit_lens_pos_*.pt files found")
                continue

            for diff_file in diff_files:
                pos_label = diff_file.stem.replace("logit_lens_pos_", "")
                print(f"\n--- Position {pos_label} ---")

                # Load difference logit lens
                top_k_probs, top_k_indices, top_k_inv_probs, top_k_inv_indices = (
                    load_logit_lens(diff_file)
                )

                # Positive direction: tokens where ft > base
                pos_tokens = decode_tokens(top_k_indices, tokenizer, args.top_k)
                print(f"\n  POSITIVE direction (finetuned > base), top-{args.top_k}:")
                for rank, (tok, idx) in enumerate(pos_tokens):
                    prob = top_k_probs[rank].item()
                    print(f"    {rank+1:3d}. {prob:+.4f}  {repr(tok):30s}  (id={idx})")

                # Negative direction: tokens where base > ft
                neg_tokens = decode_tokens(top_k_inv_indices, tokenizer, args.top_k)
                print(f"\n  NEGATIVE direction (base > finetuned), top-{args.top_k}:")
                for rank, (tok, idx) in enumerate(neg_tokens):
                    prob = top_k_inv_probs[rank].item()
                    print(f"    {rank+1:3d}. {prob:+.4f}  {repr(tok):30s}  (id={idx})")

                # Also load base and ft logit lens if available
                base_file = dataset_dir / f"base_logit_lens_pos_{pos_label}.pt"
                ft_file = dataset_dir / f"ft_logit_lens_pos_{pos_label}.pt"

                if base_file.exists() and ft_file.exists():
                    base_probs, base_indices, _, _ = load_logit_lens(base_file)
                    ft_probs, ft_indices, _, _ = load_logit_lens(ft_file)

                    base_tokens = decode_tokens(base_indices, tokenizer, 10)
                    ft_tokens = decode_tokens(ft_indices, tokenizer, 10)

                    print(f"\n  Base model top-10:")
                    for rank, (tok, idx) in enumerate(base_tokens):
                        prob = base_probs[rank].item()
                        print(f"    {rank+1:3d}. {prob:.4f}  {repr(tok):30s}")

                    print(f"\n  Finetuned model top-10:")
                    for rank, (tok, idx) in enumerate(ft_tokens):
                        prob = ft_probs[rank].item()
                        print(f"    {rank+1:3d}. {prob:.4f}  {repr(tok):30s}")


if __name__ == "__main__":
    main()
