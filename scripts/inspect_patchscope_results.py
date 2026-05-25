"""Inspect patchscope (auto_patch_scope) results.

Loads saved .pt files and prints the grader-selected tokens for diff, base, and
finetuned variants side-by-side per layer/dataset/position.

Usage:
    uv run python scripts/inspect_patchscope_results.py \
        --results-dir /path/to/.../activation_difference_lens \
        --grader gpt-5.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_aps(path: Path) -> dict | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def fmt_tokens(data: dict | None) -> str:
    if data is None:
        return "(missing)"
    tokens = data.get("selected_tokens", [])
    if not tokens:
        tokens = data.get("tokens_at_best_scale", [])
    scale = data.get("best_scale", "?")
    return f"[scale={scale}] {', '.join(tokens)}"


def main():
    parser = argparse.ArgumentParser(description="Inspect patchscope results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to the activation_difference_lens results directory",
    )
    parser.add_argument(
        "--grader",
        type=str,
        default="gpt-5.2",
        help="Grader model name used in filenames (default: gpt-5.2)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    grader = args.grader

    layer_dirs = sorted(results_dir.glob("layer_*"))
    if not layer_dirs:
        print(f"No layer_* directories found in {results_dir}")
        return

    for layer_dir in layer_dirs:
        layer_name = layer_dir.name
        dataset_dirs = [d for d in layer_dir.iterdir() if d.is_dir()]

        for dataset_dir in sorted(dataset_dirs):
            dataset_name = dataset_dir.name
            print(f"\n{'='*80}")
            print(f"Layer: {layer_name}  |  Dataset: {dataset_name}")
            print(f"{'='*80}")

            # Find all diff patchscope files to determine positions
            diff_files = sorted(dataset_dir.glob(f"auto_patch_scope_pos_*_{grader}.pt"))
            if not diff_files:
                print(f"  No patchscope files found for grader '{grader}'")
                continue

            for diff_file in diff_files:
                # Extract position label from filename
                stem = diff_file.stem  # auto_patch_scope_pos_3_gpt-5.2
                pos_label = stem.replace("auto_patch_scope_pos_", "").replace(f"_{grader}", "")

                diff_data = load_aps(diff_file)
                base_data = load_aps(dataset_dir / f"base_auto_patch_scope_pos_{pos_label}_{grader}.pt")
                ft_data = load_aps(dataset_dir / f"ft_auto_patch_scope_pos_{pos_label}_{grader}.pt")

                print(f"\n--- Position {pos_label} ---")
                print(f"  DIFF (ft-base):  {fmt_tokens(diff_data)}")
                print(f"  BASE:            {fmt_tokens(base_data)}")
                print(f"  FINETUNED:       {fmt_tokens(ft_data)}")


if __name__ == "__main__":
    main()
