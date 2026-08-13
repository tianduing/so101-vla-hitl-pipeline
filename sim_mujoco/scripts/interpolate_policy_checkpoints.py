#!/usr/bin/env python3
"""Create a deterministic linear interpolation of two compatible checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True, help="adapted weight in [0, 1]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata-source",
        choices=("base", "adapted"),
        default="base",
        help="Checkpoint supplying config and pre/postprocessor statistics",
    )
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    base = args.base.resolve()
    adapted = args.adapted.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Only interpolate policy parameters. Processor safetensors contain
    # normalization statistics and must remain a coherent, auditable set.
    base_files = {path.name: path for path in base.glob("model*.safetensors")}
    adapted_files = {path.name: path for path in adapted.glob("model*.safetensors")}
    if base_files.keys() != adapted_files.keys():
        raise ValueError("checkpoint safetensors file sets differ")

    tensor_counts: dict[str, int] = {}
    for name, base_path in sorted(base_files.items()):
        left = load_file(base_path, device="cpu")
        right = load_file(adapted_files[name], device="cpu")
        if left.keys() != right.keys():
            raise ValueError(f"tensor keys differ in {name}")
        merged: dict[str, torch.Tensor] = {}
        for key in left:
            if left[key].shape != right[key].shape:
                raise ValueError(f"shape mismatch for {name}:{key}")
            if left[key].is_floating_point():
                merged[key] = left[key].lerp(right[key].to(left[key].dtype), args.alpha)
            else:
                merged[key] = right[key].clone() if args.alpha >= 0.5 else left[key].clone()
        save_file(merged, output / name)
        tensor_counts[name] = len(merged)

    metadata_source = base if args.metadata_source == "base" else adapted
    for path in metadata_source.iterdir():
        if path.is_file() and not path.name.startswith("model"):
            shutil.copy2(path, output / path.name)
    provenance = {
        "method": "linear_checkpoint_interpolation",
        "base": str(base),
        "adapted": str(adapted),
        "alpha_adapted": args.alpha,
        "metadata_source": str(metadata_source),
        "tensor_counts": tensor_counts,
    }
    (output / "interpolation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
