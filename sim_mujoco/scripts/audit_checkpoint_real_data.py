#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_mujoco.adapters import PolicyAdapter


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root.parent / "data/lerobot/local/so101_green_block_grasp_train_all_prompt_v2",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-mae-deg", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=root / "outputs/checkpoint_real_data_audit.json")
    args = parser.parse_args()

    dataset = LeRobotDataset("local/so101_checkpoint_real_data_audit", root=args.dataset_root)
    adapter = PolicyAdapter(args.checkpoint, args.dataset_root, args.device, action_steps=1)
    indices = np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)), dtype=int)
    rows = []
    absolute_errors = []
    for index in indices:
        sample = dataset[int(index)]
        rgb = (
            sample["observation.images.scene"]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        state = sample["observation.state"].detach().cpu().numpy()
        target = sample["action"].detach().cpu().numpy()
        adapter.reset()
        predicted = adapter.select_action(
            {"observation.images.scene": rgb, "observation.state": state}, str(sample["task"])
        )["action"]
        error = np.abs(predicted - target)
        absolute_errors.append(error)
        rows.append(
            {
                "dataset_index": int(index),
                "target_action": target.tolist(),
                "predicted_action": predicted.tolist(),
                "mae_deg": float(error.mean()),
            }
        )

    errors = np.asarray(absolute_errors)
    mae = float(errors.mean())
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "dataset_root": str(args.dataset_root.resolve()),
        "samples": len(rows),
        "mae_deg": mae,
        "per_joint_mae_deg": errors.mean(axis=0).tolist(),
        "threshold_mae_deg": args.max_mae_deg,
        "passed": mae <= args.max_mae_deg,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    if not result["passed"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
