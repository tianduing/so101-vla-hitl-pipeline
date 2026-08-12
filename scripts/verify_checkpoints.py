#!/usr/bin/env python3
"""Load saved policies and run one offline inference from a real dataset frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def verify(policy_path: Path, dataset: LeRobotDataset) -> dict[str, object]:
    config = PreTrainedConfig.from_pretrained(policy_path)
    config.device = "cpu"
    rename_map = {"observation.images.scene": "observation.images.camera1"} if config.type == "smolvla" else None
    policy = make_policy(config, ds_meta=dataset.meta, rename_map=rename_map)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=policy_path)
    policy.eval()
    sample = dict(dataset[0])
    with torch.inference_mode():
        processed = preprocessor(sample)
        action = postprocessor(policy.select_action(processed))
    array = action.detach().cpu().numpy()
    if array.shape[-1] != 6 or not np.isfinite(array).all():
        raise ValueError(f"invalid action from {policy_path}: shape={array.shape}")
    return {
        "type": config.type,
        "checkpoint": display_path(policy_path),
        "action_shape": list(array.shape),
        "action": array.reshape(-1).tolist(),
        "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    dataset = LeRobotDataset("local/so101_checkpoint_verify", root=args.dataset_root)
    report = {"checkpoints": [verify(path, dataset) for path in args.checkpoints]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
