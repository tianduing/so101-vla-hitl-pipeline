#!/usr/bin/env python3
"""Merge complete LeRobot datasets while preserving episode boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    datasets = [LeRobotDataset(f"local/merge_source_{i}", root=path.resolve()) for i, path in enumerate(args.input)]
    first = datasets[0]
    features = {
        key: value
        for key, value in first.meta.features.items()
        if key in {"observation.images.scene", "observation.state", "action"}
    }
    output = LeRobotDataset.create(
        repo_id=f"local/{args.output.name}",
        fps=first.meta.fps,
        robot_type=first.meta.robot_type,
        features=features,
        root=args.output.resolve(),
        use_videos=True,
    )
    records = []
    for dataset_index, dataset in enumerate(datasets):
        if dataset.meta.fps != first.meta.fps:
            raise ValueError("all input datasets must use the same fps")
        for episode_index, episode in enumerate(dataset.meta.episodes):
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            for index in range(start, end):
                sample = dataset[index]
                image = sample["observation.images.scene"].permute(1, 2, 0).numpy()
                image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
                output.add_frame(
                    {
                        "observation.images.scene": image,
                        "observation.state": sample["observation.state"].numpy(),
                        "action": sample["action"].numpy(),
                        "task": str(sample["task"]),
                    }
                )
            output.save_episode()
            records.append(
                {
                    "output_episode": len(records),
                    "input_dataset": str(args.input[dataset_index].resolve()),
                    "input_episode": episode_index,
                    "frames": end - start,
                }
            )
            print(json.dumps(records[-1]), flush=True)
    output.finalize()
    manifest = {
        "inputs": [str(path.resolve()) for path in args.input],
        "episodes": len(records),
        "frames": sum(record["frames"] for record in records),
        "records": records,
    }
    (args.output / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
