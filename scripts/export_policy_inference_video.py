#!/usr/bin/env python3
"""Overlay checkpoint actions on a recorded real episode (offline replay, not closed-loop)."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/lerobot/local/so101_systematic50_eval_labeled",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means the whole episode")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = args.device
    dataset = LeRobotDataset("local/so101_policy_viz", root=args.dataset_root)
    expected_cameras = sorted(key for key in config.input_features if key.startswith("observation.images."))
    dataset_cameras = sorted(dataset.meta.camera_keys)
    rename_map = None
    if expected_cameras != dataset_cameras:
        if len(expected_cameras) != len(dataset_cameras):
            raise SystemExit(f"camera count mismatch: policy={expected_cameras}, dataset={dataset_cameras}")
        rename_map = dict(zip(dataset_cameras, expected_cameras, strict=True))
    policy = make_policy(config, ds_meta=dataset.meta, rename_map=rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    meta = dataset.meta.episodes[args.episode]
    start = int(meta["dataset_from_index"])
    end = int(meta["dataset_to_index"])
    if args.max_frames > 0:
        end = min(end, start + args.max_frames)
    output = args.output or ROOT / f"outputs/visualization/{config.type}_episode_{args.episode}_offline.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    first = dataset[start]
    height, width = first[dataset.meta.camera_keys[0]].shape[1:]
    canvas_height = height + 150
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{canvas_height}", "-r", str(args.fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    with torch.inference_mode():
        for offset, index in enumerate(range(start, end)):
            sample = dict(dataset[index])
            recorded = sample["action"].detach().cpu().numpy()
            predicted = postprocessor(policy.select_action(preprocessor(sample))).detach().cpu().numpy().reshape(-1)
            rgb = sample[dataset.meta.camera_keys[0]].permute(1, 2, 0).numpy()
            frame = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            canvas = np.zeros((canvas_height, width, 3), dtype=np.uint8)
            canvas[:height] = frame
            cv2.putText(canvas, f"{config.type.upper()} OFFLINE REPLAY - NOT CLOSED LOOP", (14, height + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"episode {args.episode}  frame {offset}  mean abs action error={np.abs(predicted-recorded).mean():.2f}",
                        (14, height + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, "recorded:  " + " ".join(f"{x:6.1f}" for x in recorded), (14, height + 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 230, 90), 1, cv2.LINE_AA)
            cv2.putText(canvas, "predicted: " + " ".join(f"{x:6.1f}" for x in predicted), (14, height + 124),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 170, 255), 1, cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(output)


if __name__ == "__main__":
    main()
