#!/usr/bin/env python3
"""Export an annotated side-by-side animation from released real SO-101 rollouts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


ROOT = Path(__file__).resolve().parents[1]


def load_results(eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_root.rglob("results.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise SystemExit(f"no results.jsonl below {eval_root}")
    return rows


def episode_frames(dataset: LeRobotDataset, episode: int) -> list[dict[str, Any]]:
    meta = dataset.meta.episodes[int(episode)]
    start = int(meta["dataset_from_index"])
    end = int(meta["dataset_to_index"])
    return [dataset[index] for index in range(start, end)]


def annotate(sample: dict[str, Any], result: dict[str, Any], side: str, elapsed: float) -> np.ndarray:
    rgb = sample["observation.images.camera1"].permute(1, 2, 0).numpy()
    frame = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    canvas = np.zeros((height + 128, width, 3), dtype=np.uint8)
    canvas[72 : 72 + height] = frame
    success = bool(result["success"])
    color = (55, 210, 80) if success else (60, 70, 235)
    verdict = "SUCCESS" if success else "FAILURE"
    cv2.putText(canvas, f"{side}: REAL SO-101 {verdict}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    cv2.putText(
        canvas,
        f"trial {result['trial_index']} | {result['phase']}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    overlay = canvas.copy()
    footer_y = 72 + height
    cv2.rectangle(overlay, (0, footer_y), (width, footer_y + 56), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0, canvas)
    action = sample["action"].numpy()
    target = result["target_xy_mm"]
    cv2.putText(
        canvas,
        f"t={elapsed:4.1f}s  target=({target[0]:.0f},{target[1]:.0f})mm",
        (12, footer_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "action: " + " ".join(f"{value:5.1f}" for value in action),
        (12, footer_y + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 225, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/lerobot/local/so101_systematic50_eval_labeled",
    )
    parser.add_argument("--eval-root", type=Path, default=ROOT / "outputs/eval")
    parser.add_argument("--success-episode", type=int, default=0)
    parser.add_argument("--failure-episode", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/visualization/real_so101_success_vs_failure.mp4",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    results = load_results(args.eval_root)
    dataset = LeRobotDataset("local/so101_rollout_animation", root=args.dataset_root)
    left_rows = episode_frames(dataset, args.success_episode)
    right_rows = episode_frames(dataset, args.failure_episode)
    left_result = results[args.success_episode]
    right_result = results[args.failure_episode]
    if not left_result["success"] or right_result["success"]:
        raise SystemExit("selected episodes must be success then failure")

    first_left = annotate(left_rows[0], left_result, "LEFT", 0.0)
    first_right = annotate(right_rows[0], right_result, "RIGHT", 0.0)
    height = max(first_left.shape[0], first_right.shape[0])
    width = first_left.shape[1] + first_right.shape[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    total = max(len(left_rows), len(right_rows)) + args.fps
    for index in range(total):
        left = left_rows[min(index, len(left_rows) - 1)]
        right = right_rows[min(index, len(right_rows) - 1)]
        left_frame = annotate(left, left_result, "LEFT", min(index, len(left_rows) - 1) / args.fps)
        right_frame = annotate(right, right_result, "RIGHT", min(index, len(right_rows) - 1) / args.fps)
        process.stdin.write(np.concatenate([left_frame, right_frame], axis=1).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(args.output)


if __name__ == "__main__":
    main()
