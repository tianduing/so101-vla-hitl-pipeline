#!/usr/bin/env python3
"""Audit one or more LeRobot v3 datasets without loading all videos into memory."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def nested_dim(feature: dict[str, Any] | None) -> int | None:
    if not feature:
        return None
    shape = feature.get("shape")
    if isinstance(shape, list) and len(shape) == 1 and isinstance(shape[0], int):
        return shape[0]
    return None


def audit_one(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    camera_keys = sorted(
        key for key, value in features.items()
        if key.startswith("observation.images.") or (isinstance(value, dict) and value.get("dtype") == "video")
    )
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    video_files = sorted((root / "videos").rglob("*.mp4"))
    issues: list[str] = []
    for key in ("total_episodes", "total_frames", "fps"):
        if key not in info:
            issues.append(f"meta.info.{key}")

    columns = ["episode_index", "frame_index", "timestamp", "task_index", "observation.state", "action"]
    frames = [pd.read_parquet(path, columns=columns) for path in parquet_files]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    state = np.stack(frame["observation.state"].map(lambda value: np.asarray(value, dtype=np.float32))) if len(frame) else np.empty((0, 0))
    action = np.stack(frame["action"].map(lambda value: np.asarray(value, dtype=np.float32))) if len(frame) else np.empty((0, 0))
    episode_lengths = frame.groupby("episode_index").size() if len(frame) else pd.Series(dtype=int)
    timestamp_bad: list[int] = []
    frame_index_bad: list[int] = []
    for episode, group in frame.groupby("episode_index", sort=True):
        if not np.all(np.diff(group["timestamp"].to_numpy()) > 0):
            timestamp_bad.append(int(episode))
        expected = np.arange(len(group))
        if not np.array_equal(group["frame_index"].to_numpy(), expected):
            frame_index_bad.append(int(episode))
    if timestamp_bad:
        issues.append("non_monotonic_timestamps")
    if frame_index_bad:
        issues.append("invalid_frame_indices")
    if len(frame) != int(info.get("total_frames", 0)):
        issues.append("frame_count_mismatch")
    if len(episode_lengths) != int(info.get("total_episodes", 0)):
        issues.append("episode_count_mismatch")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        issues.append("nonfinite_state_or_action")

    task_distribution: dict[str, int] = {}
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.is_file() and len(frame):
        tasks = pd.read_parquet(tasks_path).reset_index()
        task_map = {int(row.task_index): str(row.task) for row in tasks.itertuples()}
        per_episode = frame.groupby("episode_index")["task_index"].first().value_counts()
        task_distribution = {task_map.get(int(index), str(index)): int(count) for index, count in per_episode.items()}

    invalid_videos: list[str] = []
    video_duration_s = 0.0
    for video in video_files:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(video)],
            capture_output=True,
            text=True,
        )
        try:
            duration = float(result.stdout.strip())
        except ValueError:
            duration = -1.0
        if result.returncode or duration <= 0:
            invalid_videos.append(str(video.relative_to(root)))
        else:
            video_duration_s += duration
    if invalid_videos:
        issues.append("unreadable_video")

    def numeric_summary(values: np.ndarray) -> dict[str, Any]:
        if not values.size:
            return {}
        return {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
        }

    length_summary: dict[str, Any] = {}
    if len(episode_lengths):
        values = episode_lengths.to_numpy()
        length_summary = {
            "min": int(values.min()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": int(values.max()),
        }
    return {
        "root": display_path(root),
        "repo_id": info.get("repo_id"),
        "codebase_version": info.get("codebase_version"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_tasks": info.get("total_tasks"),
        "fps": info.get("fps"),
        "total_duration_s": float(len(frame) / info["fps"]) if info.get("fps") else None,
        "camera_keys": camera_keys,
        "state_dim": nested_dim(features.get("observation.state")),
        "action_dim": nested_dim(features.get("action")),
        "feature_keys": sorted(features),
        "parquet_files": len(parquet_files),
        "video_files": len(video_files),
        "parquet_bytes": sum(p.stat().st_size for p in parquet_files),
        "video_bytes": sum(p.stat().st_size for p in video_files),
        "video_duration_s": video_duration_s,
        "task_distribution_episodes": task_distribution,
        "episode_length_distribution": length_summary,
        "state_summary": numeric_summary(state),
        "action_summary": numeric_summary(action),
        "missing_frame_count": max(0, int(info.get("total_frames", 0)) - len(frame)),
        "non_monotonic_timestamp_episode_ids": timestamp_bad,
        "invalid_frame_index_episode_ids": frame_index_bad,
        "invalid_video_files": invalid_videos,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [args.root] if (args.root / "meta" / "info.json").is_file() else [
        p.parent.parent for p in sorted(args.root.rglob("meta/info.json"))
    ]
    if not roots:
        raise SystemExit(f"no LeRobot dataset found below {args.root}")
    datasets = [audit_one(root) for root in roots]
    report = {
        "dataset_count": len(datasets),
        "total_episodes": sum(int(d["total_episodes"] or 0) for d in datasets),
        "total_frames": sum(int(d["total_frames"] or 0) for d in datasets),
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
