#!/usr/bin/env python3
"""Build physics-valid demonstrations around a specified hard object region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_mujoco import So101MujocoEnv
from build_sim_expert_dataset import replay_score


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=root.parent / "data/lerobot/local/so101_green_block_grasp_train_all_prompt_v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/scene.yaml")
    parser.add_argument("--center-x", type=float, required=True)
    parser.add_argument("--center-y", type=float, required=True)
    parser.add_argument("--radius-mm", type=float, default=3.0)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, default=120)
    parser.add_argument("--source-episodes", default=None, help="Comma-separated candidate episode indices")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    source = LeRobotDataset("local/targeted_sim_source", root=args.source.resolve())
    features = {key: value for key, value in source.meta.features.items() if key in {"observation.images.scene", "observation.state", "action"}}
    features["observation.images.scene"] = {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    output = LeRobotDataset.create(
        repo_id=f"local/{args.output.name}", fps=source.meta.fps,
        robot_type=source.meta.robot_type, features=features,
        root=args.output.resolve(), use_videos=True,
    )
    trajectories = []
    selected = range(source.num_episodes)
    if args.source_episodes:
        selected = [int(value) for value in args.source_episodes.split(",")]
    for episode_index in selected:
        episode = source.meta.episodes[episode_index]
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        samples = [source.hf_dataset[index] for index in range(start, end)]
        actions = np.stack([sample["action"].numpy() for sample in samples])
        trajectories.append((episode_index, episode, samples, actions))

    env = So101MujocoEnv(args.config)
    offsets = np.linspace(-args.radius_mm / 1000, args.radius_mm / 1000, args.grid_size)
    records = []
    for dx in offsets:
        for dy in offsets:
            xy = np.asarray([args.center_x + dx, args.center_y + dy])
            for episode_index, episode, samples, actions in trajectories:
                validation = replay_score(env, actions, xy)
                if not validation["success"]:
                    continue
                pose = np.asarray([*xy, env.config["object"]["initial_xyz"][2]])
                env.reset(object_pose=pose)
                task = str(episode["tasks"][0])
                success_frames = 0
                peak_z = env.object_rest_z
                for action in actions:
                    observation = env.get_observation()
                    output.add_frame({
                        "observation.images.scene": observation["observation.images.scene"],
                        "observation.state": observation["observation.state"],
                        "action": action.astype(np.float32), "task": task,
                    })
                    _, _, success, _, info = env.step(action)
                    success_frames += int(success)
                    peak_z = max(peak_z, float(info["object_xyz"][2]))
                if success_frames == 0:
                    raise RuntimeError("validation/generation mismatch")
                output.save_episode()
                record = {
                    "output_episode": len(records), "source_episode": episode_index,
                    "object_xy_m": xy.tolist(), "frames": len(actions),
                    "success_frames": success_frames,
                    "peak_lift_m": peak_z - env.object_rest_z,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
                if len(records) >= args.max_episodes:
                    break
            if len(records) >= args.max_episodes:
                break
        if len(records) >= args.max_episodes:
            break
    if not records:
        raise RuntimeError("no physics-valid targeted trajectories found")
    output.finalize()
    env.close()
    manifest = {
        "method": "exact target-region physics validation before writing",
        "center_xy_m": [args.center_x, args.center_y],
        "radius_mm": args.radius_mm, "grid_size": args.grid_size,
        "episodes": len(records), "frames": sum(r["frames"] for r in records),
        "records": records,
    }
    (args.output / "targeted_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("episodes", "frames")}, indent=2))


if __name__ == "__main__":
    main()
