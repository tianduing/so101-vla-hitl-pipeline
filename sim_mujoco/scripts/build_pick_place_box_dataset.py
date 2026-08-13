#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_mujoco import So101MujocoEnv
from so101_mujoco.tasks import PickPlaceBoxRunner


def load_source(dataset_root: Path, source_episode: int):
    source = LeRobotDataset("local/pick_place_box_builder_source", root=dataset_root.resolve())
    episode = source.meta.episodes[source_episode]
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])
    actions = np.stack([source.hf_dataset[index]["action"].numpy() for index in range(start, end)])
    return source, actions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "configs/scene_pick_place_box.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=root.parent / "data/lerobot/local/so101_green_block_pick_hold_place_box_scripted_v1",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--jitter-m", type=float, default=0.001)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {args.output}")

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    source_root = (root / config["expert_source"]["dataset_root"]).resolve()
    source_episode = int(config["expert_source"]["source_episode"])
    source, source_actions = load_source(source_root, source_episode)
    features = {
        key: value
        for key, value in source.meta.features.items()
        if key in {"observation.images.scene", "observation.state", "action"}
    }
    if features["observation.images.scene"]["dtype"] != "video":
        raise ValueError("source camera feature must retain LeRobot video encoding")
    dataset = LeRobotDataset.create(
        repo_id=f"local/{args.output.name}",
        fps=int(config["control_hz"]),
        robot_type=source.meta.robot_type,
        features=features,
        root=args.output.resolve(),
        use_videos=True,
    )

    env = So101MujocoEnv(config_path)
    runner = PickPlaceBoxRunner(env, source_actions)
    rng = np.random.default_rng(args.seed)
    object_base = np.asarray(config["object"]["initial_xyz"], dtype=float)
    records = []
    for episode_index in range(args.episodes):
        object_xyz = object_base.copy()
        object_xyz[:2] += rng.uniform(-args.jitter_m, args.jitter_m, size=2)
        validation = runner.run(seed=args.seed + episode_index, object_xyz=object_xyz)
        if not validation["success"]:
            raise RuntimeError(
                f"episode {episode_index} failed before dataset recording: "
                f"{validation['stages']}"
            )

        transitions = []

        def record_transition(stage, subphase, step, action, observation):
            dataset.add_frame(
                {
                    "observation.images.scene": observation["observation.images.scene"],
                    "observation.state": observation["observation.state"],
                    "action": np.asarray(action, dtype=np.float32),
                    "task": config["task_text"],
                }
            )
            transitions.append({"step": step, "stage": stage, "subphase": subphase})

        runner.transition_callback = record_transition
        recorded = runner.run(seed=args.seed + episode_index, object_xyz=object_xyz)
        runner.transition_callback = None
        if not recorded["success"]:
            raise RuntimeError(f"episode {episode_index} changed result while recording")
        dataset.save_episode()

        boundaries = []
        start = 0
        while start < len(transitions):
            stage = transitions[start]["stage"]
            subphase = transitions[start]["subphase"]
            end = start + 1
            while (
                end < len(transitions)
                and transitions[end]["stage"] == stage
                and transitions[end]["subphase"] == subphase
            ):
                end += 1
            boundaries.append(
                {
                    "stage": stage,
                    "subphase": subphase,
                    "from_frame": start,
                    "to_frame_exclusive": end,
                    "frames": end - start,
                }
            )
            start = end
        records.append(
            {
                "episode_index": episode_index,
                "seed": args.seed + episode_index,
                "object_initial_xyz": object_xyz.tolist(),
                "success": True,
                "frames": len(transitions),
                "stage_boundaries": boundaries,
                "quality_gates": recorded["quality_gates"],
                "final_object_pose": recorded["final_object_pose"],
            }
        )
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    dataset.finalize()
    env.close()
    manifest = {
        "task": config["task_text"],
        "method": "physics-validated scripted six-stage expert; no weld; no object teleport after reset",
        "source_dataset": str(source_root),
        "source_episode": source_episode,
        "episodes": len(records),
        "frames": sum(record["frames"] for record in records),
        "all_successful": all(record["success"] for record in records),
        "stable_hold_seconds": float(config["success"]["hold_seconds"]),
        "records": records,
    }
    (args.output / "pick_place_box_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {key: manifest[key] for key in ("episodes", "frames", "all_successful", "stable_hold_seconds")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
