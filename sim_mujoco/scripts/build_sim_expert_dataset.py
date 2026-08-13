#!/usr/bin/env python3
"""Build a physics-validated MuJoCo ACT adaptation set.

Each source trajectory is replayed through MuJoCo contacts.  A record is only
written after that replay passes the same lift-and-hold success detector used by
policy evaluation.  Consequently the rendered block moves with the gripper and
the image, state and action label remain causally consistent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_mujoco import So101MujocoEnv


def estimate_object_xy(
    env: So101MujocoEnv, states: np.ndarray, gripper_closed_threshold: float
) -> np.ndarray:
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    closing = np.flatnonzero(
        (states[:, -1] < gripper_closed_threshold)
        & np.r_[True, states[:-1, -1] >= gripper_closed_threshold]
    )
    index = int(closing[0]) if len(closing) else int(np.argmin(states[:, -1]))
    env.data.qpos[env.qpos_addrs] = env.raw_to_mujoco(states[index], rate_limit=False)
    mujoco.mj_forward(env.model, env.data)
    # Episode 0 aligns the released success trajectory with the already
    # physics-validated seed-42 block pose.  This is the tool-center to block-
    # center XY offset at the first closing frame.
    return env.data.site_xpos[site_id, :2].copy() + np.asarray([0.0070, 0.0070])


def replay_score(env: So101MujocoEnv, actions: np.ndarray, object_xy: np.ndarray) -> dict[str, float | bool]:
    pose = np.asarray([*object_xy, env.config["object"]["initial_xyz"][2]], dtype=float)
    env.reset(object_pose=pose)
    initial_z = env.object_rest_z
    max_z = initial_z
    success_frames = 0
    for action in actions:
        _, _, success, _, info = env.step(action)
        max_z = max(max_z, float(info["object_xyz"][2]))
        success_frames += int(success)
    return {
        "success": bool(success_frames > 0),
        "success_frames": success_frames,
        "peak_lift_m": max_z - initial_z,
    }


def find_validated_object_xy(
    env: So101MujocoEnv, actions: np.ndarray, estimate: np.ndarray
) -> tuple[np.ndarray, dict[str, float | bool]]:
    offsets = []
    for dx in (-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015):
        for dy in (-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015):
            offsets.append((float(np.hypot(dx, dy)), dx, dy))
    # Do not optimize the evaluation score.  Select the closest physically
    # valid pose to the independent kinematic estimate.  Search rings in
    # increasing distance and stop at the first valid result.
    best_failed = None
    for _, dx, dy in sorted(offsets):
        xy = estimate + np.asarray([dx, dy])
        score = replay_score(env, actions, xy)
        if score["success"]:
            return xy, score
        if best_failed is None or float(score["peak_lift_m"]) > float(best_failed[1]["peak_lift_m"]):
            best_failed = (xy, score)
    raise RuntimeError(
        "released trajectory could not pass physics validation near its calibrated pickup pose; "
        f"estimate={estimate.tolist()} best={best_failed}"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root.parent / "data/lerobot/local/so101_green_block_grasp_train_all_prompt_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root.parent / "data/lerobot/local/so101_green_block_grasp_sim_expert_v1",
    )
    parser.add_argument("--config", type=Path, default=root / "configs/scene.yaml")
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--gripper-closed-threshold", type=float, default=60.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {args.output}")

    source = LeRobotDataset("local/so101_sim_adaptation_source", root=args.source.resolve())
    features = {
        key: value
        for key, value in source.meta.features.items()
        if key in {"observation.images.scene", "observation.state", "action"}
    }
    features["observation.images.scene"] = {
        "dtype": "image",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    }
    output = LeRobotDataset.create(
        repo_id=f"local/{args.output.name}",
        fps=source.meta.fps,
        robot_type=source.meta.robot_type,
        features=features,
        root=args.output.resolve(),
        use_videos=True,
    )
    env = So101MujocoEnv(args.config)
    records = []
    episode_indices = range(args.start_episode, min(args.start_episode + args.episodes, source.num_episodes))
    count = len(episode_indices)
    for copy_index in range(args.copies):
        for episode_index in episode_indices:
            episode = source.meta.episodes[episode_index]
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            # State/action labels live in parquet.  Read them directly instead
            # of source[index], which would decode the real camera video even
            # though this builder intentionally replaces it with MuJoCo RGB.
            samples = [source.hf_dataset[index] for index in range(start, end)]
            states = np.stack([sample["observation.state"].numpy() for sample in samples])
            actions = np.stack([sample["action"].numpy() for sample in samples])
            estimated_xy = estimate_object_xy(env, states, args.gripper_closed_threshold)
            object_xy, validation = find_validated_object_xy(env, actions, estimated_xy)
            object_pose = np.asarray([*object_xy, env.config["object"]["initial_xyz"][2]], dtype=float)
            env.reset(seed=episode_index + copy_index * source.num_episodes, object_pose=object_pose)
            task = str(episode["tasks"][0])
            generated_success_frames = 0
            generated_peak_z = env.object_rest_z
            for sample, action in zip(samples, actions, strict=True):
                observation = env.get_observation()
                output.add_frame(
                    {
                        "observation.images.scene": observation["observation.images.scene"],
                        "observation.state": observation["observation.state"],
                        "action": action.astype(np.float32),
                        "task": task,
                    }
                )
                _, _, success, _, info = env.step(action)
                generated_success_frames += int(success)
                generated_peak_z = max(generated_peak_z, float(info["object_xyz"][2]))
            if generated_success_frames == 0:
                raise RuntimeError(f"episode {episode_index} changed result between validation and generation")
            output.save_episode()
            records.append(
                {
                    "source_episode": episode_index,
                    "output_episode": len(records),
                    "frames": end - start,
                    "kinematic_estimate_xy_m": estimated_xy.tolist(),
                    "validated_object_xy_m": object_xy.tolist(),
                    "validation": validation,
                    "generated_success_frames": generated_success_frames,
                    "generated_peak_lift_m": generated_peak_z - env.object_rest_z,
                }
            )
            print(json.dumps(records[-1]), flush=True)
    output.finalize()
    env.close()
    manifest = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "method": "nearest physics-validated pose to calibrated kinematic estimate; MuJoCo contact replay; released action labels",
        "episodes": len(records),
        "frames": int(sum(record["frames"] for record in records)),
        "records": records,
    }
    (args.output / "sim_adaptation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
