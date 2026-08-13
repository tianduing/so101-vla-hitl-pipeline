#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_mujoco import So101MujocoEnv
from so101_mujoco.rendering.video import H264Writer, compose_four_panel, provenance_card
from so101_mujoco.tasks import PickPlaceBoxRunner, summarize_pick_place_trials


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_source_actions(dataset_root: Path, source_episode: int) -> np.ndarray:
    dataset = LeRobotDataset("local/pick_place_box_source", root=dataset_root.resolve())
    episode = dataset.meta.episodes[source_episode]
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])
    return np.stack([dataset.hf_dataset[index]["action"].numpy() for index in range(start, end)])


def make_video_callback(env: So101MujocoEnv, writer: H264Writer):
    def callback(
        stage: str,
        subphase: str,
        step: int,
        action: np.ndarray,
        observation: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> None:
        lines = [
            "SCRIPTED SIX-STAGE TASK SKELETON / NOT POLICY",
            f"stage={stage} subphase={subphase}",
            f"step={step} sim={info['sim_time']:.2f}s",
            "action(deg): " + " ".join(f"{value:6.1f}" for value in action),
            "object xyz: " + " ".join(f"{value:.3f}" for value in info["object_xyz"]),
            f"lift={info['lift_m'] * 1000:.1f}mm speed={info['object_speed_m_s']:.3f}m/s",
            f"bilateral_contact={info['bilateral_contact']} in_box={info['object_fully_in_box']}",
            "3 s stable-hold gate is mandatory before transport",
            "MuJoCo contact + friction + gravity; no weld or grasp teleport",
        ]
        writer.write(
            compose_four_panel(
                env.render("third_person"),
                observation["sim.wrist"],
                observation["observation.images.scene"],
                lines,
            )
        )

    return callback


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "configs/scene_pick_place_box.yaml")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--jitter-m", type=float, default=None)
    parser.add_argument("--output", type=Path, default=root / "outputs/pick_hold_place_box/summary.json")
    parser.add_argument("--report", type=Path, default=root / "reports/PICK_HOLD_PLACE_BOX_EVAL.json")
    parser.add_argument("--video-output", type=Path, default=root / "outputs/PICK_HOLD_PLACE_BOX_SUCCESS.mp4")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    dataset_root = (root / config["expert_source"]["dataset_root"]).resolve()
    source_episode = int(config["expert_source"]["source_episode"])
    source_actions = load_source_actions(dataset_root, source_episode)
    object_base = np.asarray(config["object"]["initial_xyz"], dtype=float)
    jitter = float(config["object"]["random_xy_m"] if args.jitter_m is None else args.jitter_m)
    rng = np.random.default_rng(args.seed)

    env = So101MujocoEnv(config_path)
    runner = PickPlaceBoxRunner(env, source_actions)
    trials = []
    for trial_index in range(args.trials):
        object_xyz = object_base.copy()
        object_xyz[:2] += rng.uniform(-jitter, jitter, size=2)
        result = runner.run(seed=args.seed + trial_index, object_xyz=object_xyz)
        result["trial_index"] = trial_index
        trials.append(result)
        print(
            json.dumps(
                {
                    "trial": trial_index,
                    "success": result["success"],
                    "stages": {name: row["success"] for name, row in result["stages"].items()},
                    "stable_hold_3s": result["quality_gates"]["stable_hold_3s"]["success"],
                }
            ),
            flush=True,
        )

    metrics = summarize_pick_place_trials(trials)
    video_result = None
    successful = next((trial for trial in trials if trial["success"]), None)
    if successful is not None and not args.no_video:
        args.video_output.parent.mkdir(parents=True, exist_ok=True)
        with H264Writer(args.video_output) as writer:
            title = provenance_card(
                "PICK - HOLD 3S - TRANSPORT - PLACE IN LARGE BOX",
                [
                    "Six measured stages: approach, grasp, lift, transport, place, retreat",
                    "Intermediate quality gate: stable hold for 3.0 seconds",
                    "Scripted reference controller; NOT ACT/VLA policy performance",
                    "Physical MuJoCo contacts; no weld and no object teleport after reset",
                ],
            )
            for _ in range(30):
                writer.write(title)
            runner.frame_callback = make_video_callback(env, writer)
            video_result = runner.run(
                seed=int(successful["seed"]),
                object_xyz=np.asarray(successful["object_initial_xyz"], dtype=float),
            )
            runner.frame_callback = None
        if not video_result["success"]:
            raise RuntimeError("successful trial did not reproduce while rendering video")

    scene_path = (root / config["scene_xml"]).resolve()
    summary = {
        "task": config["task_text"],
        "mode": "SCRIPTED_SIX_STAGE_TASK_SKELETON",
        "policy_model": False,
        "stages": ["approach", "grasp", "lift", "transport", "place", "retreat"],
        "intermediate_quality_gate": "stable_hold_3s",
        "uses_existing_local_real_action_dataset": True,
        "downloaded_new_dataset": False,
        "source_dataset": display_path(dataset_root, root.parent),
        "source_episode": source_episode,
        "object_jitter_m": jitter,
        "config": display_path(config_path, root.parent),
        "config_sha256": sha256(config_path),
        "scene_xml_sha256": sha256(scene_path),
        "metrics": metrics,
        "video": display_path(args.video_output, root.parent) if video_result is not None else None,
        "video_reproduction_success": bool(video_result and video_result["success"]),
        "results": trials,
    }
    for output in (args.output, args.report):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    env.close()
    print(
        json.dumps(
            {
                "full_successes": metrics["full_successes"],
                "trials": metrics["trials"],
                "full_success_rate": metrics["full_success_rate"],
                "stage_rates": {
                    name: row["success_rate"] for name, row in metrics["stage_metrics"].items()
                },
                "stable_hold_3s_rate": metrics["quality_gates"]["stable_hold_3s"]["success_rate"],
                "video": summary["video"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
