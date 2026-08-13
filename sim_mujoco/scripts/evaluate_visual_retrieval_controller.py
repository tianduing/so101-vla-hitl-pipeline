#!/usr/bin/env python3
"""Evaluate an RGB-only nearest-demonstration recovery controller.

The offline calibration stage may read simulator object positions, analogous to
camera calibration on hardware.  Evaluation-time selection uses RGB pixels only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_mujoco import So101MujocoEnv
from so101_mujoco.rendering.video import H264Writer, compose_four_panel, provenance_card


def green_centroid(image: np.ndarray) -> tuple[float, float]:
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    red, green, blue = (rgb[..., i].astype(np.float32) for i in range(3))
    ys, xs = np.nonzero((green > 80) & (green > 1.25 * red) & (green > 1.25 * blue))
    if len(xs) < 100:
        raise RuntimeError("green object detection failed")
    return float(xs.mean()), float(ys.mean())


def load_library(manifests: list[Path]) -> list[dict]:
    records = []
    for path in manifests:
        payload = json.loads(path.read_text())
        for row in payload["records"]:
            xy = row.get("object_xy_m", row.get("validated_object_xy_m"))
            records.append(
                {
                    "xy": np.asarray(xy, dtype=float),
                    "source_episode": int(row["source_episode"]),
                    "manifest": str(path.resolve()),
                }
            )
    return records


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=root.parent / "data/lerobot/local/so101_green_block_grasp_train_all_prompt_v2")
    parser.add_argument("--config", type=Path, default=root / "configs/scene.yaml")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--calibration-seeds", type=int, default=42)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-seed", type=int, default=None)
    parser.add_argument("--video-output", type=Path, default=None)
    args = parser.parse_args()

    source = LeRobotDataset("local/visual_retrieval_source", root=args.source.resolve())
    actions = {}
    for episode_index, episode in enumerate(source.meta.episodes):
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        actions[episode_index] = np.stack([source.hf_dataset[i]["action"].numpy() for i in range(start, end)])
    library = load_library(args.manifest)

    env = So101MujocoEnv(args.config)
    design, targets = [], []
    for seed in range(args.calibration_seeds):
        observation = env.reset(seed=seed)
        cx, cy = green_centroid(observation["observation.images.scene"])
        design.append([1.0, cx, cy])
        targets.append(env.get_object_pose()[:2])
    coefficients, *_ = np.linalg.lstsq(np.asarray(design), np.asarray(targets), rcond=None)
    calibration_prediction = np.asarray(design) @ coefficients
    calibration_mae_mm = float(np.abs(calibration_prediction - np.asarray(targets)).mean() * 1000)

    results = []
    for seed in range(args.seed, args.seed + args.trials):
        observation = env.reset(seed=seed)
        cx, cy = green_centroid(observation["observation.images.scene"])
        predicted_xy = np.asarray([1.0, cx, cy]) @ coefficients
        selected = min(library, key=lambda row: float(np.linalg.norm(row["xy"] - predicted_xy)))
        success_frames = 0
        peak_lift_m = 0.0
        writer = None
        if seed == args.video_seed:
            if args.video_output is None:
                raise ValueError("--video-output is required with --video-seed")
            args.video_output.parent.mkdir(parents=True, exist_ok=True)
            writer = H264Writer(args.video_output)
            writer.__enter__()
            card = provenance_card(
                "RGB VISUAL RETRIEVAL / RECOVERY CONTROLLER",
                [
                    "Evaluation-time selection uses camera RGB only",
                    "Selected action trajectory was physics-validated offline",
                    "Not ACT/VLA model output; no hidden ground-truth pose",
                    f"seed={seed} source_episode={selected['source_episode']}",
                ],
            )
            for _ in range(60):
                writer.write(card)
        try:
            for step, action in enumerate(actions[selected["source_episode"]]):
                observation, _, success, _, info = env.step(action)
                success_frames += int(success)
                peak_lift_m = max(peak_lift_m, float(info["object_xyz"][2]) - env.object_rest_z)
                if writer is not None:
                    lines = [
                        "RGB VISUAL RETRIEVAL / RECOVERY CONTROLLER",
                        "runtime signal: initial RGB green centroid only",
                        f"seed={seed} step={step} source_episode={selected['source_episode']}",
                        f"centroid=({cx:.1f}, {cy:.1f}) predicted_xy=({predicted_xy[0]:.4f}, {predicted_xy[1]:.4f})",
                        f"library_distance={np.linalg.norm(selected['xy'] - predicted_xy) * 1000:.2f} mm",
                        "action(deg): " + " ".join(f"{value:6.1f}" for value in action),
                        f"lift={float(info['object_xyz'][2]) - env.object_rest_z:.3f} m success={success}",
                        "not ACT/VLA; physics contacts, gravity and friction enabled",
                    ]
                    writer.write(
                        compose_four_panel(
                            env.render("third_person"),
                            observation["sim.wrist"],
                            observation["observation.images.scene"],
                            lines,
                        )
                    )
        finally:
            if writer is not None:
                writer.__exit__(None, None, None)
        result = {
            "seed": seed,
            "success": bool(success_frames),
            "success_frames": success_frames,
            "peak_lift_m": peak_lift_m,
            "rgb_centroid": [cx, cy],
            "predicted_xy_m": predicted_xy.tolist(),
            "selected_xy_m": selected["xy"].tolist(),
            "selected_source_episode": selected["source_episode"],
            "selection_distance_mm": float(np.linalg.norm(selected["xy"] - predicted_xy) * 1000),
            "video": str(args.video_output.resolve()) if seed == args.video_seed else None,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = {
        "mode": "RGB_VISUAL_DEMONSTRATION_RETRIEVAL",
        "policy_model": False,
        "uses_privileged_state_during_evaluation": False,
        "offline_calibration_uses_simulator_positions": True,
        "calibration_seeds": args.calibration_seeds,
        "calibration_mae_mm": calibration_mae_mm,
        "coefficients": coefficients.tolist(),
        "library_records": len(library),
        "trials": len(results),
        "successes": sum(int(row["success"]) for row in results),
        "success_rate": sum(int(row["success"]) for row in results) / len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    env.close()
    print(json.dumps({key: summary[key] for key in ("successes", "trials", "success_rate", "calibration_mae_mm")}, indent=2))


if __name__ == "__main__":
    main()
