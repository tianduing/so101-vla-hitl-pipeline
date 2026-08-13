#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import subprocess
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_mujoco import So101MujocoEnv
from so101_mujoco.rendering.video import H264Writer, compose_four_panel, provenance_card


def rgb_u8(tensor) -> np.ndarray:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(array * 255, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset-root", type=Path, default=root.parent / "data/lerobot/local/so101_green_block_grasp_train_all_prompt_v2")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=root / "outputs/real_trajectory_replay.mp4")
    parser.add_argument("--config", type=Path, default=root / "configs/scene.yaml")
    args = parser.parse_args()

    dataset = LeRobotDataset("local/so101_mujoco_replay", root=args.dataset_root)
    episode = dataset.meta.episodes[args.episode]
    start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
    task = str(dataset[start]["task"])
    env = So101MujocoEnv(args.config)
    env.reset(seed=42)
    logs = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with H264Writer(args.output) as writer:
        card = provenance_card("REAL TRAJECTORY REPLAY", ["Real SO-101 joint state trajectory", "MuJoCo object follows simulation physics only", "NOT CLOSED-LOOP POLICY"])
        for _ in range(60): writer.write(card)
        for local_index, index in enumerate(range(start, end)):
            sample = dataset[index]
            state = sample["observation.state"].detach().cpu().numpy()
            action = sample["action"].detach().cpu().numpy()
            obs, _, _, _, info = env.step(state)
            real_camera = rgb_u8(sample[dataset.meta.camera_keys[0]])
            third, wrist = env.render("third_person"), env.render("wrist_cam")
            lines = [
                "REAL TRAJECTORY REPLAY / NOT CLOSED-LOOP POLICY",
                f"episode={args.episode} frame={local_index} sim={info['sim_time']:.2f}s",
                f"instruction: {task}",
                "state(deg): " + " ".join(f"{x:6.1f}" for x in state),
                "recorded action: " + " ".join(f"{x:6.1f}" for x in action),
                "mapped ctrl(rad): " + " ".join(f"{x:5.2f}" for x in info["ctrl_rad"]),
                "object xyz(sim only): " + " ".join(f"{x:.3f}" for x in info["object_xyz"]),
                "No real object 6D pose was recorded.",
                "Robot joints=real trajectory; object=sim result.",
            ]
            frame = compose_four_panel(third, wrist, real_camera, lines, labels=("SIM THIRD PERSON", "SIM WRIST", "REAL CAMERA / DATASET"))
            writer.write(frame)
            logs.append({"dataset_index": index, "frame": local_index, "timestamp": float(sample["timestamp"]), "state_raw": state.tolist(), "action_raw": action.tolist(), **info})
    env.close()
    metadata = {
        "mode": "REAL_TRAJECTORY_REPLAY", "closed_loop_policy": False, "episode": args.episode, "dataset_root": str(args.dataset_root.resolve()),
        "frames": end - start, "fps": dataset.meta.fps, "task": task, "replay_source": "observation.state",
        "object_pose_provenance": "MuJoCo physics only; no real object pose available", "output": str(args.output.resolve()), "steps": logs,
    }
    meta_path = args.output.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    structured = args.output.parent / "real_trajectory_replay"
    structured.mkdir(parents=True, exist_ok=True)
    (structured / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    four_panel = structured / "replay_four_panel.mp4"
    if four_panel.exists() or four_panel.is_symlink(): four_panel.unlink()
    four_panel.symlink_to(Path("../") / args.output.name)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(args.output),
        "-vf", "crop=640:360:0:0,scale=1280:720", "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        str(structured / "replay_raw.mp4"),
    ], check=True)
    print(json.dumps({k: v for k, v in metadata.items() if k != "steps"}, indent=2))


if __name__ == "__main__":
    main()
