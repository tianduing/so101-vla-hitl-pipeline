#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from so101_mujoco import So101MujocoEnv
from so101_mujoco.rendering.video import H264Writer, draw_lines, provenance_card


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/gripper_mapping_test.mp4"
    env = So101MujocoEnv(root / "configs/scene.yaml")
    env.reset(seed=42)
    base = np.asarray(env.config["initial_state_degrees"], dtype=float)
    commands = np.concatenate([np.linspace(65, 40, 90), np.full(40, 40), np.linspace(40, 65, 90), np.full(40, 65)])
    records = []
    with H264Writer(output) as writer:
        card = provenance_card("GRIPPER MAPPING TEST", ["Raw calibrated degrees -> offset -> MuJoCo radians", "Increasing raw command increases MuJoCo joint angle", "NOT POLICY / NOT REAL ROBOT"])
        for _ in range(60): writer.write(card)
        for step, command in enumerate(commands):
            action = base.copy(); action[-1] = command
            _, _, _, _, info = env.step(action)
            frame = env.render("third_person")
            frame = __import__("cv2").resize(frame, (1280, 720))
            joint_angle = float(env.data.qpos[env.qpos_addrs[-1]])
            lines = [
                "GRIPPER MAPPING / NOT POLICY",
                f"step={step} raw command={command:.2f} deg",
                f"mapped target={env.data.ctrl[env.actuator_ids[-1]]:.4f} rad",
                f"actual gripper joint={joint_angle:.4f} rad",
                "Direction: larger raw value -> larger MJCF angle",
            ]
            writer.write(draw_lines(frame, lines, scale=0.66))
            records.append({"step": step, "raw_degrees": float(command), "ctrl_radians": float(env.data.ctrl[env.actuator_ids[-1]]), "joint_radians": joint_angle})
    env.close()
    output.with_suffix(".json").write_text(json.dumps({"mode": "GRIPPER_MAPPING_TEST", "policy": False, "records": records}, indent=2) + "\n")
    print(output)


if __name__ == "__main__": main()
