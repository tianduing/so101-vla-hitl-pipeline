#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from so101_mujoco.rendering.video import H264Writer, draw_lines, provenance_card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    root = args.root.resolve()
    xml = root / "vendor/mujoco_menagerie/robotstudio_so101/scene_box.xml"
    output = args.output or root / "outputs/scene_smoke.mp4"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    # Keep the official vendor XML unchanged; its offscreen framebuffer is 640x480.
    renderer = mujoco.Renderer(model, width=640, height=480)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.22, 0, 0.12]
    camera.distance = 0.75
    camera.azimuth = 145
    camera.elevation = -24
    frames: list[np.ndarray] = []
    means, variances, diffs = [], [], []
    frame_count = int(args.seconds * 30)
    steps_per_frame = round((1 / 30) / model.opt.timestep)
    with H264Writer(output) as writer:
        card = provenance_card("SCENE_SMOKE", ["Official Menagerie robotstudio_so101/scene_box.xml", "Real MuJoCo physics; smooth shoulder_pan target", "NOT POLICY / NOT REAL ROBOT"])
        for _ in range(60):
            writer.write(card)
        previous = None
        for frame_id in range(frame_count):
            target = 0.55 * np.sin(2 * np.pi * frame_id / frame_count)
            low, high = model.actuator_ctrlrange[0]
            data.ctrl[0] = np.clip(target, low, high)
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise FloatingPointError("non-finite qpos")
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
            rgb = cv2.resize(rgb, (1280, 720), interpolation=cv2.INTER_CUBIC)
            rgb = draw_lines(rgb, ["SCENE_SMOKE / OFFICIAL SCENE_BOX", f"sim={data.time:.2f}s  shoulder_pan={np.rad2deg(data.qpos[0]):.1f} deg", "NOT POLICY / NOT REAL ROBOT"], scale=0.62)
            means.append(float(rgb.mean())); variances.append(float(rgb.var()))
            if previous is not None:
                diffs.append(float(np.mean(np.abs(rgb.astype(np.int16) - previous.astype(np.int16)))))
            previous = rgb
            if frame_id in (0, frame_count // 2, frame_count - 1):
                frame_path = output.with_name(f"scene_smoke_frame_{frame_id:03d}.png")
                cv2.imwrite(str(frame_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                frames.append(rgb)
            writer.write(rgb)
    renderer.close()
    metrics = {
        "status": "PASS" if min(variances) > 1 and max(diffs) > 0.2 else "FAIL",
        "mujoco_version": mujoco.__version__, "xml": str(xml), "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "timestep": model.opt.timestep, "physics_seconds": args.seconds, "encoded_seconds": args.seconds + 2,
        "frame_mean_range": [min(means), max(means)], "frame_variance_min": min(variances),
        "adjacent_frame_diff_mean": float(np.mean(diffs)), "adjacent_frame_diff_max": max(diffs),
        "output": str(output),
    }
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
