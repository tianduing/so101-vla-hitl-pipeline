#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from so101_mujoco import So101MujocoEnv
from so101_mujoco.rendering.video import H264Writer, compose_four_panel, provenance_card


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "configs/scene.yaml")
    parser.add_argument("--output", type=Path, default=root / "outputs/scripted_expert_reference.mp4")
    args = parser.parse_args()
    env = So101MujocoEnv(args.config)
    # Ground-truth placement is explicitly allowed only for the scripted reference.
    env.reset(seed=42, object_pose=np.array([0.219, 0.024, 0.041]))
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    logs = []
    phases: list[tuple[str, int, np.ndarray | None, float | None]] = [
        ("close_on_ground_truth_object", 90, None, 13.6),
        ("lift", 180, np.array([0.22, 0.024, 0.15]), None),
        ("move_above_target", 360, np.array([0.22, 0.20, 0.15]), None),
        ("release", 90, None, 65.3),
        ("settle", 120, None, 65.3),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with H264Writer(args.output) as writer:
        card = provenance_card("SCRIPTED EXPERT / NOT POLICY", ["Uses ground-truth object and target pose", "Same MuJoCo contacts and success detector as Policy", "No weld, teleport during grasp, or hidden controller fallback"])
        for _ in range(60): writer.write(card)
        phase_start_gripper = float(env.get_state()["state_deg"][-1])
        held_arm_raw = np.asarray(env.get_state()["state_deg"][:5], dtype=float)
        global_step = 0
        for phase, frames, target, gripper_goal in phases:
            cartesian_start = env.data.site_xpos[site_id].copy()
            posture_reference = env.data.qpos[env.qpos_addrs[:5]].copy()
            for frame_in_phase in range(frames):
                raw_action = np.r_[held_arm_raw, phase_start_gripper]
                if target is not None:
                    fraction = (frame_in_phase + 1) / frames
                    smooth = fraction * fraction * (3 - 2 * fraction)
                    desired = cartesian_start + smooth * (target - cartesian_start)
                    jacp = np.zeros((3, env.model.nv)); jacr = np.zeros((3, env.model.nv))
                    mujoco.mj_jacSite(env.model, env.data, jacp, jacr, site_id)
                    columns = env.model.jnt_dofadr[env.joint_ids[:5]]
                    jacobian = jacp[:, columns]
                    damped_pinv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + 3e-3 * np.eye(3))
                    q = env.data.qpos[env.qpos_addrs[:5]]
                    dq = damped_pinv @ (1.2 * (desired - env.data.site_xpos[site_id]))
                    # Position-only IK has two unconstrained arm DoFs. Penalize
                    # null-space drift so it cannot jump between equivalent poses.
                    nullspace = np.eye(5) - damped_pinv @ jacobian
                    dq += nullspace @ (0.03 * (posture_reference - q))
                    target_q = np.clip(q + np.clip(dq, -0.008, 0.008), env.joint_ranges[:5, 0], env.joint_ranges[:5, 1])
                    raw_action[:5] = env.mujoco_to_raw(np.r_[target_q, env.data.ctrl[env.actuator_ids[-1]]])[:5]
                    raw_action[-1] = 13.6
                if gripper_goal is not None:
                    fraction = min(1.0, (frame_in_phase + 1) / 60)
                    smooth = fraction * fraction * (3 - 2 * fraction)
                    raw_action[-1] = phase_start_gripper + smooth * (gripper_goal - phase_start_gripper)
                obs, _, success, _, info = env.step(raw_action)
                lines = [
                    "SCRIPTED EXPERT / NOT POLICY",
                    "uses ground-truth object pose for IK",
                    f"phase={phase} step={global_step} sim={info['sim_time']:.2f}s",
                    "action(deg): " + " ".join(f"{x:6.1f}" for x in raw_action),
                    "object xyz: " + " ".join(f"{x:.3f}" for x in info["object_xyz"]),
                    f"picked={env.ever_picked} success={success} contacts={info['contacts']}",
                    "Smooth DLS IK + null-space posture hold",
                    "Physics: contact + friction + gravity; no weld",
                ]
                writer.write(compose_four_panel(env.render("third_person"), obs["sim.wrist"], obs["observation.images.scene"], lines))
                logs.append({"step": global_step, "phase": phase, "raw_action": raw_action.tolist(), **info, "success": success})
                global_step += 1
            phase_start_gripper = float(env.get_state()["state_deg"][-1])
            held_arm_raw = env.mujoco_to_raw(env.data.ctrl[env.actuator_ids])[:5]
    final_xyz = env.get_object_pose()[:3]
    target_half = np.asarray(env.config["target"]["half_extent_xy_m"])
    pick_success = bool(env.ever_picked)
    place_success = bool(np.all(np.abs(final_xyz[:2] - env.target[:2]) <= target_half) and 0.025 <= final_xyz[2] <= 0.065)
    final_success = pick_success and place_success
    result = {
        "mode": "SCRIPTED_EXPERT", "policy": False, "uses_ground_truth_object_pose": True,
        "weld_or_teleport_during_grasp": False, "success": final_success,
        "pick_success": pick_success, "place_success": place_success, "controller": "smooth_dls_with_nullspace_posture_hold",
        "steps": len(logs),
        "final_object_pose": env.get_object_pose().tolist(), "output": str(args.output.resolve()), "step_log": logs,
    }
    env.close()
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "step_log"}, indent=2))


if __name__ == "__main__":
    main()
