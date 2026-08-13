from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from typing import Any

import mujoco
import numpy as np

from so101_mujoco.envs.so101_env import So101MujocoEnv


PICK_PLACE_STAGES = ("approach", "grasp", "lift", "transport", "place", "retreat")
FrameCallback = Callable[[str, str, int, np.ndarray, dict[str, np.ndarray], dict[str, Any]], None]
TransitionCallback = Callable[[str, str, int, np.ndarray, dict[str, np.ndarray]], None]


def bilateral_object_contact(env: So101MujocoEnv) -> tuple[bool, bool, bool]:
    """Return fixed-jaw, moving-jaw and bilateral object contact flags."""
    fixed = False
    moving = False
    for contact in env.data.contact:
        names = (
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or "",
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or "",
        )
        if "object_geom" not in names:
            continue
        other = names[1] if names[0] == "object_geom" else names[0]
        fixed = fixed or other.startswith("fixed_jaw")
        moving = moving or other.startswith("moving_jaw")
    return fixed, moving, fixed and moving


def object_fully_in_box(env: So101MujocoEnv) -> bool:
    xyz = env.get_object_pose()[:3]
    half_extent = np.asarray(env.config["target"]["half_extent_xy_m"], dtype=float)
    center = np.asarray(env.config["target"]["center_xyz"], dtype=float)
    low, high = (float(value) for value in env.config["success"]["final_height_range_m"])
    return bool(np.all(np.abs(xyz[:2] - center[:2]) <= half_extent) and low <= xyz[2] <= high)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials == 0:
        return [0.0, 0.0]
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = z * sqrt((proportion * (1 - proportion) + z * z / (4 * trials)) / trials) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize_pick_place_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(trials)
    stage_metrics: dict[str, Any] = {}
    prior_pass = [True] * total
    for stage in PICK_PLACE_STAGES:
        successes = sum(int(trial["stages"][stage]["success"]) for trial in trials)
        eligible = sum(prior_pass)
        conditional_successes = sum(
            int(previous and trial["stages"][stage]["success"])
            for previous, trial in zip(prior_pass, trials, strict=True)
        )
        cumulative = [
            previous and bool(trial["stages"][stage]["success"])
            for previous, trial in zip(prior_pass, trials, strict=True)
        ]
        stage_metrics[stage] = {
            "successes": successes,
            "trials": total,
            "success_rate": successes / total if total else 0.0,
            "wilson_95": wilson_interval(successes, total),
            "conditional_successes": conditional_successes,
            "conditional_trials": eligible,
            "conditional_success_rate": conditional_successes / eligible if eligible else 0.0,
            "cumulative_pipeline_successes": sum(cumulative),
            "cumulative_pipeline_success_rate": sum(cumulative) / total if total else 0.0,
        }
        prior_pass = cumulative
        if stage == "lift":
            hold_passes = [bool(trial["quality_gates"]["stable_hold_3s"]["success"]) for trial in trials]
            prior_pass = [
                previous and hold
                for previous, hold in zip(prior_pass, hold_passes, strict=True)
            ]

    hold_successes = sum(int(trial["quality_gates"]["stable_hold_3s"]["success"]) for trial in trials)
    full_successes = sum(int(trial["success"]) for trial in trials)
    return {
        "trials": total,
        "full_successes": full_successes,
        "full_success_rate": full_successes / total if total else 0.0,
        "full_success_wilson_95": wilson_interval(full_successes, total),
        "stage_metrics": stage_metrics,
        "quality_gates": {
            "stable_hold_3s": {
                "successes": hold_successes,
                "trials": total,
                "success_rate": hold_successes / total if total else 0.0,
                "wilson_95": wilson_interval(hold_successes, total),
            }
        },
    }


@dataclass
class PickPlaceBoxRunner:
    env: So101MujocoEnv
    source_actions: np.ndarray
    frame_callback: FrameCallback | None = None
    transition_callback: TransitionCallback | None = None

    def __post_init__(self) -> None:
        self.site_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        if self.site_id < 0:
            raise ValueError("gripperframe site is missing")
        source = self.env.config["expert_source"]
        self.approach_end = int(source["approach_end_index"])
        self.grasp_end = int(source["grasp_end_index"])
        self.lift_end = int(source["lift_end_index"])
        if not (0 <= self.approach_end < self.grasp_end < self.lift_end < len(self.source_actions)):
            raise ValueError("invalid expert stage boundaries")
        self.global_step = 0
        self.initial_z = 0.0

    def _signals(self, info: dict[str, Any]) -> dict[str, Any]:
        fixed, moving, bilateral = bilateral_object_contact(self.env)
        object_xyz = np.asarray(info["object_xyz"], dtype=float)
        site_xyz = self.env.data.site_xpos[self.site_id].copy()
        return {
            "object_xyz": object_xyz.tolist(),
            "site_xyz": site_xyz.tolist(),
            "lift_m": float(object_xyz[2] - self.initial_z),
            "object_speed_m_s": float(info["object_speed"]),
            "tip_object_distance_m": float(np.linalg.norm(site_xyz - object_xyz)),
            "fixed_jaw_contact": fixed,
            "moving_jaw_contact": moving,
            "bilateral_contact": bilateral,
            "object_fully_in_box": object_fully_in_box(self.env),
        }

    def _step(self, stage: str, subphase: str, action: np.ndarray) -> dict[str, Any]:
        if self.transition_callback is not None:
            observation_before = self.env.get_observation()
            self.transition_callback(
                stage,
                subphase,
                self.global_step,
                np.asarray(action, dtype=float).copy(),
                observation_before,
            )
        observation, _, _, _, info = self.env.step(action)
        signals = self._signals(info)
        merged = {**info, **signals}
        if self.frame_callback is not None:
            self.frame_callback(
                stage,
                subphase,
                self.global_step,
                np.asarray(action, dtype=float).copy(),
                observation,
                merged,
            )
        self.global_step += 1
        return merged

    def _move_site(
        self,
        target: np.ndarray,
        frames: int,
        gripper_degrees: float,
        stage: str,
        subphase: str,
    ) -> list[dict[str, Any]]:
        controller = self.env.config["controller"]
        start = self.env.data.site_xpos[self.site_id].copy()
        posture_reference = self.env.data.qpos[self.env.qpos_addrs[:5]].copy()
        records = []
        for frame_index in range(frames):
            fraction = (frame_index + 1) / frames
            smooth = fraction * fraction * (3 - 2 * fraction)
            desired = start + smooth * (target - start)
            jacp = np.zeros((3, self.env.model.nv))
            jacr = np.zeros((3, self.env.model.nv))
            mujoco.mj_jacSite(self.env.model, self.env.data, jacp, jacr, self.site_id)
            columns = self.env.model.jnt_dofadr[self.env.joint_ids[:5]]
            jacobian = jacp[:, columns]
            damping = float(controller["ik_damping"])
            damped_pinv = jacobian.T @ np.linalg.inv(
                jacobian @ jacobian.T + damping * np.eye(3)
            )
            qpos = self.env.data.qpos[self.env.qpos_addrs[:5]]
            delta = damped_pinv @ (
                float(controller["ik_gain"]) * (desired - self.env.data.site_xpos[self.site_id])
            )
            nullspace = np.eye(5) - damped_pinv @ jacobian
            delta += nullspace @ (
                float(controller["nullspace_gain"]) * (posture_reference - qpos)
            )
            max_step = float(controller["ik_max_step_rad"])
            target_qpos = np.clip(
                qpos + np.clip(delta, -max_step, max_step),
                self.env.joint_ranges[:5, 0],
                self.env.joint_ranges[:5, 1],
            )
            action = self.env.mujoco_to_raw(
                np.r_[target_qpos, self.env.data.ctrl[self.env.actuator_ids[-1]]]
            )
            action[-1] = gripper_degrees
            records.append(self._step(stage, subphase, action))
        return records

    def run(self, seed: int, object_xyz: np.ndarray) -> dict[str, Any]:
        self.env.reset(seed=seed, object_pose=np.asarray(object_xyz, dtype=float))
        self.global_step = 0
        self.initial_z = float(self.env.object_rest_z)
        cfg = self.env.config
        success_cfg = cfg["success"]
        controller = cfg["controller"]
        target = cfg["target"]
        stages: dict[str, dict[str, Any]] = {}

        approach_records = [
            self._step("approach", "source_trajectory", action)
            for action in self.source_actions[: self.approach_end + 1]
        ]
        approach_min = min(record["tip_object_distance_m"] for record in approach_records)
        stages["approach"] = {
            "success": approach_min <= float(success_cfg["approach_tip_distance_m"]),
            "minimum_tip_object_distance_m": approach_min,
            "threshold_m": float(success_cfg["approach_tip_distance_m"]),
            "steps": len(approach_records),
        }

        grasp_records = [
            self._step("grasp", "source_trajectory", action)
            for action in self.source_actions[self.approach_end + 1 : self.grasp_end + 1]
        ]
        consecutive = 0
        max_consecutive = 0
        bilateral_frames = 0
        for record in grasp_records:
            bilateral_frames += int(record["bilateral_contact"])
            consecutive = consecutive + 1 if record["bilateral_contact"] else 0
            max_consecutive = max(max_consecutive, consecutive)
        required_contacts = int(success_cfg["bilateral_contact_frames"])
        stages["grasp"] = {
            "success": max_consecutive >= required_contacts,
            "bilateral_contact_frames": bilateral_frames,
            "max_consecutive_bilateral_contact_frames": max_consecutive,
            "required_consecutive_frames": required_contacts,
            "steps": len(grasp_records),
        }

        lift_records = [
            self._step("lift", "source_trajectory", action)
            for action in self.source_actions[self.grasp_end + 1 : self.lift_end + 1]
        ]
        peak_lift = max(record["lift_m"] for record in lift_records)
        end_lift = lift_records[-1]["lift_m"]
        lift_threshold = float(success_cfg["pick_lift_m"])
        stages["lift"] = {
            "success": end_lift >= lift_threshold,
            "peak_lift_m": peak_lift,
            "end_lift_m": end_lift,
            "threshold_m": lift_threshold,
            "steps": len(lift_records),
        }

        hold_action = np.asarray(self.env.get_state()["state_deg"], dtype=float)
        hold_action[-1] = float(controller["hold_gripper_degrees"])
        hold_steps = round(float(success_cfg["hold_seconds"]) * self.env.control_hz)
        hold_records = [
            self._step("lift", "stable_hold_3s", hold_action) for _ in range(hold_steps)
        ]
        hold_min_lift = min(record["lift_m"] for record in hold_records)
        hold_success = len(hold_records) == hold_steps and hold_min_lift >= lift_threshold
        quality_gates = {
            "stable_hold_3s": {
                "success": hold_success,
                "duration_seconds": hold_steps / self.env.control_hz,
                "required_seconds": float(success_cfg["hold_seconds"]),
                "minimum_lift_m": hold_min_lift,
                "required_lift_m": lift_threshold,
                "bilateral_contact_fraction": float(
                    np.mean([record["bilateral_contact"] for record in hold_records])
                ),
            }
        }

        transport_records = self._move_site(
            np.asarray(target["transport_site_xyz"], dtype=float),
            int(controller["transport_frames"]),
            float(controller["hold_gripper_degrees"]),
            "transport",
            "move_above_box",
        )
        transport_min_lift = min(record["lift_m"] for record in transport_records)
        transport_end_xyz = np.asarray(transport_records[-1]["object_xyz"], dtype=float)
        target_xy = np.asarray(target["center_xyz"], dtype=float)[:2]
        transport_xy_error = float(np.linalg.norm(transport_end_xyz[:2] - target_xy))
        arrival_tolerance = float(min(target["half_extent_xy_m"]))
        stages["transport"] = {
            "success": bool(
                transport_min_lift >= float(success_cfg["transport_min_lift_m"])
                and transport_xy_error <= arrival_tolerance
            ),
            "minimum_lift_m": transport_min_lift,
            "required_minimum_lift_m": float(success_cfg["transport_min_lift_m"]),
            "final_xy_error_m": transport_xy_error,
            "arrival_tolerance_m": arrival_tolerance,
            "steps": len(transport_records),
        }

        lower_records = self._move_site(
            np.asarray(target["release_site_xyz"], dtype=float),
            int(controller["lower_frames"]),
            float(controller["hold_gripper_degrees"]),
            "place",
            "lower_above_box",
        )
        arm_hold = np.asarray(self.env.get_state()["state_deg"], dtype=float)
        closed = float(controller["hold_gripper_degrees"])
        opened = float(controller["open_gripper_degrees"])
        release_records = []
        release_frames = int(controller["release_frames"])
        for frame_index in range(release_frames):
            fraction = min(1.0, (frame_index + 1) / max(1, release_frames - 15))
            smooth = fraction * fraction * (3 - 2 * fraction)
            action = arm_hold.copy()
            action[-1] = closed + smooth * (opened - closed)
            release_records.append(self._step("place", "release", action))
        settle_action = arm_hold.copy()
        settle_action[-1] = opened
        settle_records = [
            self._step("place", "settle_in_box", settle_action)
            for _ in range(int(controller["settle_frames"]))
        ]
        place_final = settle_records[-1]
        place_stable = place_final["object_speed_m_s"] <= float(success_cfg["place_max_speed_m_s"])
        stages["place"] = {
            "success": bool(place_final["object_fully_in_box"] and place_stable),
            "object_fully_in_box": bool(place_final["object_fully_in_box"]),
            "final_object_speed_m_s": float(place_final["object_speed_m_s"]),
            "max_speed_m_s": float(success_cfg["place_max_speed_m_s"]),
            "final_object_xyz": place_final["object_xyz"],
            "steps": len(lower_records) + len(release_records) + len(settle_records),
        }

        center_xy = np.asarray(target["center_xyz"], dtype=float)[:2]
        up_target = np.asarray(
            [center_xy[0], center_xy[1], target["retreat_site_xyz"][2]], dtype=float
        )
        retreat_records = self._move_site(
            up_target,
            int(controller["retreat_up_frames"]),
            opened,
            "retreat",
            "clear_box_vertically",
        )
        retreat_records += self._move_site(
            np.asarray(target["retreat_site_xyz"], dtype=float),
            int(controller["retreat_away_frames"]),
            opened,
            "retreat",
            "move_away",
        )
        retreat_final = retreat_records[-1]
        final_object = np.asarray(retreat_final["object_xyz"], dtype=float)
        final_site = np.asarray(retreat_final["site_xyz"], dtype=float)
        clearance = float(np.linalg.norm(final_site - final_object))
        remained_in_box = all(record["object_fully_in_box"] for record in retreat_records)
        retreat_stable = retreat_final["object_speed_m_s"] <= float(success_cfg["place_max_speed_m_s"])
        stages["retreat"] = {
            "success": bool(
                remained_in_box
                and retreat_stable
                and clearance >= float(success_cfg["retreat_clearance_m"])
            ),
            "object_remained_in_box": remained_in_box,
            "final_object_speed_m_s": float(retreat_final["object_speed_m_s"]),
            "final_gripper_object_clearance_m": clearance,
            "required_clearance_m": float(success_cfg["retreat_clearance_m"]),
            "steps": len(retreat_records),
        }

        stage_success = all(bool(stages[name]["success"]) for name in PICK_PLACE_STAGES)
        success = bool(stage_success and hold_success and self.env.compute_success())
        return {
            "seed": seed,
            "success": success,
            "mode": "SCRIPTED_PICK_HOLD_PLACE_BOX_EXPERT",
            "policy_model": False,
            "uses_ground_truth_for_reference_controller": True,
            "object_teleport_after_reset": False,
            "weld_or_constraint_grasp": False,
            "source_episode": int(cfg["expert_source"]["source_episode"]),
            "object_initial_xyz": np.asarray(object_xyz, dtype=float).tolist(),
            "stages": stages,
            "quality_gates": quality_gates,
            "environment_success_detector": bool(self.env.compute_success()),
            "final_object_pose": self.env.get_object_pose().tolist(),
            "total_steps": self.global_step,
            "duration_seconds": self.global_step / self.env.control_hz,
        }
