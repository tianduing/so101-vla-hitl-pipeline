from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class So101MujocoEnv:
    """Headless MuJoCo environment. Actions/states use calibrated degrees."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.root = self.config_path.parents[1]
        self.config: dict[str, Any] = yaml.safe_load(self.config_path.read_text())
        mapping = yaml.safe_load((self.root / "configs/joint_mapping.yaml").read_text())
        self.offset_deg = np.asarray([mapping["joints"][name]["offset_deg"] for name in JOINT_NAMES])
        self.sign = np.asarray([mapping["joints"][name]["sign"] for name in JOINT_NAMES])
        self.raw_scale = np.asarray([mapping["joints"][name].get("raw_scale", 1.0) for name in JOINT_NAMES])
        xml_path = (self.root / self.config["scene_xml"]).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        if bool(self.config.get("physics_match", {}).get("disable_camera_mount_collisions", False)):
            for name in ("camera_box1", "camera_box2"):
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if geom_id >= 0:
                    self.model.geom_contype[geom_id] = 0
                    self.model.geom_conaffinity[geom_id] = 0
        if bool(self.config.get("visual_match", {}).get("robot_white", False)):
            for material_id in range(self.model.nmat):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id) or ""
                if name.endswith("_material") and "sts3215" not in name:
                    self.model.mat_rgba[material_id] = [0.92, 0.92, 0.90, 1.0]
        self.data = mujoco.MjData(self.model)
        self.control_hz = float(self.config["control_hz"])
        self.substeps = max(1, round((1 / self.control_hz) / self.model.opt.timestep))
        self.joint_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES])
        self.actuator_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES])
        if np.any(self.joint_ids < 0) or np.any(self.actuator_ids < 0):
            raise ValueError("SO-101 joint/actuator schema mismatch")
        self.qpos_addrs = self.model.jnt_qposadr[self.joint_ids]
        self.ctrl_ranges = self.model.actuator_ctrlrange[self.actuator_ids].copy()
        self.joint_ranges = self.model.jnt_range[self.joint_ids].copy()
        self.object_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.object_qpos_addr = int(self.model.jnt_qposadr[self.object_joint])
        self.target = np.asarray(self.config["target"]["center_xyz"], dtype=np.float64)
        self.renderer = mujoco.Renderer(
            self.model,
            height=int(self.config["render_height"]),
            width=int(self.config["render_width"]),
        )
        self.last_raw_action = np.asarray(self.config["initial_state_degrees"], dtype=np.float64)
        self.ever_picked = False
        self.pick_hold_steps = 0
        self.reset(seed=int(self.config.get("seed", 42)))

    def reset(self, seed: int = 42, object_pose: np.ndarray | None = None) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        raw = np.asarray(self.config["initial_state_degrees"], dtype=np.float64)
        joint_rad = self.raw_to_mujoco(raw, rate_limit=False)
        self.data.qpos[self.qpos_addrs] = joint_rad
        self.data.ctrl[self.actuator_ids] = joint_rad
        xyz = np.asarray(self.config["object"]["initial_xyz"], dtype=np.float64).copy()
        if object_pose is not None:
            xyz[:] = np.asarray(object_pose, dtype=np.float64)[:3]
        else:
            spread = float(self.config["object"].get("random_xy_m", 0))
            xyz[:2] += rng.uniform(-spread, spread, 2)
        self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 7] = [*xyz, 1, 0, 0, 0]
        self.last_raw_action = raw.copy()
        self.ever_picked = False
        self.pick_hold_steps = 0
        mujoco.mj_forward(self.model, self.data)
        self.object_rest_z = float(self.get_object_pose()[2])
        return self.get_observation()

    def raw_to_mujoco(self, raw_action: np.ndarray, rate_limit: bool = True) -> np.ndarray:
        raw = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError(f"invalid 6D action: {raw}")
        if rate_limit:
            delta = float(self.config["limits"]["max_raw_action_delta_deg"])
            raw = np.clip(raw, self.last_raw_action - delta, self.last_raw_action + delta)
        radians = np.deg2rad(self.sign * self.raw_scale * raw + self.offset_deg)
        return np.clip(radians, self.ctrl_ranges[:, 0], self.ctrl_ranges[:, 1])

    def mujoco_to_raw(self, radians: np.ndarray) -> np.ndarray:
        return (np.rad2deg(np.asarray(radians, dtype=np.float64)) - self.offset_deg) / (self.sign * self.raw_scale)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        ctrl = self.raw_to_mujoco(action)
        self.data.ctrl[self.actuator_ids] = ctrl
        self.last_raw_action = self.mujoco_to_raw(ctrl)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo state became non-finite")
        success = self.compute_success()
        info = self.get_state()
        info["success"] = success
        return self.get_observation(), float(success), success, False, info

    def render(self, camera_name: str = "third_person") -> np.ndarray:
        scene_option = None
        if camera_name == "front" and self.config.get("visual_match", {}).get("hide_policy_group_4", False):
            scene_option = mujoco.MjvOption()
            scene_option.geomgroup[4] = 0
            scene_option.sitegroup[4] = 0
        self.renderer.update_scene(self.data, camera=camera_name, scene_option=scene_option)
        return self.renderer.render().copy()

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "observation.state": self.mujoco_to_raw(self.data.qpos[self.qpos_addrs]).astype(np.float32),
            "observation.images.scene": self.render("front"),
            "sim.wrist": self.render("wrist_cam"),
        }

    def get_state(self) -> dict[str, Any]:
        object_xyz = self.get_object_pose()[:3]
        object_vel = self.data.cvel[self.object_body, 3:].copy()
        contact_forces = []
        wrench = np.zeros(6)
        for contact_id in range(self.data.ncon):
            mujoco.mj_contactForce(self.model, self.data, contact_id, wrench)
            contact_forces.append(float(np.linalg.norm(wrench[:3])))
        ctrl = self.data.ctrl[self.actuator_ids]
        tolerance = 1e-5
        saturated = np.isclose(ctrl, self.ctrl_ranges[:, 0], atol=tolerance) | np.isclose(ctrl, self.ctrl_ranges[:, 1], atol=tolerance)
        return {
            "sim_time": float(self.data.time),
            "qpos_rad": self.data.qpos[self.qpos_addrs].astype(float).tolist(),
            "state_deg": self.mujoco_to_raw(self.data.qpos[self.qpos_addrs]).astype(float).tolist(),
            "ctrl_rad": self.data.ctrl[self.actuator_ids].astype(float).tolist(),
            "object_xyz": object_xyz.astype(float).tolist(),
            "object_speed": float(np.linalg.norm(object_vel)),
            "contacts": int(self.data.ncon),
            "contact_force_max_n": max(contact_forces, default=0.0),
            "contact_force_mean_n": float(np.mean(contact_forces)) if contact_forces else 0.0,
            "action_saturation_fraction": float(np.mean(saturated)),
        }

    def get_object_pose(self) -> np.ndarray:
        return self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 7].copy()

    def compute_success(self) -> bool:
        xyz = self.get_object_pose()[:3]
        cfg = self.config["success"]
        pick_lift_m = cfg.get("pick_lift_m")
        pick_threshold = (
            self.object_rest_z + float(pick_lift_m)
            if pick_lift_m is not None
            else float(cfg["pick_height_m"])
        )
        if xyz[2] >= pick_threshold:
            self.pick_hold_steps += 1
        else:
            self.pick_hold_steps = 0
        required = max(1, round(float(cfg["hold_seconds"]) * self.control_hz))
        self.ever_picked = self.ever_picked or self.pick_hold_steps >= required
        task = str(self.config.get("task_text", "")).lower()
        if ("grasp" in task or "pick" in task) and "place" not in task:
            return bool(self.ever_picked)
        half = np.asarray(self.config["target"]["half_extent_xy_m"])
        in_target = bool(np.all(np.abs(xyz[:2] - self.target[:2]) <= half))
        height_ok = float(cfg["final_height_range_m"][0]) <= xyz[2] <= float(cfg["final_height_range_m"][1])
        speed_ok = np.linalg.norm(self.data.cvel[self.object_body, 3:]) <= float(cfg["place_max_speed_m_s"])
        return bool(self.ever_picked and in_target and height_ok and speed_ok)

    def close(self) -> None:
        self.renderer.close()
