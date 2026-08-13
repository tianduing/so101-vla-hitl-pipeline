import mujoco
import numpy as np

from so101_mujoco import So101MujocoEnv
from so101_mujoco.tasks.pick_place_box import (
    PICK_PLACE_STAGES,
    object_fully_in_box,
    summarize_pick_place_trials,
)


def test_pick_place_box_scene_and_quality_gate(root):
    env = So101MujocoEnv(root / "configs/scene_pick_place_box.yaml")
    try:
        assert env.config["success"]["hold_seconds"] == 3.0
        assert tuple(PICK_PLACE_STAGES) == (
            "approach",
            "grasp",
            "lift",
            "transport",
            "place",
            "retreat",
        )
        for name in (
            "box_floor",
            "box_left_wall",
            "box_right_wall",
            "box_front_wall",
            "box_back_wall",
        ):
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0

        center = np.asarray(env.config["target"]["center_xyz"], dtype=float)
        env.data.qpos[env.object_qpos_addr : env.object_qpos_addr + 3] = center
        mujoco.mj_forward(env.model, env.data)
        assert object_fully_in_box(env)

        env.data.qpos[env.object_qpos_addr] += 0.10
        mujoco.mj_forward(env.model, env.data)
        assert not object_fully_in_box(env)
    finally:
        env.close()


def test_stage_summary_keeps_hold_gate_separate():
    def trial(hold_success: bool):
        return {
            "success": hold_success,
            "stages": {stage: {"success": True} for stage in PICK_PLACE_STAGES},
            "quality_gates": {"stable_hold_3s": {"success": hold_success}},
        }

    summary = summarize_pick_place_trials([trial(True), trial(False)])
    assert summary["stage_metrics"]["lift"]["success_rate"] == 1.0
    assert summary["quality_gates"]["stable_hold_3s"]["success_rate"] == 0.5
    assert summary["stage_metrics"]["transport"]["success_rate"] == 1.0
    assert summary["stage_metrics"]["transport"]["cumulative_pipeline_success_rate"] == 0.5
    assert summary["full_success_rate"] == 0.5
