import numpy as np

from so101_mujoco import So101MujocoEnv


def test_success_requires_pick_history(root):
    env = So101MujocoEnv(root / "configs/scene.yaml")
    env.reset(object_pose=np.array([0.22, 0.20, 0.041]))
    assert not env.compute_success()
    env.ever_picked = True
    assert env.compute_success()
    env.close()


def test_pick_threshold_is_relative_to_reset_height(root):
    env = So101MujocoEnv(root / "configs/scene.yaml")
    env.reset(object_pose=np.array([0.24, 0.0, -0.015]))
    threshold = env.object_rest_z + float(env.config["success"]["pick_lift_m"])
    env.data.qpos[env.object_qpos_addr + 2] = threshold + 0.001
    required = round(float(env.config["success"]["hold_seconds"]) * env.control_hz)
    for _ in range(required):
        env.compute_success()
    assert env.ever_picked
    env.close()
