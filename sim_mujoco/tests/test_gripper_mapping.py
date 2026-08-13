import numpy as np

from so101_mujoco import So101MujocoEnv


def test_gripper_open_direction(root):
    env = So101MujocoEnv(root / "configs/scene.yaml")
    low = np.array(env.last_raw_action); high = low.copy()
    low[-1] = 40; high[-1] = 65
    assert env.raw_to_mujoco(high, rate_limit=False)[-1] > env.raw_to_mujoco(low, rate_limit=False)[-1]
    env.close()
