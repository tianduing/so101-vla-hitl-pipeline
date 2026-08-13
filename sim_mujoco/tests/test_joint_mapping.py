import numpy as np
import pytest

from so101_mujoco import So101MujocoEnv


def test_mapping_roundtrip_and_limits(root):
    env = So101MujocoEnv(root / "configs/scene.yaml")
    raw = np.array(env.config["initial_state_degrees"])
    radians = env.raw_to_mujoco(raw, rate_limit=False)
    assert np.max(np.abs(env.mujoco_to_raw(radians) - raw)) < 1e-6
    assert np.all(radians >= env.ctrl_ranges[:, 0]) and np.all(radians <= env.ctrl_ranges[:, 1])
    with pytest.raises(ValueError): env.raw_to_mujoco(np.array([np.nan] * 6))
    env.close()


def test_rate_limit(root):
    env = So101MujocoEnv(root / "configs/scene.yaml")
    before = env.last_raw_action.copy()
    limited = env.mujoco_to_raw(env.raw_to_mujoco(before + 50))
    assert np.max(np.abs(limited - before)) <= 6.000001
    env.close()
