from so101_mujoco import So101MujocoEnv


def test_observation_schema(root):
    env = So101MujocoEnv(root / "configs/scene.yaml")
    obs = env.get_observation()
    env.close()
    assert obs["observation.state"].shape == (6,)
    assert obs["observation.images.scene"].shape == (480, 640, 3)
    assert obs["sim.wrist"].shape == (480, 640, 3)
