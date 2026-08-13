from so101_mujoco import So101MujocoEnv


def test_headless_render(root):
    env = So101MujocoEnv(root / "configs/scene.yaml")
    image = env.render("third_person")
    env.close()
    assert image.shape == (480, 640, 3)
    assert image.dtype.name == "uint8"
    assert image.var() > 100
