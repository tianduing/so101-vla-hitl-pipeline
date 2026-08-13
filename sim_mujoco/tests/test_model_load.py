import mujoco


def test_official_model_load(root):
    model = mujoco.MjModel.from_xml_path(str(root / "vendor/mujoco_menagerie/robotstudio_so101/scene_box.xml"))
    assert (model.nq, model.nv, model.nu) == (13, 12, 6)
    assert model.opt.timestep == 0.005
