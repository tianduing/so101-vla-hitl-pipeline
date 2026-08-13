import os
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")


@pytest.fixture(scope="session")
def root() -> Path:
    return Path(__file__).resolve().parents[1]
