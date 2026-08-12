import importlib.util
import sys
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("hitl_guard", Path(__file__).parents[1] / "scripts" / "hitl_guard.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
HITLGuard = module.HITLGuard


def make_guard():
    return HITLGuard(np.full(6, -1.0), np.full(6, 1.0))


def test_three_low_scores_trigger():
    guard = make_guard()
    assert guard.update(0.2, np.zeros(6))[0] is False
    assert guard.update(0.2, np.zeros(6))[0] is False
    trigger, reasons = guard.update(0.2, np.zeros(6))
    assert trigger and "low_value" in reasons


def test_joint_limit_triggers_immediately():
    trigger, reasons = make_guard().update(0.9, np.array([0, 0, 0, 0, 0, 1.1]))
    assert trigger and reasons == ["joint_limit"]


def test_invalid_signals_fail_safe():
    guard = make_guard()
    assert guard.update(float("nan"), np.zeros(6)) == (True, ["invalid_risk"])
    assert guard.update(0.9, np.full(6, np.nan)) == (True, ["nonfinite_action"])
    assert guard.update(0.9, np.zeros(5)) == (True, ["action_shape"])


def test_action_oscillation_triggers():
    guard = make_guard()
    for value in (0.1, -0.1, 0.1, -0.1, 0.1):
        assert guard.update(0.9, np.full(6, value))[0] is False
    trigger, reasons = guard.update(0.9, np.full(6, -0.1))
    assert trigger and "action_oscillation" in reasons


def test_corrective_window_has_two_second_context():
    assert module.corrective_window_bounds(90, 150, fps=30) == (30, 210)
