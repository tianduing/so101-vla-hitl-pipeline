#!/usr/bin/env python3
"""Model-independent safety trigger used by the controller-side HITL loop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HITLGuard:
    joint_low: np.ndarray
    joint_high: np.ndarray
    risk_threshold: float = 0.35
    consecutive_risk_steps: int = 3
    oscillation_window: int = 6
    oscillation_min_flips: int = 4
    oscillation_amplitude: float = 0.08
    _low_risk_count: int = 0
    _actions: deque[np.ndarray] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.joint_low = np.asarray(self.joint_low, dtype=np.float32)
        self.joint_high = np.asarray(self.joint_high, dtype=np.float32)
        if self.joint_low.shape != self.joint_high.shape or self.joint_low.ndim != 1:
            raise ValueError("joint bounds must be one-dimensional arrays with equal shape")
        if not np.all(np.isfinite(self.joint_low)) or not np.all(np.isfinite(self.joint_high)):
            raise ValueError("joint bounds must be finite")
        if np.any(self.joint_low >= self.joint_high):
            raise ValueError("every lower joint bound must be below its upper bound")

    def update(self, risk_score: float, action: np.ndarray) -> tuple[bool, list[str]]:
        action = np.asarray(action, dtype=np.float32)
        reasons: list[str] = []
        if not np.isfinite(risk_score):
            reasons.append("invalid_risk")
            self._low_risk_count = 0
        else:
            self._low_risk_count = self._low_risk_count + 1 if risk_score < self.risk_threshold else 0
        if self._low_risk_count >= self.consecutive_risk_steps:
            reasons.append("low_value")

        if action.shape != self.joint_low.shape:
            reasons.append("action_shape")
            return True, reasons
        if not np.all(np.isfinite(action)):
            reasons.append("nonfinite_action")
            return True, reasons
        if np.any(action < self.joint_low) or np.any(action > self.joint_high):
            reasons.append("joint_limit")
        self._actions.append(action.copy())
        while len(self._actions) > self.oscillation_window:
            self._actions.popleft()
        if len(self._actions) == self.oscillation_window:
            history = np.stack(self._actions)
            velocity = np.diff(history, axis=0)
            flips = np.sum(np.sign(velocity[1:]) * np.sign(velocity[:-1]) < 0, axis=0)
            amplitude = history.max(axis=0) - history.min(axis=0)
            if np.any((flips >= self.oscillation_min_flips) & (amplitude >= self.oscillation_amplitude)):
                reasons.append("action_oscillation")
        return bool(reasons), reasons


def corrective_window_bounds(intervention_frame: int, recovery_frame: int, fps: int = 30) -> tuple[int, int]:
    padding = 2 * fps
    return max(0, intervention_frame - padding), recovery_frame + padding
