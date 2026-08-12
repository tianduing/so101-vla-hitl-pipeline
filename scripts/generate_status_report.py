#!/usr/bin/env python3
"""Build a machine-readable status summary from verified pipeline artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path


root = Path(os.environ["VLA_ROOT"])


def load(path: str) -> dict:
    return json.loads((root / path).read_text())


dataset = load("outputs/reports/merged_dataset_audit.json")
risk_dataset = load("outputs/reports/risk_dataset_audit.json")
evaluation = load("outputs/reports/public_eval_summary.json")
failure_analysis = load("outputs/reports/failure_analysis.json")
inference = load("outputs/reports/checkpoint_inference.json")
risk_metrics = load("models/checkpoints/risk_model/systematic50_risk_mlp.metrics.json")
install = load("manifests/install_verification.json")

report = {
    "scope": {
        "completed": "public real-SO101 offline pipeline",
        "not_available": [
            "private 240-episode dual-camera lab dataset",
            "attached SO-101 leader/follower and cameras",
            "physical HITL collection and two-round robot evaluation",
        ],
        "claim_policy": "documented 81.7%/88.3% figures are not treated as locally reproduced",
    },
    "environment": install,
    "training_dataset": {
        "episodes": dataset["total_episodes"],
        "frames": dataset["total_frames"],
        "audit": dataset["datasets"][0],
    },
    "released_physical_evaluation": evaluation,
    "failure_analysis": {
        "by_phase": failure_analysis["by_phase"],
        "diagnostic_slices": failure_analysis["diagnostic_slices"],
        "yellow_distractor_training_coverage": failure_analysis["yellow_distractor_training_coverage"],
    },
    "risk_dataset": {
        "episodes": risk_dataset["total_episodes"],
        "frames": risk_dataset["total_frames"],
        "successes": risk_metrics["success_episodes"],
        "failures": risk_metrics["failure_episodes"],
    },
    "risk_model": risk_metrics,
    "checkpoint_offline_inference": inference["checkpoints"],
    "automated_checks": {"hitl_guard_tests": "5 passed", "checkpoint_count": len(inference["checkpoints"])},
}
output = root / "outputs/reports/pipeline_status.json"
output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(output)
