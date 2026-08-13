import json
from pathlib import Path


def test_closed_loop_policy_evidence(root):
    logs = sorted((root / "outputs/policy_closed_loop_debug_cpu").glob("trial_*.jsonl"))
    if not logs:
        return
    rows = [json.loads(line) for line in logs[-1].read_text().splitlines()]
    assert len(rows) > 1
    assert len({row["input_hash"] for row in rows}) > 1
    assert len({tuple(row["raw_policy_action"]) for row in rows}) > 1
    assert max(row["policy_call_id"] for row in rows) > 1
    assert all("raw_policy_action" in row and "ctrl_rad" in row for row in rows)


def test_replay_and_expert_not_counted_as_policy(root):
    replay = json.loads((root / "outputs/real_trajectory_replay.metadata.json").read_text())
    expert = json.loads((root / "outputs/scripted_expert_reference.json").read_text())
    assert replay["closed_loop_policy"] is False
    assert expert["policy"] is False
