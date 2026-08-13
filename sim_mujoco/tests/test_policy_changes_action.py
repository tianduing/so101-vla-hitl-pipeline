import json


def test_policy_input_and_action_change(root):
    log = sorted((root / "outputs/policy_closed_loop_act60k").glob("trial_*.jsonl"))[0]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len({row["input_hash"] for row in rows}) > 100
    assert len({tuple(row["raw_policy_action"]) for row in rows}) > 100
    assert max(row["policy_call_id"] for row in rows) == 9
