import json


def test_policy_checkpoint_was_loaded(root):
    summary = json.loads((root / "outputs/policy_closed_loop_act60k/summary.json").read_text())
    assert summary["policy_type"] == "act"
    assert len(summary["checkpoint_sha256"]) == 64
    assert summary["num_trials"] == 10
