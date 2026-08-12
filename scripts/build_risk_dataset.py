#!/usr/bin/env python3
"""Merge released rollout shards and align their episode-level success labels."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
shards = sorted((ROOT / "outputs/eval").glob("eval_so101_green_yellow_50_*"))
if len(shards) != 3:
    raise SystemExit(f"expected 3 evaluation shards, found {len(shards)}")

output = ROOT / "data/lerobot/local/so101_systematic50_eval_labeled"
labels_path = ROOT / "data/lerobot/local/so101_systematic50_eval_labels.csv"
repo_ids = [f"local/{path.name}" for path in shards]
roots = [str(path) for path in shards]

if not (output / "meta/info.json").is_file():
    if output.exists():
        raise SystemExit(f"incomplete output exists: {output}")
    subprocess.run(
        [
            "lerobot-edit-dataset",
            "--new_repo_id=local/so101_systematic50_eval_labeled",
            f"--new_root={output}",
            "--operation.type=merge",
            "--operation.concatenate_videos=false",
            "--operation.concatenate_data=false",
            f"--operation.repo_ids={repo_ids!r}",
            f"--operation.roots={roots!r}",
        ],
        check=True,
    )

rows: list[dict[str, int]] = []
offset = 0
for shard in shards:
    results = [json.loads(line) for line in (shard / "results.jsonl").read_text().splitlines() if line]
    info = json.loads((shard / "meta/info.json").read_text())
    if len(results) != info["total_episodes"]:
        raise ValueError(f"label/episode mismatch in {shard}")
    for local_episode, result in enumerate(results):
        rows.append(
            {
                "episode_index": offset + local_episode,
                "success": int(bool(result["success"])),
                "trial_index": int(result["trial_index"]),
            }
        )
    offset += info["total_episodes"]

if len(rows) != 50 or sum(row["success"] for row in rows) != 31:
    raise ValueError("unexpected systematic50 labels")
labels_path.parent.mkdir(parents=True, exist_ok=True)
with labels_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["episode_index", "success", "trial_index"])
    writer.writeheader()
    writer.writerows(rows)
print(f"risk dataset: {output} ({len(rows)} episodes, {sum(r['success'] for r in rows)} successes)")
print(f"labels: {labels_path}")
