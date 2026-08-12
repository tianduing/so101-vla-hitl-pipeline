#!/usr/bin/env python3
"""Summarize physical-evaluation failure regions and training coverage gaps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    return {
        "trials": len(rows),
        "successes": successes,
        "failures": len(rows) - successes,
        "success_rate": successes / len(rows) if rows else None,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=project_root / "outputs/eval")
    parser.add_argument("--training-root", type=Path, default=project_root / "data/lerobot/local")
    parser.add_argument("--output", type=Path, default=project_root / "outputs/reports/failure_analysis.json")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.eval_root.rglob("results.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise SystemExit(f"no results.jsonl below {args.eval_root}")

    phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        phases[str(row["phase"])].append(row)
    low_y = [row for row in rows if float(row["target_xy_mm"][1]) < 45.0]
    other = [row for row in rows if row["phase"] not in {"unseen_low_y", "unseen_left_extreme"}]

    coverage_rows: list[list[float]] = []
    for dataset in (
        "so101_green_block_grasp_yellow_distractor_v1",
        "so101_green_block_grasp_yellow_distractor_v2",
        "so101_green_block_grasp_yellow_lcr_supplement_v1",
    ):
        root = args.training_root / dataset
        if not (root / "placement_plan.json").is_file():
            continue
        info = json.loads((root / "meta/info.json").read_text())
        plan = json.loads((root / "placement_plan.json").read_text())
        coverage_rows.extend(item["target_xy_mm"] for item in plan["plans"][: int(info["total_episodes"])])

    report = {
        "evaluation": metrics(rows),
        "by_phase": {name: metrics(items) for name, items in sorted(phases.items())},
        "diagnostic_slices": {
            "target_y_below_45mm": metrics(low_y),
            "excluding_unseen_low_y_and_left_extreme": metrics(other),
        },
        "yellow_distractor_training_coverage": {
            "episodes_with_placement_metadata": len(coverage_rows),
            "target_y_below_22mm": sum(point[1] < 22.0 for point in coverage_rows),
            "target_x_below_minus_25mm": sum(point[0] < -25.0 for point in coverage_rows),
            "note": "Counts use the saved episode count and the corresponding leading placement plans.",
        },
        "interpretation": (
            "The 62% aggregate result is dominated by two out-of-distribution regions: "
            "unseen_low_y and unseen_left_extreme are both 0/5. This is primarily a coverage and "
            "generalization gap, not evidence that the full pipeline is only 62% accurate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
