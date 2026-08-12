#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    rows = []
    sources = []
    for path in sorted(args.root.rglob("results.jsonl")):
        current = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows.extend(current)
        try:
            display_path = str(path.resolve().relative_to(project_root))
        except ValueError:
            display_path = str(path)
        sources.append({"path": display_path, "trials": len(current)})
    if not rows:
        raise SystemExit(f"no results.jsonl below {args.root}")
    successes = sum(bool(row.get("success")) for row in rows)
    report = {
        "trials": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
