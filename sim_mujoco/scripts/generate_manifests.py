#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLA = ROOT.parent


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    menagerie = subprocess.check_output(["git", "-C", ROOT / "vendor/mujoco_menagerie", "rev-parse", "HEAD"], text=True).strip()
    lerobot = subprocess.check_output(["git", "-C", VLA / "src/lerobot", "rev-parse", "HEAD"], text=True).strip()
    sources = {"mujoco_menagerie": {"url": "https://github.com/google-deepmind/mujoco_menagerie.git", "commit": menagerie}, "lerobot": {"path": str(VLA / "src/lerobot"), "commit": lerobot}, "mujoco_python": "3.3.7"}
    (ROOT / "manifests/source_commits.json").write_text(json.dumps(sources, indent=2) + "\n")
    resources = []
    for path in sorted((ROOT / "outputs").rglob("*")):
        if path.is_file() and not path.is_symlink():
            resources.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    (ROOT / "manifests/resources_sha256.json").write_text(json.dumps(resources, indent=2) + "\n")
    videos = [item for item in resources if item["path"].endswith(".mp4")]
    (ROOT / "outputs/video_manifest.json").write_text(json.dumps({"videos": videos}, indent=2) + "\n")


if __name__ == "__main__": main()
