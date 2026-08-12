#!/usr/bin/env python3
"""Recover reusable CUDA wheels from pip's HTTP cache into a named wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import zipfile
from email.parser import Parser
from pathlib import Path


WANTED = {
    ("cuda-toolkit", "13.0.2"),
    ("nvidia-cublas", "13.1.0.3"),
    ("nvidia-cuda-runtime", "13.0.96"),
    ("nvidia-cufft", "12.0.0.61"),
    ("nvidia-cufile", "1.15.1.6"),
    ("nvidia-cuda-cupti", "13.0.85"),
    ("nvidia-curand", "10.4.0.35"),
    ("nvidia-cusolver", "12.0.4.66"),
    ("nvidia-cusparse", "12.6.3.3"),
    ("nvidia-nvjitlink", "13.0.88"),
    ("nvidia-cuda-nvrtc", "13.0.88"),
    ("nvidia-nvtx", "13.0.85"),
    ("nvidia-cudnn-cu13", "9.19.0.56"),
    ("nvidia-cusparselt-cu13", "0.8.0"),
    ("nvidia-nccl-cu13", "2.28.9"),
    ("nvidia-nvshmem-cu13", "3.4.5"),
}


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name)


def inspect_wheel(path: Path) -> tuple[str, str, str] | None:
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_path = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            wheel_path = next(n for n in archive.namelist() if n.endswith(".dist-info/WHEEL"))
            message = Parser().parsestr(archive.read(metadata_path).decode("utf-8", "replace"))
            wheel_message = Parser().parsestr(archive.read(wheel_path).decode("utf-8", "replace"))
            if archive.testzip() is not None:
                return None
            return message["Name"], message["Version"], wheel_message.get_all("Tag")[0]
    except (OSError, zipfile.BadZipFile, StopIteration, TypeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/pip/http-v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    restored: dict[tuple[str, str], Path] = {}
    for body in args.cache.rglob("*.body"):
        details = inspect_wheel(body)
        if not details:
            continue
        name, version, tag = details
        key = (name.lower(), version)
        if key not in WANTED or key in restored:
            continue
        filename = f"{normalized(name)}-{version}-{tag}.whl"
        target = args.output / filename
        shutil.copyfile(body, target)
        restored[key] = target
        print(f"restored {name}=={version} -> {target.name}")
    missing = sorted(WANTED - set(restored))
    manifest = args.output / "SHA256SUMS"
    with manifest.open("w") as handle:
        for target in sorted(restored.values()):
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            handle.write(f"{digest}  {target.name}\n")
    if missing:
        print("not present in cache:")
        for name, version in missing:
            print(f"  {name}=={version}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
