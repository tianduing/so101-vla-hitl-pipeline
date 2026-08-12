#!/usr/bin/env python3
"""Make the downloaded SmolVLA checkpoint self-contained for offline loading.

The policy checkpoint already contains the VLM tensors.  LeRobot's default config
nevertheless downloads the upstream 2 GB VLM once during construction, then
overwrites it with checkpoint tensors.  Pointing at local config/tokenizer files
and constructing the architecture without upstream weights removes that redundant
network dependency while preserving the checkpoint load.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


root = Path(os.environ["VLA_ROOT"])
policy_dir = root / "models/base/smolvla_base"
processor_dir = root / "models/base/smolvlm2_processor"
config_path = policy_dir / "config.json"
upstream_path = policy_dir / "config.upstream.json"
preprocessor_path = policy_dir / "policy_preprocessor.json"
preprocessor_upstream_path = policy_dir / "policy_preprocessor.upstream.json"

required = [
    processor_dir / "config.json",
    processor_dir / "preprocessor_config.json",
    processor_dir / "tokenizer.json",
    policy_dir / "model.safetensors",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing SmolVLA offline resources: {missing}")

if not upstream_path.exists():
    shutil.copy2(config_path, upstream_path)
config = json.loads(upstream_path.read_text())
config["vlm_model_name"] = str(processor_dir.resolve())
config["load_vlm_weights"] = False
config_path.write_text(json.dumps(config, indent=4, ensure_ascii=False) + "\n")

if not preprocessor_upstream_path.exists():
    shutil.copy2(preprocessor_path, preprocessor_upstream_path)
preprocessor = json.loads(preprocessor_upstream_path.read_text())
for step in preprocessor["steps"]:
    if step.get("registry_name") == "tokenizer_processor":
        step["config"]["tokenizer_name"] = str(processor_dir.resolve())
preprocessor_path.write_text(json.dumps(preprocessor, indent=2, ensure_ascii=False) + "\n")
print(f"offline SmolVLA config ready: {config_path}")
