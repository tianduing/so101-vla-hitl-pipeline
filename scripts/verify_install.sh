#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

python - <<'PY'
import json
import shutil
import subprocess
from pathlib import Path

import torch
import lerobot

root = Path(__import__('os').environ['VLA_ROOT'])
checks = {
    'python': __import__('sys').version.split()[0],
    'lerobot': getattr(lerobot, '__version__', 'source'),
    'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'gpu_count': torch.cuda.device_count(),
    'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    'ffmpeg': Path(shutil.which('ffmpeg')).name if shutil.which('ffmpeg') else None,
    'cli': {name: bool(shutil.which(name)) for name in (
        'lerobot-train', 'lerobot-record', 'lerobot-rollout',
        'lerobot-edit-dataset', 'lerobot-dataset-viz')},
}
if checks['ffmpeg']:
    checks['ffmpeg_version'] = subprocess.check_output(
        ['ffmpeg', '-version'], text=True, stderr=subprocess.STDOUT
    ).splitlines()[0]
out = root / 'manifests' / 'install_verification.json'
out.write_text(json.dumps(checks, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(checks, indent=2, ensure_ascii=False))
if not all(checks['cli'].values()):
    raise SystemExit('missing required LeRobot CLI')
PY
