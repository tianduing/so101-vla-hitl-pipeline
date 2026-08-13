#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLA_ROOT="$(cd "$ROOT/.." && pwd)"
MENAGERIE_COMMIT="da76818e269b82289eba39808e2fb91d679d6994"

mkdir -p "$ROOT"/{configs,env,manifests,reports,vendor,tests,runs,outputs}
if [[ ! -d "$ROOT/vendor/mujoco_menagerie/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$ROOT/vendor/mujoco_menagerie"
  git -C "$ROOT/vendor/mujoco_menagerie" sparse-checkout init --cone
  git -C "$ROOT/vendor/mujoco_menagerie" sparse-checkout set robotstudio_so101
fi
git -C "$ROOT/vendor/mujoco_menagerie" fetch origin "$MENAGERIE_COMMIT"
git -C "$ROOT/vendor/mujoco_menagerie" checkout --detach "$MENAGERIE_COMMIT"
if [[ ! -x "$ROOT/.sim_env/bin/python" ]]; then
  "$VLA_ROOT/.conda_env/bin/python" -m venv --system-site-packages "$ROOT/.sim_env"
fi
"$ROOT/.sim_env/bin/python" -m pip install 'mujoco==3.3.7' 'PyOpenGL==3.1.10' 'glfw==2.10.0'
source "$ROOT/env.sh"
python -c 'import mujoco; assert tuple(map(int, mujoco.__version__.split("."))) >= (3,1,3); print(mujoco.__version__)'
python -m pip freeze > "$ROOT/env/pip_freeze.txt"
python --version > "$ROOT/env/python_version.txt" 2>&1
python - "$VLA_ROOT/.conda_env/conda-meta" > "$ROOT/env/conda_explicit.txt" <<'PY'
import json
import sys
from pathlib import Path

print("# Equivalent explicit package URLs reconstructed from conda-meta; system conda is too old to run on this host.")
for path in sorted(Path(sys.argv[1]).glob("*.json")):
    item = json.loads(path.read_text())
    if item.get("url"):
        print(item["url"])
    elif item.get("channel") and item.get("subdir") and item.get("fn"):
        print(f"{item['channel'].rstrip('/')}/{item['subdir']}/{item['fn']}")
PY
"$ROOT/scripts/00_system_probe.sh"
