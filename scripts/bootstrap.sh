#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MINIFORGE="$VLA_ROOT/downloads/miniforge"
ENV_PREFIX="$VLA_ROOT/.conda_env"
INSTALLER="$VLA_ROOT/downloads/installers/Miniforge3-Linux-x86_64.sh"
PROXY_VALUE="${HTTPS_PROXY:-${https_proxy:-}}"

"$SCRIPT_DIR/fetch_sources.sh"
"$SCRIPT_DIR/apply_local_patches.sh"

if [[ ! -x "$MINIFORGE/bin/mamba" ]]; then
  mkdir -p "$(dirname "$INSTALLER")"
  if [[ ! -s "$INSTALLER" ]]; then
    https_proxy="$PROXY_VALUE" http_proxy="$PROXY_VALUE" wget -q --show-progress \
      -O "$INSTALLER" \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  fi
  bash "$INSTALLER" -b -p "$MINIFORGE"
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  https_proxy="$PROXY_VALUE" http_proxy="$PROXY_VALUE" \
    "$MINIFORGE/bin/mamba" create -y --override-channels -c conda-forge \
    -p "$ENV_PREFIX" python=3.12 pip ffmpeg=7.1.1
fi

export PIP_CACHE_DIR="$VLA_ROOT/downloads/pip_cache"
if [[ ! -x "$MINIFORGE/bin/uv" ]]; then
  https_proxy="$PROXY_VALUE" http_proxy="$PROXY_VALUE" \
    "$MINIFORGE/bin/mamba" install -y --override-channels -c conda-forge -p "$MINIFORGE" uv
fi
export UV_CACHE_DIR="$VLA_ROOT/downloads/uv_cache"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-3}"
WHEELHOUSE="$VLA_ROOT/downloads/wheelhouse_cu130"
if [[ -f "$WHEELHOUSE/torch-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl" ]]; then
  (cd "$WHEELHOUSE" && sha256sum -c SHA256SUMS)
  "$MINIFORGE/bin/uv" pip install --python "$ENV_PREFIX/bin/python" --no-deps \
    "$WHEELHOUSE"/*.whl
else
  https_proxy="$PROXY_VALUE" http_proxy="$PROXY_VALUE" \
    "$MINIFORGE/bin/uv" pip install --python "$ENV_PREFIX/bin/python" \
    'torch==2.11.0' 'torchvision==0.26.0'
fi
https_proxy="$PROXY_VALUE" http_proxy="$PROXY_VALUE" \
  "$MINIFORGE/bin/uv" pip install --python "$ENV_PREFIX/bin/python" -e \
  "$VLA_ROOT/src/lerobot[training,smolvla,diffusion,dataset_viz,core_scripts,async,feetech]" \
  pytest

source "$VLA_ROOT/env.sh"
"$SCRIPT_DIR/verify_install.sh"
python -m pip check
python -m pip freeze > "$VLA_ROOT/manifests/packages.txt"
