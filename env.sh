#!/usr/bin/env bash

set -euo pipefail

VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLA_ROOT
export HF_HOME="$VLA_ROOT/downloads/huggingface_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_LEROBOT_HOME="$VLA_ROOT/data/lerobot"
export TORCH_HOME="$VLA_ROOT/downloads/torch_cache"
export XDG_CACHE_HOME="$VLA_ROOT/downloads/xdg_cache"
export WANDB_DIR="$VLA_ROOT/logs/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE=1
export PATH="$VLA_ROOT/.conda_env/bin:$PATH"

mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR"
