#!/usr/bin/env bash
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="$(cd "$SIM_ROOT/.." && pwd)"
export SIM_ROOT VLA_ROOT
export PATH="$SIM_ROOT/.sim_env/bin:$VLA_ROOT/.conda_env/bin:$PATH"
export PYTHONPATH="$SIM_ROOT/src:$VLA_ROOT/src/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
