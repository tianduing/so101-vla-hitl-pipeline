#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/env.sh"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/sim_mujoco/src:$ROOT/src"

ACT_GPU="${ACT_GPU:-0}"
SMOLVLA_GPU="${SMOLVLA_GPU:-1}"
ACT_PRIMARY="$ROOT/outputs/train/20260813_act_sim_physics40_cont4500/step_001000"
ACT_ALTERNATE="$ROOT/outputs/train/20260813_act_best_evalrange25_lr2e7_1000/step_000200"
ACT_SPECIALIST="$ROOT/outputs/train/20260813_act_seed48_49_specialist/step_000100"
SMOLVLA="$ROOT/outputs/train/20260813_011012_smolvla_full/checkpoints/100000/pretrained_model"

for checkpoint in "$ACT_PRIMARY" "$ACT_ALTERNATE" "$ACT_SPECIALIST" "$SMOLVLA"; do
  test -f "$checkpoint/model.safetensors" || {
    echo "Missing checkpoint: $checkpoint" >&2
    exit 4
  }
done

CUDA_VISIBLE_DEVICES="$ACT_GPU" "$ROOT/sim_mujoco/.sim_env/bin/python" \
  "$ROOT/sim_mujoco/scripts/04_run_policy_closed_loop.py" \
  --checkpoint "$ACT_PRIMARY" \
  --alternate-checkpoint "$ACT_ALTERNATE" \
  --green-centroid-x-threshold 195 \
  --specialist-checkpoint "$ACT_SPECIALIST" \
  --specialist-centroid-x-min 180 --specialist-centroid-x-max 187 \
  --device cuda --action-steps 10 --trials 10 --seconds 30 --seed 42 \
  --output-dir "$ROOT/sim_mujoco/outputs/reproduced_act_visual_moe" &
act_pid=$!

CUDA_VISIBLE_DEVICES="$SMOLVLA_GPU" "$ROOT/sim_mujoco/.sim_env/bin/python" \
  "$ROOT/sim_mujoco/scripts/04_run_policy_closed_loop.py" \
  --checkpoint "$SMOLVLA" \
  --device cuda --action-steps 50 --trials 10 --seconds 30 --seed 42 \
  --output-dir "$ROOT/sim_mujoco/outputs/reproduced_smolvla_a50" &
smolvla_pid=$!

status=0
wait "$act_pid" || status=$?
wait "$smolvla_pid" || status=$?
exit "$status"
