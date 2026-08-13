#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/env.sh"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$ROOT/sim_mujoco/src:$ROOT/src"

manifest_args=(
  --manifest "$ROOT/data/lerobot/local/so101_green_block_grasp_sim_target_leftwide_v1/targeted_manifest.json"
  --manifest "$ROOT/data/lerobot/local/so101_green_block_grasp_sim_target_left_highy_v1/targeted_manifest.json"
  --manifest "$ROOT/data/lerobot/local/so101_green_block_grasp_sim_target_left_lowy_v1/targeted_manifest.json"
  --manifest "$ROOT/data/lerobot/local/so101_green_block_grasp_sim_target_seed48_49_v1/targeted_manifest.json"
  --manifest "$ROOT/data/lerobot/local/so101_green_block_grasp_sim_source65_fullgrid_v1/targeted_manifest.json"
  --manifest "$ROOT/data/lerobot/local/so101_sim_physics_v4_shard_000_039/sim_adaptation_manifest.json"
  --manifest "$ROOT/data/lerobot/local/so101_sim_physics_v4_shard_040_079/sim_adaptation_manifest.json"
  --manifest "$ROOT/data/lerobot/local/so101_sim_physics_v4_shard_080_119/sim_adaptation_manifest.json"
  --manifest "$ROOT/data/lerobot/local/so101_sim_physics_v4_shard_120_159/sim_adaptation_manifest.json"
)

"$ROOT/sim_mujoco/.sim_env/bin/python" \
  "$ROOT/sim_mujoco/scripts/evaluate_visual_retrieval_controller.py" \
  "${manifest_args[@]}" \
  --calibration-seeds 42 --seed 52 --trials 20 \
  --output "$ROOT/sim_mujoco/outputs/eval_rgb_visual_retrieval_seed52_71_reproduced.json"
