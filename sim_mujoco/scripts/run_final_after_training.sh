#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
TRAIN_UNIT="so101-vla-fsdp4-full-train.service"

while [[ "$(systemctl --user is-active "$TRAIN_UNIT" 2>/dev/null || true)" == "active" ]]; do sleep 30; done

complete_checkpoint() {
  find "$VLA_ROOT/outputs/train" \
    \( -path "*_${1}_full_fsdp4/checkpoints/100000/pretrained_model/config.json" \
       -o -path "*_${1}_full/checkpoints/100000/pretrained_model/config.json" \) \
    -printf '%h\n' 2>/dev/null | sort | tail -n 1
}

parallel_unit_for_policy() {
  case "$1" in
    diffusion) echo "so101-vla-diffusion-full-parallel.service" ;;
    smolvla) echo "so101-vla-smolvla-full-parallel.service" ;;
    *) echo "" ;;
  esac
}

for policy in act diffusion smolvla; do
  checkpoint="$(complete_checkpoint "$policy")"
  parallel_unit="$(parallel_unit_for_policy "$policy")"
  while [[ -z "$checkpoint" && -n "$parallel_unit" ]] \
    && [[ "$(systemctl --user is-active "$parallel_unit" 2>/dev/null || true)" == "active" ]]; do
    echo "$(date --iso-8601=seconds) waiting for parallel $policy training: $parallel_unit"
    sleep 30
    checkpoint="$(complete_checkpoint "$policy")"
  done
  if [[ -z "$checkpoint" ]]; then
    echo "$(date --iso-8601=seconds) resuming missing full policy: $policy"
    "$VLA_ROOT/scripts/run_distributed_when_ready.sh" "$VLA_ROOT/scripts/train_policies_fsdp.sh" full "$policy"
    checkpoint="$(complete_checkpoint "$policy")"
  fi
  [[ -n "$checkpoint" ]] || { echo "full checkpoint still missing: $policy" >&2; exit 4; }
  echo "$policy=$checkpoint"
done

smolvla_checkpoint="$(complete_checkpoint smolvla)"
min_free_mib="${SIM_POLICY_MIN_FREE_MIB:-5000}"
while true; do
  gpu="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, -v min="$min_free_mib" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2>=min) print $2, $1}' | sort -nr | head -n1 | awk '{print $2}')"
  [[ -n "$gpu" ]] && break
  echo "$(date --iso-8601=seconds) waiting for one GPU with ${min_free_mib}MiB free"
  sleep 30
done

echo "running final SmolVLA MuJoCo closed loop on physical GPU $gpu"
if ! CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" "$SCRIPT_DIR/run_closed_loop.sh" \
    --checkpoint "$smolvla_checkpoint" --device cuda --trials 10 --seconds 30 \
    --output-dir "$SIM_ROOT/outputs/policy_closed_loop_smolvla_final"; then
  echo "GPU evaluation failed; preserving evidence and retrying on CPU"
  MUJOCO_EGL_DEVICE_ID=0 "$SCRIPT_DIR/run_closed_loop.sh" \
    --checkpoint "$smolvla_checkpoint" --device cpu --trials 10 --seconds 30 \
    --output-dir "$SIM_ROOT/outputs/policy_closed_loop_smolvla_final_cpu"
fi

"$SCRIPT_DIR/generate_manifests.py"
"$SCRIPT_DIR/render_showcase.sh"
