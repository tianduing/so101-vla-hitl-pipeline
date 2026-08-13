#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
[[ -f "$VLA_ROOT/configs/project.env" ]] && source "$VLA_ROOT/configs/project.env"

MODE="${1:-smoke}"
POLICY="${2:-all}"
NUM_GPUS="${FSDP_NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DATASET_ID="${DATASET_ID:-local/so101_green_block_grasp_train_all_prompt_v2}"
DATASET_ROOT="${DATASET_ROOT:-$VLA_ROOT/data/lerobot/local/so101_green_block_grasp_train_all_prompt_v2}"
SMOLVLA_BASE="${SMOLVLA_BASE:-$VLA_ROOT/models/base/smolvla_base}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ "$MODE" == "smoke" ]]; then
  STEPS="${STEPS:-2}"
  SAVE_FREQ="${SAVE_FREQ:-2}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
else
  STEPS="${STEPS:-100000}"
  SAVE_FREQ="${SAVE_FREQ:-20000}"
  NUM_WORKERS="${NUM_WORKERS:-4}"
fi

[[ -f "$DATASET_ROOT/meta/info.json" ]] || { echo "dataset missing: $DATASET_ROOT" >&2; exit 2; }
[[ "$(awk -F, '{print NF}' <<< "$GPU_IDS")" -eq "$NUM_GPUS" ]] || {
  echo "GPU_IDS must contain exactly FSDP_NUM_GPUS=$NUM_GPUS entries: $GPU_IDS" >&2
  exit 2
}

run_policy() {
  local name="$1"
  local per_gpu_batch="$2"
  shift 2
  local fsdp_policy="$name"
  [[ "$name" == "act_corrected" ]] && fsdp_policy="act"
  local fsdp_config="$VLA_ROOT/configs/accelerate/fsdp_4gpu_${fsdp_policy}.yaml"
  local output="$VLA_ROOT/outputs/train/${STAMP}_${name}_${MODE}_fsdp${NUM_GPUS}"
  local metadata="$VLA_ROOT/outputs/train/.run_metadata/${STAMP}_${name}_${MODE}_fsdp${NUM_GPUS}"
  local checkpoint_link="$VLA_ROOT/models/checkpoints/$name/${STAMP}_${MODE}_fsdp${NUM_GPUS}"
  local ignored_modules=""
  local -a rename_args=()
  if [[ "$name" == "smolvla" ]]; then
    rename_args+=(--rename_map='{"observation.images.scene":"observation.images.camera1"}')
    # The frozen VLM is bf16 while the trainable action expert is fp32. Keep the
    # frozen VLM replicated and shard only trainable parameters to avoid FSDP1
    # mixed-dtype FlatParameter failures.
    ignored_modules='model\.vlm_with_expert\.vlm'
  fi
  mkdir -p "$metadata"
  local -a command=(
    accelerate launch
    --config_file "$fsdp_config"
    --num_processes "$NUM_GPUS"
    -m lerobot.scripts.lerobot_train
    "$@"
    --dataset.repo_id="$DATASET_ID"
    --dataset.root="$DATASET_ROOT"
    "${rename_args[@]}"
    --batch_size="$per_gpu_batch"
    --steps="$STEPS"
    --save_freq="$SAVE_FREQ"
    --env_eval_freq=0
    --eval_steps=0
    --log_freq=1
    --num_workers="$NUM_WORKERS"
    --output_dir="$output"
    --job_name="so101_${name}_${MODE}_fsdp${NUM_GPUS}"
    --policy.device=cuda
    --policy.push_to_hub=false
    --wandb.enable=false
  )

  printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU_IDS" "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    "FSDP_IGNORED_MODULES=$ignored_modules" \
    "${command[@]}" > "$metadata/command.txt"
  printf '\n' >> "$metadata/command.txt"
  cp "$fsdp_config" "$metadata/accelerate_config.yaml"
  {
    printf 'physical_gpu_ids=%s\n' "$GPU_IDS"
    printf 'num_processes=%s\n' "$NUM_GPUS"
    printf 'per_gpu_batch_size=%s\n' "$per_gpu_batch"
    printf 'effective_batch_size=%s\n' "$((per_gpu_batch * NUM_GPUS))"
  } > "$metadata/distributed.txt"
  python -m pip freeze > "$metadata/pip_freeze.txt"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free \
    --format=csv,noheader > "$metadata/gpu_before.txt"

  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FSDP_IGNORED_MODULES="$ignored_modules" \
    "${command[@]}" 2>&1 | tee "$metadata/train.log"

  if [[ "$name" == "smolvla" ]]; then
    python - "$output/checkpoints/last/pretrained_model/policy_preprocessor.json" "$VLA_ROOT/models/base/smolvlm2_processor" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
tokenizer = str(Path(sys.argv[2]).resolve())
config = json.loads(path.read_text())
for step in config["steps"]:
    if step.get("registry_name") == "tokenizer_processor":
        step["config"]["tokenizer_name"] = tokenizer
path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
PY
  fi

  mv "$metadata"/* "$output"/
  rmdir "$metadata"
  find "$output/checkpoints" -type f -name '*.safetensors' -print0 \
    | sort -z | xargs -0 -r sha256sum > "$output/checkpoint_SHA256SUMS"
  mkdir -p "$(dirname "$checkpoint_link")"
  ln -s "$output" "$checkpoint_link"
}

case "$POLICY" in
  act) run_policy act "${ACT_BATCH_PER_GPU:-2}" --policy.type=act ;;
  act_corrected) run_policy act_corrected "${ACT_BATCH_PER_GPU:-2}" \
    --policy.type=act \
    --policy.use_vae=false \
    --policy.chunk_size=50 \
    --policy.n_action_steps=10 \
    --dataset.image_transforms.enable=true ;;
  diffusion) run_policy diffusion "${DIFFUSION_BATCH_PER_GPU:-2}" --policy.type=diffusion ;;
  smolvla) run_policy smolvla "${SMOLVLA_BATCH_PER_GPU:-1}" --policy.path="$SMOLVLA_BASE" ;;
  all)
    run_policy act "${ACT_BATCH_PER_GPU:-2}" --policy.type=act
    run_policy diffusion "${DIFFUSION_BATCH_PER_GPU:-2}" --policy.type=diffusion
    run_policy smolvla "${SMOLVLA_BATCH_PER_GPU:-1}" --policy.path="$SMOLVLA_BASE"
    ;;
  *) echo "usage: $0 [smoke|full] [act|act_corrected|diffusion|smolvla|all]" >&2; exit 2 ;;
esac

if [[ "$MODE" == "full" && "$POLICY" == "all" ]]; then
  "$SCRIPT_DIR/postprocess_fsdp_training.sh"
fi
