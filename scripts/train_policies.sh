#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
[[ -f "$VLA_ROOT/configs/project.env" ]] && source "$VLA_ROOT/configs/project.env"

MODE="${1:-smoke}"
POLICY="${2:-all}"
DATASET_ID="${DATASET_ID:-local/so101_green_block_grasp_train_all_prompt_v2}"
DATASET_ROOT="${DATASET_ROOT:-$VLA_ROOT/data/lerobot/local/so101_green_block_grasp_train_all_prompt_v2}"
SMOLVLA_BASE="${SMOLVLA_BASE:-$VLA_ROOT/models/base/smolvla_base}"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEVICE="${DEVICE:-cuda}"

if [[ "$MODE" == "smoke" ]]; then
  STEPS="${STEPS:-20}"
  SAVE_FREQ="${SAVE_FREQ:-20}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  NUM_WORKERS="${NUM_WORKERS:-2}"
else
  STEPS="${STEPS:-100000}"
  # Five checkpoints per full run keeps the project within the shared disk budget.
  SAVE_FREQ="${SAVE_FREQ:-20000}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  NUM_WORKERS="${NUM_WORKERS:-8}"
fi

[[ -f "$DATASET_ROOT/meta/info.json" ]] || { echo "dataset missing: $DATASET_ROOT" >&2; exit 2; }

run_policy() {
  local name="$1"
  shift
  local output="$VLA_ROOT/outputs/train/${STAMP}_${name}_${MODE}"
  local metadata="$VLA_ROOT/outputs/train/.run_metadata/${STAMP}_${name}_${MODE}"
  local checkpoint_link="$VLA_ROOT/models/checkpoints/$name/${STAMP}_${MODE}"
  local -a rename_args=()
  if [[ "$name" == "smolvla" ]]; then
    rename_args+=(--rename_map='{"observation.images.scene":"observation.images.camera1"}')
  fi
  mkdir -p "$metadata"
  local -a command=(
    lerobot-train "$@"
    --dataset.repo_id="$DATASET_ID"
    --dataset.root="$DATASET_ROOT"
    "${rename_args[@]}"
    --batch_size="$BATCH_SIZE"
    --steps="$STEPS"
    --save_freq="$SAVE_FREQ"
    --env_eval_freq=0
    --eval_steps=0
    --log_freq=1
    --num_workers="$NUM_WORKERS"
    --output_dir="$output"
    --job_name="so101_${name}_${MODE}"
    --policy.device="$DEVICE"
    --policy.push_to_hub=false
    --wandb.enable=false
  )
  printf '%q ' "${command[@]}" > "$metadata/command.txt"
  printf '\n' >> "$metadata/command.txt"
  {
    printf 'lerobot '
    git -C "$VLA_ROOT/src/lerobot" rev-parse HEAD
    printf 'public_so101_reference '
    git -C "$VLA_ROOT/src/public_so101_reference" rev-parse HEAD
  } > "$metadata/git_commits.txt"
  python -m pip freeze > "$metadata/pip_freeze.txt"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader > "$metadata/gpu_info.txt" 2>/dev/null || true

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
  act) run_policy act --policy.type=act ;;
  diffusion) run_policy diffusion --policy.type=diffusion ;;
  smolvla) run_policy smolvla --policy.path="$SMOLVLA_BASE" ;;
  all)
    run_policy act --policy.type=act
    run_policy diffusion --policy.type=diffusion
    run_policy smolvla --policy.path="$SMOLVLA_BASE"
    ;;
  *) echo "usage: $0 [smoke|full] [act|diffusion|smolvla|all]" >&2; exit 2 ;;
esac
