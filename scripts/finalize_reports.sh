#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

python "$SCRIPT_DIR/summarize_public_eval.py" \
  --root "$VLA_ROOT/outputs/eval" \
  --output "$VLA_ROOT/outputs/reports/public_eval_summary.json"
python "$SCRIPT_DIR/analyze_eval_failures.py"

python "$SCRIPT_DIR/audit_dataset.py" \
  --root "$VLA_ROOT/data/lerobot/local/so101_green_block_grasp_train_all_prompt_v2" \
  --output "$VLA_ROOT/outputs/reports/merged_dataset_audit.json"
python "$SCRIPT_DIR/audit_dataset.py" \
  --root "$VLA_ROOT/data/lerobot/local/so101_systematic50_eval_labeled" \
  --output "$VLA_ROOT/outputs/reports/risk_dataset_audit.json"

checkpoints=()
for policy in act diffusion smolvla; do
  latest="$(find "$VLA_ROOT/outputs/train" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${policy}_*" -printf '%f\n' | sort | tail -n 1)"
  [[ -n "$latest" ]] || {
    echo "no completed $policy training directory found" >&2
    exit 3
  }
  checkpoint="$VLA_ROOT/outputs/train/$latest/checkpoints/last/pretrained_model"
  [[ -f "$checkpoint/config.json" ]] || {
    echo "incomplete checkpoint: $checkpoint" >&2
    exit 3
  }
  checkpoints+=("$checkpoint")
done

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python "$SCRIPT_DIR/verify_checkpoints.py" \
  --dataset-root "$VLA_ROOT/data/lerobot/local/so101_green_block_grasp_train_all_prompt_v2" \
  --output "$VLA_ROOT/outputs/reports/checkpoint_inference.json" \
  "${checkpoints[@]}"

"$SCRIPT_DIR/verify_install.sh"
python "$SCRIPT_DIR/generate_status_report.py"
"$SCRIPT_DIR/capture_manifests.sh"
