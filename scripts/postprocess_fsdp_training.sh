#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
[[ -f "$VLA_ROOT/configs/project.env" ]] && source "$VLA_ROOT/configs/project.env"

NUM_GPUS="${FSDP_NUM_GPUS:-4}"
MARKER="$VLA_ROOT/outputs/reports/fsdp4_postprocess_complete.txt"
declare -a runs=()

for policy in act diffusion smolvla; do
  latest="$(find "$VLA_ROOT/outputs/train" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${policy}_full_fsdp${NUM_GPUS}" -printf '%f\n' | sort | tail -n 1)"
  [[ -n "$latest" ]] || { echo "no completed full FSDP run for $policy" >&2; exit 3; }
  runs+=("$latest")
done

signature="$(printf '%s\n' "${runs[@]}")"
if [[ -f "$MARKER" ]] && [[ "$(sed -n '/^runs:$/,$p' "$MARKER" | tail -n +2)" == "$signature" ]]; then
  echo "postprocessing already complete for current full FSDP runs"
  exit 0
fi

"$SCRIPT_DIR/finalize_reports.sh"

policies=(act diffusion smolvla)
for index in "${!policies[@]}"; do
  policy="${policies[$index]}"
  latest="${runs[$index]}"
  checkpoint="$VLA_ROOT/outputs/train/$latest/checkpoints/last/pretrained_model"
  output="$VLA_ROOT/outputs/visualization/${latest}_episode0_offline.mp4"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python "$SCRIPT_DIR/export_policy_inference_video.py" \
    --checkpoint "$checkpoint" \
    --episode 0 \
    --device cpu \
    --max-frames "${VIZ_INFERENCE_FRAMES:-30}" \
    --output "$output"
done

{
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'runs:\n%s\n' "$signature"
} > "$MARKER"
echo "FSDP reports and offline inference videos are complete."
