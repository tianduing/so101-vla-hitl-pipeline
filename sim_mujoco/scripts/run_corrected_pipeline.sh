#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

checkpoint="$(find "$VLA_ROOT/outputs/train" -path '*_act_corrected_full_fsdp4/checkpoints/100000/pretrained_model/config.json' -printf '%h\n' 2>/dev/null | sort | tail -n1)"
if [[ -z "$checkpoint" ]]; then
  "$VLA_ROOT/scripts/run_distributed_when_ready.sh" \
    "$VLA_ROOT/scripts/train_policies_fsdp.sh" full act_corrected
  checkpoint="$(find "$VLA_ROOT/outputs/train" -path '*_act_corrected_full_fsdp4/checkpoints/100000/pretrained_model/config.json' -printf '%h\n' | sort | tail -n1)"
fi
[[ -n "$checkpoint" ]] || { echo "corrected ACT checkpoint missing" >&2; exit 4; }

python "$SCRIPT_DIR/audit_checkpoint_real_data.py" \
  --checkpoint "$checkpoint" \
  --device cpu \
  --samples 32 \
  --max-mae-deg 8 \
  --output "$SIM_ROOT/outputs/act_corrected_real_data_audit.json"

python "$SCRIPT_DIR/04_run_policy_closed_loop.py" \
  --checkpoint "$checkpoint" \
  --device cpu \
  --action-steps 10 \
  --trials 10 \
  --seconds 30 \
  --output-dir "$SIM_ROOT/outputs/policy_closed_loop_act_corrected_final"

"$SCRIPT_DIR/generate_manifests.py"
"$SCRIPT_DIR/render_showcase.sh"

# Preserve the previously requested Diffusion/SmolVLA completion sequence.
exec "$SCRIPT_DIR/run_final_after_training.sh"
