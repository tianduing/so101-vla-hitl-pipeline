#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

if [[ "${1:-}" == "--checkpoint" ]]; then
  exec python "$SCRIPT_DIR/04_run_policy_closed_loop.py" "$@"
fi

checkpoint="$(find "$VLA_ROOT/outputs/train" -path '*_smolvla_full_fsdp4/checkpoints/last/pretrained_model/config.json' -printf '%h\n' 2>/dev/null | sort | tail -n 1)"
if [[ -z "$checkpoint" ]]; then
  checkpoint="$(find "$VLA_ROOT/outputs/train" -path '*_act_full_fsdp4/checkpoints/*/pretrained_model/config.json' -printf '%h\n' | sort | tail -n 1)"
fi
[[ -n "$checkpoint" ]] || { echo "no usable full checkpoint" >&2; exit 3; }
echo "selected checkpoint: $checkpoint"
exec python "$SCRIPT_DIR/04_run_policy_closed_loop.py" --checkpoint "$checkpoint" "$@"
