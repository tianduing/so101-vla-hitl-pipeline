#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

ROUND="${1:?usage: build_replay_dataset.sh ROUND BASE_ROOT CORRECTIVE_ROOT}"
BASE_ROOT="${2:?missing base dataset root}"
CORRECTIVE_ROOT="${3:?missing corrective dataset root}"
OUT="$VLA_ROOT/data/lerobot/local/so101_hitl_round_${ROUND}"

[[ -f "$BASE_ROOT/meta/info.json" ]] || { echo "invalid base dataset" >&2; exit 2; }
[[ -f "$CORRECTIVE_ROOT/meta/info.json" ]] || { echo "invalid corrective dataset" >&2; exit 2; }
[[ -e "$OUT" ]] && { echo "refusing to overwrite immutable replay dataset: $OUT" >&2; exit 3; }

lerobot-edit-dataset \
  --new_repo_id "local/so101_hitl_round_${ROUND}" --new_root "$OUT" \
  --operation.type merge \
  --operation.repo_ids "['local/base_round_${ROUND}','local/corrective_round_${ROUND}']" \
  --operation.roots "['$BASE_ROOT','$CORRECTIVE_ROOT']"
python "$SCRIPT_DIR/audit_dataset.py" \
  --root "$OUT" --output "$VLA_ROOT/outputs/reports/hitl_round_${ROUND}_audit.json"
