#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

"$SCRIPT_DIR/bootstrap.sh"
source "$VLA_ROOT/env.sh"
"$SCRIPT_DIR/download_resources.sh"
"$SCRIPT_DIR/extract_public_data.sh"
"$SCRIPT_DIR/build_public_dataset.sh"
python "$SCRIPT_DIR/build_risk_dataset.py"
if [[ ! -f "$VLA_ROOT/models/checkpoints/risk_model/systematic50_risk_mlp.pt" ]]; then
  python "$SCRIPT_DIR/train_risk_model.py" \
    --dataset-root "$VLA_ROOT/data/lerobot/local/so101_systematic50_eval_labeled" \
    --labels "$VLA_ROOT/data/lerobot/local/so101_systematic50_eval_labels.csv" \
    --output "$VLA_ROOT/models/checkpoints/risk_model/systematic50_risk_mlp.pt" \
    --device cpu
fi
python -m pytest -q "$VLA_ROOT/tests"
"$SCRIPT_DIR/run_when_gpu_free.sh" "$SCRIPT_DIR/train_policies.sh" smoke all
"$SCRIPT_DIR/finalize_reports.sh"
