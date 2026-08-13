#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

MODE="${1:-rrd}"
EPISODE="${2:-0}"
DATASET_ID="${VIZ_DATASET_ID:-local/so101_systematic50_eval_labeled}"
DATASET_ROOT="${VIZ_DATASET_ROOT:-$VLA_ROOT/data/lerobot/local/so101_systematic50_eval_labeled}"

case "$MODE" in
  rrd)
    mkdir -p "$VLA_ROOT/outputs/visualization/rrd"
    RERUN_TELEMETRY_ENABLED=false lerobot-dataset-viz \
      --repo-id "$DATASET_ID" --root "$DATASET_ROOT" --episode-index "$EPISODE" \
      --save 1 --output-dir "$VLA_ROOT/outputs/visualization/rrd" --num-workers 0
    ;;
  rerun-server)
    RERUN_TELEMETRY_ENABLED=false lerobot-dataset-viz \
      --repo-id "$DATASET_ID" --root "$DATASET_ROOT" --episode-index "$EPISODE" \
      --mode distant --web-port "${RERUN_WEB_PORT:-9090}" --grpc-port "${RERUN_GRPC_PORT:-9876}" \
      --num-workers 0
    ;;
  foxglove)
    lerobot-dataset-viz \
      --repo-id "$DATASET_ID" --root "$DATASET_ROOT" --episode-index "$EPISODE" \
      --display-mode foxglove --host 0.0.0.0 --web-port "${FOXGLOVE_PORT:-8765}" --num-workers 0
    ;;
  animation)
    python "$SCRIPT_DIR/export_real_rollout_animation.py"
    ;;
  *) echo "usage: $0 [rrd|rerun-server|foxglove|animation] [episode]" >&2; exit 2 ;;
esac
