#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

POLICY="${1:?usage: train_single_gpu_with_batch_backoff.sh POLICY BATCH [BATCH ...]}"
shift
(( $# > 0 )) || { echo "at least one batch size is required" >&2; exit 2; }

for batch in "$@"; do
  echo "$(date --iso-8601=seconds) starting policy=$POLICY batch_size=$batch gpu=${CUDA_VISIBLE_DEVICES:-unset}"
  if BATCH_SIZE="$batch" NUM_WORKERS="${NUM_WORKERS:-8}" \
      "$SCRIPT_DIR/train_policies.sh" full "$POLICY"; then
    echo "$(date --iso-8601=seconds) completed policy=$POLICY batch_size=$batch"
    exit 0
  fi
  status=$?
  echo "$(date --iso-8601=seconds) failed policy=$POLICY batch_size=$batch status=$status; trying lower batch" >&2
  sleep 5
done

echo "all batch sizes failed for policy=$POLICY" >&2
exit 4
