#!/usr/bin/env bash

set -euo pipefail

GPU_ID="${1:?usage: wait_for_specific_gpu.sh GPU_ID MIN_FREE_MIB COMMAND [ARG ...]}"
MIN_FREE_MIB="${2:?usage: wait_for_specific_gpu.sh GPU_ID MIN_FREE_MIB COMMAND [ARG ...]}"
shift 2
(( $# > 0 )) || { echo "command is required" >&2; exit 2; }
POLL_SECONDS="${GPU_CLAIM_POLL_SECONDS:-2}"

while true; do
  free_mib="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  echo "$(date --iso-8601=seconds) gpu=$GPU_ID free_mib=$free_mib required=$MIN_FREE_MIB"
  if (( free_mib >= MIN_FREE_MIB )); then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "$(date --iso-8601=seconds) claiming physical_gpu=$GPU_ID free_mib=$free_mib command=$*"
    exec "$@"
  fi
  sleep "$POLL_SECONDS"
done
