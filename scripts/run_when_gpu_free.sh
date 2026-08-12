#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
[[ -f "$VLA_ROOT/configs/project.env" ]] && source "$VLA_ROOT/configs/project.env"

MIN_FREE="${GPU_MIN_FREE_MIB:-18000}"
POLL="${GPU_POLL_SECONDS:-60}"
LOG="$VLA_ROOT/logs/system/gpu_wait.log"
mkdir -p "$(dirname "$LOG")"

while true; do
  GPU_LINE="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n1)"
  GPU_INDEX="${GPU_LINE%%,*}"
  FREE_MIB="${GPU_LINE##*,}"
  GPU_INDEX="${GPU_INDEX// /}"
  FREE_MIB="${FREE_MIB// /}"
  printf '%s gpu=%s free_mib=%s required=%s\n' "$(date --iso-8601=seconds)" "$GPU_INDEX" "$FREE_MIB" "$MIN_FREE" >> "$LOG"
  if (( FREE_MIB >= MIN_FREE )); then
    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    exec "$@"
  fi
  sleep "$POLL"
done
