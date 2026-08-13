#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
[[ -f "$VLA_ROOT/configs/project.env" ]] && source "$VLA_ROOT/configs/project.env"

NUM_GPUS="${FSDP_NUM_GPUS:-4}"
MIN_FREE_MIB="${FSDP_MIN_FREE_MIB:-7000}"
MAX_UTIL="${FSDP_MAX_GPU_UTIL:-85}"
POLL="${GPU_POLL_SECONDS:-60}"
LOG="$VLA_ROOT/logs/system/fsdp_gpu_wait.log"
mkdir -p "$(dirname "$LOG")"

(( NUM_GPUS >= 2 )) || { echo "FSDP_NUM_GPUS must be at least 2" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }

while true; do
  mapfile -t candidates < <(
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
      | awk -F, -v min_free="$MIN_FREE_MIB" -v max_util="$MAX_UTIL" '
          {
            gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)
            if (($2 + 0) >= min_free && ($3 + 0) <= max_util) print $1 "," $2 "," $3
          }' \
      | sort -t, -k2,2nr
  )

  snapshot="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | tr '\n' ';' | sed 's/;$//')"
  printf '%s eligible=%s/%s min_free_mib=%s max_util=%s snapshot=%s\n' \
    "$(date --iso-8601=seconds)" "${#candidates[@]}" "$NUM_GPUS" "$MIN_FREE_MIB" "$MAX_UTIL" "$snapshot" \
    >> "$LOG"

  if (( ${#candidates[@]} >= NUM_GPUS )); then
    selected=()
    for ((i = 0; i < NUM_GPUS; i++)); do
      selected+=("${candidates[$i]%%,*}")
    done
    GPU_IDS="$(IFS=,; echo "${selected[*]}")"
    export GPU_IDS
    export FSDP_NUM_GPUS="$NUM_GPUS"
    printf '%s launching on physical GPUs %s\n' "$(date --iso-8601=seconds)" "$GPU_IDS" | tee -a "$LOG"
    exec "$@"
  fi
  sleep "$POLL"
done
