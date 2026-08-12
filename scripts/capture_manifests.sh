#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

{
  uname -a
  cat /etc/os-release
  lscpu
  free -h
  df -h "$VLA_ROOT"
  nvidia-smi
} > "$VLA_ROOT/manifests/system_info.txt"

{
  printf 'lerobot '
  git -C "$VLA_ROOT/src/lerobot" rev-parse HEAD
  printf 'public_so101_reference '
  git -C "$VLA_ROOT/src/public_so101_reference" rev-parse HEAD
} > "$VLA_ROOT/manifests/git_commits.txt"

python -m pip freeze > "$VLA_ROOT/manifests/packages.txt"
find "$VLA_ROOT/models" "$VLA_ROOT/outputs/train" -type f -print0 \
  | sort -z | xargs -0 -r sha256sum > "$VLA_ROOT/manifests/artifact_SHA256SUMS"
