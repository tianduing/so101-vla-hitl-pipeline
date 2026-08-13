#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEROBOT="$VLA_ROOT/src/lerobot"
PATCH="$VLA_ROOT/patches/lerobot-fsdp-trainable-fp32.patch"
MARKER='Normalizing mixed-dtype trainable parameters to fp32 for FSDP'

[[ -d "$LEROBOT/.git" ]] || { echo "LeRobot checkout missing: $LEROBOT" >&2; exit 2; }
if rg -Fq "$MARKER" "$LEROBOT/src/lerobot/scripts/lerobot_train.py"; then
  echo "local LeRobot FSDP compatibility patch already applied"
else
  git -C "$LEROBOT" apply "$PATCH"
  echo "applied local LeRobot FSDP compatibility patch"
fi
