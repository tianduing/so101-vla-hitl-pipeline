#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

UNIT="${1:-so101-vla-fsdp4-full-train.service}"
POLL_SECONDS="${POSTPROCESS_POLL_SECONDS:-30}"

while true; do
  state="$(systemctl --user is-active "$UNIT" 2>/dev/null || true)"
  case "$state" in
    active|activating|reloading) sleep "$POLL_SECONDS" ;;
    *) break ;;
  esac
done

result="$(systemctl --user show "$UNIT" --property=Result --value 2>/dev/null || true)"
if [[ "$result" != "success" ]]; then
  echo "training unit did not finish successfully: unit=$UNIT state=$state result=${result:-unknown}" >&2
  exit 4
fi

exec "$SCRIPT_DIR/postprocess_fsdp_training.sh"
