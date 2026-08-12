#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

SOURCE="${1:?usage: ingest_lab_dataset.sh /path/to/so101_vla_240eps}"
DEST="$VLA_ROOT/data/lerobot/so101_vla_240eps"
[[ -f "$SOURCE/meta/info.json" ]] || { echo "source is not a LeRobot dataset: $SOURCE" >&2; exit 2; }
[[ -e "$DEST" ]] && { echo "destination already exists; refusing to overwrite: $DEST" >&2; exit 3; }

mkdir -p "$DEST"
rsync -a --partial --info=progress2 "$SOURCE/" "$DEST/"
python "$SCRIPT_DIR/audit_dataset.py" \
  --root "$DEST" --output "$VLA_ROOT/outputs/reports/lab_240eps_audit.json"
find "$DEST" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$VLA_ROOT/manifests/lab_dataset_SHA256SUMS"
