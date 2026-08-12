#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

PUBLIC_DIR="$VLA_ROOT/data/external/so101_smolvla_thesis"
TRAIN_ARCHIVE="$PUBLIC_DIR/so101_smolvla_training_datasets_20260430.tar.zst"
EVAL_ARCHIVE="$PUBLIC_DIR/so101_smolvla_systematic50_eval_yellow_20260430.tar.zst"

[[ -s "$TRAIN_ARCHIVE" ]] || { echo "missing $TRAIN_ARCHIVE" >&2; exit 2; }
[[ -s "$EVAL_ARCHIVE" ]] || { echo "missing $EVAL_ARCHIVE" >&2; exit 2; }

cd "$PUBLIC_DIR"
sha256sum -c SHA256SUMS.txt

extract_once() {
  local archive="$1"
  local marker="$2"
  local digest
  digest="$(sha256sum "$archive" | cut -d' ' -f1)"
  if [[ -f "$marker" ]] && [[ "$(<"$marker")" == "$digest" ]]; then
    echo "already extracted and verified: $(basename "$archive")"
    return
  fi
  if tar --zstd -tf "$archive" | awk '/^\// || /(^|\/)\.\.($|\/)/ { bad=1 } END { exit bad ? 0 : 1 }'; then
    echo "unsafe archive member detected: $archive" >&2
    exit 3
  fi
  tar --zstd -xf "$archive" -C "$VLA_ROOT"
  printf '%s\n' "$digest" > "$marker"
}

extract_once "$TRAIN_ARCHIVE" "$PUBLIC_DIR/.training_extracted.sha256"
extract_once "$EVAL_ARCHIVE" "$PUBLIC_DIR/.evaluation_extracted.sha256"
# macOS AppleDouble sidecars are archive metadata, not Parquet/video payloads.
find "$VLA_ROOT/data/lerobot/local" "$VLA_ROOT/outputs/eval" \
  -type f -name '._*' -delete
python "$SCRIPT_DIR/audit_dataset.py" \
  --root "$VLA_ROOT/data/lerobot/local" \
  --output "$VLA_ROOT/outputs/reports/public_dataset_audit.json"
python "$SCRIPT_DIR/summarize_public_eval.py" \
  --root "$VLA_ROOT/outputs/eval" \
  --output "$VLA_ROOT/outputs/reports/public_eval_summary.json"
