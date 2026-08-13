#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "usage: $0 RUN_ID [--execute]" >&2; exit 2; }
TARGET="$(realpath -m "$ROOT/runs/$RUN_ID/frames")"
case "$TARGET" in "$ROOT"/runs/*/frames) ;; *) echo "refusing unsafe path: $TARGET" >&2; exit 3 ;; esac
find "$TARGET" -mindepth 1 -maxdepth 1 -type f -print 2>/dev/null || true
if [[ "${2:-}" == "--execute" && -d "$TARGET" ]]; then
  find "$TARGET" -mindepth 1 -maxdepth 1 -type f -delete
else
  echo "dry-run only; append --execute to delete listed regenerable frames"
fi
