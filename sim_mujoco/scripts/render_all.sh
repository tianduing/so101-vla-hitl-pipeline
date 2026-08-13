#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_smoke.sh"
"$SCRIPT_DIR/run_replay.sh"
"$SCRIPT_DIR/run_expert_reference.sh"
"$SCRIPT_DIR/run_closed_loop.sh" "$@"
