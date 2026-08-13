#!/usr/bin/env bash
set -euo pipefail
systemctl --user status so101-vla-fsdp4-full-train.service --no-pager -n 5 || true
systemctl --user status so101-mujoco-final-after-training.service --no-pager -n 5 || true
systemctl --user status so101-mujoco-act60k-eval.service --no-pager -n 5 || true
