#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
CONFIG="${1:-$VLA_ROOT/configs/robot/so101_controller.env}"
[[ -f "$CONFIG" ]] || { echo "copy and fill configs/robot/so101_controller.example.env" >&2; exit 2; }
source "$CONFIG"

for device in "$FOLLOWER_PORT" "$FRONT_CAMERA" "$WRIST_CAMERA"; do
  [[ -e "$device" ]] || { echo "missing controller device: $device" >&2; exit 3; }
done

lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{front: {type: opencv, index_or_path: '$FRONT_CAMERA', width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: '$WRIST_CAMERA', width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" \
  --task="$TASK" --policy.path="$POLICY_PATH" --display_data=true
