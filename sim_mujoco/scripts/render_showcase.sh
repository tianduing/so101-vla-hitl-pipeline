#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

policy_video="$(find "$ROOT/outputs" -path '*policy_closed_loop_smolvla_final*/representative_failure.mp4' -o -path '*policy_closed_loop_smolvla_final*/best_success.mp4' | sort | head -n1)"
if [[ -z "$policy_video" ]]; then
  policy_video="$(find "$ROOT/outputs" -path '*policy_closed_loop_act_corrected_final/best_success.mp4' -o -path '*policy_closed_loop_act_corrected_final/representative_failure.mp4' | sort | head -n1)"
fi
if [[ -z "$policy_video" ]]; then
  policy_video="$(find "$ROOT/outputs" -path '*policy_closed_loop_act100k/closest_attempt.mp4' -o -path '*policy_closed_loop_act100k/best_success.mp4' -o -path '*policy_closed_loop_act100k/representative_failure.mp4' | sort | head -n1)"
fi
if [[ -z "$policy_video" ]]; then
  policy_video="$(find "$ROOT/outputs" -path '*policy_closed_loop_act60k/representative_failure.mp4' -o -path '*policy_closed_loop_act60k/best_success.mp4' | sort | head -n1)"
fi
if [[ -z "$policy_video" ]]; then
  policy_video="$(find "$ROOT/outputs" -path '*policy_closed_loop_debug_cpu/trial_*.mp4' | sort | head -n1)"
fi
[[ -n "$policy_video" ]] || { echo "no policy video available" >&2; exit 3; }

ffmpeg -y -loglevel error \
  -i "$ROOT/outputs/scene_smoke.mp4" \
  -i "$ROOT/outputs/real_trajectory_replay.mp4" \
  -i "$ROOT/outputs/scripted_expert_reference.mp4" \
  -i "$policy_video" \
  -filter_complex '[0:v]fps=30,scale=1280:720,setsar=1[v0];[1:v]fps=30,scale=1280:720,setsar=1[v1];[2:v]fps=30,scale=1280:720,setsar=1[v2];[3:v]fps=30,scale=1280:720,setsar=1[v3];[v0][v1][v2][v3]concat=n=4:v=1:a=0[out]' \
  -map '[out]' -c:v libx264 -preset medium -crf 21 -pix_fmt yuv420p -movflags +faststart \
  "$ROOT/outputs/so101_mujoco_showcase.mp4"

ffmpeg -y -loglevel error \
  -ss 3 -i "$ROOT/outputs/scene_smoke.mp4" \
  -ss 3 -i "$ROOT/outputs/real_trajectory_replay.mp4" \
  -ss 8 -i "$ROOT/outputs/scripted_expert_reference.mp4" \
  -ss 5 -i "$policy_video" \
  -filter_complex '[0:v]scale=640:360[a];[1:v]scale=640:360[b];[2:v]scale=640:360[c];[3:v]scale=640:360[d];[a][b]hstack[top];[c][d]hstack[bottom];[top][bottom]vstack[out]' \
  -map '[out]' -frames:v 1 "$ROOT/outputs/contact_sheet.png"

"$ROOT/scripts/generate_manifests.py"
echo "$ROOT/outputs/so101_mujoco_showcase.mp4"
