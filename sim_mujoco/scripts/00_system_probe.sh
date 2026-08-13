#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/reports/system_audit.txt}"
mkdir -p "$(dirname "$OUT")"
{
  date --iso-8601=seconds
  hostname
  whoami
  id
  uname -a
  cat /etc/os-release
  lscpu
  free -h
  df -h "$ROOT"
  nvidia-smi
  command -v python git git-lfs ffmpeg cmake conda mamba docker podman || true
  python --version
  git --version
  ffmpeg -version
  cmake --version
  printf 'DISPLAY=%s WAYLAND_DISPLAY=%s\n' "${DISPLAY:-}" "${WAYLAND_DISPLAY:-}"
  ldconfig -p 2>/dev/null | grep -E 'lib(EGL|GL|OSMesa|glfw)' || true
  git -C "$ROOT/vendor/mujoco_menagerie" rev-parse HEAD
  "$ROOT/.sim_env/bin/python" -c 'import mujoco; print("mujoco", mujoco.__version__)'
} > "$OUT" 2>&1
echo "$OUT"
