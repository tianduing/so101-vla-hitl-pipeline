#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LEROBOT_URL="https://github.com/huggingface/lerobot.git"
LEROBOT_COMMIT="30da8e687a6dfc617fcd94afc367ac7071c376ce"
REFERENCE_URL="https://github.com/Shaibk/so101-smolvla-thesis.git"
REFERENCE_COMMIT="648de23235c15085ae0ce5887c7c0c2e8908b0ac"

checkout_locked() {
  local url="$1"
  local destination="$2"
  local commit="$3"

  if [[ -d "$destination/.git" ]]; then
    local current
    current="$(git -C "$destination" rev-parse HEAD)"
    if [[ "$current" != "$commit" ]]; then
      echo "source checkout has unexpected commit: $destination" >&2
      echo "expected $commit, found $current; move it aside and rerun" >&2
      exit 3
    fi
    echo "locked source already present: $destination ($commit)"
    return
  fi
  [[ ! -e "$destination" ]] || {
    echo "refusing to replace non-git path: $destination" >&2
    exit 3
  }

  mkdir -p "$(dirname "$destination")"
  git clone --filter=blob:none --no-checkout "$url" "$destination"
  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

checkout_locked "$LEROBOT_URL" "$VLA_ROOT/src/lerobot" "$LEROBOT_COMMIT"
checkout_locked "$REFERENCE_URL" "$VLA_ROOT/src/public_so101_reference" "$REFERENCE_COMMIT"

mkdir -p "$VLA_ROOT/manifests"
{
  printf 'lerobot %s\n' "$LEROBOT_COMMIT"
  printf 'public_so101_reference %s\n' "$REFERENCE_COMMIT"
} > "$VLA_ROOT/manifests/git_commits.txt"
