#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

PUBLIC_DIR="$VLA_ROOT/data/external/so101_smolvla_thesis"
mkdir -p "$PUBLIC_DIR" "$VLA_ROOT/models/base/smolvla_base"

hf download lerobot/smolvla_base \
  --local-dir "$VLA_ROOT/models/base/smolvla_base"
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  config.json generation_config.json added_tokens.json chat_template.json \
  merges.txt preprocessor_config.json processor_config.json \
  special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json \
  --local-dir "$VLA_ROOT/models/base/smolvlm2_processor"
python "$SCRIPT_DIR/prepare_smolvla_offline.py"
hf download Shaibk/so101-smolvla-thesis \
  so101_smolvla_training_datasets_20260430.tar.zst \
  so101_smolvla_systematic50_eval_yellow_20260430.tar.zst \
  SHA256SUMS.txt release_manifest.json \
  --repo-type dataset --local-dir "$PUBLIC_DIR"

cd "$PUBLIC_DIR"
sha256sum -c SHA256SUMS.txt
