#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

DATA="$VLA_ROOT/data/lerobot/local"
ALL_ROOT="$DATA/so101_green_block_grasp_train_all_prompt_v2"
YELLOW_ROOT="$DATA/so101_green_block_grasp_yellow_focus_prompt_v2"

NO_YELLOW=(
  so101_green_block_grasp_no_board_random_v1
  so101_green_block_grasp_no_board_lateral_v1
  so101_green_block_grasp_no_board_edge_v1
)
YELLOW=(
  so101_green_block_grasp_yellow_distractor_v1
  so101_green_block_grasp_yellow_distractor_v2
  so101_green_block_grasp_yellow_lcr_supplement_v1
)

for ds in "${NO_YELLOW[@]}"; do
  out="${ds}_prompt_v2"
  [[ -f "$DATA/$out/meta/info.json" ]] && continue
  [[ -e "$DATA/$out" ]] && { echo "incomplete output exists: $DATA/$out" >&2; exit 3; }
  cp -a --reflink=auto "$DATA/$ds" "$DATA/$out"
  lerobot-edit-dataset \
    --repo_id "local/$out" --root "$DATA/$out" \
    --operation.type modify_tasks --operation.new_task "grasp the green block"
done

for ds in "${YELLOW[@]}"; do
  out="${ds}_prompt_v2"
  [[ -f "$DATA/$out/meta/info.json" ]] && continue
  [[ -e "$DATA/$out" ]] && { echo "incomplete output exists: $DATA/$out" >&2; exit 3; }
  cp -a --reflink=auto "$DATA/$ds" "$DATA/$out"
  lerobot-edit-dataset \
    --repo_id "local/$out" --root "$DATA/$out" \
    --operation.type modify_tasks \
    --operation.new_task "grasp the green block, ignore the yellow block"
done

PROMPT_DATASETS=()
PROMPT_ROOTS=()
for ds in "${NO_YELLOW[@]}" "${YELLOW[@]}"; do
  PROMPT_DATASETS+=("local/${ds}_prompt_v2")
  PROMPT_ROOTS+=("$DATA/${ds}_prompt_v2")
done

if [[ ! -f "$ALL_ROOT/meta/info.json" ]]; then
  [[ -e "$ALL_ROOT" ]] && { echo "incomplete merged output exists: $ALL_ROOT" >&2; exit 3; }
  DATASET_LIST="$(printf "'%s'," "${PROMPT_DATASETS[@]}")"
  ROOT_LIST="$(printf "'%s'," "${PROMPT_ROOTS[@]}")"
  lerobot-edit-dataset \
    --new_repo_id local/so101_green_block_grasp_train_all_prompt_v2 \
    --new_root "$ALL_ROOT" --operation.type merge \
    --operation.concatenate_videos=false \
    --operation.concatenate_data=false \
    --operation.repo_ids "[${DATASET_LIST%,}]" \
    --operation.roots "[${ROOT_LIST%,}]"
fi

if [[ ! -f "$YELLOW_ROOT/meta/info.json" ]]; then
  [[ -e "$YELLOW_ROOT" ]] && { echo "incomplete merged output exists: $YELLOW_ROOT" >&2; exit 3; }
  YELLOW_IDS=()
  YELLOW_ROOTS=()
  for ds in "${YELLOW[@]}"; do
    YELLOW_IDS+=("local/${ds}_prompt_v2")
    YELLOW_ROOTS+=("$DATA/${ds}_prompt_v2")
  done
  DATASET_LIST="$(printf "'%s'," "${YELLOW_IDS[@]}")"
  ROOT_LIST="$(printf "'%s'," "${YELLOW_ROOTS[@]}")"
  lerobot-edit-dataset \
    --new_repo_id local/so101_green_block_grasp_yellow_focus_prompt_v2 \
    --new_root "$YELLOW_ROOT" --operation.type merge \
    --operation.concatenate_videos=false \
    --operation.concatenate_data=false \
    --operation.repo_ids "[${DATASET_LIST%,}]" \
    --operation.roots "[${ROOT_LIST%,}]"
fi

python "$SCRIPT_DIR/audit_dataset.py" \
  --root "$ALL_ROOT" --output "$VLA_ROOT/outputs/reports/merged_dataset_audit.json"
