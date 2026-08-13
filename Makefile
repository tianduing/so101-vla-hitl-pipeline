SHELL := /usr/bin/env bash

.PHONY: bootstrap download extract dataset risk verify test smoke full fsdp-smoke fsdp-full report viz-animation viz-rrd status pipeline

bootstrap:
	./scripts/bootstrap.sh

download:
	source ./env.sh && ./scripts/download_resources.sh

extract:
	source ./env.sh && ./scripts/extract_public_data.sh

dataset:
	source ./env.sh && ./scripts/build_public_dataset.sh

risk:
	source ./env.sh && python ./scripts/build_risk_dataset.py
	source ./env.sh && python ./scripts/train_risk_model.py --dataset-root data/lerobot/local/so101_systematic50_eval_labeled --labels data/lerobot/local/so101_systematic50_eval_labels.csv --output models/checkpoints/risk_model/systematic50_risk_mlp.pt --device cpu

verify:
	source ./env.sh && ./scripts/verify_install.sh

test:
	source ./env.sh && python -m pytest -q tests

smoke:
	source ./env.sh && ./scripts/run_when_gpu_free.sh ./scripts/train_policies.sh smoke all

full:
	source ./env.sh && ./scripts/run_when_gpu_free.sh ./scripts/train_policies.sh full all
	source ./env.sh && ./scripts/finalize_reports.sh

fsdp-smoke:
	source ./env.sh && FSDP_NUM_GPUS=4 GPU_IDS=0,1,2,3 ./scripts/train_policies_fsdp.sh smoke all

fsdp-full:
	source ./env.sh && ./scripts/run_distributed_when_ready.sh ./scripts/train_policies_fsdp.sh full all

viz-animation:
	source ./env.sh && ./scripts/visualize_episode.sh animation

viz-rrd:
	source ./env.sh && ./scripts/visualize_episode.sh rrd 0

report:
	source ./env.sh && ./scripts/finalize_reports.sh

status:
	systemctl --user show so101-vla-fsdp4-full-train --property=ActiveState,SubState,Result,ExecMainStartTimestamp --no-pager || true
	systemctl --user show so101-vla-fsdp4-postprocess --property=ActiveState,SubState,Result --no-pager || true
	tail -n 10 logs/system/fsdp_gpu_wait.log 2>/dev/null || true
	find outputs/train/.run_metadata -type f -name train.log -print0 2>/dev/null | xargs -0 -r ls -1t | head -n 1 | xargs -r tail -n 5

pipeline:
	./scripts/run_pipeline.sh
