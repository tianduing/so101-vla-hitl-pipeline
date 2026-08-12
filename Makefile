SHELL := /usr/bin/env bash

.PHONY: bootstrap download extract dataset risk verify test smoke full report status pipeline

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

report:
	source ./env.sh && ./scripts/finalize_reports.sh

status:
	systemctl --user status so101-vla-full-train --no-pager -l || true
	tail -n 20 logs/system/gpu_wait.log

pipeline:
	./scripts/run_pipeline.sh
