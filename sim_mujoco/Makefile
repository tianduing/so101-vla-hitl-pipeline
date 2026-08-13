SHELL := /usr/bin/env bash
.PHONY: setup smoke gripper replay expert closed-loop pick-place-box pick-place-box-dataset tests showcase status

setup:
	./scripts/setup.sh
smoke:
	./scripts/run_smoke.sh
gripper:
	source ./env.sh && python ./scripts/02_gripper_mapping_test.py
replay:
	./scripts/run_replay.sh
expert:
	./scripts/run_expert_reference.sh
closed-loop:
	./scripts/run_closed_loop.sh
pick-place-box:
	./scripts/run_pick_place_box.sh
pick-place-box-dataset:
	./scripts/build_pick_place_box_dataset.sh
tests:
	./scripts/run_tests.sh
showcase:
	./scripts/render_showcase.sh
status:
	./scripts/resume.sh
