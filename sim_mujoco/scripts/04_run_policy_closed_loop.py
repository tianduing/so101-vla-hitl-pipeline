#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from so101_mujoco import So101MujocoEnv
from so101_mujoco.adapters import PolicyAdapter
from so101_mujoco.rendering.video import H264Writer, compose_four_panel, provenance_card


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def green_object_centroid(image: np.ndarray) -> tuple[float, float, int]:
    """Locate the green block using only the RGB observation."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    red, green, blue = (rgb[..., index].astype(np.float32) for index in range(3))
    mask = (green > 80) & (green > 1.25 * red) & (green > 1.25 * blue)
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise RuntimeError(f"green object detection failed: only {len(xs)} pixels")
    return float(xs.mean()), float(ys.mean()), int(len(xs))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alternate-checkpoint", type=Path, default=None)
    parser.add_argument("--green-centroid-x-threshold", type=float, default=195.0)
    parser.add_argument("--specialist-checkpoint", type=Path, default=None)
    parser.add_argument("--specialist-centroid-x-min", type=float, default=180.0)
    parser.add_argument("--specialist-centroid-x-max", type=float, default=187.0)
    parser.add_argument("--dataset-root", type=Path, default=root.parent / "data/lerobot/local/so101_green_block_grasp_train_all_prompt_v2")
    parser.add_argument("--config", type=Path, default=root / "configs/scene.yaml")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/policy_closed_loop")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-steps", type=int, default=None, help="Override checkpoint action queue length without changing weights")
    parser.add_argument("--action-repeat", type=int, default=1, help="Repeat each policy action for N control ticks")
    parser.add_argument("--arm-action-alpha", type=float, default=1.0, help="EMA gain for the five arm joints; gripper is not smoothed")
    parser.add_argument("--max-arm-target-step-deg", type=float, default=None, help="Optional per-tick arm target slew limit")
    parser.add_argument("--post-success-seconds", type=float, default=0.0, help="Freeze arm and maintain grip after first success, then verify stable hold")
    parser.add_argument("--hold-gripper-deg", type=float, default=40.0, help="Closed gripper target used by the post-success hold controller")
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None, help="ACT temporal ensembling coefficient; replans every step")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.action_repeat < 1:
        raise ValueError("action-repeat must be >= 1")
    if not 0 < args.arm_action_alpha <= 1:
        raise ValueError("arm-action-alpha must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter = PolicyAdapter(
        args.checkpoint,
        args.dataset_root,
        args.device,
        action_steps=args.action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    alternate_adapter = None
    if args.alternate_checkpoint is not None:
        alternate_adapter = PolicyAdapter(
            args.alternate_checkpoint, args.dataset_root, args.device,
            action_steps=args.action_steps,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        )
    specialist_adapter = None
    if args.specialist_checkpoint is not None:
        specialist_adapter = PolicyAdapter(
            args.specialist_checkpoint, args.dataset_root, args.device,
            action_steps=args.action_steps,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        )
    diag = adapter.diagnostics()
    task = "grasp the green block"
    warmup_env = So101MujocoEnv(args.config)
    warmup_obs = warmup_env.reset(seed=args.seed)
    warmup_latencies = adapter.warmup(warmup_obs, task, count=20)
    if alternate_adapter is not None:
        alternate_adapter.warmup(warmup_obs, task, count=20)
    if specialist_adapter is not None:
        specialist_adapter.warmup(warmup_obs, task, count=20)
    warmup_env.close()
    diag["warmup_calls"] = 20
    diag["warmup_latency_ms_p50"] = percentile(warmup_latencies, 50)
    diag["warmup_latency_ms_p95"] = percentile(warmup_latencies, 95)
    results = []
    all_latencies: list[float] = []
    for trial in range(args.trials):
        seed = args.seed + trial
        env = So101MujocoEnv(args.config)
        obs = env.reset(seed=seed)
        trial_adapter = adapter
        route = {"expert": "primary", "green_centroid_x": None, "green_centroid_y": None, "green_pixels": None}
        if alternate_adapter is not None:
            centroid_x, centroid_y, green_pixels = green_object_centroid(obs["observation.images.scene"])
            if centroid_x < args.green_centroid_x_threshold:
                trial_adapter = alternate_adapter
                route["expert"] = "alternate"
            route.update(green_centroid_x=centroid_x, green_centroid_y=centroid_y, green_pixels=green_pixels)
            if (
                specialist_adapter is not None
                and args.specialist_centroid_x_min <= centroid_x < args.specialist_centroid_x_max
            ):
                trial_adapter = specialist_adapter
                route["expert"] = "specialist"
        route.update(
            selected_checkpoint=str(trial_adapter.checkpoint),
            selected_checkpoint_sha256=trial_adapter.checkpoint_sha256,
        )
        trial_adapter.reset()
        step_logs = []
        latencies: list[float] = []
        replan_latencies: list[float] = []
        termination = "timeout"
        success = False
        first_success_step = None
        post_success_remaining = 0
        post_success_min_lift_m = None
        hold_action = None
        cached_policy_result = None
        last_command = np.asarray(obs["observation.state"], dtype=float).copy()
        trial_wall_start = time.perf_counter()
        selected_sha = trial_adapter.checkpoint_sha256
        video_path = args.output_dir / f"trial_{trial:02d}_{diag['policy_type']}_{selected_sha[:10]}.mp4"
        with H264Writer(video_path) as writer:
            card = provenance_card("POLICY CLOSED LOOP", [f"Policy: {diag['policy_type']}", f"Expert: {route['expert']}", f"Checkpoint SHA256: {selected_sha}", "Every ctrl action comes from Policy select_action/action queue", "MuJoCo simulation only / never real hardware"])
            for _ in range(60): writer.write(card)
            for step in range(int(args.seconds * env.control_hz)):
                try:
                    if hold_action is not None:
                        raw_action = hold_action.copy()
                        policy_result = {
                            "input_hash": "post_success_hold_controller",
                            "latency_ms": 0.0,
                            "select_action_call_id": trial_adapter.select_action_calls,
                            "policy_call_id": trial_adapter.model_replans,
                            "model_replan": False,
                            "executor_repeat": False,
                            "post_success_hold_controller": True,
                        }
                    elif cached_policy_result is None or step % args.action_repeat == 0:
                        policy_result = trial_adapter.select_action(obs, task)
                        raw_action = policy_result.pop("action")
                        cached_policy_result = {**policy_result, "action": raw_action.copy()}
                        policy_result["executor_repeat"] = False
                        policy_result["post_success_hold_controller"] = False
                    else:
                        raw_action = cached_policy_result["action"].copy()
                        policy_result = {key: value for key, value in cached_policy_result.items() if key != "action"}
                        policy_result["latency_ms"] = 0.0
                        policy_result["model_replan"] = False
                        policy_result["executor_repeat"] = True
                        policy_result["post_success_hold_controller"] = False
                    filtered_action = raw_action.copy()
                    filtered_action[:5] = last_command[:5] + args.arm_action_alpha * (raw_action[:5] - last_command[:5])
                    if args.max_arm_target_step_deg is not None:
                        delta = float(args.max_arm_target_step_deg)
                        filtered_action[:5] = np.clip(filtered_action[:5], last_command[:5] - delta, last_command[:5] + delta)
                    last_command = filtered_action.copy()
                    obs, _, instantaneous_success, _, info = env.step(filtered_action)
                except (ValueError, FloatingPointError, RuntimeError) as error:
                    termination = f"invalid_or_unstable:{type(error).__name__}:{error}"
                    break
                latency = float(policy_result["latency_ms"])
                latencies.append(latency); all_latencies.append(latency)
                if policy_result["model_replan"]:
                    replan_latencies.append(latency)
                current_lift_m = float(info["object_xyz"][2]) - float(env.object_rest_z)
                if first_success_step is None and instantaneous_success:
                    first_success_step = step
                    if args.post_success_seconds > 0:
                        post_success_remaining = max(1, round(args.post_success_seconds * env.control_hz))
                        hold_action = np.asarray(info["state_deg"], dtype=float)
                        # A policy may cross the lift threshold while its queued
                        # gripper target is still only partly closed.  Reusing
                        # that transient target can let the block slowly slip.
                        # Use the demonstrated fully-closed target during the
                        # optional stability test instead.
                        hold_action[-1] = min(
                            float(args.hold_gripper_deg),
                            float(raw_action[-1]),
                            float(hold_action[-1]),
                        )
                        post_success_min_lift_m = current_lift_m
                    else:
                        success = True
                elif hold_action is not None:
                    post_success_min_lift_m = min(float(post_success_min_lift_m), current_lift_m)
                    post_success_remaining -= 1
                    if post_success_remaining <= 0:
                        threshold = float(env.config["success"]["pick_lift_m"])
                        success = bool(post_success_min_lift_m >= threshold)
                        termination = "stable_hold_success" if success else "post_success_drop"
                log = {"trial": trial, "seed": seed, "step": step, "observation_timestamp": info["sim_time"], "task": task,
                       "raw_policy_action": raw_action.tolist(), "executed_filtered_action": filtered_action.tolist(),
                       "instantaneous_success": bool(instantaneous_success), **policy_result, **info}
                step_logs.append(log)
                lines = [
                    "POLICY CLOSED LOOP",
                    f"policy={diag['policy_type']} expert={route['expert']} checkpoint={selected_sha[:12]}",
                    f"trial={trial}/{args.trials-1} seed={seed} step={step} sim={info['sim_time']:.2f}s",
                    f"select_call={policy_result['select_action_call_id']} model_replan={policy_result['policy_call_id']} fresh={policy_result['model_replan']}",
                    f"latency={latency:.1f}ms control={env.control_hz:.0f}Hz physics={1/env.model.opt.timestep:.0f}Hz",
                    "state(deg): " + " ".join(f"{x:6.1f}" for x in info["state_deg"]),
                    "policy action: " + " ".join(f"{x:6.1f}" for x in raw_action),
                    "executed filt: " + " ".join(f"{x:6.1f}" for x in filtered_action),
                    "object xyz: " + " ".join(f"{x:.3f}" for x in info["object_xyz"]),
                    f"picked={env.ever_picked} success={success} hold_left={post_success_remaining} contacts={info['contacts']}",
                    f"instruction: {task}",
                ]
                writer.write(compose_four_panel(env.render("third_person"), obs["sim.wrist"], obs["observation.images.scene"], lines))
                if success or termination == "post_success_drop":
                    if termination == "timeout": termination = "success"
                    break
        env.close()
        wall_seconds = time.perf_counter() - trial_wall_start
        contact_max = max((row["contact_force_max_n"] for row in step_logs), default=0.0)
        contact_mean = statistics.fmean((row["contact_force_mean_n"] for row in step_logs)) if step_logs else 0.0
        saturation_mean = statistics.fmean((row["action_saturation_fraction"] for row in step_logs)) if step_logs else 0.0
        initial_object_z_m = float(step_logs[0]["object_xyz"][2]) if step_logs else 0.0
        max_object_z_m = max((float(row["object_xyz"][2]) for row in step_logs), default=0.0)
        result = {
            "trial": trial, "seed": seed, "success": success, "termination": termination, "steps": len(step_logs),
            "first_success_step": first_success_step,
            "post_success_seconds_requested": args.post_success_seconds,
            "post_success_min_lift_m": post_success_min_lift_m,
            "select_action_calls": trial_adapter.select_action_calls, "model_replans": trial_adapter.model_replans,
            "visual_expert_route": route,
            "latency_ms_p50": percentile(latencies, 50), "latency_ms_p95": percentile(latencies, 95),
            "model_replan_latency_ms_p50": percentile(replan_latencies, 50),
            "model_replan_latency_ms_p95": percentile(replan_latencies, 95),
            "wall_seconds": wall_seconds,
            "sim_seconds": float(step_logs[-1]["sim_time"]) if step_logs else 0.0,
            "realtime_factor": (float(step_logs[-1]["sim_time"]) / wall_seconds) if step_logs and wall_seconds else 0.0,
            "action_saturation_fraction_mean": saturation_mean,
            "contact_force_max_n": contact_max,
            "contact_force_mean_n": contact_mean,
            "initial_object_z_m": initial_object_z_m,
            "max_object_z_m": max_object_z_m,
            "peak_lift_m": max_object_z_m - initial_object_z_m,
            "video": str(video_path.resolve()), "step_log": str(video_path.with_suffix(".jsonl").resolve()),
        }
        with video_path.with_suffix(".jsonl").open("w") as stream:
            for row in step_logs: stream.write(json.dumps(row) + "\n")
        video_path.with_suffix(".result.json").write_text(json.dumps({**diag, **result}, indent=2) + "\n")
        results.append(result)
        print(json.dumps(result))
    summary = {
        **diag, "mode": "POLICY_CLOSED_LOOP", "task": task, "num_trials": len(results),
        "successes": sum(int(r["success"]) for r in results), "success_rate": sum(int(r["success"]) for r in results) / len(results),
        "latency_ms_p50": percentile(all_latencies, 50), "latency_ms_p95": percentile(all_latencies, 95), "results": results,
        "executor": {"action_repeat": args.action_repeat, "arm_action_alpha": args.arm_action_alpha,
                     "max_arm_target_step_deg": args.max_arm_target_step_deg,
                     "post_success_seconds": args.post_success_seconds,
                     "hold_gripper_deg": args.hold_gripper_deg},
    }
    if alternate_adapter is not None:
        summary["visual_expert_routing"] = {
            "uses_simulator_privileged_state": False,
            "signal": "initial observation.images.scene green RGB centroid",
            "primary_if_centroid_x_ge": args.green_centroid_x_threshold,
            "alternate_checkpoint": str(args.alternate_checkpoint.resolve()),
            "alternate_checkpoint_sha256": alternate_adapter.checkpoint_sha256,
        }
    if specialist_adapter is not None:
        summary["visual_expert_routing"].update({
            "specialist_centroid_x_range": [args.specialist_centroid_x_min, args.specialist_centroid_x_max],
            "specialist_checkpoint": str(args.specialist_checkpoint.resolve()),
            "specialist_checkpoint_sha256": specialist_adapter.checkpoint_sha256,
        })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    failures = [Path(r["video"]) for r in results if not r["success"]]
    successes = [Path(r["video"]) for r in results if r["success"]]
    if failures:
        link = args.output_dir / "representative_failure.mp4"
        if not link.exists(): link.symlink_to(failures[0].name)
    if successes:
        link = args.output_dir / "best_success.mp4"
        if not link.exists(): link.symlink_to(successes[0].name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
