#!/usr/bin/env python3
"""Low-memory ACT adaptation on MuJoCo-rendered successful trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from safetensors.torch import save_file
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--save-freq", type=int, default=1000)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    delta_timestamps = {"action": [index / 30 for index in range(50)]}
    dataset = LeRobotDataset(
        "local/so101_act_sim_adaptation",
        root=args.dataset_root.resolve(),
        delta_timestamps=delta_timestamps,
    )
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = str(checkpoint)
    config.device = args.device
    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**config.input_features, **config.output_features},
                "norm_map": config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": config.output_features,
                "norm_map": config.normalization_mapping,
            }
        },
    )
    groups = policy.get_optim_params()
    for index, group in enumerate(groups):
        group["lr"] = args.backbone_lr if index == 1 else args.lr
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    history = []
    iterator = iter(loader)
    policy.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = preprocessor(batch)
        loss, loss_dict = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % args.log_freq == 0:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                **{
                    key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                    for key, value in loss_dict.items()
                },
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        if step % args.save_freq == 0 or step == args.steps:
            save_dir = output / f"step_{step:06d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            # save_pretrained also writes the exact runtime config used here.
            policy.save_pretrained(save_dir)
            preprocessor.save_pretrained(save_dir)
            postprocessor.save_pretrained(save_dir)
            (save_dir / "adaptation.json").write_text(
                json.dumps(
                    {
                        "source_checkpoint": str(checkpoint),
                        "dataset_root": str(args.dataset_root.resolve()),
                        "steps": step,
                        "lr": args.lr,
                        "backbone_lr": args.backbone_lr,
                        "batch_size": args.batch_size,
                        "history": history,
                    },
                    indent=2,
                )
                + "\n"
            )
    (output / "summary.json").write_text(json.dumps({"history": history}, indent=2) + "\n")


if __name__ == "__main__":
    main()
