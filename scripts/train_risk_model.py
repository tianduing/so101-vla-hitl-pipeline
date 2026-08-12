#!/usr/bin/env python3
"""Train a compact trajectory-success model from LeRobot state/action parquet rows."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class RiskMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_labels(path: Path) -> dict[int, float]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        labels = {int(row["episode_index"]): float(row["success"]) for row in rows}
    if not labels:
        raise ValueError("label file is empty")
    return labels


def as_vector(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="CSV columns: episode_index,success")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    labels = load_labels(args.labels)
    parts = []
    required = ["episode_index", "observation.state", "action"]
    for parquet in sorted((args.dataset_root / "data").rglob("*.parquet")):
        frame = pd.read_parquet(parquet, columns=required)
        parts.append(frame[frame["episode_index"].isin(labels)])
    if not parts:
        raise SystemExit("no matching parquet rows")
    frame = pd.concat(parts, ignore_index=True)
    state = np.stack(frame["observation.state"].map(as_vector))
    action = np.stack(frame["action"].map(as_vector))
    delta = np.zeros_like(action)
    for _, indices in frame.groupby("episode_index").groups.items():
        ordered = np.asarray(list(indices))
        delta[ordered[1:]] = action[ordered[1:]] - action[ordered[:-1]]
    x = np.concatenate([state, action, delta], axis=1)
    y = frame["episode_index"].map(labels).to_numpy(np.float32)

    episodes = sorted(labels)
    random.shuffle(episodes)
    validation = set(episodes[: max(1, len(episodes) // 5)])
    val_mask = frame["episode_index"].isin(validation).to_numpy()
    train_mask = ~val_mask
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0).clip(min=1e-6)
    x = (x - mean) / std

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = RiskMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    x_train = torch.from_numpy(x[train_mask]).to(device)
    y_train = torch.from_numpy(y[train_mask]).to(device)
    for _ in range(args.epochs):
        order = torch.randperm(len(x_train), device=device)
        for start in range(0, len(order), 512):
            idx = order[start : start + 512]
            loss = criterion(model(x_train[idx]), y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(x[val_mask]).to(device))
        val_y = torch.from_numpy(y[val_mask]).to(device)
        val_loss = float(criterion(val_logits, val_y))
        val_accuracy = float(((val_logits.sigmoid() >= 0.5) == (val_y >= 0.5)).float().mean())
        val_probabilities = val_logits.sigmoid().cpu().numpy()
    val_frame = frame.loc[val_mask, ["episode_index"]].copy()
    val_frame["probability"] = val_probabilities
    episode_probability = val_frame.groupby("episode_index")["probability"].mean()
    episode_truth = np.asarray([labels[int(index)] for index in episode_probability.index])
    episode_prediction = (episode_probability.to_numpy() >= 0.5).astype(np.float32)
    episode_accuracy = float((episode_prediction == episode_truth).mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.cpu().state_dict(),
        "input_dim": x.shape[1],
        "mean": mean,
        "std": std,
        "state_dim": state.shape[1],
        "action_dim": action.shape[1],
        "threshold": 0.35,
        "consecutive_steps": 3,
    }, args.output)
    metrics = {
        "rows": len(frame),
        "episodes": len(episodes),
        "success_episodes": int(sum(value >= 0.5 for value in labels.values())),
        "failure_episodes": int(sum(value < 0.5 for value in labels.values())),
        "validation_episodes": len(validation),
        "validation_episode_ids": sorted(validation),
        "device": str(device),
        "val_loss": val_loss,
        "val_frame_accuracy": val_accuracy,
        "val_episode_accuracy": episode_accuracy,
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
