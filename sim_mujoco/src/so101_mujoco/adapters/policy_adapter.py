from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from safetensors import safe_open


class PolicyAdapter:
    def __init__(
        self,
        checkpoint: str | Path,
        dataset_root: str | Path,
        device: str = "cuda",
        action_steps: int | None = None,
        temporal_ensemble_coeff: float | None = None,
    ):
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {self.checkpoint}")
        self.dataset = LeRobotDataset("local/so101_mujoco_policy_schema", root=Path(dataset_root).resolve())
        self.config = PreTrainedConfig.from_pretrained(self.checkpoint)
        # The serialized config records how training was initialized (None for
        # scratch policies, or a base model for VLA fine-tuning).  It does not
        # necessarily point back to the checkpoint that contains the newly
        # trained weights.  make_policy() only loads weights when
        # config.pretrained_path is set, so inference must explicitly select
        # the checkpoint requested by the caller.
        self.config.pretrained_path = str(self.checkpoint)
        self.config.device = device
        if action_steps is not None:
            if action_steps < 1 or action_steps > int(getattr(self.config, "chunk_size", 1)):
                raise ValueError(f"action_steps must be in [1, {self.config.chunk_size}], got {action_steps}")
            self.config.n_action_steps = action_steps
        if temporal_ensemble_coeff is not None:
            if self.config.type != "act":
                raise ValueError("temporal ensembling is only supported for ACT")
            self.config.n_action_steps = 1
            self.config.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.device = device
        self.rename_map = {"observation.images.scene": "observation.images.camera1"} if self.config.type == "smolvla" else None
        self.policy = make_policy(self.config, ds_meta=self.dataset.meta, rename_map=self.rename_map)
        self.loaded_weight_probe = self._verify_checkpoint_weights_loaded()
        overrides: dict[str, Any] = {"device_processor": {"device": device}}
        if self.rename_map:
            overrides["rename_observations_processor"] = {"rename_map": self.rename_map}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config, pretrained_path=self.checkpoint, preprocessor_overrides=overrides
        )
        self.policy.eval()
        self.checkpoint_sha256 = self._weight_hash()
        self.select_action_calls = 0
        self.model_replans = 0

    def _weight_hash(self) -> str:
        digest = hashlib.sha256()
        weights = sorted(self.checkpoint.glob("*.safetensors"))
        if not weights:
            raise FileNotFoundError(f"no safetensors in {self.checkpoint}")
        for path in weights:
            digest.update(path.name.encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def _verify_checkpoint_weights_loaded(self) -> dict[str, Any]:
        """Fail fast if inference silently instantiated the wrong policy weights."""
        model_files = sorted(self.checkpoint.glob("model*.safetensors"))
        if not model_files:
            raise FileNotFoundError(f"no model safetensors in {self.checkpoint}")

        state = self.policy.state_dict()
        candidates: list[tuple[int, Path, str]] = []
        for path in model_files:
            with safe_open(path, framework="pt", device="cpu") as tensors:
                for key in tensors.keys():
                    if key in state:
                        shape = tuple(tensors.get_slice(key).get_shape())
                        candidates.append((int(np.prod(shape)), path, key))
        if not candidates:
            raise RuntimeError(
                f"checkpoint tensor names do not match the instantiated policy: {self.checkpoint}"
            )

        _, path, key = min(candidates, key=lambda item: item[0])
        with safe_open(path, framework="pt", device="cpu") as tensors:
            expected = tensors.get_tensor(key)
        actual = state[key].detach().cpu()
        if expected.shape != actual.shape or not torch.equal(expected, actual):
            raise RuntimeError(
                "policy weights do not match the requested checkpoint "
                f"(probe tensor {key!r} from {path.name})"
            )
        return {
            "file": path.name,
            "tensor": key,
            "shape": list(expected.shape),
            "verified": True,
        }

    def reset(self) -> None:
        self.policy.reset()
        self.select_action_calls = 0
        self.model_replans = 0

    def _queue_empty(self) -> bool:
        queue = getattr(self.policy, "_action_queue", None)
        if queue is not None:
            return len(queue) == 0
        queues = getattr(self.policy, "_queues", None)
        if isinstance(queues, dict) and ACTION in queues:
            return len(queues[ACTION]) == 0
        checker = getattr(self.policy, "is_action_queue_empty", None)
        if callable(checker):
            return bool(checker())
        return True

    def build_batch(self, observation: dict[str, np.ndarray], task_text: str) -> tuple[dict[str, Any], str]:
        rgb = np.asarray(observation["observation.images.scene"])
        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"front RGB schema mismatch: {rgb.shape} {rgb.dtype}")
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError(f"state schema mismatch: {state}")
        image_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255)
        sample: dict[str, Any] = {
            "observation.images.scene": image_tensor,
            "observation.state": torch.from_numpy(state.copy()),
            "task": task_text,
        }
        digest = hashlib.sha256(rgb.tobytes() + state.tobytes() + task_text.encode()).hexdigest()
        return self.preprocessor(sample), digest

    def select_action(self, observation: dict[str, np.ndarray], task_text: str) -> dict[str, Any]:
        batch, input_hash = self.build_batch(observation, task_text)
        replan = self._queue_empty()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.policy.select_action(batch)
            action = self.postprocessor(output)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        self.select_action_calls += 1
        if replan:
            self.model_replans += 1
        raw = action.detach().cpu().numpy().reshape(-1)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError(f"policy emitted invalid action: {raw}")
        return {
            "action": raw,
            "input_hash": input_hash,
            "latency_ms": latency_ms,
            "select_action_call_id": self.select_action_calls,
            "policy_call_id": self.model_replans,
            "model_replan": replan,
        }

    def warmup(self, observation: dict[str, np.ndarray], task_text: str, count: int = 20) -> list[float]:
        latencies = []
        for _ in range(count):
            self.reset()
            result = self.select_action(observation, task_text)
            latencies.append(float(result["latency_ms"]))
        self.reset()
        return latencies

    def diagnostics(self) -> dict[str, Any]:
        return {
            "policy_type": self.config.type,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "weights_loaded_from": str(self.config.pretrained_path),
            "loaded_weight_probe": self.loaded_weight_probe,
            "device": self.device,
            "input_features": {k: list(v.shape) for k, v in self.config.input_features.items()},
            "output_features": {k: list(v.shape) for k, v in self.config.output_features.items()},
            "n_action_steps": int(getattr(self.config, "n_action_steps", 1)),
            "chunk_size": int(getattr(self.config, "chunk_size", 1)),
            "temporal_ensemble_coeff": getattr(self.config, "temporal_ensemble_coeff", None),
            "camera_rename_map": self.rename_map,
        }
