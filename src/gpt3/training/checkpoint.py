"""Versioned, atomic checkpoints containing only tensors and Python primitives."""

import os
import random
from collections.abc import Buffer
from hashlib import sha256
from io import RawIOBase
from pathlib import Path
from typing import Any, BinaryIO, cast
from uuid import uuid4

import numpy as np
import torch

from gpt3.config import ExperimentConfig
from gpt3.model import GPT3


def random_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Replace the destination only after serialization finishes successfully."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save({"format_version": 1, **payload}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return payload


def load_model(
    path: str | Path,
    device: torch.device,
    *,
    expected_config: ExperimentConfig | None = None,
) -> tuple[GPT3, ExperimentConfig]:
    """Load a model for inference, optionally checking model and tokenizer settings first."""
    payload = read_checkpoint(path)
    config = ExperimentConfig.model_validate(payload["config"])
    if expected_config is not None and (
        expected_config.model != config.model or expected_config.tokenizer != config.tokenizer
    ):
        raise ValueError("Model/tokenizer settings must match the checkpoint")
    model = GPT3(config.model)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), config


def check_resume_config(saved: dict[str, Any], current: ExperimentConfig) -> None:
    """Allow reporting/output changes; keep the original schedule and data contract."""
    old = ExperimentConfig.model_validate(saved)
    for name in ("model", "data", "tokenizer", "optimizer", "hardware"):
        if getattr(old, name) != getattr(current, name):
            raise ValueError(f"Cannot resume with changed {name} configuration")
    for name in ("seed", "batch_size", "gradient_accumulation_steps", "max_steps"):
        if getattr(old.training, name) != getattr(current.training, name):
            raise ValueError(f"Cannot resume with changed training.{name}")


class _CheckpointDigest(RawIOBase):
    """Hash serialized state without holding a second copy in memory."""

    def __init__(self) -> None:
        super().__init__()
        self.digest = sha256()

    def writable(self) -> bool:
        return True

    def write(self, data: Buffer) -> int:
        self.digest.update(data)
        return memoryview(data).nbytes


def distributed_checkpoint_identity(payload: dict[str, Any]) -> str:
    """Identify a run, using replicated training state for older checkpoints."""
    run_id = payload.get("run_id")
    if isinstance(run_id, str):
        return f"run:{run_id}"
    # Older files have no run ID. Rank-local data and RNG state differ by design.
    state = tuple(payload[name] for name in ("model", "optimizer", "scaler"))
    with _CheckpointDigest() as writer:
        # torch.save only needs write() and flush() on file-like objects.
        torch.save(state, cast(BinaryIO, writer))
        return f"state:{writer.digest.hexdigest()}"
