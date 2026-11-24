"""torchrun initialization and small collective helpers for data parallel training."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist

from gpt3.config import ExperimentConfig


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def sum_values(values: list[float], device: torch.device) -> list[float]:
    if world_size() == 1:
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.tolist()


def gather_rank_values[T](value: T) -> list[T]:
    if world_size() == 1:
        return [value]
    values: list[T] = [value] * world_size()
    dist.all_gather_object(values, value)
    return values


@contextmanager
def synchronize_rank_errors(operation: str, device: torch.device) -> Iterator[None]:
    """Let all ranks leave a phase together if a rank raises a local error."""
    error = None
    try:
        yield
    except Exception as exc:
        error = exc

    if not sum_values([float(error is not None)], device)[0]:
        return
    if world_size() == 1:
        assert error is not None
        raise error

    local_message = f"{type(error).__name__}: {error}" if error is not None else None
    messages = gather_rank_values(local_message)
    if error is not None:
        raise error
    details = ", ".join(f"rank {i}: {message}" for i, message in enumerate(messages) if message)
    raise RuntimeError(f"{operation} failed: {details}")


@contextmanager
def distributed_session(config: ExperimentConfig) -> Iterator[None]:
    started = int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized()
    if started:
        use_cuda = config.hardware.device != "cpu" and torch.cuda.is_available()
        if config.hardware.device == "mps":
            raise ValueError("DDP requires CPU/Gloo or CUDA/NCCL")
        if use_cuda:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo", timeout=timedelta(seconds=120)
        )
    try:
        yield
    finally:
        if started:
            dist.destroy_process_group()
