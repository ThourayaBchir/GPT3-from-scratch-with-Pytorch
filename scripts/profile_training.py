"""Profile synthetic training batches without loading a dataset or tokenizer."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from gpt3.config import ExperimentConfig, load_config
from gpt3.model import GPT3
from gpt3.training.engine import (
    autocast_context,
    cuda_matmul_context,
    make_optimizer,
    resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    device = resolve_device(config)
    with cuda_matmul_context(device, config.hardware.allow_tf32):
        _profile(config, device, args)


def _profile(config: ExperimentConfig, device: torch.device, args: argparse.Namespace) -> None:
    torch.manual_seed(config.training.seed)
    model = GPT3(config.model).to(device).train()
    model.activation_checkpointing = config.hardware.gradient_checkpointing
    model.loss_chunk_size = config.hardware.loss_chunk_size
    optimizer = make_optimizer(model, config)
    ids = torch.randint(
        config.model.vocab_size,
        (config.training.batch_size, config.model.context_length + 1),
        device=device,
    )
    batch = {
        "input_ids": ids[:, :-1],
        "labels": ids[:, 1:],
        "attention_mask": torch.ones_like(ids[:, :-1], dtype=torch.bool),
    }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def step():
        optimizer.zero_grad(set_to_none=True)
        with record_function("forward"), autocast_context(device, config.hardware.precision):
            loss = model(**batch, return_logits=config.hardware.loss_chunk_size == 0).loss
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()

    for _ in range(3):
        step()
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(args.steps):
        start = time.perf_counter()
        step()
        synchronize()
        durations.append(time.perf_counter() - start)
    saved_bytes = 0

    def pack(tensor):
        nonlocal saved_bytes
        saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    optimizer.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        model(**batch, return_logits=config.hardware.loss_chunk_size == 0).loss.backward()
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
        step()
    profiler.export_chrome_trace(str(args.output / "trace.json"))
    (args.output / "operators.txt").write_text(
        profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
    )
    summary = {
        "config": config.model_dump(),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "threads": args.threads,
        "median_step_seconds": statistics.median(durations),
        "tokens_per_second": batch["input_ids"].numel() / statistics.median(durations),
        "saved_tensor_bytes": saved_bytes,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "config"}, indent=2))


if __name__ == "__main__":
    main()
