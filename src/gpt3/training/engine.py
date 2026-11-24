"""Train and validate GPT-3 models on a single device or across DDP ranks.

Each batch contains three tensors of shape [batch, time]. input_ids and
labels use torch.long. The loader shifts labels one token ahead and uses
-100 for ignored targets. In attention_mask, True marks a real token
and False marks right padding. Loss is weighted by the number of scored
targets across all participating ranks.
"""

import json
import math
import random
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any, TextIO, cast
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from gpt3.config import ExperimentConfig
from gpt3.data import load_tokenizer
from gpt3.evaluation.generation import generate
from gpt3.model import GPT3, GPT3Output
from gpt3.training.checkpoint import (
    check_resume_config,
    distributed_checkpoint_identity,
    random_state,
    read_checkpoint,
    restore_random_state,
    save_checkpoint,
)
from gpt3.training.distributed import (
    gather_rank_values,
    rank,
    sum_values,
    synchronize_rank_errors,
    world_size,
)

Batch = Mapping[str, Tensor]
_BATCH_FIELDS = ("input_ids", "labels", "attention_mask")


@dataclass(frozen=True)
class ValidationMetrics:
    """Held-out metrics aggregated over the evaluated batches.

    Attributes:
        loss: Mean next-token cross-entropy in nats per scored token.
        perplexity: Exponential of loss, or None on floating-point overflow.
        target_tokens: Total scored targets across ranks.
    """
    loss: float
    perplexity: float | None
    target_tokens: int


@dataclass(frozen=True)
class TrainResult:
    """Result of a training run.

    Attributes:
        steps: Total completed optimizer updates, including restored updates.
        run_dir: Directory for this run's configuration, metrics, and checkpoints.
        final_validation: Held-out metrics at the final model weights.
    """
    steps: int
    run_dir: Path
    final_validation: ValidationMetrics


@dataclass
class _TrainingProgress:
    """Checkpointed counters for replay and interval reporting.

    Attributes:
        completed: Number of successful optimizer updates.
        batches_seen: Microbatches consumed on this rank, including skipped updates.
        tokens_seen: Global scored targets used in successful updates.
        interval_loss: On-device sum of token losses since the last training log.
        interval_tokens: Scored targets contributing to interval_loss.
        interval_train_seconds: Accumulated update time in seconds, or None if
            a legacy checkpoint contains untimed tokens in the current interval.
    """
    interval_loss: Tensor
    completed: int = 0
    batches_seen: int = 0
    tokens_seen: int = 0
    interval_tokens: int = 0
    interval_train_seconds: float | None = 0.0


def _loss_total(device: torch.device, value: float = 0.0) -> Tensor:
    """Create an on-device loss total at the highest supported precision."""
    dtype = torch.float32 if device.type == "mps" else torch.float64
    return torch.full((), value, device=device, dtype=dtype)


def resolve_device(config: ExperimentConfig) -> torch.device:
    """Select the training device for the current rank.

    Args:
        config: Requested device and precision. Auto selection prefers CUDA,
            then single-process MPS, then CPU.

    Returns:
        The selected device, including the current device index for CUDA.

    Raises:
        ValueError: If the requested device, precision, or DDP combination
            is unsupported.
    """
    kind = config.hardware.device
    if kind == "auto":
        if torch.cuda.is_available():
            kind = "cuda"
        elif world_size() == 1 and torch.backends.mps.is_available():
            kind = "mps"
        else:
            kind = "cpu"
    elif kind == "mps" and world_size() > 1:
        raise ValueError("DDP requires CPU or CUDA")
    if kind == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if kind == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    precision = config.hardware.precision
    if precision == "float16" and kind != "cuda":
        raise ValueError("float16 mixed precision requires CUDA")
    if precision == "bfloat16" and kind not in ("cuda", "cpu"):
        raise ValueError("bfloat16 mixed precision requires CUDA or CPU")
    if precision == "bfloat16" and kind == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16")
    return (
        torch.device("cuda", torch.cuda.current_device()) if kind == "cuda" else torch.device(kind)
    )


def autocast_context(device: torch.device, precision: str) -> AbstractContextManager[Any]:
    """Select the precision context for the forward pass and loss.

    Args:
        device: Device on which the model runs.
        precision: Configured float32, float16, or bfloat16 precision.

    Returns:
        A no-op context for float32 or a PyTorch autocast context for mixed precision.
    """
    if precision == "float32":
        return nullcontext()
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


@contextmanager
def cuda_matmul_context(device: torch.device, allow_tf32: bool) -> Iterator[None]:
    """Allow TF32 for CUDA float32 matmuls within the training or profiling run."""
    if not allow_tf32:
        yield
        return
    if device.type != "cuda":
        raise ValueError("hardware.allow_tf32 requires CUDA")

    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def learning_rate_at_step(config: ExperimentConfig, step: int) -> float:
    """Compute the learning rate for a scheduled optimizer update.

    Args:
        config: Peak and minimum rates, warmup length, and training horizon.
        step: One-based optimizer update to evaluate.

    Returns:
        The rate from the linear warmup and cosine decay schedule.

    Raises:
        ValueError: If step is outside [1, training.max_steps].
    """
    if not 1 <= step <= config.training.max_steps:
        raise ValueError("step must be in [1, training.max_steps]")
    peak = config.optimizer.learning_rate
    floor = config.optimizer.min_learning_rate
    warmup = config.optimizer.warmup_steps
    if step <= warmup:
        return peak * step / warmup
    decay_steps = config.training.max_steps - warmup - 1
    progress = (step - warmup - 1) / decay_steps if decay_steps > 0 else 0.0
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: GPT3, config: ExperimentConfig) -> torch.optim.AdamW:
    """Build AdamW parameter groups for the model.

    Args:
        model: Model whose trainable parameters the optimizer will update.
        config: AdamW settings and whether to use the CUDA fused implementation.

    Returns:
        An optimizer applying weight decay to tensors of rank two or higher
        and zero decay to scalars and vectors.

    Raises:
        ValueError: If fused AdamW is requested for parameters outside CUDA.
    """
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            if config.hardware.fused_adamw and not parameter.is_cuda:
                raise ValueError("hardware.fused_adamw requires CUDA")
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.optimizer.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.eps,
        fused=True if config.hardware.fused_adamw else None,
    )


def _count_targets(batch: Batch, device: torch.device, vocab_size: int) -> int:
    """Check batch structure and count the targets contributing to loss.

    Args:
        batch: Engine batch following the module's [batch, time] contract.
        device: Destination device. CUDA batches receive a vocabulary range check.
        vocab_size: Exclusive upper bound for token IDs.

    Returns:
        Number of labels other than -100, possibly zero.

    Raises:
        ValueError: If required tensors are missing, their shapes or token dtypes
            are invalid, or a CUDA input or scored target lies outside the vocabulary.
    """
    missing = [name for name in _BATCH_FIELDS if name not in batch]
    if missing:
        raise ValueError(f"Batch is missing {missing}")
    for name in _BATCH_FIELDS:
        if not isinstance(batch[name], Tensor):
            raise ValueError(f"{name} must be a tensor")
    ids = batch["input_ids"]
    if ids.ndim != 2 or ids.dtype != torch.long:
        raise ValueError("input_ids must be a [batch, time] torch.long tensor")
    labels = batch["labels"]
    if labels.dtype != torch.long or labels.shape != ids.shape:
        raise ValueError("labels must match input_ids shape and have torch.long dtype")
    if batch["attention_mask"].shape != ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    # A CUDA indexing assertion can disable the device before peers exchange errors.
    if device.type == "cuda":
        if ((ids < 0) | (ids >= vocab_size)).any():
            raise ValueError("input_ids contain a token outside the model vocabulary")
        if ((labels != -100) & ((labels < 0) | (labels >= vocab_size))).any():
            raise ValueError("labels contain a target outside the model vocabulary")
    return int((labels != -100).sum().item())


def _move_batch(batch: Batch, device: torch.device) -> dict[str, Tensor]:
    """Transfer the model inputs to the execution device.

    Args:
        batch: Validated batch containing input_ids, labels, and attention_mask.
        device: Destination device. CUDA transfers request nonblocking copies.

    Returns:
        The three model inputs on device with their original shapes and dtypes.
    """
    return {
        name: batch[name].to(device, non_blocking=device.type == "cuda")
        for name in _BATCH_FIELDS
    }


def _batch_loss(model: torch.nn.Module, batch: Batch, target_count: int) -> Tensor:
    """Compute a scalar loss for one microbatch.

    Args:
        model: Decoder, optionally compiled or wrapped in DDP.
        batch: Model inputs already on the execution device.
        target_count: Number of scored targets on this rank.

    Returns:
        Mean cross-entropy for scored targets. An empty-target training rank
        receives a zero connected to the model graph so it participates in backward.
    """
    result: GPT3Output
    if target_count:
        result = model(**batch, return_logits=False)
        assert result.loss is not None
        return result.loss

    # A training rank with no targets still needs a gradient path for DDP backward.
    result = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    assert result.logits is not None
    return result.logits.sum() * 0


class _TrainingBatches(Iterator[Batch]):
    """Iterate over training batches, restarting the source at the end of each pass.

    A finite source must support fresh iteration to supply multiple passes.
    """

    def __init__(self, source: Iterable[Batch]) -> None:
        """Open the first pass over the training data.

        Args:
            source: Rank-local training loader.
        """
        self.source = source
        self.iterator = iter(source)

    def __next__(self) -> Batch:
        """Read the next microbatch, reopening an exhausted source once.

        Returns:
            The next batch in loader order.

        Raises:
            ValueError: If the new pass also yields no batches.
        """
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.source)
        try:
            return next(self.iterator)
        except StopIteration as exc:
            raise ValueError("Training loader contains no batches") from exc


def _rank_checkpoint_path(path: Path, local_rank: int) -> Path:
    """Resolve a rank's file from the rank-zero checkpoint path.

    Args:
        path: Checkpoint path used by rank zero.
        local_rank: Global process rank that owns the checkpoint.

    Returns:
        The original path for rank zero or its rank-suffixed sibling.
    """
    if local_rank == 0:
        return path
    return path.with_name(f"{path.stem}.rank{local_rank}{path.suffix}")


def _restore_training(
    config: ExperimentConfig,
    model: GPT3,
    optimizer: torch.optim.AdamW,
    scaler: torch.amp.GradScaler | None,
    train_loader: Iterable[Batch],
    device: torch.device,
) -> tuple[_TrainingProgress, Iterator[Batch], dict[str, Any]]:
    """Restore a checkpoint and position the data iterator for the next update.

    Args:
        config: Current experiment with training.resume_from set to the rank-zero file.
        model: Model to populate with saved weights.
        optimizer: Optimizer to populate with saved parameter groups and moments.
        scaler: Active FP16 gradient scaler, or None.
        train_loader: Loader that reproduces this rank's original batch order.
        device: Execution device used to coordinate failures across ranks.

    Returns:
        Restored progress, the iterator positioned after the consumed batches,
        and the initial data RNG state needed for future checkpoints.

    Raises:
        ValueError: If the checkpoint configuration, rank, or progress is incompatible.
    """
    assert config.training.resume_from is not None
    local_rank = rank()
    resume_path = _rank_checkpoint_path(Path(config.training.resume_from), local_rank)
    distributed = world_size() > 1
    with synchronize_rank_errors("Checkpoint validation", device):
        checkpoint = read_checkpoint(resume_path)
        if checkpoint.get("world_size", 1) != world_size():
            raise ValueError("Resume requires the original world size")
        if checkpoint.get("rank", 0) != local_rank:
            raise ValueError(f"Checkpoint rank does not match rank {local_rank}: {resume_path}")
        check_resume_config(checkpoint["config"], config)
        saved = checkpoint["progress"]
        interval_tokens = saved["interval_tokens"]
        progress = _TrainingProgress(
            interval_loss=_loss_total(device, saved["interval_loss"]),
            completed=saved["completed"],
            batches_seen=saved["batches_seen"],
            tokens_seen=saved["tokens_seen"],
            interval_tokens=interval_tokens,
            # Older checkpoints can carry untimed tokens into the first log.
            interval_train_seconds=saved.get(
                "interval_train_seconds", None if interval_tokens else 0.0
            ),
        )
        if checkpoint["scheduler"]["completed"] != progress.completed:
            raise ValueError("Checkpoint scheduler and progress disagree")
        if distributed:
            identity = distributed_checkpoint_identity(checkpoint)
        data_random_state = checkpoint["data_random_state"]
        training_random_state = checkpoint["random_state"]

    if distributed:
        local_progress = (identity, progress.completed, progress.batches_seen)
        if any(value != local_progress for value in gather_rank_values(local_progress)):
            raise ValueError(
                "Rank checkpoints must come from the same run and have matching progress"
            )

    with synchronize_rank_errors("Checkpoint restore", device):
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        restore_random_state(data_random_state)

    with synchronize_rank_errors("Training data replay", device):
        batches = _TrainingBatches(train_loader)
        for _ in range(progress.batches_seen):
            next(batches)
        restore_random_state(training_random_state)
    return progress, batches, data_random_state


@torch.no_grad()
def validate(
    model: GPT3,
    loader: Iterable[Batch],
    config: ExperimentConfig,
    device: torch.device,
) -> ValidationMetrics:
    """Evaluate token-weighted loss and perplexity across ranks.

    All ranks participate, including those with shorter validation tails. The
    call restores the model's training mode, loss chunk size, and global RNG state.

    Args:
        model: Decoder already on device.
        loader: Held-out batches for this rank.
        config: Precision, loss chunk size, and per-rank evaluation batch limit.
        device: Execution device for the model inputs and collective reductions.

    Returns:
        Metrics over scored tokens from all evaluated batches.

    Raises:
        ValueError: If the batches are malformed or supply no scored targets globally.
        FloatingPointError: If the aggregate validation loss is nonfinite.
    """
    training_rng = random_state()
    previous_chunk_size = model.loss_chunk_size
    was_training = model.training
    model.loss_chunk_size = config.hardware.loss_chunk_size
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        with synchronize_rank_errors("Validation loader initialization", device):
            batches = islice(loader, config.evaluation.max_batches)
        while True:
            with synchronize_rank_errors("Validation batch", device):
                try:
                    raw_batch = next(batches)
                except StopIteration:
                    has_batch = False
                else:
                    has_batch = True
                    token_count = _count_targets(raw_batch, device, config.model.vocab_size)
                    if token_count:
                        batch = _move_batch(raw_batch, device)
                        with autocast_context(device, config.hardware.precision):
                            batch_loss = _batch_loss(model, batch, token_count)
                        total_loss += batch_loss.item() * token_count
                        total_tokens += token_count
            # Exhausted ranks participate until every rank has finished its tail.
            if not sum_values([float(has_batch)], device)[0]:
                break
    finally:
        model.train(was_training)
        model.loss_chunk_size = previous_chunk_size
        restore_random_state(training_rng)
    total_loss, token_sum = sum_values([total_loss, float(total_tokens)], device)
    total_tokens = int(token_sum)
    if total_tokens == 0:
        raise ValueError("Validation loader contains no target tokens")
    loss = total_loss / total_tokens
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite validation loss")
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = None
    return ValidationMetrics(loss=loss, perplexity=perplexity, target_tokens=total_tokens)


def _accumulate_gradients(
    model: torch.nn.Module,
    scaler: torch.amp.GradScaler | None,
    config: ExperimentConfig,
    device: torch.device,
    batches: Iterator[Batch],
) -> tuple[Tensor, int]:
    """Accumulate gradients of the global mean token loss for one update.

    Args:
        model: Training decoder, optionally compiled or wrapped in DDP.
        scaler: Active FP16 gradient scaler, or None.
        config: Microbatches per update, vocabulary size, and forward precision.
        device: Execution device for the forward pass and collective reductions.
        batches: Rank-local iterator advanced by one accumulation window.

    Returns:
        On-device global sum of token losses and global number of scored targets.

    Raises:
        ValueError: If the accumulation window contains no scored targets globally.
        FloatingPointError: If a microbatch loss is nonfinite.
    """
    microbatches: list[tuple[Batch, int]] = []
    for _ in range(config.training.gradient_accumulation_steps):
        with synchronize_rank_errors("Training data loading", device):
            batch = next(batches)
            count = _count_targets(batch, device, config.model.vocab_size)
            microbatches.append((batch, count))
    local_tokens = sum(count for _, count in microbatches)
    target_count = int(sum_values([float(local_tokens)], device)[0])
    if target_count == 0:
        raise ValueError("Training accumulation window contains no target tokens")

    ranks = world_size()
    loss_sum = _loss_total(device)
    for index, (raw_batch, count) in enumerate(microbatches):
        sync = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and index < len(microbatches) - 1
            else nullcontext()
        )
        with sync:
            with synchronize_rank_errors("Training batch transfer", device):
                batch = _move_batch(raw_batch, device)
            with (
                synchronize_rank_errors("Training forward pass", device),
                autocast_context(device, config.hardware.precision),
            ):
                loss = _batch_loss(model, batch, count)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite training loss")
            loss_sum.add_(loss.detach().to(dtype=loss_sum.dtype), alpha=count)
            # DDP divides gradients by the number of ranks.
            weighted_loss = loss * (count * ranks / target_count)
            if scaler is not None:
                weighted_loss = scaler.scale(weighted_loss)
            torch.autograd.backward(weighted_loss)
    if ranks > 1:
        dist.all_reduce(loss_sum)
    return loss_sum, target_count


def _step_optimizer(
    model: GPT3,
    optimizer: torch.optim.AdamW,
    scaler: torch.amp.GradScaler | None,
    learning_rate: float,
    grad_clip_norm: float,
) -> Tensor | None:
    """Clip accumulated gradients and attempt one AdamW update.

    Args:
        model: Model holding the accumulated gradients.
        optimizer: AdamW instance whose parameter groups receive learning_rate.
        scaler: Active FP16 gradient scaler, or None.
        learning_rate: Rate scheduled for this update.
        grad_clip_norm: Maximum gradient norm after clipping.

    Returns:
        Scalar gradient norm measured before clipping, or None if the scaler
        skipped the update.
    """
    if scaler is not None:
        scaler.unscale_(optimizer)
    grad_norm = clip_grad_norm_(
        model.parameters(), grad_clip_norm, error_if_nonfinite=scaler is None
    )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    if scaler is None:
        optimizer.step()
    else:
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            return None
    return grad_norm


def _record(
    log_file: TextIO | None,
    event: str,
    step: int,
    **metrics: float | int | str | list[int] | None,
) -> None:
    """Write and flush one metrics event to the run log and stdout.

    Args:
        log_file: Rank-zero JSONL stream, or None to skip output on this rank.
        event: Event name, such as train or validation.
        step: Number of completed optimizer updates.
        **metrics: JSON fields for this event.
    """
    if log_file is None:
        return
    line = json.dumps(
        {
            "event": event,
            "step": step,
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            **metrics,
        },
        allow_nan=False,
    )
    log_file.write(line + "\n")
    log_file.flush()
    print(line, flush=True)


def _record_samples(
    log_file: TextIO,
    step: int,
    config: ExperimentConfig,
    model: GPT3,
    device: torch.device,
    tokenizer: Tokenizer | None,
) -> Tokenizer | None:
    """Generate and log completions for rank zero's validation prompts.

    Args:
        log_file: Open metrics stream for the run.
        step: Completed update associated with this validation.
        config: Prompts, decoding options, tokenizer settings, and precision.
        model: Unwrapped decoder used for generation.
        device: Execution device for autocast.
        tokenizer: Previously loaded tokenizer, or None to load it on demand.

    Returns:
        The tokenizer for reuse at the next validation, or None if loading failed.
        Loading and generation failures are recorded as sample_error events.
    """
    if tokenizer is None:
        try:
            tokenizer = load_tokenizer(config)
        except Exception as exc:
            _record(log_file, "sample_error", step, error=str(exc))
            return None

    # Generation uses its own RNG and restores the model's training mode.
    for prompt in config.evaluation.sample_prompts:
        try:
            with autocast_context(device, config.hardware.precision):
                sample = generate(
                    model,
                    tokenizer,
                    prompt,
                    config.evaluation.max_new_tokens,
                    config.tokenizer.eos_token_id,
                    options=config.generation,
                )
        except Exception as exc:
            _record(log_file, "sample_error", step, prompt=prompt, error=str(exc))
        else:
            _record(
                log_file,
                "sample",
                step,
                prompt=prompt,
                completion=sample.text,
                token_ids=sample.token_ids,
                finish_reason=sample.finish_reason,
            )
    return tokenizer


def _save_training_checkpoint(
    run_dir: Path,
    config: ExperimentConfig,
    model: GPT3,
    optimizer: torch.optim.AdamW,
    scaler: torch.amp.GradScaler | None,
    progress: _TrainingProgress,
    data_random_state: dict[str, Any],
) -> Path:
    """Atomically save the current rank's training checkpoint.

    Args:
        run_dir: Output directory whose name identifies the run.
        config: Experiment settings to serialize for resume checks.
        model: Unwrapped decoder supplying model weights.
        optimizer: AdamW instance supplying optimizer state.
        scaler: Active FP16 gradient scaler, or None.
        progress: Update, data replay, and logging counters.
        data_random_state: RNG snapshot from before the initial training iterator
            was created. The current RNG state is saved separately.
    """
    local_rank = rank()
    path = _rank_checkpoint_path(run_dir / f"checkpoint-{progress.completed:08d}.pt", local_rank)
    save_checkpoint(
        path,
        {
            "world_size": world_size(),
            "rank": local_rank,
            "run_id": run_dir.name,
            "config": config.model_dump(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": {
                "completed": progress.completed,
                "name": config.optimizer.scheduler,
                "max_steps": config.training.max_steps,
            },
            "scaler": scaler.state_dict() if scaler is not None else None,
            "progress": vars(progress) | {"interval_loss": float(progress.interval_loss.item())},
            "data_random_state": data_random_state,
            "random_state": random_state(),
        },
    )
    return path


def train(
    config: ExperimentConfig,
    model: GPT3,
    train_loader: Iterable[Batch],
    validation_loader: Iterable[Batch],
) -> TrainResult:
    """Train a decoder through the configured number of successful optimizer updates.

    Each rank calls this function with its own batches. Validation, logging, and
    checkpoint intervals count completed updates. Resuming creates a new run directory.

    Args:
        config: Model, data, optimization, hardware, evaluation, and output settings.
        model: Initialized decoder to move to the training device and update in place.
        train_loader: Rank-local training batches. A finite loader must support
            fresh iteration if more than one pass is needed.
        validation_loader: Rank-local held-out batches, reusable across evaluations.

    Returns:
        Completed update count, run directory, and final validation metrics.

    Raises:
        ValueError: If model settings disagree, resume checks fail, or an
            accumulation window has no scored targets globally.
        FloatingPointError: If a training or validation loss is nonfinite.
        RuntimeError: If the FP16 scaler skips 100 consecutive updates.
    """
    if model.config != config.model:
        raise ValueError("Model configuration does not match experiment configuration")
    device = resolve_device(config)
    with cuda_matmul_context(device, config.hardware.allow_tf32):
        return _train_on_device(config, model, train_loader, validation_loader, device)


def _train_on_device(
    config: ExperimentConfig,
    model: GPT3,
    train_loader: Iterable[Batch],
    validation_loader: Iterable[Batch],
    device: torch.device,
) -> TrainResult:
    seed = config.training.seed + rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    model.activation_checkpointing = config.hardware.gradient_checkpointing
    model.loss_chunk_size = config.hardware.loss_chunk_size
    # Keep the original decoder for validation, sampling, and checkpoint state.
    training_model: torch.nn.Module = model
    if config.hardware.compile_model:
        training_model = cast(torch.nn.Module, torch.compile(model))
    if world_size() > 1:
        training_model = DistributedDataParallel(
            training_model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda") if config.hardware.precision == "float16" else None
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    if world_size() > 1:
        shared = [run_id]
        dist.broadcast_object_list(shared, src=0)
        run_id = shared[0]
    run_dir = Path(config.training.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if rank() == 0:
        (run_dir / "config.json").write_text(
            config.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    if config.training.resume_from:
        progress, batches, data_random_state = _restore_training(
            config, model, optimizer, scaler, train_loader, device
        )
    else:
        progress = _TrainingProgress(interval_loss=_loss_total(device))
        data_random_state = random_state()
        with synchronize_rank_errors("Training data initialization", device):
            batches = _TrainingBatches(train_loader)
    skipped_updates = 0
    final_validation: ValidationMetrics | None = None
    sample_tokenizer: Tokenizer | None = None
    previous_checkpoint: Path | None = None
    generate_samples = config.training.generate_samples and bool(config.evaluation.sample_prompts)

    log_context = (
        (run_dir / "metrics.jsonl").open("w", encoding="utf-8") if rank() == 0 else nullcontext()
    )
    with log_context as log_file:
        while progress.completed < config.training.max_steps:
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_sum, target_count = _accumulate_gradients(
                training_model, scaler, config, device, batches
            )
            progress.batches_seen += config.training.gradient_accumulation_steps

            step = progress.completed + 1
            lr = learning_rate_at_step(config, step)
            grad_norm = _step_optimizer(
                model, optimizer, scaler, lr, config.optimizer.grad_clip_norm
            )
            if grad_norm is None:
                skipped_updates += 1
                if skipped_updates >= 100:
                    raise RuntimeError(
                        "100 consecutive optimizer updates were skipped by GradScaler"
                    )
                continue
            skipped_updates = 0

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_started

            progress.completed = step
            progress.tokens_seen += target_count
            progress.interval_loss.add_(loss_sum)
            progress.interval_tokens += target_count
            if progress.interval_train_seconds is not None:
                progress.interval_train_seconds += step_seconds

            final_step = step == config.training.max_steps
            if final_step or step % config.training.log_interval == 0:
                if log_file is not None:
                    seconds = progress.interval_train_seconds
                    _record(
                        log_file,
                        "train",
                        step,
                        loss=float(progress.interval_loss.item()) / progress.interval_tokens,
                        learning_rate=lr,
                        grad_norm=float(grad_norm.item()),
                        target_tokens=progress.interval_tokens,
                        train_seconds=seconds,
                        tokens_per_second=progress.interval_tokens / seconds if seconds else None,
                    )
                progress.interval_loss.zero_()
                progress.interval_tokens = 0
                progress.interval_train_seconds = 0.0
            if final_step or step % config.training.eval_interval == 0:
                final_validation = validate(model, validation_loader, config, device)
                _record(log_file, "validation", step, **asdict(final_validation))
                if log_file is not None and generate_samples:
                    sample_tokenizer = _record_samples(
                        log_file, step, config, model, device, sample_tokenizer
                    )
            if final_step or step % config.training.checkpoint_interval == 0:
                with synchronize_rank_errors("Checkpoint save", device):
                    checkpoint = _save_training_checkpoint(
                        run_dir, config, model, optimizer, scaler, progress, data_random_state
                    )
                if config.training.keep_last_checkpoint and previous_checkpoint is not None:
                    with synchronize_rank_errors("Checkpoint retention", device):
                        previous_checkpoint.unlink(missing_ok=True)
                previous_checkpoint = checkpoint

    if final_validation is None:
        final_validation = validate(model, validation_loader, config, device)

    return TrainResult(steps=progress.completed, run_dir=run_dir, final_validation=final_validation)
