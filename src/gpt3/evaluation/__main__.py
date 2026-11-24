"""Evaluate a checkpoint; loading Hub data/tokenizers may require downloads."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from gpt3.config import load_config
from gpt3.data import build_dataloader, load_tokenizer
from gpt3.evaluation.generation import generate
from gpt3.evaluation.tasks import evaluate_task
from gpt3.training.checkpoint import load_model
from gpt3.training.engine import resolve_device, validate


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete_on_close=False
    ) as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
        # Publish only a completed file. Linking fails if another writer won.
        os.link(handle.name, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Evaluation report already exists: {args.output}")
    config = load_config(args.config)
    device = resolve_device(config)
    model, saved_config = load_model(args.checkpoint, device)
    if config.model != saved_config.model or config.tokenizer != saved_config.tokenizer:
        raise ValueError("Evaluation model/tokenizer settings must match the checkpoint")
    model.loss_chunk_size = config.hardware.loss_chunk_size
    tokenizer = load_tokenizer(config)
    metrics = validate(model, build_dataloader(config, args.split), config, device)
    samples = []
    for prompt in config.evaluation.sample_prompts:
        result = generate(
            model,
            tokenizer,
            prompt,
            config.evaluation.max_new_tokens,
            config.tokenizer.eos_token_id,
            options=config.generation,
        )
        samples.append(
            {
                "prompt": prompt,
                "completion": result.text,
                "token_ids": result.token_ids,
                "finish_reason": result.finish_reason,
            }
        )
    report = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "config": config.model_dump(),
        "held_out": asdict(metrics),
        "samples": samples,
        "tasks": [evaluate_task(model, tokenizer, task) for task in config.evaluation.tasks],
    }
    _write_report(args.output, report)
    perplexity = "overflow" if metrics.perplexity is None else f"{metrics.perplexity:.4f}"
    print(f"Held-out loss={metrics.loss:.4f}, perplexity={perplexity}; {args.output}")


if __name__ == "__main__":
    main()
