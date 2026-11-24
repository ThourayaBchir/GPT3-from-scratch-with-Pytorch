"""Run real train/evaluate/resume/generate validation; all artifacts stay in this project."""

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cache = root / ".cache"
    temporary = cache / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    # Set before importing Hugging Face/PyTorch; never consult external caches.
    for name, path in {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "HF_DATASETS_CACHE": cache / "huggingface/datasets",
        "HF_XET_CACHE": cache / "huggingface/xet",
        "XDG_CACHE_HOME": cache,
        "TORCH_HOME": cache / "torch",
        "TMPDIR": temporary,
    }.items():
        os.environ[name] = str(path)
    tempfile.tempdir = str(temporary)
    os.environ["HF_HUB_OFFLINE"] = "0" if args.allow_downloads else "1"
    os.environ["HF_DATASETS_OFFLINE"] = "0" if args.allow_downloads else "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from huggingface_hub import HfApi

    from gpt3.config import load_config
    from gpt3.data import build_dataloader, load_tokenizer
    from gpt3.evaluation.generation import generate
    from gpt3.model import GPT3
    from gpt3.training.checkpoint import load_model, read_checkpoint
    from gpt3.training.engine import train, validate

    torch.set_num_threads(2)
    config = load_config(root / "configs/end_to_end.toml")
    revisions_file = cache / "end_to_end_revisions.json"
    if args.allow_downloads:
        api = HfApi()
        revisions = {
            "dataset": api.dataset_info(config.data.dataset_id).sha,
            "tokenizer": api.model_info(config.tokenizer.name).sha,
        }
        revisions_file.write_text(json.dumps(revisions, indent=2))
    else:
        if not revisions_file.exists():
            raise SystemExit(
                "No project-local revisions found; approve downloads for the first run"
            )
        revisions = json.loads(revisions_file.read_text())
    dataset_revision = revisions["dataset"]
    tokenizer_revision = revisions["tokenizer"]
    config = config.model_copy(
        update={
            "data": config.data.model_copy(update={"revision": dataset_revision}),
            "tokenizer": config.tokenizer.model_copy(update={"revision": tokenizer_revision}),
        }
    )
    session = root / "artifacts/end_to_end" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    session.mkdir(parents=True)
    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "output_dir": str(session / "full"),
                }
            )
        }
    )
    (session / "config.json").write_text(config.model_dump_json(indent=2))
    print(f"Session: {session}", flush=True)
    tokenizer = load_tokenizer(config)
    training = build_dataloader(config, "train")
    validation = build_dataloader(config, "validation")
    torch.manual_seed(config.training.seed)
    model = GPT3(config.model)
    device = torch.device("cpu")
    baseline = validate(model, validation, config, device)
    print(f"Initial validation: {baseline}", flush=True)
    full = train(config, model, training, validation)
    midpoint = full.run_dir / "checkpoint-00000015.pt"
    final_checkpoint = full.run_dir / "checkpoint-00000030.pt"
    resume_config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "resume_from": str(midpoint),
                    "output_dir": str(session / "resumed"),
                }
            )
        }
    )
    resumed_model = GPT3(config.model)
    resumed = train(resume_config, resumed_model, training, validation)
    resumed_checkpoint = resumed.run_dir / "checkpoint-00000030.pt"
    a, b = read_checkpoint(final_checkpoint), read_checkpoint(resumed_checkpoint)
    exact_weights = all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    exact_optimizer = all(
        torch.equal(value, b["optimizer"]["state"][index][key])
        for index, state in a["optimizer"]["state"].items()
        for key, value in state.items()
    )
    exact_progress = a["progress"] == b["progress"] and a["scheduler"] == b["scheduler"]
    # Evaluate and generate using reloaded checkpoint weights, not the in-memory model.
    reloaded, _ = load_model(resumed_checkpoint, device)
    held_out = validate(reloaded, build_dataloader(config, "test"), config, device)
    samples = [
        {
            "prompt": prompt,
            **asdict(
                generate(
                    reloaded,
                    tokenizer,
                    prompt,
                    config.evaluation.max_new_tokens,
                    config.tokenizer.eos_token_id,
                    options=config.generation,
                )
            ),
        }
        for prompt in config.evaluation.sample_prompts
    ]
    report = {
        "configuration": config.model_dump(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_validation": asdict(baseline),
        "final_validation": asdict(full.final_validation),
        "resumed_validation": asdict(resumed.final_validation),
        "held_out_test": asdict(held_out),
        "exact_resume": {
            "weights": exact_weights,
            "optimizer": exact_optimizer,
            "progress_and_schedule": exact_progress,
        },
        "checkpoints": {
            "midpoint": str(midpoint),
            "full": str(final_checkpoint),
            "resumed": str(resumed_checkpoint),
        },
        "samples": samples,
    }
    (session / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not all((exact_weights, exact_optimizer, exact_progress)):
        raise RuntimeError("Resumed run differs from uninterrupted training; see report")


if __name__ == "__main__":
    main()
