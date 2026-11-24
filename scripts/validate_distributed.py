"""Two-rank CPU DDP, resume, and streaming data checks without Hub downloads."""

import argparse
import json
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers

from gpt3.config import ExperimentConfig
from gpt3.data import PackedTokenDataset, build_dataloader, collate_tokens, tokenize_split
from gpt3.model import GPT3
from gpt3.training.checkpoint import read_checkpoint
from gpt3.training.distributed import distributed_session, rank
from gpt3.training.engine import train, validate


def check_streaming_loader(base: ExperimentConfig) -> None:
    local_rank = rank()
    config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={"streaming": True, "append_eos": False, "drop_remainder": False}
            ),
            "training": base.training.model_copy(update={"batch_size": 2}),
            "evaluation": base.evaluation.model_copy(update={"batch_size": 2}),
        }
    )
    tokens = [*range(config.model.vocab_size), 0]
    documents = [{"input_ids": tokens}]
    streaming_source = Dataset.from_dict(
        {"text": [" ".join(f"token{i}" for i in tokens)]}
    ).to_iterable_dataset()
    tokenizer = Tokenizer(models.WordLevel({f"token{i}": i for i in tokens}))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    with (
        patch("gpt3.data.pipeline.load_text_split", return_value=streaming_source) as load_split,
        patch("gpt3.data.pipeline.load_tokenizer", return_value=tokenizer),
        patch("gpt3.data.pipeline.tokenize_split", wraps=tokenize_split) as tokenize,
    ):
        for split in ("validation", "train"):
            expected = PackedTokenDataset(
                documents,
                config.model.context_length,
                drop_remainder=False,
                rank=local_rank,
                world_size=dist.get_world_size(),
                repeat=split == "train",
            )
            loader = build_dataloader(config, split)
            examples = list(islice(expected, 6))
            batches = [collate_tokens(examples[i : i + 2], 1) for i in range(0, len(examples), 2)]
            actual = list(islice(loader, 3)) if split == "train" else list(loader)
            for result, reference in zip(actual, batches, strict=True):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
            if split == "train":
                replay = list(islice(loader, 1, 3))
                for result, reference in zip(replay, batches[1:], strict=True):
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)

    assert load_split.call_count == tokenize.call_count == (2 if local_rank == 0 else 0)


def check_loader_failure(config: ExperimentConfig) -> None:
    with patch("gpt3.data.pipeline._packed_split", side_effect=ValueError("Invalid token shard")):
        try:
            train(config, GPT3(config.model), build_dataloader(config, "train"), [])
        except (ValueError, RuntimeError) as exc:
            expected = "Rank 0 failed to prepare data" if rank() else "Invalid token shard"
            assert expected in str(exc)
        else:
            raise AssertionError("Training accepted a broken data source")


def check_malformed_batches(config, sample):
    model = GPT3(config.model)
    device = torch.device("cpu")
    missing_mask = {key: value for key, value in sample.items() if key != "attention_mask"}
    local_batch = missing_mask if rank() else sample
    try:
        train(config, model, [local_batch], [sample])
    except (ValueError, RuntimeError) as exc:
        assert "attention_mask" in str(exc)
    else:
        raise AssertionError("Training accepted a malformed batch")

    # Rank zero has finished when rank one encounters its second batch.
    local_validation = [sample, missing_mask] if rank() else [sample]
    try:
        validate(model, local_validation, config, device)
    except (ValueError, RuntimeError) as exc:
        assert "attention_mask" in str(exc)
    else:
        raise AssertionError("Validation accepted a malformed batch")

    metrics = validate(model, [sample, sample] if rank() else [], config, device)
    assert metrics.target_tokens == 2 * int((sample["labels"] != -100).sum())


def check_rejected_resumes(config: ExperimentConfig) -> None:
    def unreadable_loader():
        raise AssertionError("Training data replay started before checkpoint validation")
        yield

    for field, value, expected in (
        ("rank", 0, "Checkpoint rank does not match"),
        ("run_id", "different-run", "same run and have matching progress"),
        ("batches_seen", 0, "same run and have matching progress"),
    ):
        def read_mismatched_checkpoint(path, field=field, value=value):
            checkpoint = read_checkpoint(path)
            if rank() == 1:
                target = checkpoint["progress"] if field == "batches_seen" else checkpoint
                target[field] = value
            return checkpoint

        with patch("gpt3.training.engine.read_checkpoint", side_effect=read_mismatched_checkpoint):
            try:
                train(config, GPT3(config.model), unreadable_loader(), [])
            except (ValueError, RuntimeError) as exc:
                assert expected in str(exc)
            else:
                raise AssertionError(f"Resume accepted a mismatched {field}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = ExperimentConfig.model_validate(
        {
            "model": {
                "vocab_size": 13,
                "context_length": 3,
                "local_window": 2,
                "n_layers": 2,
                "d_model": 12,
                "n_heads": 2,
                "head_dim": 6,
            },
            "tokenizer": {"eos_token_id": 1},
            "optimizer": {"warmup_steps": 0},
            "hardware": {"device": "cpu", "gradient_checkpointing": True, "loss_chunk_size": 2},
            "training": {
                "max_steps": 2,
                "gradient_accumulation_steps": 2,
                "checkpoint_interval": 1,
                "eval_interval": 1,
                "output_dir": str(args.output / "distributed"),
            },
        }
    )
    samples = [
        {
            "input_ids": torch.tensor([[i, i + 1, i + 2]]),
            "labels": torch.tensor([[i + 1, i + 2, -100 if i % 2 == 0 else i + 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        }
        for i in (2, 3, 4, 5)
    ]
    with distributed_session(config):
        if dist.get_world_size() != 2:
            raise ValueError("Launch with --nproc-per-node=2")
        check_streaming_loader(config)
        local_rank = rank()
        torch.manual_seed(21)
        model = GPT3(config.model)
        local = samples[local_rank::2]
        # Exercise a rank-local accumulation group with no targets; it must still
        # participate in backward while other ranks provide valid targets.
        if local_rank == 0:
            local = [
                {**sample, "labels": torch.full_like(sample["labels"], -100)} for sample in local
            ]
        result = train(config, model, local, samples[local_rank::2])
        reference_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        resume_config = config.model_copy(
            update={
                "training": config.training.model_copy(
                    update={
                        "resume_from": str(result.run_dir / "checkpoint-00000001.pt"),
                        "output_dir": str(args.output / "resumed"),
                    }
                )
            }
        )
        def read_legacy_checkpoint(path):
            checkpoint = read_checkpoint(path)
            del checkpoint["run_id"]
            return checkpoint

        for reader in (read_checkpoint, read_legacy_checkpoint):
            with patch("gpt3.training.engine.read_checkpoint", side_effect=reader):
                resumed_model = GPT3(config.model)
                train(resume_config, resumed_model, local, samples[local_rank::2])
            for name, value in resumed_model.state_dict().items():
                torch.testing.assert_close(value, reference_state[name], rtol=0, atol=0)
        check_rejected_resumes(resume_config)
        check_loader_failure(config)
        check_malformed_batches(config, samples[0])
        dist.barrier()
    if local_rank == 0:
        torch.manual_seed(21)
        reference = GPT3(config.model)
        merged = []
        for first in (0, 2):
            batch = {
                name: torch.cat([samples[first][name], samples[first + 1][name]])
                for name in samples[first]
            }
            batch["labels"][0].fill_(-100)
            merged.append(batch)
        config = config.model_copy(
            update={
                "training": config.training.model_copy(
                    update={
                        "output_dir": str(args.output / "reference"),
                    }
                )
            }
        )
        train(config, reference, merged, merged)
        maximum = 0.0
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(reference_state[name], value, atol=2e-6, rtol=2e-4)
            maximum = max(maximum, float((value - reference_state[name]).abs().max()))
        report = {
            "world_size": 2,
            "streaming_loader_matches_previous_batch_order": True,
            "resume_exact": True,
            "reference_max_parameter_difference": maximum,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(report)


if __name__ == "__main__":
    main()
