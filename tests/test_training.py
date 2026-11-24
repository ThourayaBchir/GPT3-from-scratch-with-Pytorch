"""Training integration and accumulation equivalence on synthetic CPU batches."""

import json
import math
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from gpt3.config import ExperimentConfig
from gpt3.model import GPT3, GPT3Output
from gpt3.training.engine import cuda_matmul_context, learning_rate_at_step, make_optimizer, train


def test_accumulation_matches_combined_batch_and_logs_validation(tmp_path):
    config = ExperimentConfig.model_validate(
        {
            "model": {
                "vocab_size": 13,
                "context_length": 3,
                "n_layers": 2,
                "d_model": 12,
                "n_heads": 2,
                "head_dim": 6,
                "local_window": 2,
            },
            "tokenizer": {"eos_token_id": 1},
            "optimizer": {"warmup_steps": 0},
            "training": {
                "max_steps": 1,
                "gradient_accumulation_steps": 2,
                "output_dir": str(tmp_path / "accum"),
            },
            "hardware": {"device": "cpu"},
        }
    )
    torch.manual_seed(42)
    model = GPT3(config.model)
    other = GPT3(config.model)
    other.load_state_dict(model.state_dict())
    batches = [
        {"input_ids": torch.tensor([[2, 3, 4]]), "labels": torch.tensor([[3, 4, 5]])},
        {"input_ids": torch.tensor([[6, 7, 1]]), "labels": torch.tensor([[7, 8, -100]])},
    ]
    for batch in batches:
        batch["attention_mask"] = batch["labels"] != -100
    merged = {key: torch.cat([b[key] for b in batches]) for key in batches[0]}
    model.eval()
    result = train(config, model, batches, batches)
    combined_config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"gradient_accumulation_steps": 1, "output_dir": str(tmp_path / "combined")}
            )
        }
    )
    combined_result = train(combined_config, other, [merged], [merged])
    for actual, expected in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
    records = [
        json.loads(line) for line in (result.run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    combined_record = json.loads(
        (combined_result.run_dir / "metrics.jsonl").read_text().splitlines()[0]
    )
    assert records[0]["target_tokens"] == 5
    assert records[0]["loss"] == pytest.approx(combined_record["loss"])
    assert records[-1]["event"] == "validation"
    assert model.training
    assert result.steps == combined_result.steps == 1
    assert abs(result.final_validation.loss - combined_result.final_validation.loss) < 1e-5


def test_schedule_warmup_and_decay_endpoints():
    config = ExperimentConfig.model_validate(
        {"training": {"max_steps": 6}, "optimizer": {"warmup_steps": 2}}
    )
    rates = [learning_rate_at_step(config, i) for i in range(1, 7)]
    assert rates[0] == config.optimizer.learning_rate / 2
    assert rates[1] == config.optimizer.learning_rate
    assert rates[-1] == config.optimizer.min_learning_rate
    assert rates[2:] == sorted(rates[2:], reverse=True)


def test_training_preserves_loader_errors_and_reports_empty_loaders(tmp_path):
    config = ExperimentConfig.model_validate(
        {
            "model": {"vocab_size": 8, "d_model": 8, "n_layers": 1, "n_heads": 2, "head_dim": 4},
            "tokenizer": {"eos_token_id": 1},
            "hardware": {"device": "cpu"},
            "training": {"max_steps": 1, "output_dir": str(tmp_path)},
            "optimizer": {"warmup_steps": 0},
        }
    )

    def invalid_tokens():
        yield from ()
        raise ValueError("Token IDs outside model vocabulary")

    with pytest.raises(ValueError, match="Token IDs outside model vocabulary"):
        train(config, GPT3(config.model), invalid_tokens(), [])
    with pytest.raises(ValueError, match="Training loader contains no batches"):
        train(config, GPT3(config.model), [], [])


def test_training_samples_at_validation_and_final_step_without_changing_weights(tmp_path):
    config = ExperimentConfig.model_validate(
        {
            "model": {
                "vocab_size": 8,
                "context_length": 3,
                "local_window": 2,
                "n_layers": 1,
                "d_model": 8,
                "n_heads": 2,
                "head_dim": 4,
                "dropout": 0.2,
            },
            "tokenizer": {"eos_token_id": 1},
            "optimizer": {"warmup_steps": 0},
            "training": {
                "max_steps": 3,
                "eval_interval": 2,
                "gradient_accumulation_steps": 1,
                "generate_samples": True,
                "output_dir": str(tmp_path),
            },
            "evaluation": {"sample_prompts": ["2 3 4 5", "2"], "max_new_tokens": 2},
            "generation": {"temperature": 0.8},
            "hardware": {"device": "cpu"},
        }
    )
    tokenizer = Tokenizer(models.WordLevel({str(i): i for i in range(8)}))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    batches = [
        {
            "input_ids": torch.tensor([[2, 3, 4]]),
            "labels": torch.tensor([[3, 4, 5]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.bool),
        }
    ]
    model, baseline = GPT3(config.model), GPT3(config.model)
    baseline.load_state_dict(model.state_dict())
    with patch("gpt3.training.engine.load_tokenizer", return_value=tokenizer) as load:
        run = train(config, model, batches, batches)
        disabled = config.model_copy(
            update={"training": config.training.model_copy(update={"generate_samples": False})}
        )
        train(disabled, baseline, batches, batches)
        load.assert_called_once_with(config)

    lines = (run.run_dir / "metrics.jsonl").read_text().splitlines()
    metrics = [json.loads(line) for line in lines]
    assert all(datetime.fromisoformat(record["timestamp"]).tzinfo == UTC for record in metrics)
    samples = [record for record in metrics if record["event"] == "sample"]
    assert [(sample["step"], sample["prompt"]) for sample in samples] == [(2, "2"), (3, "2")]
    assert all(1 <= len(sample["token_ids"]) <= 2 for sample in samples)
    errors = [record for record in metrics if record["event"] == "sample_error"]
    assert [(error["step"], error["prompt"]) for error in errors] == [
        (2, "2 3 4 5"),
        (3, "2 3 4 5"),
    ]
    torch.testing.assert_close(model.state_dict(), baseline.state_dict(), rtol=0, atol=0)
    assert model.training


def test_validation_limit_does_not_fetch_extra_batch_and_restores_mode():
    from gpt3.training.engine import validate

    config = ExperimentConfig.model_validate(
        {
            "model": {
                "vocab_size": 13,
                "context_length": 3,
                "local_window": 2,
                "n_layers": 1,
                "d_model": 8,
                "n_heads": 2,
                "head_dim": 4,
            },
            "tokenizer": {"eos_token_id": 1},
            "evaluation": {"max_batches": 1},
        }
    )
    model = GPT3(config.model).train()
    model.loss_chunk_size = 7
    rng_before = torch.get_rng_state()

    def batches():
        torch.rand(1)
        yield {
            "input_ids": torch.tensor([[2, 3]]),
            "labels": torch.tensor([[3, 4]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        raise AssertionError("Validation fetched an extra batch")

    metrics = validate(model, batches(), config, torch.device("cpu"))
    assert metrics.target_tokens == 2
    assert model.training
    assert model.loss_chunk_size == 7
    assert torch.equal(torch.get_rng_state(), rng_before)

    def broken_loader():
        torch.rand(1)
        yield {}

    with pytest.raises(ValueError, match="Batch is missing"):
        validate(model, broken_loader(), config, torch.device("cpu"))
    assert model.training
    assert model.loss_chunk_size == 7
    assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize("loss", [709.0, 710.0])
def test_large_validation_loss_is_logged_and_checkpointed(tmp_path, loss):
    config = ExperimentConfig.model_validate({
        "model": {"vocab_size": 8, "n_layers": 1, "d_model": 8, "n_heads": 2, "head_dim": 4},
        "tokenizer": {"eos_token_id": 1},
        "optimizer": {"warmup_steps": 0},
        "training": {"max_steps": 1, "gradient_accumulation_steps": 1, "output_dir": str(tmp_path)},
        "hardware": {"device": "cpu"},
    })
    model = GPT3(config.model)
    forward = model.forward

    def controlled_loss(*args, **kwargs):
        if model.training:
            return forward(*args, **kwargs)
        return GPT3Output(logits=None, loss=torch.tensor(loss))

    ids = torch.tensor([[2, 3]])
    batch = {"input_ids": ids, "labels": ids + 1, "attention_mask": torch.ones_like(ids).bool()}
    with patch.object(model, "forward", side_effect=controlled_loss):
        run = train(config, model, [batch], [batch])
    record = json.loads((run.run_dir / "metrics.jsonl").read_text().splitlines()[-1])
    assert record["loss"] == loss
    assert record["perplexity"] == (math.exp(709) if loss == 709 else None)
    assert (run.run_dir / "checkpoint-00000001.pt").exists()


def test_rejects_unsupported_cuda_bfloat16():
    from unittest.mock import patch

    import pytest

    from gpt3.training.engine import resolve_device

    config = ExperimentConfig.model_validate(
        {"hardware": {"device": "cuda", "precision": "bfloat16"}}
    )
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.is_bf16_supported", return_value=False),
        pytest.raises(ValueError, match="bfloat16"),
    ):
        resolve_device(config)


def test_fused_adamw_requires_cuda_parameters():
    config = ExperimentConfig.model_validate({"model": {"n_layers": 1}})
    model = GPT3(config.model)
    assert make_optimizer(model, config).defaults["fused"] is None

    enabled = config.model_copy(
        update={"hardware": config.hardware.model_copy(update={"fused_adamw": True})}
    )
    with pytest.raises(ValueError, match="fused_adamw requires CUDA"):
        make_optimizer(model, enabled)
    with patch.object(torch.nn.Parameter, "is_cuda", True):
        assert make_optimizer(model, enabled).defaults["fused"] is True


def test_training_scopes_cuda_tf32_and_restores_precision_on_failure():
    config = ExperimentConfig.model_validate(
        {"model": {"n_layers": 1}, "hardware": {"device": "cuda", "allow_tf32": True}}
    )
    with pytest.raises(ValueError, match="allow_tf32 requires CUDA"):
        with cuda_matmul_context(torch.device("cpu"), allow_tf32=True):
            pytest.fail("TF32 context accepted a CPU device")

    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")

        def fail_during_training(*args):
            assert torch.get_float32_matmul_precision() == "high"
            raise RuntimeError("training failed")

        with (
            patch("gpt3.training.engine.resolve_device", return_value=torch.device("cuda")),
            patch("gpt3.training.engine._train_on_device", side_effect=fail_during_training),
            pytest.raises(RuntimeError, match="training failed"),
        ):
            train(config, GPT3(config.model), [], [])
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(previous)
