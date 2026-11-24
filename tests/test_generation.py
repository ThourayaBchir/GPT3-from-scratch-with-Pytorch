"""Distribution filtering, reproducibility, stopping, and generation state."""

import json
import sys
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError
from tokenizers import Tokenizer, models, pre_tokenizers

from gpt3 import generate as generation_cli
from gpt3.config import ExperimentConfig, GenerationConfig, ModelConfig
from gpt3.evaluation.generation import generate, sampling_probabilities
from gpt3.model import GPT3, GPT3Output
from gpt3.training.checkpoint import save_checkpoint


class ScriptedModel(torch.nn.Module):
    def __init__(self, tokens=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = ModelConfig(vocab_size=5, context_length=3, local_window=2)
        self.tokens = tokens
        self.calls = 0
        self.lengths = []

    def forward(self, ids):
        self.lengths.append(ids.shape[1])
        logits = torch.zeros(1, ids.shape[1], 5)
        if self.tokens is not None:
            logits.fill_(-1000)
            logits[..., self.tokens[self.calls]] = 0
        self.calls += 1
        return GPT3Output(logits)


def tokenizer():
    result = Tokenizer(models.WordLevel({"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}, unk_token="a"))
    result.pre_tokenizer = pre_tokenizers.Whitespace()
    return result


def test_temperature_and_top_k_top_p():
    logits = torch.log(torch.tensor([0.6, 0.3, 0.1]))
    original = logits.clone()
    p = sampling_probabilities(logits, GenerationConfig(temperature=1.0, top_p=0.7))
    torch.testing.assert_close(p, torch.tensor([2 / 3, 1 / 3, 0.0]))
    p = sampling_probabilities(logits, GenerationConfig(temperature=1.0, top_k=2, top_p=0.6))
    torch.testing.assert_close(p, torch.tensor([1.0, 0.0, 0.0]))
    soft = sampling_probabilities(logits, GenerationConfig(temperature=2.0))
    sharp = sampling_probabilities(logits, GenerationConfig(temperature=0.5))
    assert sharp[0] > soft[0]
    torch.testing.assert_close(logits, original)
    # Ties still retain exactly k entries, and oversized k is safe.
    assert (
        sampling_probabilities(torch.zeros(4), GenerationConfig(temperature=1.0, top_k=2)) > 0
    ).sum() == 2
    assert sampling_probabilities(
        logits, GenerationConfig(temperature=1.0, top_k=100)
    ).sum() == pytest.approx(1)


def test_seeded_sampling_is_repeatable_without_changing_global_rng():
    options = GenerationConfig(temperature=1.0, seed=17)
    state = torch.get_rng_state().clone()
    first = generate(ScriptedModel(), tokenizer(), "a", 20, 4, options=options)
    second = generate(ScriptedModel(), tokenizer(), "a", 20, 4, options=options)
    assert first == second
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize(
    "options",
    [GenerationConfig(temperature=5e-324), GenerationConfig(temperature=1.0, top_p=5e-324)],
)
def test_small_positive_sampling_parameters(options):
    probabilities = sampling_probabilities(torch.tensor([2.0, 1.0]), options)
    torch.testing.assert_close(probabilities, torch.tensor([1.0, 0.0]))


@pytest.mark.parametrize(
    ("tokens", "options", "reason", "text"),
    [
        ([1, 4], GenerationConfig(), "eos", "b"),
        ([1, 3], GenerationConfig(stop_token_ids=[3]), "stop_token", "b"),
        ([1, 2], GenerationConfig(stop_strings=["b c"]), "stop_string", ""),
        ([1, 2, 3], GenerationConfig(), "length", "b c d"),
    ],
)
def test_stopping_and_rolling_context(tokens, options, reason, text):
    model = ScriptedModel(tokens).train()
    result = generate(model, tokenizer(), "a b c", len(tokens), 4, options=options)
    assert result.finish_reason == reason
    assert result.text == text
    assert result.token_ids == tokens
    assert model.calls == len(tokens)
    assert all(length == 3 for length in model.lengths)
    assert model.training


def test_model_mode_restored_on_error():
    model = ScriptedModel([]).eval()
    with pytest.raises(IndexError):
        generate(model, tokenizer(), "a", 1, 4)
    assert not model.training


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": -1.0},
        {"temperature": float("nan")},
        {"top_k": 0},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"stop_strings": [""]},
        {"stop_token_ids": [-1]},
    ],
)
def test_invalid_configuration(settings):
    with pytest.raises(ValidationError):
        GenerationConfig.model_validate(settings)


def test_nonfinite_logits_rejected():
    with pytest.raises(FloatingPointError):
        sampling_probabilities(torch.tensor([0.0, float("nan")]), GenerationConfig())


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_cli_rejects_nonpositive_token_limit_before_loading(limit, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate", "unused.toml", "unused.pt", "--prompt", "a", "--max-new-tokens", limit],
    )
    with (
        patch.object(generation_cli, "load_config") as load_config,
        pytest.raises(SystemExit) as exc,
    ):
        generation_cli.main()
    assert exc.value.code == 2
    assert "--max-new-tokens must be positive" in capsys.readouterr().err
    load_config.assert_not_called()


@pytest.mark.parametrize("mismatch", ["model", "tokenizer"])
def test_cli_rejects_incompatible_checkpoint_before_constructing_model(
    mismatch, tmp_path, monkeypatch
):
    saved = ExperimentConfig()
    different_settings = (
        saved.model.model_copy(update={"context_length": 128})
        if mismatch == "model"
        else saved.tokenizer.model_copy(update={"revision": "other"})
    )
    config = saved.model_copy(
        update={
            "hardware": saved.hardware.model_copy(update={"device": "cpu"}),
            mismatch: different_settings,
        }
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {"config": saved.model_dump()})
    monkeypatch.setattr(sys, "argv", ["generate", "unused.toml", str(path), "--prompt", "a"])
    with (
        patch.object(generation_cli, "load_config", return_value=config),
        patch("gpt3.training.checkpoint.GPT3") as construct_model,
        patch.object(generation_cli, "load_tokenizer") as load_tokenizer,
        pytest.raises(ValueError, match="Model/tokenizer settings must match"),
    ):
        generation_cli.main()
    construct_model.assert_not_called()
    load_tokenizer.assert_not_called()


@pytest.mark.parametrize(("flags", "token_count"), [([], 3), (["--max-new-tokens", "2"], 2)])
def test_cli_uses_configured_precision_and_token_limit(
    flags, token_count, tmp_path, monkeypatch, capsys
):
    config = ExperimentConfig.model_validate(
        {
            "model": {
                "vocab_size": 5,
                "context_length": 3,
                "local_window": 2,
                "n_layers": 1,
                "d_model": 8,
                "n_heads": 2,
                "head_dim": 4,
            },
            "tokenizer": {"eos_token_id": 4},
            "evaluation": {"max_new_tokens": 3},
            "hardware": {"device": "cpu", "precision": "bfloat16"},
        }
    )
    path = tmp_path / "checkpoint.pt"
    model = GPT3(config.model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    save_checkpoint(path, {"config": config.model_dump(), "model": model.state_dict()})
    forward = GPT3.forward

    def check_precision(model, *args, **kwargs):
        assert torch.is_autocast_enabled("cpu")
        result = forward(model, *args, **kwargs)
        assert result.logits.dtype == torch.bfloat16
        return result

    monkeypatch.setattr(GPT3, "forward", check_precision)
    monkeypatch.setattr(
        sys, "argv", ["generate", "unused.toml", str(path), "--prompt", "b", *flags]
    )
    with (
        patch.object(generation_cli, "load_config", return_value=config),
        patch.object(generation_cli, "load_tokenizer", return_value=tokenizer()),
    ):
        generation_cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["token_ids"] == [0] * token_count
    assert result["finish_reason"] == "length"
