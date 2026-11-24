"""Continuation likelihoods, few-shot prompts, and greedy sample inspection."""

import json
import math
import sys
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from gpt3.config import ModelConfig, TaskConfig
from gpt3.evaluation import __main__ as evaluation_cli
from gpt3.evaluation.generation import generate_sample
from gpt3.evaluation.tasks import continuation_score, evaluate_task
from gpt3.model import GPT3


def fixtures():
    tokenizer = Tokenizer(models.WordLevel({"a": 0, "b": 1, "c": 2, "d": 3}, unk_token="a"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    config = ModelConfig(
        vocab_size=4,
        context_length=16,
        local_window=2,
        n_layers=2,
        d_model=8,
        n_heads=2,
        head_dim=4,
    )
    model = GPT3(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model, tokenizer


def test_continuation_scoring_ignores_prompt_and_preserves_mode():
    model, _ = fixtures()
    score = continuation_score(model, [0, 1, 2], [3, 1])
    assert score == pytest.approx(-2 * math.log(4))
    score = continuation_score(model, [0, 1, 2], [3, 1], length_normalize=True)
    assert score == pytest.approx(-math.log(4))
    assert model.training
    with pytest.raises(ValueError, match="exceeds"):
        continuation_score(model, [0] * 16, [1, 2])


def test_zero_and_few_shot_tasks_report_accuracy_and_reject_leakage(tmp_path):
    model, tokenizer = fixtures()
    evaluation = tmp_path / "test.jsonl"
    demonstration = tmp_path / "train.jsonl"
    evaluation.write_text(json.dumps({"prompt": "a ", "choices": ["b", "c"], "answer": 0}) + "\n")
    demonstration.write_text(
        json.dumps({"prompt": "d ", "choices": ["b", "c"], "answer": 1}) + "\n"
    )
    zero = TaskConfig(name="toy", evaluation_file=str(evaluation))
    assert evaluate_task(model, tokenizer, zero)["accuracy"] == 1.0
    few = TaskConfig(
        name="toy", evaluation_file=str(evaluation), shots=1, demonstrations_file=str(demonstration)
    )
    result = evaluate_task(model, tokenizer, few)
    assert result["shots"] == 1
    assert result["examples"] == 1
    demonstration.write_text(evaluation.read_text())
    with pytest.raises(ValueError, match="overlap"):
        evaluate_task(model, tokenizer, few)


def test_choice_boundary_uses_joint_encoding_and_chunked_loss(tmp_path):
    tokens = ["A", ":", "Ġ", "y", "e", "s", "t", "ye", "yes", "yet", "Ġyes", "Ġyet"]
    tokenizer = Tokenizer(
        models.BPE(
            {token: i for i, token in enumerate(tokens)},
            [("y", "e"), ("ye", "s"), ("ye", "t"), ("Ġ", "yes"), ("Ġ", "yet")],
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    base, _ = fixtures()
    model = GPT3(base.config.model_copy(update={"vocab_size": len(tokens)}))
    model.loss_chunk_size = 1
    path = tmp_path / "choices.jsonl"
    task = TaskConfig(name="boundary", evaluation_file=str(path))

    def scores(prompt, choices):
        path.write_text(json.dumps({"prompt": prompt, "choices": choices, "answer": 0}))
        return evaluate_task(model, tokenizer, task)["predictions"][0]["scores"]

    with patch.object(model, "_chunked_loss", wraps=model._chunked_loss) as loss:
        actual = scores("A: ", ["yes", "yet"])
        assert actual == scores("A:", [" yes", " yet"])
        assert actual == scores("A: y", ["es", "et"])
    assert loss.call_count == 6
    assert actual == [continuation_score(model, [0, 1], [token]) for token in (10, 11)]


def test_report_write_is_atomic_and_refuses_overwrite(tmp_path):
    path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="JSON compliant"):
        evaluation_cli._write_report(path, {"first": "written", "loss": math.nan})
    assert list(tmp_path.iterdir()) == []

    evaluation_cli._write_report(path, {"loss": 1.0})
    with pytest.raises(FileExistsError):
        evaluation_cli._write_report(path, {"loss": 2.0})
    assert json.loads(path.read_text()) == {"loss": 1.0}
    assert list(tmp_path.iterdir()) == [path]


def test_evaluation_cli_rejects_existing_report_before_loading(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.touch()
    monkeypatch.setattr(
        sys, "argv", ["evaluation", "unused.toml", "unused.pt", "--output", str(path)]
    )
    with pytest.raises(FileExistsError, match="report already exists"):
        evaluation_cli.main()


def test_generation_stops_at_eos_and_restores_mode():
    model, tokenizer = fixtures()
    assert generate_sample(model, tokenizer, "b", 10, eos_token_id=0) == ""
    assert model.training
