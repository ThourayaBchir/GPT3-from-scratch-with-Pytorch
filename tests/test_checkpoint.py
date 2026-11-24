"""Resume equivalence with dropout, optimizer moments, and iterator replay."""

import json
import random
from functools import partial
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from gpt3.config import ExperimentConfig
from gpt3.model import GPT3
from gpt3.training.checkpoint import (
    load_model,
    random_state,
    read_checkpoint,
    restore_random_state,
    save_checkpoint,
)
from gpt3.training.engine import train


@pytest.mark.parametrize(
    ("legacy_checkpoint", "compile_model"),
    [(False, False), (True, False), (False, True)],
)
def test_resume_matches_uninterrupted_training(
    tmp_path, legacy_checkpoint, compile_model, monkeypatch
):
    compiler = Mock(wraps=partial(torch.compile, backend="aot_eager"))
    monkeypatch.setattr(torch, "compile", compiler)
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
                "dropout": 0.2,
            },
            "tokenizer": {"eos_token_id": 1},
            "optimizer": {"warmup_steps": 0},
            "training": {
                "max_steps": 3,
                "gradient_accumulation_steps": 2,
                "checkpoint_interval": 1,
                "eval_interval": 1,
                "log_interval": 2,
                "output_dir": str(tmp_path / "full"),
            },
            "hardware": {"device": "cpu", "compile_model": compile_model},
        }
    )
    batches = [
        {
            "input_ids": torch.tensor([[i, i + 1, i + 2]]),
            "labels": torch.tensor([[i + 1, i + 2, i + 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        }
        for i in (2, 4, 6)
    ]
    full_model = GPT3(config.model)
    validation_loader = torch.utils.data.DataLoader(batches, batch_size=None)
    clock = SimpleNamespace(perf_counter=count().__next__)
    with patch("gpt3.training.engine.time", clock):
        full = train(config, full_model, batches, validation_loader)
    if compile_model:
        compiler.assert_called_once_with(full_model)
    else:
        compiler.assert_not_called()
    resume_path = full.run_dir / "checkpoint-00000001.pt"
    saved = read_checkpoint(resume_path)
    assert isinstance(saved["progress"]["interval_loss"], float)
    assert saved["progress"]["interval_loss"] > 0
    if legacy_checkpoint:
        del saved["run_id"]
        del saved["progress"]["interval_train_seconds"]
        del saved["config"]["hardware"]["fused_adamw"]
        del saved["config"]["hardware"]["allow_tf32"]
        resume_path = tmp_path / "legacy.pt"
        save_checkpoint(resume_path, saved)
    resume = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "resume_from": str(resume_path),
                    "eval_interval": 3,
                    "keep_last_checkpoint": True,
                    "output_dir": str(tmp_path / "resumed"),
                }
            )
        }
    )
    resumed_model = GPT3(config.model)
    with patch("gpt3.training.engine.time", clock):
        resumed = train(resume, resumed_model, batches, validation_loader)
    assert sorted(path.name for path in resumed.run_dir.glob("checkpoint-*.pt")) == [
        "checkpoint-00000003.pt"
    ]
    assert resume_path.is_file()
    logs = [
        [
            entry
            for line in (run.run_dir / "metrics.jsonl").read_text().splitlines()
            if (entry := json.loads(line))["event"] == "train"
        ]
        for run in (full, resumed)
    ]
    full_first, resumed_first = logs[0][0], logs[1][0]
    assert full_first["target_tokens"] == resumed_first["target_tokens"]
    assert full_first["loss"] == resumed_first["loss"]
    for key in ("train_seconds", "tokens_per_second"):
        assert resumed_first[key] == (None if legacy_checkpoint else full_first[key])
        assert logs[1][1][key] == logs[0][1][key]
    for expected, actual in zip(full_model.parameters(), resumed_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    a = read_checkpoint(full.run_dir / "checkpoint-00000003.pt")
    b = read_checkpoint(resumed.run_dir / "checkpoint-00000003.pt")
    assert a["progress"] == b["progress"]
    assert a["scheduler"] == b["scheduler"]
    for index in a["optimizer"]["state"]:
        for key in a["optimizer"]["state"][index]:
            torch.testing.assert_close(
                a["optimizer"]["state"][index][key],
                b["optimizer"]["state"][index][key],
                rtol=0,
                atol=0,
            )
    loaded, _ = load_model(resumed.run_dir / "checkpoint-00000003.pt", torch.device("cpu"))
    assert loaded.lm_head.weight is loaded.token_embedding.weight
    changed = resume.model_copy(
        update={"training": resume.training.model_copy(update={"max_steps": 4})}
    )
    with pytest.raises(ValueError, match="max_steps"):
        train(changed, GPT3(config.model), batches, batches)


def test_random_state_roundtrip():
    state = random_state()
    expected = (random.random(), float(np.random.random()), torch.rand(2))
    restore_random_state(state)
    actual = (random.random(), float(np.random.random()), torch.rand(2))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)


def test_failed_save_preserves_existing_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {"value": 1})
    with patch("torch.save", side_effect=OSError("Disk full")), pytest.raises(OSError):
        save_checkpoint(path, {"value": 2})
    assert read_checkpoint(path)["value"] == 1
    assert list(tmp_path.iterdir()) == [path]
