"""Exercise user-facing configuration errors and the shipped presets."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from gpt3.config import ExperimentConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["tiny", "gpt3_125m", "gpt3_175b"])
def test_shipped_presets_load(name: str) -> None:
    config = load_config(ROOT / "configs" / f"{name}.toml")
    assert config.model.d_ff == 4 * config.model.d_model


def test_paper_reference_dimensions() -> None:
    config = load_config(ROOT / "configs/gpt3_175b.toml")
    assert (config.model.n_layers, config.model.d_model, config.model.n_heads) == (96, 12288, 96)
    assert config.model.head_dim == 128
    assert config.model.context_length == 2048


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"model": {"n_layer": 4}}, "Extra inputs"),
        ({"model": {"n_layers": "4"}}, "valid integer"),
        ({"training": {"batch_size": 0}}, "greater than 0"),
        ({"model": {"local_window": 512}}, "local_window"),
        ({"tokenizer": {"eos_token_id": 50257}}, "eos_token_id"),
        ({"optimizer": {"warmup_steps": 1000}}, "warmup_steps"),
        ({"optimizer": {"min_learning_rate": 1.0}}, "min_learning_rate"),
        ({"data": {"validation_split": "train"}}, "distinct"),
        ({"hardware": {"device": "gpu"}}, "Input should be"),
    ],
)
def test_invalid_settings_are_rejected(settings: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(settings)


def test_toml_unknown_section_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text("[trainng]\nbatch_size = 2\n")
    with pytest.raises(ValidationError, match="trainng"):
        load_config(path)


def test_effective_tokens_per_device() -> None:
    config = load_config(ROOT / "configs/tiny.toml")
    assert config.tokens_per_optimizer_step_per_device == 4096
