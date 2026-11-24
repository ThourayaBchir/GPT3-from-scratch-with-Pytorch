"""Strict experiment configuration, independent of PyTorch and network access."""

import argparse
import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)]
NonEmptyString = Annotated[str, Field(min_length=1)]


class ConfigSection(BaseModel):
    """Reject typos and implicit type coercion; keep loaded settings immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelConfig(ConfigSection):
    vocab_size: PositiveInt = 50_257
    context_length: PositiveInt = 256
    n_layers: PositiveInt = 4
    d_model: PositiveInt = 256
    n_heads: PositiveInt = 4
    head_dim: PositiveInt = 64
    mlp_ratio: PositiveInt = 4
    attention_pattern: Literal["dense", "alternating"] = "alternating"
    # Project convention: dense first, then local; not a paper-specified band size.
    local_window: PositiveInt = 128
    dropout: Probability = 0.0
    layer_norm_eps: PositiveFloat = 1e-5
    initializer_std: PositiveFloat = 0.02
    tie_embeddings: bool = True

    @property
    def d_ff(self) -> int:
        return self.d_model * self.mlp_ratio

    @model_validator(mode="after")
    def check_local_window(self) -> Self:
        if self.attention_pattern == "alternating" and self.local_window > self.context_length:
            raise ValueError("model.local_window must not exceed model.context_length")
        return self


class DataConfig(ConfigSection):
    backend: Literal["huggingface", "numpy"] = "huggingface"
    train_shards: list[NonEmptyString] = Field(default_factory=list)
    validation_shards: list[NonEmptyString] = Field(default_factory=list)
    test_shards: list[NonEmptyString] = Field(default_factory=list)
    dataset_id: NonEmptyString = "Salesforce/wikitext"
    dataset_config: NonEmptyString | None = "wikitext-2-raw-v1"
    revision: NonEmptyString = "main"
    data_files: dict[NonEmptyString, Annotated[list[NonEmptyString], Field(min_length=1)]] = (
        Field(default_factory=dict)
    )
    text_column: NonEmptyString = "text"
    train_split: NonEmptyString = "train"
    validation_split: NonEmptyString = "validation"
    test_split: NonEmptyString = "test"
    streaming: bool = False
    cache_dir: NonEmptyString = ".cache/huggingface"
    num_workers: NonNegativeInt = 0
    shuffle_buffer_size: PositiveInt = 10_000
    append_eos: bool = True
    document_mode: Literal["rows", "wikitext_articles"] = "rows"
    # Pack without padding and discard the final incomplete sequence.
    drop_remainder: bool = True

    @model_validator(mode="after")
    def check_splits(self) -> Self:
        if self.backend == "numpy":
            if self.streaming or self.num_workers:
                raise ValueError("NumPy shards require streaming=false and num_workers=0")
            for split in ("train", "validation"):
                if not getattr(self, f"{split}_shards"):
                    raise ValueError(f"data.{split}_shards must be configured for NumPy training")
            return self
        splits = (self.train_split, self.validation_split, self.test_split)
        if len(set(splits)) != 3:
            raise ValueError("data split expressions must be distinct")
        return self


class TokenizerConfig(ConfigSection):
    name: NonEmptyString = "openai-community/gpt2"
    revision: NonEmptyString = "main"
    eos_token_id: NonNegativeInt = 50_256


class OptimizerConfig(ConfigSection):
    name: Literal["adamw"] = "adamw"
    learning_rate: PositiveFloat = 6e-4
    min_learning_rate: NonNegativeFloat = 6e-5
    beta1: Probability = 0.9
    beta2: Probability = 0.95
    eps: PositiveFloat = 1e-8
    weight_decay: NonNegativeFloat = 0.1
    grad_clip_norm: PositiveFloat = 1.0
    scheduler: Literal["cosine"] = "cosine"
    warmup_steps: NonNegativeInt = 100

    @model_validator(mode="after")
    def check_learning_rates(self) -> Self:
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("optimizer.min_learning_rate must not exceed learning_rate")
        return self


class TrainingConfig(ConfigSection):
    seed: NonNegativeInt = 42
    batch_size: PositiveInt = 2
    gradient_accumulation_steps: PositiveInt = 8
    max_steps: PositiveInt = 1_000
    log_interval: PositiveInt = 10
    eval_interval: PositiveInt = 100
    generate_samples: bool = False
    checkpoint_interval: PositiveInt = 100
    keep_last_checkpoint: bool = False
    output_dir: NonEmptyString = "artifacts/tiny"
    resume_from: NonEmptyString | None = None


class TaskConfig(ConfigSection):
    name: NonEmptyString
    evaluation_file: NonEmptyString
    demonstrations_file: NonEmptyString | None = None
    shots: NonNegativeInt = 0
    instruction: str = ""
    length_normalize: bool = False

    @model_validator(mode="after")
    def check_demonstrations(self) -> Self:
        if self.shots and not self.demonstrations_file:
            raise ValueError("few-shot tasks require demonstrations_file")
        return self


class EvaluationConfig(ConfigSection):
    batch_size: PositiveInt = 2
    max_batches: PositiveInt | None = 50
    sample_prompts: list[str] = Field(default_factory=list)
    max_new_tokens: PositiveInt = 32
    tasks: list[TaskConfig] = Field(default_factory=list)


class GenerationConfig(ConfigSection):
    # Zero selects greedy decoding; positive values enable sampling.
    temperature: NonNegativeFloat = 0.0
    top_k: PositiveInt | None = None
    top_p: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 1.0
    seed: NonNegativeInt | None = 42
    stop_token_ids: list[NonNegativeInt] = Field(default_factory=list)
    stop_strings: list[NonEmptyString] = Field(default_factory=list)


class HardwareConfig(ConfigSection):
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    fused_adamw: bool = False
    allow_tf32: bool = False
    compile_model: bool = False
    gradient_checkpointing: bool = False
    loss_chunk_size: NonNegativeInt = 0


class ExperimentConfig(ConfigSection):
    name: NonEmptyString = "tiny"
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.tokenizer.eos_token_id >= self.model.vocab_size:
            raise ValueError("tokenizer.eos_token_id must be smaller than model.vocab_size")
        if self.optimizer.warmup_steps >= self.training.max_steps:
            raise ValueError("optimizer.warmup_steps must be smaller than training.max_steps")
        return self

    @property
    def tokens_per_optimizer_step_per_device(self) -> int:
        return (
            self.training.batch_size
            * self.training.gradient_accumulation_steps
            * self.model.context_length
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load TOML, apply defaults, and validate without downloading or allocating a model."""
    with Path(path).open("rb") as handle:
        return ExperimentConfig.model_validate(tomllib.load(handle))


def main() -> None:
    """Print resolved settings for review before a future training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to an experiment TOML file")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except (OSError, ValueError, ValidationError) as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    print(config.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
