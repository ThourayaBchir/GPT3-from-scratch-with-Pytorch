"""Offline shard loading, boundaries, validation, and repeated rank streams."""

from itertools import islice
from unittest.mock import patch

import numpy as np
import pytest

from gpt3.config import ExperimentConfig
from gpt3.data import PackedTokenDataset, build_dataloader
from gpt3.data.pipeline import NumpyTokenDocuments


def shard_config(tmp_path, tokens):
    path = tmp_path / "tokens.npy"
    np.save(path, tokens)
    return ExperimentConfig.model_validate({
        "model": {"vocab_size": 20, "context_length": 3, "local_window": 3},
        "tokenizer": {"eos_token_id": 1},
        "data": {"backend": "numpy", "train_shards": [str(path)],
                 "validation_shards": [str(path)],
                 "drop_remainder": False},
        "evaluation": {"batch_size": 2},
    })


def test_offline_loader_preserves_shard_boundaries_and_tail(tmp_path):
    config = shard_config(tmp_path, np.arange(4, dtype=np.uint16))
    second = tmp_path / "second.npy"
    np.save(second, np.arange(4, 9, dtype=np.int32))
    config = config.model_copy(update={"data": config.data.model_copy(update={
        "validation_shards": [str(tmp_path / "tokens.npy"), str(second)]
    })})
    with patch("gpt3.data.pipeline.load_tokenizer", side_effect=AssertionError("network")):
        batches = list(build_dataloader(config, "validation"))
    assert batches[0]["input_ids"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert batches[0]["labels"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert batches[1]["labels"].tolist() == [[7, 8]]


def test_numpy_training_repeats_before_rank_sharding(tmp_path):
    config = shard_config(tmp_path, np.arange(16))
    source = NumpyTokenDocuments(config, "validation")
    streams = [list(islice(PackedTokenDataset(
        source, 3, rank=rank, world_size=2, repeat=True
    ), 10)) for rank in range(2)]
    assert [int(ex["input_ids"][0]) for pair in zip(*streams, strict=True)
            for ex in pair] == [0, 3, 6, 9, 12] * 4


@pytest.mark.parametrize("tokens", [np.array([1.5]), np.ones((2, 2), dtype=int),
                                    np.array([-1, 2]), np.array([20])])
def test_invalid_arrays_fail(tmp_path, tokens):
    config = shard_config(tmp_path, tokens)
    with pytest.raises(ValueError):
        list(build_dataloader(config, "validation"))


def test_missing_and_overlapping_paths_fail(tmp_path):
    config = shard_config(tmp_path, np.arange(4))
    missing = config.model_copy(update={"data": config.data.model_copy(update={
        "validation_shards": [str(tmp_path / "missing*.npy")]
    })})
    with pytest.raises(ValueError, match="No NumPy shards match"):
        build_dataloader(missing, "validation")


def test_only_requested_split_patterns_are_resolved(tmp_path):
    config = shard_config(tmp_path, np.arange(4))
    config = config.model_copy(update={"data": config.data.model_copy(update={
        "test_shards": [str(tmp_path / "missing*.npy")]
    })})
    batches = list(build_dataloader(config, "validation"))
    assert batches


def test_empty_shards_fail_with_clear_error(tmp_path):
    config = shard_config(tmp_path, np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="contains no tokens"):
        list(build_dataloader(config, "validation"))
