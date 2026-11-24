"""Checkpointed gradients, chunked vocabulary loss, and disjoint rank shards."""

from collections import Counter
from itertools import islice

import pytest
import torch

from gpt3.config import ModelConfig
from gpt3.data import PackedTokenDataset
from gpt3.model import GPT3


def test_checkpointing_and_chunked_loss_preserve_gradients():
    config = ModelConfig(
        vocab_size=13,
        context_length=6,
        local_window=2,
        n_layers=2,
        d_model=12,
        n_heads=2,
        head_dim=6,
        dropout=0.2,
    )
    original, optimized = GPT3(config), GPT3(config)
    optimized.load_state_dict(original.state_dict())
    optimized.activation_checkpointing = True
    optimized.loss_chunk_size = 2
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    labels = torch.tensor([[3, 4, 5, -100, -100, -100]])
    torch.manual_seed(23)
    before = original(ids, labels=labels)
    before.loss.backward()
    torch.manual_seed(23)
    after = optimized(ids, labels=labels, return_logits=False)
    after.loss.backward()
    assert after.logits is None
    torch.testing.assert_close(before.loss, after.loss)
    for first, second in zip(original.parameters(), optimized.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad, atol=1e-6, rtol=1e-4)


def test_rank_shards_partition_packed_examples_without_duplicates():
    documents = [{"input_ids": list(range(20))}]
    full = list(PackedTokenDataset(documents, 3, drop_remainder=False))
    shards = [
        list(PackedTokenDataset(documents, 3, drop_remainder=False, rank=rank, world_size=3))
        for rank in range(3)
    ]
    for rank, shard in enumerate(shards):
        assert len(shard) == len(full[rank::3])
        for expected, actual in zip(full[rank::3], shard, strict=True):
            torch.testing.assert_close(expected["input_ids"], actual["input_ids"])
            torch.testing.assert_close(expected["labels"], actual["labels"])


def test_repeated_training_shards_preserve_global_example_frequency():
    # Five examples across two ranks used to repeat the smaller shard faster.
    documents = [{"input_ids": list(range(16))}]
    shards = [
        PackedTokenDataset(documents, 3, rank=rank, world_size=2, repeat=True)
        for rank in range(2)
    ]
    streams = [list(islice(shard, 15)) for shard in shards]
    ordered = [
        int(example["input_ids"][0]) for pair in zip(*streams, strict=True) for example in pair
    ]
    assert ordered == [0, 3, 6, 9, 12] * 6
    assert Counter(ordered) == {start: 6 for start in (0, 3, 6, 9, 12)}
    # Replaying from a new iterator preserves checkpoint batch skipping.
    replay = list(islice(shards[0], 7, 15))
    for expected, actual in zip(streams[0][7:], replay, strict=True):
        torch.testing.assert_close(expected["input_ids"], actual["input_ids"])


def test_repeated_shards_support_fewer_examples_than_ranks():
    documents = [{"input_ids": [0, 1, 2, 3]}]
    for rank in range(3):
        shard = PackedTokenDataset(documents, 3, rank=rank, world_size=3, repeat=True)
        assert len(list(islice(shard, 4))) == 4


def test_repeated_empty_stream_terminates():
    assert list(PackedTokenDataset([], 3, repeat=True)) == []


def test_one_shot_document_source_requires_single_pass():
    documents = [{"input_ids": [0, 1, 2, 3]}]
    with pytest.raises(ValueError, match="re-iterable document source"):
        PackedTokenDataset(iter(documents), 3, repeat=True)

    assert len(list(PackedTokenDataset(iter(documents), 3))) == 1
