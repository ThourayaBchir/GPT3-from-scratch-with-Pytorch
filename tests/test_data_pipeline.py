"""Behavioral checks for document boundaries, packing, and split isolation."""

from collections.abc import Iterator
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers

from gpt3.config import ExperimentConfig, load_config
from gpt3.data import PackedTokenDataset, build_dataloader, collate_tokens, tokenize_split
from gpt3.data.pipeline import load_text_split, load_tokenizer, wikitext_articles


def make_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        models.WordLevel(
            {"<unk>": 0, "<|endoftext|>": 1, "one": 2, "two": 3, "three": 4},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def test_wikitext_article_boundaries_and_partial_slice():
    rows = [
        "",
        " leading fragment \n",
        " = one = \n",
        " one two \n",
        "",
        " = = Section = = \n",
        " three \n",
        " = two = \n",
        " two three \n",
    ]
    articles = list(wikitext_articles(({"text": row} for row in rows), "text"))
    assert [article["text"] for article in articles] == [
        "leading fragment",
        "= one =\n\none two\n\n= = Section = =\n\nthree",
        "= two =\n\ntwo three",
    ]
    assert list(wikitext_articles([{"text": " \n"}], "text")) == []


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("split", ["train", "validation"])
def test_wikitext_groups_before_shuffle_and_adds_one_eos(tmp_path, streaming, split):
    rows = [
        " = one = \n",
        " one two \n",
        " = = Section = = \n",
        " three \n",
        " = two = \n",
        " two three \n",
    ]
    if streaming:
        rows = ["outside the slice", *rows, "also outside"]
    source = Dataset.from_dict({"text": rows})
    if streaming:
        source = source.to_iterable_dataset()
    config = ExperimentConfig.model_validate(
        {
            "data": {
                "document_mode": "wikitext_articles",
                "streaming": streaming,
                "cache_dir": str(tmp_path),
                f"{split}_split": f"{split}[1:7]" if streaming else split,
            },
            "model": {"vocab_size": 5},
            "tokenizer": {"eos_token_id": 1},
        }
    )
    with patch("gpt3.data.pipeline.load_dataset", return_value=source) as load_split:
        documents = load_text_split(config, split)
    assert load_split.call_args.kwargs["split"] == split
    texts = [row["text"] for row in documents]
    assert sorted(texts) == sorted(
        ["= one =\n\none two\n\n= = Section = =\n\nthree", "= two =\n\ntwo three"]
    )
    tokenized = list(tokenize_split(documents, config, make_tokenizer()))
    assert len(tokenized) == 2
    assert all(row["input_ids"].count(1) == 1 for row in tokenized)
    assert all(row["input_ids"][-1] == 1 for row in tokenized)


@pytest.mark.parametrize("row_slice", ["[:2]", "[1:3]", "[1:]"])
def test_streaming_row_slice_matches_cached_slice(tmp_path, row_slice):
    text_file = tmp_path / "validation.txt"
    text_file.write_text("one\ntwo\nthree\none two\n", encoding="utf-8")
    settings = {
        "dataset_id": "text",
        "dataset_config": None,
        "data_files": {"validation": [str(text_file)]},
        "validation_split": f"validation{row_slice}",
        "cache_dir": str(tmp_path),
    }

    def read_ids(streaming):
        config = ExperimentConfig.model_validate(
            {
                "data": {**settings, "streaming": streaming},
                "model": {"vocab_size": 5},
                "tokenizer": {"eos_token_id": 1},
            }
        )
        source = load_text_split(config, "validation")
        return [row["input_ids"] for row in tokenize_split(source, config, make_tokenizer())]

    assert read_ids(streaming=True) == read_ids(streaming=False)


def test_fineweb_preset_uses_explicit_shards_and_disjoint_splits(tmp_path):
    config = load_config("configs/fineweb_edu.toml")
    files = config.data.data_files
    assert files == {
        "train": [f"sample/10BT/{i:03d}_00000.parquet" for i in range(13)],
        "holdout": ["sample/10BT/013_00000.parquet"],
    }

    train_file = tmp_path / "train.parquet"
    holdout_file = tmp_path / "holdout.parquet"
    pq.write_table(pa.table({"text": ["training document"]}), train_file)
    pq.write_table(pa.table({"text": [str(i) for i in range(2001)]}), holdout_file)
    local_files = {"train": [str(train_file)], "holdout": [str(holdout_file)]}

    def load_local(path, **kwargs):
        assert path == "HuggingFaceFW/fineweb-edu"
        assert kwargs["name"] == "sample-10BT"
        assert kwargs["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
        assert kwargs["data_files"] == files
        assert kwargs["streaming"] is False
        return load_dataset(
            "parquet", **{**kwargs, "data_files": local_files, "cache_dir": str(tmp_path / "cache")}
        )

    with patch("gpt3.data.pipeline.load_dataset", side_effect=load_local):
        assert list(load_text_split(config, "train")) == [{"text": "training document"}]
        for split, start in [("validation", 0), ("test", 1000)]:
            assert [row["text"] for row in load_text_split(config, split)] == [
                str(i) for i in range(start, start + 1000)
            ]


def test_article_preprocessing_change_rejects_old_resume():
    from gpt3.training.checkpoint import check_resume_config

    old = ExperimentConfig()
    current = ExperimentConfig.model_validate({"data": {"document_mode": "wikitext_articles"}})
    with pytest.raises(ValueError, match="changed data"):
        check_resume_config(old.model_dump(), current)


def test_tokenize_and_pack_documents_without_losing_boundary_target() -> None:
    config = ExperimentConfig.model_validate(
        {
            "model": {"vocab_size": 5, "context_length": 3, "local_window": 3},
            "tokenizer": {"eos_token_id": 1},
        }
    )
    source = Dataset.from_dict({"text": ["one two", "", "three one"]})
    tokenized = tokenize_split(source, config, make_tokenizer())
    assert tokenized.column_names == ["input_ids"]
    assert tokenized["input_ids"] == [[2, 3, 1], [], [4, 2, 1]]

    examples = list(PackedTokenDataset(tokenized, context_length=3, drop_remainder=False))
    assert [example["input_ids"].tolist() for example in examples] == [[2, 3, 1], [4, 2]]
    assert [example["labels"].tolist() for example in examples] == [[3, 1, 4], [2, 1]]

    batch = collate_tokens(examples, pad_token_id=1)
    assert batch["input_ids"].tolist() == [[2, 3, 1], [4, 2, 1]]
    assert batch["labels"].tolist() == [[3, 1, 4], [2, 1, -100]]
    assert batch["attention_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["labels"].dtype == torch.long


def test_streamed_and_cached_paths_pack_identically() -> None:
    config = ExperimentConfig.model_validate(
        {
            "model": {"vocab_size": 5, "context_length": 2, "local_window": 2},
            "tokenizer": {"eos_token_id": 1},
        }
    )
    source = Dataset.from_dict({"text": ["one", "two three", "one"]})
    tokenizer = make_tokenizer()
    cached = tokenize_split(source, config, tokenizer)
    streamed = tokenize_split(source.to_iterable_dataset(), config, tokenizer)
    cached_examples = list(PackedTokenDataset(cached, 2))
    streamed_examples = list(PackedTokenDataset(streamed, 2))
    assert len(cached_examples) == len(streamed_examples) == 3
    assert [ex["labels"].tolist() for ex in cached_examples] == [
        ex["labels"].tolist() for ex in streamed_examples
    ]


def test_load_tokenizer_checks_vocab_and_eos() -> None:
    tokenizer = make_tokenizer()
    config = ExperimentConfig.model_validate(
        {"model": {"vocab_size": 5}, "tokenizer": {"eos_token_id": 1}}
    )
    with patch("gpt3.data.pipeline.Tokenizer.from_pretrained", return_value=tokenizer):
        assert load_tokenizer(config) is tokenizer
    wrong_vocab = config.model_copy(
        update={"model": config.model.model_copy(update={"vocab_size": 6})}
    )
    with patch("gpt3.data.pipeline.Tokenizer.from_pretrained", return_value=tokenizer):
        with pytest.raises(ValueError, match="vocabulary"):
            load_tokenizer(wrong_vocab)


def test_build_dataloader_uses_only_requested_split() -> None:
    config = ExperimentConfig.model_validate(
        {
            "model": {"vocab_size": 5, "context_length": 2, "local_window": 2},
            "tokenizer": {"eos_token_id": 1},
            "training": {"batch_size": 1},
            "evaluation": {"batch_size": 1},
        }
    )
    source = Dataset.from_dict({"text": ["one two three"]})
    with (
        patch("gpt3.data.pipeline.load_tokenizer", return_value=make_tokenizer()),
        patch("gpt3.data.pipeline.load_text_split", return_value=source) as load_split,
    ):
        batches = list(build_dataloader(config, "validation"))
    load_split.assert_called_once_with(config, "validation")
    assert len(batches) == 1
    assert batches[0]["input_ids"].tolist() == [[2, 3]]
    assert batches[0]["labels"].tolist() == [[3, 4]]


def test_invalid_rows_and_empty_batch_fail_clearly() -> None:
    def rows() -> Iterator[dict[str, object]]:
        yield {"input_ids": [1, "bad"]}

    with pytest.raises(TypeError, match="integer input_ids"):
        list(PackedTokenDataset(rows(), 2))
    with pytest.raises(ValueError, match="empty batch"):
        collate_tokens([], pad_token_id=1)
