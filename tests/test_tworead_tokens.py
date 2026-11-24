"""Token training, generation, and checkpoint compatibility without Hub access."""

import runpy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "operator_memory_tworead.py"))
Config = script["Config"]
LanguageModel = script["LanguageModel"]
TextCodec = script["TextCodec"]


@pytest.fixture
def codec():
    vocabulary = {f"t{i}": i for i in range(298)}
    vocabulary.update({"<unk>": 298, "<|endoftext|>": 299})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<|endoftext|>"])
    return TextCodec("gpt2", "test-tokenizer", "test-revision", tokenizer.to_str())


def config_for(codec):
    return Config(
        width=16,
        heads=2,
        layers=4,
        chunk_size=3,
        window=5,
        memory_reads=2,
        vocab_size=codec.vocab_size,
        tokenization="gpt2",
        tokenizer_name=codec.name,
        tokenizer_revision=codec.revision,
        eos_token_id=codec.eos_token_id,
    )


def test_wikitext_articles_and_eos_are_preserved(codec):
    args = SimpleNamespace(
        data=None,
        dataset="Salesforce/wikitext",
        dataset_config="wikitext-2-raw-v1",
        dataset_revision="main",
        cache_dir="unused",
        document_separator="",
    )
    with patch("datasets.load_dataset") as loader:
        loader.side_effect = [
            {
                "text": [
                    " = t270 = \n",
                    " t271 t272 \n",
                    "",
                    " t273 \n",
                    " = t280 = \n",
                    " t281 t282 \n",
                ]
            },
            {"text": [" = t290 = \n", " t291 t292 \n"]},
        ]
        training, validation = script["load_training_documents"](args)
    assert training == ["= t270 =\n\nt271 t272\n\nt273", "= t280 =\n\nt281 t282"]
    assert validation == ["= t290 =\n\nt291 t292"]
    assert [call.kwargs["split"] for call in loader.call_args_list] == ["train", "validation"]
    corpus = script["token_corpus"](training, codec, 2)
    assert all(doc[-1].item() == 299 and doc.tolist().count(299) == 1 for doc in corpus.docs)
    # Every sampled window belongs to one article, including windows ending in EOS.
    expected = {tuple(doc[i : i + 3].tolist()) for doc in corpus.docs for i in range(len(doc) - 2)}
    sampled = corpus.batch(100, torch.Generator().manual_seed(1), "cpu")
    assert all(tuple(row) in expected for row in sampled.tolist())
    with pytest.raises(ValueError, match="outside the vocabulary"):
        script["Corpus"].from_documents([[1, 300, 2]], 2, 300)


def test_local_unicode_split_precedes_tokenization(tmp_path):
    path = tmp_path / "text.txt"
    text = "é🙂" * 100
    path.write_text(text, encoding="utf-8")
    args = SimpleNamespace(data=str(path), document_separator="")
    training, validation = script["load_training_documents"](args)
    assert training[0] + validation[0] == text
    assert len(training[0]) == 180
    path.write_text("article one|article two|article three", encoding="utf-8")
    args.document_separator = "|"
    assert script["load_training_documents"](args) == (
        ["article one", "article two"],
        ["article three"],
    )


def test_large_ids_streaming_and_loss_gradients(codec):
    model = LanguageModel(config_for(codec)).double()
    ids = torch.tensor([[270, 271, 272, 273, 274, 275, 276]])
    full, _ = model(ids[:, :-1])
    first, state = model(ids[:, :3])
    second, _ = model(ids[:, 3:-1], state)
    assert full.shape == (1, 6, 300)
    torch.testing.assert_close(full, torch.cat((first, second), dim=1))
    chunk_loss = script["segment_loss"](model, ids, 3)
    reference_loss = torch.nn.functional.cross_entropy(full.flatten(0, 1), ids[:, 1:].flatten())
    torch.testing.assert_close(chunk_loss, reference_loss)
    chunk_grad = torch.autograd.grad(chunk_loss, tuple(model.parameters()))
    full_grad = torch.autograd.grad(reference_loss, tuple(model.parameters()))
    for a, b in zip(chunk_grad, full_grad, strict=True):
        torch.testing.assert_close(a, b)


def test_generation_decodes_large_ids_and_stops_at_eos(codec):
    class ScriptedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = config_for(codec)
            self.calls = 0

        def forward(self, ids, states=None):
            logits = torch.full((1, ids.size(1), 300), -1000.0)
            logits[:, -1, 270 if self.calls == 0 else 299] = 0
            self.calls += 1
            return logits, None

    model = ScriptedModel()
    rng = torch.random.get_rng_state()
    result = script["sample_text"](model, "t271", 10, top_k=300, codec=codec)
    assert result == "t271 t270"
    assert model.calls == 2 and model.training
    assert torch.equal(rng, torch.random.get_rng_state())


def test_checkpoint_roundtrip_and_legacy_byte_loading(tmp_path, codec):
    config = config_for(codec)
    model = LanguageModel(config)
    path = tmp_path / "tokens.pt"
    payload = {
        "format_version": 3,
        "config": asdict(config),
        "model": model.state_dict(),
        "tokenizer_json": codec.to_json(),
    }
    torch.save(payload, path)
    with patch("tokenizers.Tokenizer.from_pretrained", side_effect=AssertionError("network")):
        restored, restored_codec = script["load_checkpoint"](path, "cpu")
    ids = torch.tensor([[270, 271]])
    torch.testing.assert_close(model(ids)[0], restored(ids)[0])
    assert restored_codec.encode("t270 t271") == [270, 271]
    payload["tokenizer_json"] = None
    torch.save(payload, path)
    with pytest.raises(ValueError, match="missing its tokenizer"):
        script["load_checkpoint"](path, "cpu")
    payload["tokenizer_json"] = codec.to_json()
    payload["config"]["vocab_size"] = 301
    torch.save(payload, path)
    with pytest.raises(ValueError, match="does not match"):
        script["load_checkpoint"](path, "cpu")

    legacy_config = Config(width=16, heads=2, layers=4)
    legacy = LanguageModel(legacy_config)
    metadata = asdict(legacy_config)
    for name in (
        "vocab_size",
        "tokenization",
        "tokenizer_name",
        "tokenizer_revision",
        "eos_token_id",
    ):
        metadata.pop(name)
    torch.save({"format_version": 2, "config": metadata, "model": legacy.state_dict()}, path)
    restored, byte_codec = script["load_checkpoint"](path, "cpu")
    assert byte_codec.mode == "bytes"
    assert byte_codec.decode(byte_codec.encode("café🙂")) == "café🙂"
    torch.testing.assert_close(legacy(ids % 256)[0], restored(ids % 256)[0])


def test_training_cli_saves_tokenizer_and_token_metrics(tmp_path, codec, capsys):
    data = tmp_path / "corpus.txt"
    data.write_text("t270 t271 t272 t273 " * 40, encoding="utf-8")
    out = tmp_path / "trained.pt"
    argv = [
        "operator_memory_tworead.py",
        "train",
        "--data",
        str(data),
        "--out",
        str(out),
        "--device",
        "cpu",
        "--steps",
        "1",
        "--batch",
        "1",
        "--seq",
        "2",
        "--bptt-chunks",
        "2",
        "--width",
        "16",
        "--heads",
        "2",
        "--eval-batches",
        "1",
        "--sample-every",
        "1",
        "--sample-tokens",
        "2",
    ]
    with (
        patch("sys.argv", argv),
        patch("tokenizers.Tokenizer.from_pretrained", return_value=codec.tokenizer),
    ):
        script["main"]()
    output = capsys.readouterr().out
    assert '"val_perplexity"' in output and '"val_bits_per_byte"' not in output
    saved = torch.load(out, weights_only=True)
    assert saved["format_version"] == 3
    assert saved["config"]["tokenization"] == "gpt2"
    assert saved["config"]["memory_reads"] == 2
    assert saved["tokenizer_json"] == codec.to_json()
    model, restored_codec = script["load_checkpoint"](out, "cpu")
    assert isinstance(script["sample_text"](model, "t270", 2, codec=restored_codec), str)
