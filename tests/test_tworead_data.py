"""Offline checks for the standalone two-read model's input sources."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

load_training_bytes = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "operator_memory_tworead.py")
)["load_training_bytes"]


def test_wikitext_uses_official_splits_and_utf8():
    args = SimpleNamespace(
        data=None,
        dataset="Salesforce/wikitext",
        dataset_config="wikitext-2-raw-v1",
        dataset_revision="pinned-revision",
        cache_dir="local-cache",
    )
    with patch("datasets.load_dataset") as loader:
        loader.side_effect = [{"text": ["café", "", "train"]}, {"text": ["held out"]}]
        training, validation = load_training_bytes(args)

    assert training == "café\n\ntrain".encode()
    assert validation == b"held out"
    assert [call.kwargs["split"] for call in loader.call_args_list] == ["train", "validation"]
    for call in loader.call_args_list:
        assert call.args == (args.dataset, args.dataset_config)
        assert call.kwargs["revision"] == args.dataset_revision
        assert call.kwargs["cache_dir"] == args.cache_dir


def test_local_file_preserves_chronological_split(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"0123456789")
    with patch("datasets.load_dataset") as loader:
        assert load_training_bytes(SimpleNamespace(data=str(path))) == (b"012345678", b"9")
    loader.assert_not_called()
