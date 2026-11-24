"""Hugging Face text loading, tokenization, and packed next-token examples."""

import argparse
import glob
import re
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from itertools import islice
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, Features, Value, load_dataset  # type: ignore[import-untyped]
from datasets import IterableDataset as HfIterableDataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset

from gpt3.config import ExperimentConfig, load_config

Split = Literal["train", "validation", "test"]


class TokenExample(TypedDict):
    """One unpadded example with two aligned token sequences.

    Attributes:
        input_ids: Input token IDs, a torch.long tensor of shape [tokens].
        labels: Next-token targets, a torch.long tensor of shape [tokens].
    """

    input_ids: torch.Tensor
    labels: torch.Tensor


class TokenBatch(TokenExample):
    """Right-padded examples stacked along the batch dimension.

    Attributes:
        input_ids: Input IDs, a torch.long tensor of shape [batch, tokens].
        labels: Targets, a torch.long tensor of shape [batch, tokens].
            Padding positions contain -100.
        attention_mask: Boolean tensor of shape [batch, tokens] with True
            for real tokens and False for padding.
    """

    attention_mask: torch.Tensor


@dataclass(frozen=True)
class TokenizeDocuments:
    """Picklable map function so Dataset.map can cache tokenization.

    Args:
        tokenizer: Tokenizer used to encode each document.
        text_column: Name of the input column containing document strings.
        append_eos: Whether to append an EOS token to nonblank documents.
        eos_token_id: Token ID to append when append_eos is enabled.
    """

    tokenizer: Tokenizer
    text_column: str
    append_eos: bool
    eos_token_id: int

    def __call__(self, batch: Mapping[str, Sequence[object]]) -> dict[str, list[list[int]]]:
        """Encode a batch of text rows.

        Args:
            batch: Column names mapped to sequences of row values. The text
                column contains one string per document.

        Returns:
            A mapping with an input_ids column containing one variable-length
            list of token IDs per document.
        """
        texts = batch[self.text_column]
        if not all(isinstance(text, str) for text in texts):
            raise TypeError(f"Column {self.text_column!r} must contain text strings")
        # WikiText has empty separator rows; skip them without inserting EOS.
        useful = [text if text.strip() else "" for text in cast(Sequence[str], texts)]
        encodings = self.tokenizer.encode_batch(useful, add_special_tokens=False)
        ids = [encoding.ids for encoding in encodings]
        if self.append_eos:
            for text, document in zip(useful, ids, strict=True):
                if text:
                    document.append(self.eos_token_id)
        return {"input_ids": ids}


def load_tokenizer(config: ExperimentConfig) -> Tokenizer:
    """Load a tokenizer and check vocabulary compatibility.

    Args:
        config: Settings for the tokenizer name, revision, EOS ID, and model
            vocabulary size.

    Returns:
        The loaded tokenizer.

    Raises:
        ValueError: The tokenizer vocabulary size or EOS ID differs from config.
    """
    tokenizer = Tokenizer.from_pretrained(config.tokenizer.name, revision=config.tokenizer.revision)
    actual_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_size != config.model.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary has {actual_size} tokens; model.vocab_size is "
            f"{config.model.vocab_size}"
        )
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id != config.tokenizer.eos_token_id:
        raise ValueError(
            f"Tokenizer EOS ID is {eos_id}; tokenizer.eos_token_id is "
            f"{config.tokenizer.eos_token_id}"
        )
    return tokenizer


def wikitext_articles(
    source: Iterable[Mapping[str, object]], text_column: str
) -> Iterator[dict[str, str]]:
    """Assemble ordered WikiText rows before shuffling or batched tokenization.

    Single-equals headings start articles; nested headings remain within them.
    A split slice starting mid-article retains its leading fragment. Strip row
    padding so standalone trailing spaces do not become newline predictors.

    Args:
        source: Raw WikiText rows in their original order.
        text_column: Key containing the text in each row.

    Yields:
        A mapping from text_column to the assembled text of one article or
        article fragment.
    """
    paragraphs: list[str] = []
    for row in source:
        value = row[text_column]
        if not isinstance(value, str):
            raise TypeError(f"Column {text_column!r} must contain text strings")
        text = value.strip()
        if not text:
            continue
        if re.fullmatch(r"=\s+[^=\s].*?\s+=", text) and paragraphs:
            yield {text_column: "\n\n".join(paragraphs)}
            paragraphs = []
        paragraphs.append(text)
    if paragraphs:
        yield {text_column: "\n\n".join(paragraphs)}


def load_text_split(config: ExperimentConfig, split: Split) -> Dataset | HfIterableDataset:
    """Load one Hub split and prepare its text rows.

    Args:
        config: Dataset, split, cache, document grouping, and shuffle settings.
        split: Logical split to load, one of train, validation, or test.

    Returns:
        A cached Dataset or streaming IterableDataset of text rows. In
        WikiText article mode, each row contains an assembled article.
    """
    split_expression = getattr(config.data, f"{split}_split")
    split_name = split_expression
    start = 0
    stop: int | None = None
    if config.data.streaming:
        match = re.fullmatch(r"([^\[\]+]+)(?:\[([0-9]*):([0-9]*)\])?", split_expression)
        if match is None:
            raise ValueError(
                f"Unsupported streaming split {split_expression!r}; "
                "use a split name or a row slice with non-negative bounds, such as train[:400]"
            )
        split_name, start_bound, stop_bound = match.groups()
        start = int(start_bound) if start_bound else 0
        stop = int(stop_bound) if stop_bound else None
    result = load_dataset(
        config.data.dataset_id,
        name=config.data.dataset_config,
        split=split_name,
        revision=config.data.revision,
        streaming=config.data.streaming,
        cache_dir=config.data.cache_dir,
        **({"data_files": config.data.data_files} if config.data.data_files else {}),
    )
    if not isinstance(result, (Dataset, HfIterableDataset)):
        raise TypeError(f"Expected a single dataset for {split!r}, got {type(result).__name__}")
    columns = result.column_names
    if columns is not None and config.data.text_column not in columns:
        raise ValueError(
            f"Dataset split {split!r} has no {config.data.text_column!r} column; "
            f"available: {columns}"
        )
    if isinstance(result, HfIterableDataset):
        # Slice raw rows before grouping articles or shuffling the training split.
        if start:
            result = result.skip(start)
        if stop is not None:
            result = result.take(max(0, stop - start))
    if config.data.document_mode == "wikitext_articles":
        features = Features({config.data.text_column: Value("string")})
        generator_kwargs = {"source": result, "text_column": config.data.text_column}
        if isinstance(result, Dataset):
            result = Dataset.from_generator(
                wikitext_articles,
                gen_kwargs=generator_kwargs,
                features=features,
                cache_dir=config.data.cache_dir,
            )
        else:
            result = HfIterableDataset.from_generator(
                wikitext_articles,
                gen_kwargs=generator_kwargs,
                features=features,
            )
    if split == "train":
        result = result.shuffle(
            seed=config.training.seed,
            **(
                {"buffer_size": config.data.shuffle_buffer_size}
                if isinstance(result, HfIterableDataset)
                else {}
            ),
        )
    return result


def tokenize_split(
    source: Dataset | HfIterableDataset,
    config: ExperimentConfig,
    tokenizer: Tokenizer,
) -> Dataset | HfIterableDataset:
    """Tokenize in batches and reuse the map cache for Arrow datasets.

    Args:
        source: Cached or streaming dataset with a text column.
        config: Text-column and end-of-document settings.
        tokenizer: Tokenizer used to encode the rows.

    Returns:
        A dataset of the same kind as source with a variable-length input_ids
        list in each row.
    """
    columns = source.column_names
    if columns is not None and config.data.text_column not in columns:
        raise ValueError(f"Missing text column: {config.data.text_column!r}")
    tokenize = TokenizeDocuments(
        tokenizer=tokenizer,
        text_column=config.data.text_column,
        append_eos=config.data.append_eos,
        eos_token_id=config.tokenizer.eos_token_id,
    )
    kwargs = {
        "batched": True,
        "batch_size": 1_000,
        "remove_columns": columns,
    }
    if isinstance(source, Dataset):
        return source.map(
            tokenize,
            num_proc=config.data.num_workers or None,
            desc="Tokenizing documents",
            **kwargs,
        )
    return source.map(tokenize, **kwargs)


class NumpyTokenDocuments:
    """Re-iterable, memory-mapped shards read in bounded token chunks."""

    def __init__(self, config: ExperimentConfig, split: Split) -> None:
        """Resolve the shard paths for a split.

        Args:
            config: Shard path patterns and model vocabulary size.
            split: Logical split whose shard paths should be resolved.
        """
        patterns = getattr(config.data, f"{split}_shards")
        self.paths: list[Path] = []
        for pattern in patterns:
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise ValueError(f"No NumPy shards match {pattern!r} for {split}")
            self.paths.extend(Path(match).resolve() for match in matches)
        if len(set(self.paths)) != len(self.paths):
            raise ValueError(f"NumPy shards for {split} contain duplicate files")
        if not self.paths:
            raise ValueError(f"No NumPy shards configured for {split}")
        self.vocab_size = config.model.vocab_size

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        """Read the selected shards in order.

        Yields:
            A mapping with input_ids containing a one-dimensional list of up
            to 65,536 token IDs from one shard.
        """
        for path in self.paths:
            if path.suffix != ".npy":
                raise ValueError(f"Expected a .npy token shard: {path}")
            tokens = np.load(path, mmap_mode="r", allow_pickle=False)
            if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
                raise ValueError(f"Shard must be a one-dimensional integer array: {path}")
            if len(tokens) == 0:
                raise ValueError(f"Shard contains no tokens: {path}")
            for start in range(0, len(tokens), 65_536):
                chunk = tokens[start : start + 65_536]
                if chunk.min() < 0 or chunk.max() >= self.vocab_size:
                    raise ValueError(f"Token IDs outside model vocabulary in {path}")
                yield {"input_ids": chunk.tolist()}


class PackedTokenDataset(TorchIterableDataset[TokenExample]):
    """Pack a continuous token stream into causal examples with one-token overlap.

    Example:
        With context_length=3 and drop_remainder=False, source rows [1, 2]
        and [3, 4, 5, 6] produce two examples:

            input_ids = [1, 2, 3]    labels = [2, 3, 4]
            input_ids = [4, 5]       labels = [5, 6]

        Token 4 is the last target of the first example and the first input
        of the second. The source row boundary does not break the stream.
    """

    def __init__(
        self,
        documents: Iterable[Mapping[str, object]],
        context_length: int,
        *,
        drop_remainder: bool = True,
        rank: int = 0,
        world_size: int = 1,
        repeat: bool = False,
    ) -> None:
        """Configure packing, sharding, and repetition.

        Args:
            documents: Rows with an input_ids list or tuple of integers.
                With repeat=True, this source must support fresh iteration.
            context_length: Maximum number of input–target pairs in an example.
            drop_remainder: Whether to discard a final short example.
            rank: Zero-based index of the shard to yield.
            world_size: Number of round-robin shards.
            repeat: Whether to start another pass after each nonempty pass.
        """
        super().__init__()
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if repeat and isinstance(documents, Iterator):
            raise ValueError("repeat=True requires a re-iterable document source")
        self.rank = rank
        self.world_size = world_size
        self.documents = documents
        self.context_length = context_length
        self.drop_remainder = drop_remainder
        self.repeat = repeat

    def __iter__(self) -> Iterator[TokenExample]:
        """Iterate over this rank's examples in stream order.

        Yields:
            TokenExample values with torch.long input_ids and labels of shape
            [tokens], where tokens is at most context_length.
        """
        # Repeat the global stream before sharding, retaining the offset across
        # passes so a short final shard cannot cycle faster than the other ranks.
        index = 0
        while True:
            nonempty = False
            for example in self._examples():
                nonempty = True
                if index % self.world_size == self.rank:
                    yield example
                index += 1
            if not self.repeat or not nonempty:
                return

    def _examples(self) -> Iterator[TokenExample]:
        """Pack a single pass through the source.

        Yields:
            Unpadded examples with input_ids and labels of shape [tokens].
            Each label is the next token after its paired input in the stream.
        """
        pending: deque[int] = deque()
        for row in self.documents:
            ids = row["input_ids"]
            if not isinstance(ids, (list, tuple)) or not all(isinstance(i, int) for i in ids):
                raise TypeError("Each tokenized row must contain a list of integer input_ids")
            pending.extend(ids)
            while len(pending) > self.context_length:
                window = [pending.popleft() for _ in range(self.context_length)]
                next_token = pending[0]
                yield {
                    "input_ids": torch.tensor(window, dtype=torch.long),
                    "labels": torch.tensor(window[1:] + [next_token], dtype=torch.long),
                }
        if not self.drop_remainder and len(pending) > 1:
            tail = list(pending)
            yield {
                "input_ids": torch.tensor(tail[:-1], dtype=torch.long),
                "labels": torch.tensor(tail[1:], dtype=torch.long),
            }


def collate_tokens(examples: Sequence[TokenExample], pad_token_id: int) -> TokenBatch:
    """Stack examples and right-pad them to the longest sequence in the batch.

    Args:
        examples: Nonempty collection of examples, each with torch.long
            input_ids and labels of the same nonempty shape [tokens].
        pad_token_id: Value used for padded input IDs.

    Returns:
        A TokenBatch with all tensors of shape [batch, max_tokens]. Padded
        targets are -100, and attention_mask is True at real input positions.

    Example:
        Collating examples of lengths 3 and 2 with pad_token_id=0 can yield:

            input_ids      = [[1,    2,    3   ], [4,    5,    0    ]]
            labels         = [[2,    3,    4   ], [5,    6,    -100 ]]
            attention_mask = [[True, True, True], [True, True, False]]
    """
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    max_length = max(example["input_ids"].numel() for example in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for index, example in enumerate(examples):
        length = example["input_ids"].numel()
        if length != example["labels"].numel() or length == 0:
            raise ValueError("Each example needs equally sized, nonempty inputs and labels")
        input_ids[index, :length] = example["input_ids"]
        labels[index, :length] = example["labels"]
        attention_mask[index, :length] = True
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _packed_split(config: ExperimentConfig, split: Split) -> PackedTokenDataset:
    """Connect the configured token source to the packer.

    Args:
        config: Data backend, tokenizer, and packing settings.
        split: Logical split to prepare.

    Returns:
        A dataset of unpadded TokenExample values. The training split repeats
        and the validation and test splits are finite.
    """
    documents: Iterable[Mapping[str, object]]
    if config.data.backend == "numpy":
        documents = NumpyTokenDocuments(config, split)
    else:
        tokenizer = load_tokenizer(config)
        source = load_text_split(config, split)
        documents = tokenize_split(source, config, tokenizer)
    return PackedTokenDataset(
        documents,
        config.model.context_length,
        drop_remainder=config.data.drop_remainder,
        repeat=split == "train",
    )


class _RoundStatus(IntEnum):
    """Status sent to every rank for a distributed loader round."""

    SOURCE_ERROR = -1
    FULL = 0
    FINAL = 1


class DistributedPackedDataset(TorchIterableDataset[TokenExample]):
    """Pack once on rank 0 and scatter one batch of examples per rank at a time.

    Example:
        With two ranks and batch_size=2, rank 0 packs examples 0 through 3.
        Rank 0 receives examples [0, 2] and rank 1 receives [1, 3].
    """

    def __init__(self, config: ExperimentConfig, split: Split, batch_size: int) -> None:
        """Configure a distributed split.

        Args:
            config: Source and packing settings used on rank zero.
            split: Logical split to distribute.
            batch_size: Number of examples per rank in a full round.
        """
        super().__init__()
        self.config = config
        self.split = split
        self.batch_size = batch_size
        self.packed: PackedTokenDataset | None = None

    def __iter__(self) -> Iterator[TokenExample]:
        """Receive this rank's examples in collective rounds.

        Yields:
            Unpadded examples with torch.long input_ids and labels of shape
            [tokens], where tokens is at most the model context length.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        batch_size = self.batch_size
        width = self.config.model.context_length
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )
        # One packet per rank, with one row per example and round status in row zero.
        status_column, length_column, inputs_start = 0, 1, 2
        labels_start = inputs_start + width
        shape = (batch_size, labels_start + width)
        round_capacity = batch_size * world_size
        examples: Iterator[TokenExample] | None = None
        while True:
            outgoing = None
            source_error = None
            if rank == 0:
                packets = torch.zeros((world_size, *shape), dtype=torch.long)
                try:
                    if examples is None:
                        if self.packed is None:
                            self.packed = _packed_split(self.config, self.split)
                        examples = iter(self.packed)
                    count = 0
                    for example in islice(examples, round_capacity):
                        row = packets[count % world_size, count // world_size]
                        length = example["input_ids"].numel()
                        row[length_column] = length
                        row[inputs_start : inputs_start + length] = example["input_ids"]
                        row[labels_start : labels_start + length] = example["labels"]
                        count += 1
                    # Every rank learns about a partial final round, including
                    # ranks that have no examples left to yield.
                    packets[:, 0, status_column] = (
                        _RoundStatus.FINAL if count < round_capacity else _RoundStatus.FULL
                    )
                except Exception as exc:
                    source_error = exc
                    packets[:, 0, status_column] = _RoundStatus.SOURCE_ERROR
                outgoing = list(packets.to(device).unbind())

            incoming = torch.empty(shape, dtype=torch.long, device=device)
            dist.scatter(incoming, scatter_list=outgoing, src=0)
            packet = incoming.cpu()
            status = int(packet[0, status_column])
            if status == _RoundStatus.SOURCE_ERROR:
                if source_error is not None:
                    raise source_error
                raise RuntimeError(f"Rank 0 failed to prepare data for the {self.split} split")
            for row in packet:
                length = int(row[length_column])
                if length:
                    yield {
                        "input_ids": row[inputs_start : inputs_start + length],
                        "labels": row[labels_start : labels_start + length],
                    }
            if status == _RoundStatus.FINAL:
                return


def build_dataloader(config: ExperimentConfig, split: Split) -> DataLoader[TokenExample]:
    """Build a loader that collates packed examples into model inputs.

    Args:
        config: Data, model, batch size, and device settings.
        split: Logical split to load, one of train, validation, or test.

    Returns:
        A DataLoader yielding TokenBatch values. Each tensor has shape
        [batch, max_tokens]. In distributed mode, batch size is per rank and
        all ranks must advance their loaders together.
    """
    if config.data.num_workers and config.data.streaming:
        raise ValueError(
            "Streaming requires data.num_workers=0 to prevent duplicated packed tokens"
        )
    batch_size = config.training.batch_size if split == "train" else config.evaluation.batch_size
    packed: TorchIterableDataset[TokenExample]
    if dist.is_initialized() and dist.get_world_size() > 1:
        # Rank 0 tokenizes and packs the stream once, preserving the global order.
        # All ranks advance together to receive their own round-robin batches.
        # Debug from the repo root on macOS:
        # TMPDIR="$PWD/.cache" GLOO_SOCKET_IFNAME=lo0 TORCH_DISTRIBUTED_DEBUG=INFO \
        #   .venv/bin/python -m torch.distributed.run --nnodes=1 --nproc-per-node=2 \
        #   --master-addr=127.0.0.1 --master-port=29501 \
        #   scripts/validate_distributed.py --output artifacts/distributed-debug
        packed = DistributedPackedDataset(config, split, batch_size)
    else:
        packed = _packed_split(config, split)
    # Tokenization workers are handled by Dataset.map; packing runs in one process.
    return DataLoader(
        packed,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=config.hardware.device in ("auto", "cuda") and torch.cuda.is_available(),
        collate_fn=lambda examples: collate_tokens(examples, config.tokenizer.eos_token_id),
    )


def main() -> None:
    """Inspect one or more batches from a configured data split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("configs/end_to_end.toml"),
        help="TOML configuration path (default: %(default)s)",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--batches", type=int, default=1)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")
    config = load_config(args.config)
    loader = build_dataloader(config, args.split)
    for index, batch in enumerate(loader):
        print(
            f"batch={index} split={args.split} "
            f"shape={tuple(batch['input_ids'].shape)} "
            f"valid_targets={int(batch['attention_mask'].sum())}"
        )
        if index + 1 >= args.batches:
            break


if __name__ == "__main__":
    main()
