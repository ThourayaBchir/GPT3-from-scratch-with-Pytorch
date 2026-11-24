"""Run training; this command may download configured Hub data and tokenizer."""

import argparse

import torch

from gpt3.config import load_config
from gpt3.data import build_dataloader
from gpt3.model import GPT3
from gpt3.training.distributed import distributed_session, rank
from gpt3.training.engine import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_config(args.config)
    with distributed_session(config):
        torch.manual_seed(config.training.seed)
        result = train(
            config,
            GPT3(config.model),
            build_dataloader(config, "train"),
            build_dataloader(config, "validation"),
        )
        if rank() == 0:
            print(f"Completed {result.steps} updates; results: {result.run_dir}")


if __name__ == "__main__":
    main()
