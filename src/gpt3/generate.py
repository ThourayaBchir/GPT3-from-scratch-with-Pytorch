"""Generate from a local checkpoint; loading the configured tokenizer may download files."""

import argparse
import json
from dataclasses import asdict

from gpt3.config import load_config
from gpt3.data import load_tokenizer
from gpt3.evaluation.generation import generate
from gpt3.training.checkpoint import load_model
from gpt3.training.engine import autocast_context, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    config = load_config(args.config)
    device = resolve_device(config)
    model, _ = load_model(args.checkpoint, device, expected_config=config)
    tokenizer = load_tokenizer(config)
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else config.evaluation.max_new_tokens
    )
    with autocast_context(device, config.hardware.precision):
        result = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens,
            config.tokenizer.eos_token_id,
            options=config.generation,
        )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
