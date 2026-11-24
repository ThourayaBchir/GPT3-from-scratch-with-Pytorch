"""Full-sequence generation with sampling and explicit stopping conditions."""

from dataclasses import dataclass
from typing import Literal

import torch
from tokenizers import Tokenizer

from gpt3.config import GenerationConfig
from gpt3.model import GPT3


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    finish_reason: Literal["length", "eos", "stop_token", "stop_string"]


def sampling_probabilities(logits: torch.Tensor, options: GenerationConfig) -> torch.Tensor:
    """Apply temperature, exact top-k, then nucleus filtering to one vocabulary vector."""
    if logits.ndim != 1 or not logits.numel():
        raise ValueError("Expected a nonempty one-dimensional logits vector")
    scores = logits.detach().float().cpu()
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Generation logits must be finite")
    if options.temperature == 0:
        probabilities = torch.zeros_like(scores)
        probabilities[scores.argmax()] = 1.0
        return probabilities
    # Use float64 for scaling and filtering so small config values stay positive.
    scores = scores.double()
    scores = (scores - scores.max()) / options.temperature
    if options.top_k is not None:
        indices = scores.topk(min(options.top_k, scores.numel())).indices
        filtered = torch.full_like(scores, -torch.inf)
        filtered[indices] = scores[indices]
        scores = filtered
    if options.top_p < 1.0:
        sorted_scores, indices = scores.sort(descending=True)
        probabilities = sorted_scores.softmax(dim=-1)
        # Keep the smallest prefix reaching p, including the crossing token.
        remove = probabilities.cumsum(dim=-1) - probabilities >= options.top_p
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        scores = torch.empty_like(scores).scatter(0, indices, sorted_scores)
    return scores.softmax(dim=-1).float()


@torch.no_grad()
def generate(
    model: GPT3,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_token_id: int,
    *,
    options: GenerationConfig | None = None,
) -> GenerationResult:
    """Generate one continuation; local CPU sampling RNG leaves global RNG untouched."""
    options = options or GenerationConfig()
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if (
        any(token >= model.config.vocab_size for token in [eos_token_id, *options.stop_token_ids])
        or eos_token_id < 0
    ):
        raise ValueError("Stop token IDs must be within the model vocabulary")
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not ids:
        raise ValueError("Sample prompt must encode to at least one token")
    if len(ids) > model.config.context_length:
        raise ValueError("Sample prompt exceeds context length")
    generator = torch.Generator(device="cpu")
    if options.seed is None:
        generator.seed()
    else:
        generator.manual_seed(options.seed)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    continuation: list[int] = []
    text = ""
    reason: Literal["length", "eos", "stop_token", "stop_string"] = "length"
    try:
        for _ in range(max_new_tokens):
            context = torch.tensor(
                [ids[-model.config.context_length :]], dtype=torch.long, device=device
            )
            logits = model(context).logits
            assert logits is not None
            probabilities = sampling_probabilities(logits[0, -1], options)
            next_token = int(torch.multinomial(probabilities, 1, generator=generator))
            ids.append(next_token)
            continuation.append(next_token)
            if next_token == eos_token_id or next_token in options.stop_token_ids:
                reason = "eos" if next_token == eos_token_id else "stop_token"
                break
            text = tokenizer.decode(continuation, skip_special_tokens=True)
            matches = [text.find(stop) for stop in options.stop_strings if stop in text]
            if matches:
                text = text[: min(matches)]
                reason = "stop_string"
                break
        return GenerationResult(text=text, token_ids=continuation, finish_reason=reason)
    finally:
        model.train(was_training)


def generate_sample(
    model: GPT3,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_token_id: int,
    *,
    options: GenerationConfig | None = None,
) -> str:
    """Convenience wrapper returning completion text without stop markers."""
    return generate(model, tokenizer, prompt, max_new_tokens, eos_token_id, options=options).text
