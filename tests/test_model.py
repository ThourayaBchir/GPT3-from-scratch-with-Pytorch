"""Behavioral checks for causal attention, local windows, and training loss."""

from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from gpt3.config import ModelConfig
from gpt3.model import GPT3, CausalSelfAttention


def small_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 23,
        "context_length": 8,
        "n_layers": 2,
        "d_model": 12,
        "n_heads": 2,
        "head_dim": 4,
        "local_window": 2,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig.model_validate(values)


def test_model_shapes_weight_tying_loss_and_backward() -> None:
    torch.manual_seed(5)
    config = small_config()
    model = GPT3(config)
    assert [block.attention.local for block in model.blocks] == [False, True]
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.blocks[0].attention.qkv_proj.out_features == 3 * 2 * 4
    assert model.blocks[0].mlp.in_proj.out_features == 4 * config.d_model

    ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    labels = torch.tensor([[3, 4, 5, 6], [7, 8, 9, -100]])
    result = model(ids, labels=labels)
    assert result.logits.shape == (2, 4, config.vocab_size)
    assert result.loss is not None
    expected = F.cross_entropy(
        result.logits.float().reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100
    )
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert model.blocks[0].attention.qkv_proj.weight.grad is not None
    assert model.blocks[1].mlp.in_proj.weight.grad is not None
    assert torch.isfinite(model.blocks[1].mlp.in_proj.weight.grad).all()


def test_future_tokens_cannot_change_earlier_logits() -> None:
    torch.manual_seed(7)
    model = GPT3(small_config()).eval()
    original = torch.tensor([[2, 3, 4, 5, 6, 7]])
    changed = torch.tensor([[2, 3, 4, 5, 6, 22]])
    with torch.no_grad():
        first = model(original).logits
        second = model(changed).logits
    torch.testing.assert_close(first[:, :5], second[:, :5], rtol=0, atol=1e-6)


def test_local_attention_excludes_tokens_outside_window() -> None:
    torch.manual_seed(9)
    config = small_config(local_window=2)
    local = CausalSelfAttention(config, local=True).eval()
    dense = CausalSelfAttention(config, local=False).eval()
    dense.load_state_dict(local.state_dict())
    x = torch.randn(1, 7, config.d_model)
    changed = x.clone()
    changed[:, 0] += 100
    with torch.no_grad():
        local_before, local_after = local(x), local(changed)
        dense_before, dense_after = dense(x), dense(changed)
    torch.testing.assert_close(local_before[:, 2:], local_after[:, 2:], rtol=0, atol=1e-6)
    assert not torch.allclose(dense_before[:, 2], dense_after[:, 2])


def test_chunked_local_attention_matches_full_reference_mask() -> None:
    torch.manual_seed(10)
    config = small_config(local_window=3)
    attention = CausalSelfAttention(config, local=True).eval()
    x = torch.randn(2, 7, config.d_model)
    padding = torch.tensor([[True] * 7, [True, True, True, False, False, False, False]])
    with torch.no_grad():
        actual = attention(x, padding)
        qkv = attention.qkv_proj(x)
        q, k, v = (
            part.view(2, 7, config.n_heads, config.head_dim).transpose(1, 2)
            for part in qkv.split(config.n_heads * config.head_dim, dim=-1)
        )
        positions = torch.arange(7)
        distance = positions[:, None] - positions[None, :]
        local_mask = (distance >= 0) & (distance < config.local_window)
        allowed = local_mask[None, None] & (padding[:, None, None, :] | ~padding[:, None, :, None])
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        reference = reference.transpose(1, 2).reshape(2, 7, config.n_heads * config.head_dim)
        reference = attention.out_proj(reference)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


def test_right_padding_stays_finite_and_does_not_change_valid_logits() -> None:
    torch.manual_seed(11)
    model = GPT3(small_config()).eval()
    short = torch.tensor([[2, 3]])
    padded = torch.tensor([[2, 3, 1, 1, 1, 1, 1, 1]])
    mask = torch.tensor([[True, True, False, False, False, False, False, False]])
    labels = torch.tensor([[3, 4, -100, -100, -100, -100, -100, -100]])
    with torch.no_grad():
        short_logits = model(short).logits
        padded_output = model(padded, labels=labels, attention_mask=mask)
    assert torch.isfinite(padded_output.logits).all()
    assert padded_output.loss is not None and torch.isfinite(padded_output.loss)
    torch.testing.assert_close(padded_output.logits[:, :2], short_logits, rtol=1e-5, atol=1e-5)


def test_dense_only_and_untied_projection() -> None:
    config = small_config(attention_pattern="dense", tie_embeddings=False)
    model = GPT3(config)
    assert all(not block.attention.local for block in model.blocks)
    assert model.lm_head.weight is not model.token_embedding.weight
    assert model(torch.tensor([[2]])).logits.shape == (1, 1, config.vocab_size)


@pytest.mark.parametrize(
    ("ids", "labels", "mask", "message"),
    [
        (torch.tensor([[1] * 9]), None, None, "input length"),
        (torch.tensor([[1, 2]]), torch.tensor([[2]]), None, "labels"),
        (torch.tensor([[1, 2]]), None, torch.tensor([[False, True]]), "first token"),
        (torch.tensor([[1, 2, 3]]), None, torch.tensor([[True, False, True]]), "right padding"),
        (torch.tensor([[1, 2]]), torch.tensor([[-100, -100]]), None, "target token"),
    ],
)
def test_invalid_inputs_fail_clearly(
    ids: torch.Tensor, labels: torch.Tensor | None, mask: torch.Tensor | None, message: str
) -> None:
    model = GPT3(small_config())
    with (
        patch.object(
            model.token_embedding, "forward", side_effect=AssertionError("input reached embeddings")
        ),
        pytest.raises(ValueError, match=message),
    ):
        model(ids, labels=labels, attention_mask=mask)
