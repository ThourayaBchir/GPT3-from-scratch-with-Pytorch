"""End-to-end correctness check from packed tokens through optimization."""

import torch
from torch.nn import functional as F

from gpt3.config import ModelConfig
from gpt3.data import PackedTokenDataset, collate_tokens
from gpt3.model import GPT3


def test_packed_targets_loss_gradients_and_tiny_batch_overfit() -> None:
    torch.manual_seed(1234)
    config = ModelConfig.model_validate(
        {
            "vocab_size": 13,
            "context_length": 6,
            "n_layers": 2,
            "d_model": 24,
            "n_heads": 3,
            "head_dim": 8,
            "local_window": 3,
            "dropout": 0.0,
        }
    )
    documents = [{"input_ids": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]}]
    examples = list(PackedTokenDataset(documents, config.context_length))
    batch = collate_tokens(examples, pad_token_id=1)
    assert batch["input_ids"].tolist() == [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 1]]
    assert batch["labels"].tolist() == [[3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 1, 2]]
    assert batch["attention_mask"].all()

    model = GPT3(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    first_loss = None
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        assert output.logits.shape == (2, 6, config.vocab_size)
        assert output.loss is not None
        if first_loss is None:
            first_loss = output.loss.item()
            expected = F.cross_entropy(
                output.logits.reshape(-1, config.vocab_size), batch["labels"].reshape(-1)
            )
            torch.testing.assert_close(output.loss, expected)
        output.loss.backward()
        if step == 0:
            # The first backward pass must reach both attention patterns, MLPs,
            # learned positions, and tied token/output embeddings.
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
            assert model.lm_head.weight is model.token_embedding.weight
        optimizer.step()

    with torch.no_grad():
        result = model(**batch)
    assert result.loss is not None
    assert first_loss is not None
    assert result.loss.item() < 0.05
    assert result.loss.item() < first_loss / 10
    assert torch.equal(result.logits.argmax(dim=-1), batch["labels"])
