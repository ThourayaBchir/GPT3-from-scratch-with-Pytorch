"""Decoder-only GPT-3-style Transformer built from PyTorch layers."""

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from gpt3.config import ModelConfig


@dataclass(frozen=True)
class GPT3Output:
    """Vocabulary logits and optional next-token cross-entropy loss."""

    logits: Tensor | None
    loss: Tensor | None = None


class CausalSelfAttention(nn.Module):
    """Causal multi-head attention with dense or bounded local key ranges."""

    def __init__(self, config: ModelConfig, *, local: bool) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.inner_dim = config.n_heads * config.head_dim
        self.local = local
        self.local_window = config.local_window
        self.dropout = config.dropout
        self.qkv_proj = nn.Linear(config.d_model, 3 * self.inner_dim)  # One nn.Linear computes all three projections and returns them concatenated along the last dimension in [Q | K | V] order. Each has its own learned weights and bias within that layer.
        self.out_proj = nn.Linear(self.inner_dim, config.d_model)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Apply causal self-attention to normalized hidden states.

        The combined QKV projection has shape
        [batch_size, sequence_length, 3 * n_heads * head_dim].

        Args:
            x: Hidden states of shape
                [batch_size, sequence_length, d_model], one vector per
                token position.
            attention_mask: Optional boolean tensor of shape
                [batch_size, sequence_length]. True marks a real token
                and False marks right padding. None treats all positions
                as real tokens.

        Returns:
            Tensor of shape [batch_size, sequence_length, d_model],
            ready for residual addition.
        """
        batch_size, length, _ = x.shape    #.  [batch_size, sequence_length, d_model]
        qkv = self.qkv_proj(x)             #  [batch_size, sequence_length, 3 * (config.n_heads * config.head_dim)]
        q, k, v = (
            part.view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2)
            for part in qkv.split(self.inner_dim, dim=-1)
        )
        # part              [batch, tokens, heads * head_dim]   [2, 256, 256]
        # .view(...)        [batch, tokens, heads, head_dim]    [2, 256, 4, 64]
        # .transpose(1, 2)  [batch, heads, tokens, head_dim]    [2, 4, 256, 64]

        dropout_p = self.dropout if self.training else 0.0
        if self.local:
            attended = self._local_attention(q, k, v, attention_mask, dropout_p)
        else:
            attended = self._full_attention(q, k, v, attention_mask, dropout_p)

        attended = attended.transpose(1, 2).contiguous().view(batch_size, length, self.inner_dim)
        # This line joins the attention heads back into one vector per token so out_proj can mix the heads’ outputs for that token.
        # With the tiny preset, attention returns [2, 4, 256, 64] — 2 sequences, 4 heads, 256 tokens, and 64 features per head.
        # attended          [2, 4, 256, 64]   [batch, heads, tokens, head_dim]
        # .transpose(1, 2)  [2, 256, 4, 64]   [batch, tokens, heads, head_dim]
        # .contiguous()     [2, 256, 4, 64]   same shape, reordered in memory
        # .view(...)        [2, 256, 256]     [batch, tokens, heads * head_dim]


        # W^O from the original Transformer (Attention Is All You Need, 2017, §3.2.2).
        # https://arxiv.org/html/1706.03762#S3.SS2.SSS2
        # Mix the concatenated head outputs within each token and project to d_model
        # so the block can add the result to the residual stream.
        # [batch, tokens, d_model] - tokens is seq_length
        return cast(Tensor, self.out_proj(attended))

    def _local_attention(
        self, q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor | None, dropout_p: float
    ) -> Tensor:
        """Attend within the local window.

        Args:
            q, k, v: Queries, keys, and values, each of shape
                [batch_size, n_heads, sequence_length, head_dim].
            attention_mask: Optional token mask as described in forward.
            dropout_p: Attention dropout probability, zero in eval mode.

        Returns:
            Tensor with the same shape as q.
        """
        length = q.size(-2)
        window = self.local_window
        positions = torch.arange(length, device=q.device)
        chunks = []
        # Each query chunk needs at most 2 * window - 1 keys.
        for start in range(0, length, window):
            end = min(start + window, length)
            queries = slice(start, end)
            keys = slice(max(0, start - window + 1), end)

            distance = positions[queries, None] - positions[None, keys]
            allowed = (distance >= 0) & (distance < window)
            if attention_mask is not None:
                allowed = self._mask_padded_keys(
                    allowed, attention_mask[:, queries], attention_mask[:, keys]
                )

            chunk = F.scaled_dot_product_attention(
                q[:, :, queries, :],
                k[:, :, keys, :],
                v[:, :, keys, :],
                attn_mask=allowed,
                dropout_p=dropout_p,
            )
            chunks.append(chunk)
        return torch.cat(chunks, dim=-2)

    def _full_attention(
        self, q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor | None, dropout_p: float
    ) -> Tensor:
        """Attend over the full causal prefix.

        Args:
            q, k, v: Queries, keys, and values, each of shape
                [batch_size, n_heads, sequence_length, head_dim].
            attention_mask: Optional token mask as described in forward.
            dropout_p: Attention dropout probability, zero in eval mode.

        Returns:
            Tensor with the same shape as q.
        """
        if attention_mask is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)

        positions = torch.arange(q.size(-2), device=q.device)
        causal = positions[:, None] >= positions[None, :]
        allowed = self._mask_padded_keys(causal, attention_mask, attention_mask)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=dropout_p)

    @staticmethod
    def _mask_padded_keys(allowed: Tensor, query_valid: Tensor, key_valid: Tensor) -> Tensor:
        # Real queries exclude padded keys. Padded queries keep their causal range,
        # including themselves, so they always have a key. Their targets are ignored.
        return allowed[None, None, :, :] & (
            key_valid[:, None, None, :] | ~query_valid[:, None, :, None]
        )


class FeedForward(nn.Module):
    """GPT-2-style two-layer MLP with GELU activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.in_proj = nn.Linear(config.d_model, config.d_ff)
        self.activation = nn.GELU(approximate="tanh")
        self.out_proj = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.dropout(self.out_proj(self.activation(self.in_proj(x)))))


class TransformerBlock(nn.Module):
    """Pre-normalized attention and MLP, each with a residual connection."""

    def __init__(self, config: ModelConfig, *, local: bool) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attention = CausalSelfAttention(config, local=local)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.mlp_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = x + self.residual_dropout(self.attention(self.attention_norm(x), attention_mask))
        return cast(Tensor, x + self.mlp(self.mlp_norm(x)))


class GPT3(nn.Module):
    """Autoregressive Transformer whose output predicts the next token at each position."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.activation_checkpointing = False
        self.loss_chunk_size = 0
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        # GPT-3 kept GPT-2's architecture and alternated dense and local attention.
        # https://arxiv.org/html/2005.14165v4#S2.SS1
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config,
                local=config.attention_pattern == "alternating" and layer_index % 2 == 1,
            )
            for layer_index in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Residual projections use the GPT-2 scaled initialization.
        residual_std = config.initializer_std / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            assert isinstance(block, TransformerBlock)
            nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.out_proj.weight, mean=0.0, std=residual_std)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        return_logits: bool = True,
    ) -> GPT3Output:
        """Compute next-token logits and loss.

        Args:
            input_ids: Nonempty [batch, time] torch.long token IDs.
                The sequence length must fit the context window.
            labels: Optional next-token targets with the input shape
                and dtype. Use -100 for ignored targets, including
                right padding.
            attention_mask: Optional mask with the input shape. True
                marks real tokens and False marks right padding.
                Each row must start with a real token.
            return_logits: Whether to include vocabulary logits.

        Returns:
            A GPT3Output with logits of shape [batch, time, vocab_size]
            when requested and mean loss over valid targets when labels
            are supplied.

        Raises:
            ValueError: If inputs violate the shape, dtype, length,
                or padding requirements, or all labels are ignored.
        """
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a [batch, time] torch.long tensor")
        batch_size, length = input_ids.shape
        if batch_size < 1 or not 1 <= length <= self.config.context_length:
            raise ValueError(f"input length must be between 1 and {self.config.context_length}")
        if labels is not None and (labels.shape != input_ids.shape or labels.dtype != torch.long):
            raise ValueError("labels must match input_ids shape and have torch.long dtype")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape")
            attention_mask = attention_mask.to(dtype=torch.bool)
            if not bool(attention_mask[:, 0].all()):
                raise ValueError("attention_mask must mark the first token of each row as valid")
            # Right padding preserves the packed sequence's position indices.
            if bool((~attention_mask[:, :-1] & attention_mask[:, 1:]).any()):
                raise ValueError("attention_mask must represent right padding")

        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None
        if labels is not None and not bool((labels != -100).any()):
            raise ValueError("labels must contain at least one target token")

        positions = torch.arange(length, device=input_ids.device)
        x = self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden) if return_logits else None
        loss = None
        if labels is not None:
            if self.loss_chunk_size > 0 and logits is None:
                loss = self._chunked_loss(hidden, labels)
            else:
                logits_for_loss = logits if logits is not None else self.lm_head(hidden)
                loss = F.cross_entropy(
                    logits_for_loss.float().reshape(-1, self.config.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
        return GPT3Output(logits=logits, loss=loss)

    def _chunked_loss(self, hidden: Tensor, labels: Tensor) -> Tensor:
        flat_hidden = hidden.reshape(-1, self.config.d_model)
        flat_labels = labels.reshape(-1)
        recompute = self.training and torch.is_grad_enabled()

        def chunk_loss(states: Tensor, targets: Tensor) -> Tensor:
            logits = self.lm_head(states).float()
            return F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")

        # Sum in FP32 and divide once so each valid target has the same weight.
        total_loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, flat_labels.numel(), self.loss_chunk_size):
            end = start + self.loss_chunk_size
            states = flat_hidden[start:end]
            targets = flat_labels[start:end]
            if recompute:
                loss = checkpoint(chunk_loss, states, targets, use_reentrant=False)
            else:
                loss = chunk_loss(states, targets)
            total_loss = total_loss + loss
        return total_loss / (flat_labels != -100).sum()
