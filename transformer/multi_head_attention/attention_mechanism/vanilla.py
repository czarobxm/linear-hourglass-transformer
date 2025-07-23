from typing import Tuple, Optional
import math
import torch

from transformer.multi_head_attention.attention_mechanism.base import (
    BaseAttentionMechanism,
)


class VanillaAttention(BaseAttentionMechanism):
    """
    Vanilla attention mechanism.
    """

    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__(d_model=d_model, num_heads=num_heads)
        self.head_dim = d_model // num_heads
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

        self.cache_pos = 0
        self.max_seq_len = 16384

        if self.head_dim * num_heads != self.d_model:
            raise ValueError(
                f"embed_dim {d_model} not divisible by num_heads {num_heads}"
            )

    def multihead_reshape(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-head reshaping - Split heads from size [B, L, D] to size: [B, Nh, L, Dh]
        """
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return query, key, value

    def undo_multihead_reshape(self, attn_output: torch.Tensor) -> torch.Tensor:
        """
        Undo multi-head reshaping: Split heads from size [B, Nh, L, Dh] to size [B, L, D]
        """
        batch_size = attn_output.size(0)
        output = attn_output.contiguous().view(batch_size, -1, self.d_model)
        return output

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))

        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if causal:
            if S > L:
                causal_mask = torch.triu(
                    torch.ones(L, L, dtype=torch.bool, device=query.device), diagonal=1
                )
                block_size = S // L
                causal_mask = causal_mask.repeat_interleave(block_size, dim=1)
            elif L > S:
                causal_mask = torch.triu(
                    torch.ones(S, S, dtype=torch.bool, device=query.device), diagonal=1
                )
                block_size = L // S
                causal_mask = causal_mask.repeat_interleave(block_size, dim=0)
            else:
                causal_mask = torch.triu(
                    torch.ones(S, S, dtype=torch.bool, device=query.device), diagonal=1
                )

            attn_bias.masked_fill_(causal_mask, float("-inf"))

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        output = attn_weight @ value
        return output.transpose(1, 2)

    def clear_cache(self) -> None:
        """Clear the KV cache."""
        self.k_cache = None
        self.v_cache = None

    def inference(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficient inference with pre-allocated KV caching.

        Args:
            query (torch.Tensor): Query tensor for the current token, size [B, Nh, 1, Dh]
            key (torch.Tensor): Key tensor for the current token, size [B, Nh, 1, Dh]
            value (torch.Tensor): Value tensor for the current token, size [B, Nh, 1, Dh]
        """
        with torch.no_grad():
            batch_size, _, seq_len, _ = key.shape
            assert (
                seq_len == 1
            ), "KV caching is implemented for single-token generation only."

            # Initialize cache on the first inference pass
            if self.k_cache is None:
                self.k_cache = torch.zeros(
                    (batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                    dtype=key.dtype,
                    device=key.device,
                )
                self.v_cache = torch.zeros(
                    (batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                    dtype=value.dtype,
                    device=value.device,
                )

            if self.cache_pos >= self.max_seq_len:
                raise ValueError(
                    "KV Cache is full. Increase max_seq_len or clear the cache."
                )

            # Update cache in-place
            self.k_cache[:, :, self.cache_pos : self.cache_pos + 1, :] = key
            self.v_cache[:, :, self.cache_pos : self.cache_pos + 1, :] = value
            self.cache_pos += 1

            # Get the active part of the cache
            keys = self.k_cache[:, :, : self.cache_pos, :]
            values = self.v_cache[:, :, : self.cache_pos, :]

            # Apply scaled dot product attention
            output = self.scaled_dot_product_attention(query, keys, values, causal=False)

            # The original code had a transpose here, which is now handled by undo_multihead_reshape
            # so we return the standard [B, Nh, L, Dh] output.
            return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
        inference: bool = False,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Vanilla attention mechanism.

        Args:
            query (torch.Tensor): Query tensor of size [B, Nh, L, Dh]
            key (torch.Tensor): Key tensor of size [B, Nh, L, Dh]
            value (torch.Tensor): Value tensor of size [B, Nh, L, Dh]
            causal (bool): Whether to apply causal mask

        Returns:
            torch.Tensor: Attention mechanism output of size [B, L, D]

        Where:
            B - batch size
            L - sequence length/attention dimension
            D - embedding dimension
        """
        # Scaled dot product attention
        if inference:
            # Inference mode uses KV caching
            with torch.no_grad():
                return self.inference(
                    query.detach(), key.detach(), value.detach()
                ).detach()
        output = self.scaled_dot_product_attention(query, key, value, causal=causal)

        return output
