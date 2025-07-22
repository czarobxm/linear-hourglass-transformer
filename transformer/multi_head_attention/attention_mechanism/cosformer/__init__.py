# pylint: disable=attribute-defined-outside-init,no-member
"""
Cosformer attention implementation based on the official implementation:
https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py
"""

from typing import List, Optional
import math

import torch
from torch.nn import functional as F
from torch import nn
import numpy as np

from transformer.multi_head_attention.attention_mechanism.base import (
    BaseAttentionMechanism,
)
from transformer.multi_head_attention.attention_mechanism.cosformer.attention_noncausal import (
    attention_noncausal,
)
from transformer.multi_head_attention.attention_mechanism.cosformer.attention_causal import (
    attention_causal,
)
from transformer.multi_head_attention.attention_mechanism.cosformer.multihead_reshape import (
    multihead_reshape,
)


def get_index(seq_len: int, start_pos: int, m: int) -> nn.Parameter:
    """Create array of indices for the cosformer attention mechanism."""
    index = torch.arange(start_pos, start_pos + seq_len + 1).reshape(1, -1, 1)
    weights = np.pi / 2 * index / m
    weights[index > m] = np.pi / 2
    return nn.Parameter(index, requires_grad=False)


def query_key_feature_map(
    x: torch.Tensor, weight_index: torch.Tensor, seq_len: int, m: int
) -> torch.Tensor:
    """
    Compute the query and key feature map for the cosformer attention mechanism.

    Args:
        x (torch.Tensor): Input tensor of shape [B, Nh, L, Dh]
        weight_index (torch.Tensor): Weight index tensor
        seq_len (int): Sequence length
        m (int): Maximum of source and target sequence lengths

    Returns:
        torch.Tensor: Feature map of shape [B, Nh, L, 2 * Dh]
    """
    sin_part = x * torch.sin(weight_index[:, :seq_len, :] / m)
    cos_part = x * torch.cos(weight_index[:, :seq_len, :] / m)
    return torch.cat([sin_part, cos_part], dim=-1)


class Cosformer(BaseAttentionMechanism):
    """
    Cosformer linear attention mechanism - https://arxiv.org/abs/2202.08791.

    This class provides efficient non-causal attention for deep learning models
    and supports multiple hardware platforms:
    - MPS: Implementation based on the official repository
    - CPU and CUDA: Implementation based on causal_dot_product function from FastTransformers
    """

    def __init__(
        self, d_model: int, num_heads: int, m: int, eps: float = 1e-6, device: str = "cpu"
    ) -> None:
        """Creates instance and buffers for the cosformer attention mechanism."""
        super().__init__(d_model=d_model, num_heads=num_heads)
        self.eps = eps
        self.device = device
        self.m = m
        self.register_buffer(
            "kv",
            torch.zeros(
                1, self.num_heads, 2 * self.dim_head, self.dim_head, device=device
            ),
        )
        self.register_buffer("k_", torch.zeros(1, self.num_heads, device=device))
        self.to(device)

    def reset_cache(self) -> None:
        """Reset the internal states of the attention mechanism."""
        self.kv.zero_()  # pylint: disable=no-member
        self.k_.zero_()  # pylint: disable=no-member

    def inference(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        kv = torch.einsum("bnld,bnlm->bnldm", key, value)
        kv_cum = torch.sum(kv, dim=2)

        # Update internal states
        self.kv = (  # pylint: disable=attribute-defined-outside-init
            self.kv + kv_cum  # pylint: disable=no-member
        )
        # Calculate denominator: [B, L, 2 * Dh], [B, 2 * Dh] -> [B, L]
        denom = torch.clamp_min(torch.einsum("bnlm,bnlm->bnl", query, key), self.eps)
        denom = 1 / torch.sum(denom, dim=2)

        self.k_ = self.k_ + denom

        # Compute attention output: [B, L, 2 * Dh], [B, 2 * Dh, Dh], [B, L] -> [B, L, Dh]
        return torch.einsum("bnld,bndm,bn->nlm", query, self.kv, self.k_).unsqueeze(0)

    def multihead_reshape(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> List[torch.Tensor]:
        """Reshape the input tensors for the multi-head attention mechanism."""
        return (
            multihead_reshape(query, self.num_heads, self.dim_head),
            multihead_reshape(key, self.num_heads, self.dim_head),
            multihead_reshape(value, self.num_heads, self.dim_head),
        )

    def undo_multihead_reshape(self, attn_output: torch.Tensor) -> torch.Tensor:
        """
        Undo the reshape operation for multi-head attention output.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, Nh, Dh]

        Returns:
            torch.Tensor: Reshaped tensor of shape [B, L, D]

        Where:
            B: Batch size
            Nh: Number of heads
            L: Sequence length
            Dh: Dimension of each head
            D: Model dimension (D = Nh * Dh)
        """
        batch_size, seq_len, num_heads, dim_head = attn_output.size()
        return attn_output.contiguous().view(batch_size, seq_len, num_heads * dim_head)

    def feature_map(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        tgt_len: int,
        src_len: int,
        start_pos: int,
    ) -> List[torch.Tensor]:
        """Feature map for the cosformer attention mechanism."""
        seq_len = max(src_len, tgt_len)
        weight_index = get_index(seq_len, start_pos, self.m).to(query)
        q_ = query_key_feature_map(
            query, weight_index, tgt_len, self.m
        )  # [B * Nh, L, 2 * h]
        k_ = query_key_feature_map(
            key, weight_index, src_len, self.m
        )  # [B * Nh, S, 2 * Dh]
        return q_, k_

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
        inference: bool = False,
        start_pos: int = 1,
    ) -> torch.Tensor:
        """
        Cosformer attention mechanism - https://arxiv.org/abs/2202.08791

        Args:
            query (torch.Tensor): Query tensor of shape [B, Nh, L, Dh]
            key (torch.Tensor): Key tensor of shape [B, Nh, L, Dh]
            value (torch.Tensor): Value tensor of shape [B, Nh, L, Dh]
            causal (bool): Whether to use causal attention
            inference (bool): Whether to use inference mode
            start_pos (int): Starting position for the feature map

        Returns:
            torch.Tensor: Attention mechanism output of shape [B, L, D]
        """
        tgt_len, src_len = query.size(2), key.size(2)
        q_, k_ = self.feature_map(query, key, tgt_len, src_len, start_pos)

        if inference:
            return self.inference(q_, k_, value)

        if causal:
            out = attention_causal(q_, k_, value, self.eps, self.kv, self.k_, self.device)
        else:
            out = attention_noncausal(q_, k_, value, self.eps)

        return out

    def left_product(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        causal: bool = True,
        start_pos: int = 1,
    ):
        """
        Left-product Cosformer attention mechanism (of n^2 complexity) - https://arxiv.org/abs/2202.08791

        Args:
            query (torch.Tensor): Query tensor of shape [B, Nh, L, Dh]
            key (torch.Tensor): Key tensor of shape [B, Nh, L, Dh]
            value (torch.Tensor): Value tensor of shape [B, Nh, L, Dh]
            causal (bool): Whether to use causal attention
            inference (bool): Whether to use inference mode
            start_pos (int): Starting position for the feature map

        Returns:
            torch.Tensor: Attention mechanism output of shape [B, L, D]
        """
        tgt_len, src_len = query.size(2), key.size(2)

        q_, k_ = self.feature_map(query, key, tgt_len, src_len, start_pos)
        attn_weight = q_ @ k_.transpose(-2, -1)

        attn_bias = torch.zeros(tgt_len, src_len, dtype=query.dtype, device=query.device)
        if causal:
            causal_mask = torch.triu(
                torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device),
                diagonal=1,
            )
            attn_bias.masked_fill_(causal_mask, float("-inf"))
            attn_weight = attn_weight.masked_fill(attn_bias == float("-inf"), 0)

        denom = torch.clamp_min(attn_weight.sum(dim=-1, keepdim=True), self.eps)
        attn_weight = attn_weight / denom

        attn_output = attn_weight @ value
        return attn_output.transpose(1, 2)
