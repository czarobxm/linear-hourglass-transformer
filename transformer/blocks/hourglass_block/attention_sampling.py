from typing import Union

import torch
from torch import nn

from transformer.blocks.single_layer import TransformerLayer
from transformer.multi_head_attention.attention_mechanism.attn_params import (
    VanillaParams,
    LinearAttnParams,
    PerformerParams,
    CosformerParams,
)


class AttentionSampling(nn.Module):
    def __init__(
        self,
        d_model: int,
        factor: int,
        sampling_type: str,
        post_norm: bool,
        method_params: Union[
            LinearAttnParams,
            VanillaParams,
            PerformerParams,
            CosformerParams,
        ],
        act_fun: nn.Module,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.sampling_type = sampling_type
        self.d_model = d_model
        self.post_norm = post_norm
        self.method_params = method_params
        self.act_fun = act_fun

        self.attention = TransformerLayer(
            d_model=d_model,
            num_heads=8,
            method_params=VanillaParams(),
            act_fun=act_fun,
            apply_rotary_pos_enc=False,
            dropout=0.0,
            apply_linear=False,
            device="cuda" if torch.cuda.is_available() else "mps",
        )

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor, causal: bool = True
    ) -> torch.Tensor:
        return self.attention(query, key_value, causal=causal, inference=False)
