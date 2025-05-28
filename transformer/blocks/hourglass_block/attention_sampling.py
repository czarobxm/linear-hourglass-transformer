from functools import partial

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention


class AttentionSampling(nn.Module):
    def __init__(
        self,
        d_model: int,
        factor: int,
        sampling_type: str,
        use_full_attention: bool,
        post_norm: bool,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.sampling_type = sampling_type
        self.d_model = d_model
        self.use_full_attention = use_full_attention
        self.post_norm = post_norm

        if self.sampling_type == "downsampling" and not self.use_full_attention:
            self.attention = self.attention_downsampling
        elif self.sampling_type == "upsampling" and not self.use_full_attention:
            self.attention = self.attention_upsampling
        elif self.use_full_attention:
            self.attention = partial(scaled_dot_product_attention, is_causal=True)

        self.norm1 = nn.LayerNorm(d_model)

    def attention_downsampling(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        batch_size, seq_len, d_model = key.size()
        key = key.view(
            batch_size,
            seq_len // self.factor,
            self.factor,
            d_model,
        )
        weights = torch.einsum("bsd,bsfd->bsf", query, key).flatten(1)
        attn_output = torch.einsum("bs,bsd->bsd", weights, value)
        attn_output = attn_output.view(
            batch_size,
            seq_len // self.factor,
            self.factor,
            d_model,
        )
        attn_output = attn_output.sum(dim=2)
        return attn_output

    def attention_upsampling(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        batch_size, seq_len, d_model = query.size()
        query = query.view(
            batch_size,
            seq_len // self.factor,
            self.factor,
            d_model,
        )
        weights = torch.einsum("bsfd,bsd->bsf", query, key)
        attn_output = torch.einsum("bsf,bsd->bsfd", weights, value)
        return attn_output.view(batch_size, seq_len, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        print(self.attention)
        # Attention
        if self.post_norm:
            output = self.norm1(query + self.attention(query, key, value))
        else:
            output = query + self.attention(
                self.norm1(query), self.norm1(key), self.norm1(value)
            )

        return output
