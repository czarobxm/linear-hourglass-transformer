import math

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


from models.base import BaseModel
from transformer.blocks.transformer_block import Block

from transformer.multi_head_attention.attention_mechanism.attn_params import (
    CosformerParams,
)


# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def cast_tuple(val, depth=1):
    return val if isinstance(val, tuple) else ((val,) * depth)


# factory


def get_hourglass_transformer(
    dim,
    *,
    depth,
    shorten_factor,
    attn_resampling,
    updown_sample_type,
    device,
    heads,
    **kwargs,
):
    assert isinstance(depth, int) or (
        isinstance(depth, tuple) and len(depth) == 3
    ), "depth must be either an integer or a tuple of 3, indicating (pre_transformer_depth, <nested-hour-glass-config>, post_transformer_depth)"
    assert not (
        isinstance(depth, int) and shorten_factor
    ), "there does not need to be a shortening factor when only a single transformer block is indicated (depth of one integer value)"

    if isinstance(depth, int):
        return Block(
            n_layers=depth,
            d_model=dim,
            num_heads=heads,
            method_params=CosformerParams,
            apply_rotary_pos_enc=False,
            dropout=0.0,
            act_fun=torch.nn.ReLU(),
            post_norm=False,
            device=device,
        )

    return HourglassTransformer(
        dim=dim,
        depth=depth,
        shorten_factor=shorten_factor,
        attn_resampling=attn_resampling,
        updown_sample_type=updown_sample_type,
        device=device,
        **kwargs,
    )


# up and down sample classes


class NaiveDownsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return reduce(x, "b (n s) d -> b n d", "mean", s=self.shorten_factor)


class NaiveUpsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return repeat(x, "b n d -> b (n s) d", s=self.shorten_factor)


class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, "b (n s) d -> b n (s d)", s=self.shorten_factor)
        return self.proj(x)


class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "b n (s d) -> b (n s) d", s=self.shorten_factor)


# classes


class HourglassTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        shorten_factor=2,
        attn_resampling=True,
        updown_sample_type="naive",
        heads=8,
        dim_head=64,
        causal=True,
        norm_out=False,
        device: str = "cpu",
    ):
        super().__init__()
        assert len(depth) == 3, "depth should be a tuple of length 3"
        assert updown_sample_type in {
            "naive",
            "linear",
        }, "downsample / upsample type must be either naive (average pool and repeat) or linear (linear projection and reshape)"

        pre_layers_depth, valley_depth, post_layers_depth = depth

        if isinstance(shorten_factor, (tuple, list)):
            shorten_factor, *rest_shorten_factor = shorten_factor
        elif isinstance(valley_depth, int):
            shorten_factor, rest_shorten_factor = shorten_factor, None
        else:
            shorten_factor, rest_shorten_factor = shorten_factor, shorten_factor

        transformer_kwargs = dict(dim=dim, heads=heads, dim_head=dim_head)

        self.causal = causal
        self.shorten_factor = shorten_factor

        if updown_sample_type == "naive":
            self.downsample = NaiveDownsample(shorten_factor).to(device)
            self.upsample = NaiveUpsample(shorten_factor).to(device)
        elif updown_sample_type == "linear":
            self.downsample = LinearDownsample(dim, shorten_factor).to(device)
            self.upsample = LinearUpsample(dim, shorten_factor).to(device)
        else:
            raise ValueError(
                f"unknown updown_sample_type keyword value - must be either naive or linear for now"
            )

        self.valley_transformer = get_hourglass_transformer(
            shorten_factor=rest_shorten_factor,
            depth=valley_depth,
            attn_resampling=attn_resampling,
            updown_sample_type=updown_sample_type,
            causal=causal,
            device=device,
            **transformer_kwargs,
        )

        self.pre_transformer = Block(
            n_layers=pre_layers_depth,
            d_model=dim,
            num_heads=heads,
            method_params=CosformerParams,
            apply_rotary_pos_enc=False,
            dropout=0.0,
            act_fun=torch.nn.ReLU(),
            post_norm=False,
            device=device,
        )

        self.post_transformer = Block(
            n_layers=post_layers_depth,
            d_model=dim,
            num_heads=heads,
            method_params=CosformerParams,
            apply_rotary_pos_enc=False,
            dropout=0.0,
            act_fun=torch.nn.ReLU(),
            post_norm=False,
            device=device,
        )
        self.norm_out = nn.LayerNorm(dim, device=device) if norm_out else nn.Identity()

    def forward(self, x):
        # b : batch, n : sequence length, d : feature dimension, s : shortening factor

        s, b, n = self.shorten_factor, *x.shape[:2]

        # top half of hourglass, pre-transformer layers

        x = self.pre_transformer(x)

        # pad to multiple of shortening factor, in preparation for pooling

        x = pad_to_multiple(x, s, dim=-2)

        # save the residual, and for "attention resampling" at downsample and upsample

        x_residual = x.clone()

        # if autoregressive, do the shift by shortening factor minus one

        if self.causal:
            shift = s - 1
            x = F.pad(x, (0, 0, shift, -shift), value=0.0)

        # naive average pool

        downsampled = self.downsample(x)

        # the "valley" - either a regular transformer or another hourglass

        x = self.valley_transformer(downsampled)

        # naive repeat upsample

        x = self.upsample(x)

        # add the residual

        x = x + x_residual

        # bring sequence back to original length, if it were padded for pooling

        x = x[:, :n]

        # post-valley transformers

        x = self.post_transformer(x)
        return self.norm_out(x)


# main class


class HourglassTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        shorten_factor=None,
        heads=8,
        dim_head=64,
        attn_resampling=True,
        updown_sample_type="naive",
        causal=True,
        device="cuda",
    ):
        super().__init__()

        self.device = device

        self.token_emb = nn.Embedding(num_tokens, dim, device=self.device)

        self.transformer = get_hourglass_transformer(
            dim=dim,
            depth=depth,
            shorten_factor=shorten_factor,
            attn_resampling=attn_resampling,
            updown_sample_type=updown_sample_type,
            dim_head=dim_head,
            heads=heads,
            causal=causal,
            norm_out=True,
            device=device,
        )

        self.to_logits = nn.Linear(dim, num_tokens, device=self.device)

    def forward(self, x):
        x = self.token_emb(x)
        x = self.transformer(x)
        return self.to_logits(x)
