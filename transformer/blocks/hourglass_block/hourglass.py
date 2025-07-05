from typing import Union, List, Optional

import torch
from torch import nn

from transformer.multi_head_attention.attention_mechanism.attn_params import (
    LinearAttnParams,
    VanillaParams,
    PerformerParams,
    CosformerParams,
)
from transformer.blocks.transformer_block import Block
from transformer.blocks.hourglass_block.utils import ShiftRight
from transformer.blocks.hourglass_block.downsampling import DownsamplingLayer
from transformer.blocks.hourglass_block.upsampling import UpsamplingLayer
from transformer.blocks.hourglass_block.attention_sampling import (
    AttentionSampling,
)


class HourglassBlock(nn.Module):
    """
    Hourglass block with downsampling and upsampling layers connected with residual
    connections. Described in https://arxiv.org/pdf/2110.13711.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        sizes: int,
        num_heads: int,
        method_params: Union[
            LinearAttnParams,
            VanillaParams,
            PerformerParams,
            CosformerParams,
        ],
        apply_rotary_pos_enc: bool = True,
        dropout: float = 0.1,
        act_fun: nn.Module = None,
        post_norm: bool = False,
        hourglass_downsampling_type: str = "linear",
        hourglass_upsampling_type: str = "linear",
        hourglass_attention_downsampling: bool = False,
        hourglass_attention_upsampling: bool = False,
        hourglass_upsampling_residual: bool = True,
        hourglass_sampling_post_norm: bool = True,
        hourglass_attention_sampling_full_attention: bool = True,
        device: str = "cpu",
        **kwargs
    ) -> None:

        super().__init__()
        self.d_model = d_model
        self.sizes = sizes
        self.method_params = method_params
        self.act_fun = act_fun
        self.downsampling_type = hourglass_downsampling_type
        self.upsampling_type = hourglass_upsampling_type
        self.attention_downsampling = hourglass_attention_downsampling
        self.attention_upsampling = hourglass_attention_upsampling
        self.upsampling_residual = hourglass_upsampling_residual
        self.sampling_post_norm = hourglass_sampling_post_norm
        self.sampling_full_attention = hourglass_attention_sampling_full_attention
        self.device = device

        self._validate_inputs(n_layers, sizes)

        if self.attention_downsampling:
            self.attention_downsampling_layers = (
                self._create_attention_downsampling_layers()
            )
        if self.attention_upsampling:
            self.attention_upsampling_layers = self._create_attention_upsampling_layers()

        self.downsampling_layers, self.upsampling_layers = self._create_sampling_layers()
        self.shift_right_layers = self._create_shift_right_layers()
        self.decoder_chunks = self._create_decoder_chunks(
            n_layers,
            num_heads,
            method_params,
            apply_rotary_pos_enc,
            dropout,
            act_fun,
            post_norm,
        )

        self.to(device)

    def _validate_inputs(self, n_layers: List[int], sizes: List[int]) -> None:
        """Validate input parameters."""
        assert len(n_layers) == len(sizes), "n_layers and sizes must have the same length"
        for i in range(len(sizes) - 1):
            if sizes[i] > sizes[i + 1]:
                assert sizes[i] % sizes[i + 1] == 0, "Adjacent sizes must be divisible"
            else:
                assert sizes[i + 1] % sizes[i] == 0, "Adjacent sizes must be divisible"

    def _create_attention_downsampling_layers(self) -> nn.ModuleList:
        """Create attention downsamplinglayers."""
        attention_downsampling_layers = nn.ModuleList()
        for i in range(len(self.sizes) - 1):
            if self.sizes[i] > self.sizes[i + 1]:
                factor = self.sizes[i] // self.sizes[i + 1]
                attention_downsampling_layers.append(
                    AttentionSampling(
                        self.d_model,
                        factor,
                        sampling_type="downsampling",
                        post_norm=self.sampling_post_norm,
                        method_params=self.method_params,
                        act_fun=self.act_fun,
                    )
                )
        return attention_downsampling_layers

    def _create_attention_upsampling_layers(self) -> nn.ModuleList:
        """Create attention upsampling layers."""
        attention_upsampling_layers = nn.ModuleList()
        for i in range(len(self.sizes) - 1):
            if self.sizes[i] <= self.sizes[i + 1]:
                factor = self.sizes[i + 1] // self.sizes[i]
                attention_upsampling_layers.append(
                    AttentionSampling(
                        self.d_model,
                        factor,
                        sampling_type="upsampling",
                        post_norm=self.sampling_post_norm,
                        method_params=self.method_params,
                        act_fun=self.act_fun,
                    )
                )
        return attention_upsampling_layers

    def _create_sampling_layers(self) -> nn.ModuleList:
        """Create downsampling and upsampling layers."""
        downsampling_layers = nn.ModuleList()
        upsampling_layers = nn.ModuleList()
        for i in range(len(self.sizes) - 1):
            if self.sizes[i] > self.sizes[i + 1]:
                factor = self.sizes[i] // self.sizes[i + 1]
                downsampling_layers.append(
                    DownsamplingLayer(self.d_model, factor, self.downsampling_type)
                )
            else:
                factor = self.sizes[i + 1] // self.sizes[i]
                upsampling_layers.append(
                    UpsamplingLayer(self.d_model, factor, self.upsampling_type)
                )
        return downsampling_layers, upsampling_layers

    def _create_shift_right_layers(self) -> nn.ModuleList:
        """Create shift right layers."""
        shift_right_layers = nn.ModuleList()
        for i in range(len(self.sizes) - 1):
            if self.sizes[i] > self.sizes[i + 1]:
                factor = self.sizes[i] // self.sizes[i + 1]
                shift_right_layers.append(ShiftRight(shift=factor - 1))
        return shift_right_layers

    def _create_decoder_chunks(
        self,
        n_layers: List[int],
        num_heads: int,
        method_params: Union[
            LinearAttnParams, VanillaParams, PerformerParams, CosformerParams
        ],
        apply_rotary_pos_enc: bool,
        dropout: float,
        act_fun: Optional[nn.Module],
        post_norm: bool,
    ) -> nn.ModuleList:
        """Create decoder chunks."""
        return nn.ModuleList(
            [
                Block(
                    n_layers=n,
                    d_model=self.d_model,
                    num_heads=num_heads,
                    method_params=method_params,
                    apply_rotary_pos_enc=apply_rotary_pos_enc,
                    dropout=dropout,
                    act_fun=act_fun,
                    post_norm=post_norm,
                    device=self.device,
                )
                for n in n_layers
            ]
        )

    def forward(
        self, x: torch.Tensor, causal: bool = True, inference: bool = False
    ) -> torch.Tensor:
        residuals = []

        n_downsampling_layers = len(self.downsampling_layers)

        x = self.decoder_chunks[0](x, causal=causal, inference=inference)
        residuals.append(x)

        # Downsampling path
        for i, (dec, downsample) in enumerate(
            zip(
                self.decoder_chunks[1 : n_downsampling_layers + 1],
                self.downsampling_layers,
            )
        ):
            x = self.shift_right_layers[i](x)
            x_downsampled = downsample(x)

            if self.attention_downsampling:
                x_downsampled = x_downsampled + self.attention_downsampling_layers[i](
                    x_downsampled, key_value=x, causal=causal
                )

            x = dec(x_downsampled, causal=causal, inference=inference)

            if i < n_downsampling_layers - 1:
                residuals.append(x)

        # Upsampling path
        for i, (dec, upsample, residual) in enumerate(
            zip(
                self.decoder_chunks[n_downsampling_layers + 1 :],
                self.upsampling_layers,
                reversed(residuals),
            )
        ):
            x_upsampled = upsample(x)

            if self.upsampling_residual:
                x_upsampled = residual + x_upsampled[:, : residual.size(1), :]

            if self.attention_upsampling:
                x_upsampled = x_upsampled[
                    :, : residual.size(1), :
                ] + self.attention_upsampling_layers[i](
                    x_upsampled[:, : residual.size(1), :], key_value=x, causal=causal
                )

            x = dec(
                x_upsampled[:, : residual.size(1), :], causal=causal, inference=inference
            )

        return x
