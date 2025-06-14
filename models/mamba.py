from functools import partial

import torch
import torch.nn as nn

from models.utils import parse_structure

from transformer.blocks.hourglass_block.utils import ShiftRight
from transformer.blocks.hourglass_block.downsampling import DownsamplingLayer
from transformer.blocks.hourglass_block.upsampling import UpsamplingLayer

from mamba_ssm.models.mixer_seq_simple import _init_weights, create_block

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        """MixerModel without the embedding."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def forward(self, x, inference_params=None, **mixer_kwargs):
        hidden_states = x
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        return hidden_states


class MambaHourglass(nn.Module):
    def __init__(
        self,
        structure: str,
        d_model: int = 512,
        vocab_size: int = 256,
        d_state: int = 256,
        d_conv: int = 4,
        expand: int = 4,
        rms_norm: bool = True,
        hourglass_upsampling_residual: bool = True,
        hourglass_upsampling_type: str = "linear",
        hourglass_downsampling_type: str = "linear",
    ):
        super().__init__()
        self.n_layers, self.sizes = parse_structure(structure=structure)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.rms_norm = rms_norm
        self.upsampling_residual = hourglass_upsampling_residual
        self.hourglass_upsampling_type = hourglass_upsampling_type
        self.hourglass_downsampling_type = hourglass_downsampling_type

        # Embedders
        self.embedder = nn.Embedding(vocab_size, d_model)

        # Shift right
        self.shift_right = ShiftRight(shift=1)

        # Create sampling layers
        self.downsampling_layers, self.upsampling_layers = self._create_sampling_layers()

        # Create shift right layers
        self.shift_right_layers = self._create_shift_right_layers()

        # Create blocks of Mamba
        self.decoder_chunks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    n_layer=n_layer,
                    d_intermediate=0,  # No MLP in Mamba
                    ssm_cfg=dict(
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    ),
                    rms_norm=rms_norm,
                    attn_layer_idx=None,
                    device=None,  # Device will be set later
                )
                for _, n_layer in enumerate(self.n_layers)
            ]
        )

    def _create_sampling_layers(self) -> nn.ModuleList:
        """Create downsampling and upsampling layers."""
        downsampling_layers = nn.ModuleList()
        upsampling_layers = nn.ModuleList()
        for i in range(len(self.sizes) - 1):
            if self.sizes[i] > self.sizes[i + 1]:
                factor = self.sizes[i] // self.sizes[i + 1]
                downsampling_layers.append(
                    DownsamplingLayer(
                        self.d_model, factor, self.hourglass_downsampling_type
                    )
                )
            else:
                factor = self.sizes[i + 1] // self.sizes[i]
                upsampling_layers.append(
                    UpsamplingLayer(self.d_model, factor, self.hourglass_upsampling_type)
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

    def forward(
        self, x: torch.Tensor, causal: bool = True, inference: bool = False
    ) -> torch.Tensor:
        residuals = []

        n_downsampling_layers = len(self.downsampling_layers)

        x = self.decoder_chunks[0](x)
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

            x = dec(x_downsampled)

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
                x_upsampled = residual + x_upsampled

            x = dec(x_upsampled)

        return x
