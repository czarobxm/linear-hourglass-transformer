"""Base model class."""

from __future__ import annotations
from typing import Dict, Any

import torch
from torch import nn

from conf.definitions.model import ModelCfg


class BaseModel(nn.Module):
    """Base model class.

    This class is used as a base class for all models in the project.
    :param d_model: dimension of embedding, size of one vector in a sequence
    :param vocab_size: size of the vocabulary
    :param structure: structure of the model
    :param num_heads: number of heads
    :param method_params: dictionary with method parameters
    :param apply_rotary_pos_enc: flag determining whether to use rotary position encoding
    :param dropout: dropout rate
    :param max_length: maximum length of the sequence
    :param act_fun: activation function
    :param norm_before: flag determining whether to use PostNorm or PreNorm
    :param pos_enc_type: type of positional encoding
    :param use_embedding: flag determining whether to use embedding
    :param device: device
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        structure: str,
        num_heads: int,
        num_classes: int,
        method_params: Dict[str, Any],
        apply_rotary_pos_enc: bool,
        dropout: float,
        act_fun: str,
        post_norm: bool,
        pos_enc_type: str,
        use_embedding: bool,
        hourglass_downsampling_type: str,
        hourglass_upsampling_type: str,
        hourglass_attention_downsampling: bool,
        hourglass_attention_upsampling: bool,
        hourglass_upsampling_residual: bool,
        hourglass_sampling_post_norm: bool,
        device: str,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.structure = structure
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.method_params = method_params
        self.apply_rotary_pos_enc = apply_rotary_pos_enc
        self.dropout = dropout
        self.pos_enc_type = pos_enc_type
        self.act_fun_name = act_fun
        if act_fun == "gelu":
            self.act_fun = nn.GELU()
        elif act_fun == "relu":
            self.act_fun = nn.ReLU()
        elif act_fun == "none":
            self.act_fun = nn.Identity()
        self.post_norm = post_norm
        self.use_embedding = use_embedding
        self.hourglass_attention_downsampling = hourglass_attention_downsampling
        self.hourglass_attention_upsampling = hourglass_attention_upsampling
        self.hourglass_upsampling_residual = hourglass_upsampling_residual
        self.hourglass_downsampling_type = hourglass_downsampling_type
        self.hourglass_upsampling_type = hourglass_upsampling_type
        self.hourglass_sampling_post_norm = hourglass_sampling_post_norm

        self.device = device

        self.n_layers, self.sizes = self.parse_structure()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError

    @classmethod
    def from_cfg(
        cls, cfg_model: ModelCfg, vocab_size: int, device: str = "cuda"
    ) -> BaseModel:
        """Initialize the model from the cfguration."""
        return cls(
            d_model=cfg_model.mha.d_model,
            vocab_size=vocab_size,
            structure=cfg_model.structure,
            num_heads=cfg_model.mha.num_heads,
            num_classes=cfg_model.num_classes,
            method_params=cfg_model.mha.method_params,
            apply_rotary_pos_enc=cfg_model.mha.apply_rotary_pos_enc,
            dropout=cfg_model.mha.dropout,
            act_fun=cfg_model.mha.act_fun,
            post_norm=cfg_model.mha.post_norm,
            pos_enc_type=cfg_model.pos_enc_type,
            use_embedding=cfg_model.use_embedding,
            hourglass_downsampling_type=cfg_model.hourglass.downsampling_type,
            hourglass_upsampling_type=cfg_model.hourglass.upsampling_type,
            hourglass_attention_downsampling=cfg_model.hourglass.attention_downsampling,
            hourglass_attention_upsampling=cfg_model.hourglass.attention_upsampling,
            hourglass_upsampling_residual=cfg_model.hourglass.upsampling_residual,
            hourglass_sampling_post_norm=cfg_model.hourglass.sampling_post_norm,
            device=device,
        )

    def parse_structure(self):
        """Parse the structure string and return the number of layers and sizes."""
        blocks = self.structure.split(",")
        n_layers = []
        sizes = []
        for block in blocks:
            n_l, size = block.split("x")
            n_layers.append(int(n_l))
            sizes.append(int(size))
        return n_layers, sizes

    def get_hyperparams(self):
        """Return hyperparameters of the model."""
        params_dict = {
            # Model parameters
            "structure": self.structure,
            "vocab_size": self.vocab_size,
            "use_embedding": self.use_embedding,
            "embedder_type": "learnable",
            "pos_enc_type": self.pos_enc_type,
            "number_of_params": sum(p.numel() for p in self.parameters()),
            "device": self.device,
            # Hourglass parameters
            "downsampling_type": self.hourglass_downsampling_type,
            "upsampling_type": self.hourglass_upsampling_type,
            "attention_downsampling": self.hourglass_attention_downsampling,
            "attention_upsampling": self.hourglass_attention_upsampling,
            "upsampling_residual": self.hourglass_upsampling_residual,
            "sampling_post_norm": self.hourglass_sampling_post_norm,
            # MHA parameters
            "mha_type": self.method_params.method,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "act_fun": self.act_fun_name,
            "apply_rotary_pos_enc": self.apply_rotary_pos_enc,
            "post_norm": self.post_norm,
        }

        return params_dict
