"""Module containing the transformer-decoder-only model."""

import torch
from torch import nn

from models.base import BaseModel
from transformer.blocks.transformer_block import Block
from transformer.blocks.hourglass_block import HourglassBlock
from transformer.positional_encoding import PositionalEncoding

from transformer.blocks.hourglass_block.utils import ShiftRight


class DecoderOnlyTransformer(BaseModel):
    """Decoder-only transformer model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        kwargs["n_layers"] = self.n_layers
        kwargs["sizes"] = self.sizes
        kwargs["act_fun"] = self.act_fun

        # Embedders
        self.embedder = nn.Embedding(self.vocab_size, self.d_model)

        # Positional Encodings
        self.pos_enc = PositionalEncoding(
            self.sizes[0], self.d_model, self.pos_enc_type, device=self.device
        )
        # Shift right
        self.shift_right = ShiftRight(shift=1)
        # Decoder
        if len(self.n_layers) == len(self.sizes) == 1:
            self.decoder_block = Block(**kwargs)
        else:
            self.decoder_block = HourglassBlock(**kwargs)
        # Classifier
        self.classifier = nn.Linear(self.d_model, self.vocab_size)

        # Device
        self.to(self.device)

    def forward(self, x: torch.Tensor, inference: bool = False):
        """Produces the output of the decoder block."""
        # Shift right
        x = self.shift_right(x)
        # Embedding
        if self.use_embedding:
            x = self.embedder(x)
        # Positional Encoding
        x = self.pos_enc(x)
        # Decoder
        x = self.decoder_block(x, causal=True, inference=inference)
        # Linear
        x = self.classifier(x)
        return x
