from models import DecoderOnlyTransformer, ClassifierTransformer, MambaHourglass, Mamba
from models.new_decoder_only import HourglassTransformerLM
from models.base import BaseModel
from conf.definitions import ModelCfg


def initialize_model(cfg_model: ModelCfg, vocab_size: int, device: str) -> BaseModel:

    if cfg_model.type == "sequence_modelling":
        return DecoderOnlyTransformer.from_cfg(
            cfg_model, vocab_size=vocab_size, device=device
        )
    if cfg_model.type == "classification":
        return ClassifierTransformer.from_cfg(
            cfg_model, vocab_size=vocab_size, device=device
        )
    if cfg_model.type == "mamba":
        return Mamba.from_cfg(cfg_model, vocab_size=vocab_size, device=device)
    if cfg_model.type == "mamba_hourglass":
        MambaHourglass.from_cfg(cfg_model, vocab_size=vocab_size, device=device)

    if cfg_model.type == "lucidrains":
        return HourglassTransformerLM(
            num_tokens=vocab_size,
            dim=cfg_model.mha.d_model,
            heads=cfg_model.mha.num_heads,
            shorten_factor=2,
            depth=(2, 4, 2),
            attn_resampling=False,
            updown_sample_type="linear",
            causal=True,
            device=device,
        )

    else:
        raise ValueError(f"Model {cfg_model.type} not implemented.")
