from models import DecoderOnlyTransformer, ClassifierTransformer

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
    else:
        raise ValueError(f"Model {cfg_model.type} not implemented.")
