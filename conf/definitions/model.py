from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class MultiheadAttentionCfg:
    method_params: Dict[str, Any]
    d_model: int
    num_heads: int
    dropout: float
    act_fun: str
    apply_rotary_pos_enc: bool
    post_norm: bool


@dataclass
class HourglassCfg:
    downsampling_type: str
    upsampling_type: str
    attention_downsampling: bool
    attention_upsampling: bool
    upsampling_residual: bool
    sampling_post_norm: bool
    sampling_use_linear: bool
    sampling_use_feedforward: bool


@dataclass
class ModelCfg:
    type: str
    structure: str
    pos_enc_type: str
    use_embedding: bool
    mha: MultiheadAttentionCfg
    hourglass: HourglassCfg
