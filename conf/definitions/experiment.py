from typing import Dict, Any
from dataclasses import dataclass

from conf.definitions.model import ModelCfg
from conf.definitions.neptune import NeptuneCfg
from conf.definitions.training import TrainingCfg


@dataclass
class Experiment:
    task: str
    model: ModelCfg
    training: TrainingCfg
    tokenizer: Dict[str, Any]
    dataset: Dict[str, Any]
    neptune: NeptuneCfg
    device: str
