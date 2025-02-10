from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class SchedulerCfg:
    scheduler: bool
    lr_warmup_fraction: float
    scheduler_steps_fraction: float
    final_lr_fraction: float


@dataclass
class TrainingCfg:
    optimizer: Dict[str, Any]
    scheduler: SchedulerCfg
    criterion: Dict[str, Any]
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    use_validation: bool
