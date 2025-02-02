from typing import Dict, Any

import torch
import hydra

from training_setup.scheduler import get_cosine_scheduler_with_warmup
from conf.definitions import TrainingCfg


def setup_training(cfg_training: TrainingCfg, model: torch.nn.Module) -> Dict[str, Any]:
    optimizer = hydra.utils.instantiate(cfg_training.optimizer, params=model.parameters())

    if cfg_training.scheduler:
        scheduler = get_cosine_scheduler_with_warmup(
            optimizer,
            num_all_steps=cfg_training.scheduler.num_all_steps,
            num_warmup_steps=cfg_training.scheduler.lr_warmup_steps,
            final_lr_fraction=cfg_training.scheduler.final_lr_fraction,
        )
    else:
        scheduler = None

    loss_fn = hydra.utils.instantiate(cfg_training.criterion)
    return {"optimizer": optimizer, "scheduler": scheduler, "loss_fn": loss_fn}
