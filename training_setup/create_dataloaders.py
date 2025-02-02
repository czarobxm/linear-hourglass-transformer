from typing import Tuple, Dict, Any

import hydra
from torch.utils.data import DataLoader

from conf import TrainingCfg


def create_dataloaders(
    cfg_dataset: Dict[str, Any], cfg_training: TrainingCfg
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:

    train, val, test = hydra.utils.instantiate(cfg_dataset)

    train_loader = DataLoader(train, batch_size=cfg_training.batch_size)
    test_loader = DataLoader(test, batch_size=cfg_training.batch_size)
    if cfg_training.use_validation:
        val_loader = DataLoader(val, batch_size=cfg_training.batch_size)
        return train_loader, val_loader, test_loader
    return train_loader, None, test_loader
