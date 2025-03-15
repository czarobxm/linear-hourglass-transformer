import torch
import numpy as np
import random

from conf import ExperimentCfg


def setup_random_seed(cfg_experiment: ExperimentCfg) -> None:
    seed = cfg_experiment.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
