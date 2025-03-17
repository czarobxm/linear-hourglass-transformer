import torch
import numpy as np
import random


def setup_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Ensures seed is set for CUDA
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
