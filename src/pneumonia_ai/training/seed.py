"""Shared reproducibility controls for training and data loading."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed one DataLoader worker from PyTorch's worker-specific seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
