"""Reproducibility helpers and device selection."""

import random
import os
import numpy as np
import torch


def seed_everything(seed):
    """Fix all sources of randomness for fully reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # force deterministic CUDA kernels
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    # suppress oneDNN non-determinism
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def seed_worker(_worker_id):
    """Propagate the torch worker seed to numpy and random for full reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
