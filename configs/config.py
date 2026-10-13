"""Reproducibility helpers, config loading, and device selection."""

import argparse
import random
import os
import numpy as np
import torch
import yaml

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    """Load a YAML model configuration file into a dict."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_config(default_name, description=None):
    """Parse --config (plus optional path overrides) and return the merged config dict.

    default_name: YAML file name under configs/ used when --config is not given.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default=os.path.join(CONFIG_DIR, default_name),
                        help=f'YAML config file (default: configs/{default_name})')
    parser.add_argument('--dataset_csv', default=None, help='override DATASET_CSV')
    parser.add_argument('--output_base', default=None, help='override OUTPUT_BASE')
    parser.add_argument('--io_checkpoint_base', default=None,
                        help='override IO_CHECKPOINT_BASE (multimodal only)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key, value in [('DATASET_CSV', args.dataset_csv),
                       ('OUTPUT_BASE', args.output_base),
                       ('IO_CHECKPOINT_BASE', args.io_checkpoint_base)]:
        if value is not None:
            cfg[key] = value
    return cfg


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
