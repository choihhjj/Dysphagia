"""Standalone script to ensemble test predictions across folds for all three model pipelines.

Usage: python scripts/cv_ensemble.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv_ensemble import save_ensemble

MODELS = {
    'image_only':   'outputs/image_only',
    'feature_only': 'outputs/feature_only',
    'multimodal':   'outputs/multimodal',
}
N_FOLDS   = 5
THRESHOLD = 0.5

if __name__ == '__main__':
    for model_name, output_base in MODELS.items():
        if not os.path.isdir(output_base):
            print(f'[{model_name}] Directory not found: {output_base}, skipping.')
            continue
        save_ensemble(output_base, N_FOLDS, THRESHOLD)
