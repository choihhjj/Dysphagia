"""Standalone script to generate cv_summary.csv for all three model pipelines.

Usage: python scripts/cv_summary.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv_summary import save_cv_summary

MODELS = {
    'image_only':   'outputs/image_only',
    'feature_only': 'outputs/feature_only',
    'multimodal':   'outputs/multimodal',
}

if __name__ == '__main__':
    for model_name, output_base in MODELS.items():
        if not os.path.isdir(output_base):
            print(f'[{model_name}] Directory not found: {output_base}, skipping.')
            continue
        save_cv_summary(model_name, output_base)
        print()
