"""ROC curves (mean ± SD over 5 folds) and confusion matrices (5-fold ensemble) on the test set.

Usage: python scripts/plot_roc_cm.py [--image_only DIR] [--feature_only DIR] [--multimodal DIR]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib
matplotlib.use('Agg')
import pandas as pd

from src.figures import set_style, plot_roc_cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_only', default='outputs/image_only')
    parser.add_argument('--feature_only', default='outputs/feature_only')
    parser.add_argument('--multimodal', default='outputs/multimodal')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--save_path', default='outputs/figures/roc_cm')
    args = parser.parse_args()

    model_dfs = {
        name: pd.read_csv(os.path.join(path, 'test_inference_ensemble.csv'))
        for name, path in [('Image-only', args.image_only),
                           ('Feature-only', args.feature_only),
                           ('Multimodal', args.multimodal)]
    }

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    set_style()
    plot_roc_cm(model_dfs, args.save_path, args.n_folds)
    print(f'saved -> {args.save_path}.png / .tif')


if __name__ == '__main__':
    main()
