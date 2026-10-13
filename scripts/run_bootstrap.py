"""Paired bootstrap (95% CI, p-value) of the 5-fold ensembles on the test set.

Resampling is done at the patient level using the patient_id column of the dataset CSV.

Usage: python scripts/run_bootstrap.py --dataset_csv path/to/final_dataset.csv
                                       [--model1 outputs/image_only] [--model2 outputs/multimodal]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import pandas as pd

from src.bootstrap import bootstrap_compare, format_table

MODEL_NAMES = {'image_only': 'Image-only', 'feature_only': 'Feature-only', 'multimodal': 'Multimodal'}


def load_ensemble(output_base):
    df = pd.read_csv(os.path.join(output_base, 'test_inference_ensemble.csv'))
    return df[['image_path', 'label', 'prob_ensemble']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', default='outputs/image_only')
    parser.add_argument('--model2', default='outputs/multimodal')
    parser.add_argument('--dataset_csv', default='path/to/final_dataset.csv')
    parser.add_argument('--n_boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', default='outputs/bootstrap')
    args = parser.parse_args()

    key1 = os.path.basename(os.path.normpath(args.model1))
    key2 = os.path.basename(os.path.normpath(args.model2))
    name1, name2 = MODEL_NAMES.get(key1, key1), MODEL_NAMES.get(key2, key2)

    df = load_ensemble(args.model1).merge(
        load_ensemble(args.model2), on=['image_path', 'label'], suffixes=('_1', '_2'))

    dataset = pd.read_csv(args.dataset_csv)
    patient_ids = dict(zip(dataset['img_dir'].str.split('/').str[-1], dataset['patient_id']))
    groups = df['image_path'].map(patient_ids)
    if groups.isna().any():
        raise ValueError('patient_id not found for some test images')

    summary, raw = bootstrap_compare(
        df['label'], df['prob_ensemble_1'], df['prob_ensemble_2'], groups,
        threshold=args.threshold, n_boot=args.n_boot, seed=args.seed,
    )
    table = format_table(summary, name1, name2)

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, f'bootstrap_{key1}_vs_{key2}')
    summary.to_csv(prefix + '.csv', index=False)
    raw.to_csv(prefix + '_raw.csv', index=False)
    table.to_csv(prefix + '_table.csv', index=False, encoding='utf-8-sig')

    print(f'{len(df)} test images, {len(raw)} valid bootstrap samples')
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
