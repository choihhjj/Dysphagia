"""Ensemble utilities for merging per-fold test predictions."""

import json
import os

import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, recall_score, precision_score, f1_score,
)


def load_fold_inference(output_base, fold):
    """Load test_inference.csv for one fold; rename prob/pred with fold suffix."""
    path = os.path.join(output_base, f'fold{fold}', 'test_inference.csv')
    df = pd.read_csv(path)
    return df.rename(columns={
        'prob': f'prob_fold{fold}',
        'pred': f'pred_fold{fold}',
    })


def build_ensemble(output_base, n_folds=5, threshold=0.5):
    """Merge per-fold test inferences and return DataFrame with ensemble prob/pred."""
    merged = load_fold_inference(output_base, 1)
    for fold in range(2, n_folds + 1):
        merged = merged.merge(
            load_fold_inference(output_base, fold),
            on=['image_path', 'label'],
            how='inner',
        )

    prob_cols = [f'prob_fold{k}' for k in range(1, n_folds + 1)]
    pred_cols = [f'pred_fold{k}' for k in range(1, n_folds + 1)]

    merged['prob_ensemble'] = merged[prob_cols].mean(axis=1)
    merged['pred_ensemble'] = (merged['prob_ensemble'] >= threshold).astype(int)

    col_order = ['image_path', 'label'] + prob_cols + pred_cols + ['prob_ensemble', 'pred_ensemble']
    return merged[col_order]


def compute_ensemble_metrics(df):
    """Compute binary classification metrics for the ensemble predictions."""
    trues, probs, preds = df['label'], df['prob_ensemble'], df['pred_ensemble']
    return {
        'auc':         roc_auc_score(trues, probs),
        'prauc':       average_precision_score(trues, probs),
        'acc':         accuracy_score(trues, preds),
        'sensitivity': recall_score(trues, preds, pos_label=1, zero_division=0),
        'specificity': recall_score(trues, preds, pos_label=0, zero_division=0),
        'precision':   precision_score(trues, preds, zero_division=0),
        'f1':          f1_score(trues, preds, zero_division=0),
    }


def save_ensemble(output_base, n_folds=5, threshold=0.5):
    """Build ensemble and save test_inference_ensemble.csv and metrics_test_ensemble.json."""
    print(f'\n[cv_ensemble] Building {n_folds}-fold ensemble from {output_base} ...')
    ensemble_df = build_ensemble(output_base, n_folds, threshold)
    print(f'[cv_ensemble] Ensemble test set size: {len(ensemble_df)} samples')

    csv_path = os.path.join(output_base, 'test_inference_ensemble.csv')
    ensemble_df.to_csv(csv_path, index=False)
    print(f'[cv_ensemble] Saved → {csv_path}')

    metrics = compute_ensemble_metrics(ensemble_df)
    print('[cv_ensemble] Ensemble metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')

    json_path = os.path.join(output_base, 'metrics_test_ensemble.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'[cv_ensemble] Saved → {json_path}')
