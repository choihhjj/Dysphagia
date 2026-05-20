"""CV summary utilities for all three pipelines."""

import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score,
)

METRICS = ['acc', 'auc', 'prauc', 'sen', 'spe', 'ppv', 'f1']


def calc_metrics(df, model_name, fold, split):
    """Compute binary classification metrics for one fold/split DataFrame."""
    return {
        'model':     model_name,
        'datasplit': split,
        'fold':      fold,
        'acc':       accuracy_score(df['label'], df['pred']),
        'auc':       roc_auc_score(df['label'], df['prob']),
        'prauc':     average_precision_score(df['label'], df['prob']),
        'sen':       recall_score(df['label'], df['pred'], pos_label=1, zero_division=0),
        'spe':       recall_score(df['label'], df['pred'], pos_label=0, zero_division=0),
        'ppv':       precision_score(df['label'], df['pred'], zero_division=0),
        'f1':        f1_score(df['label'], df['pred'], zero_division=0),
    }


def get_metrics(model_name, output_base, fold):
    """Load val and test CSVs for one fold and return a two-row metrics DataFrame."""
    fold_dir = os.path.join(output_base, f'fold{fold}')
    val  = pd.read_csv(os.path.join(fold_dir, 'results_val.csv'))
    test = pd.read_csv(os.path.join(fold_dir, 'test_inference.csv'))
    return pd.DataFrame([
        calc_metrics(val,  model_name, fold, 'val'),
        calc_metrics(test, model_name, fold, 'test'),
    ])


def cv_results(model_name, output_base, n_folds=5):
    """Build an n-fold CV summary DataFrame with per-fold, mean, sd, and mean+/-sd rows."""
    res_all = pd.concat(
        [get_metrics(model_name, output_base, fold) for fold in range(1, n_folds + 1)],
        axis=0,
    ).reset_index(drop=True)

    summary = res_all.groupby('datasplit')[METRICS].agg(['mean', 'std'])

    mean_sd_df = pd.DataFrame()
    for m in METRICS:
        mean_sd_df[m] = (
            summary[(m, 'mean')].map(lambda x: f"{x:.2f}")
            + '±'
            + summary[(m, 'std')].map(lambda x: f"{x:.2f}")
        )
    mean_sd_df = mean_sd_df.reset_index()
    mean_sd_df.insert(0, 'fold', 'mean±sd')
    mean_sd_df.insert(0, 'model', model_name)

    mean_df = summary.xs('mean', axis=1, level=1).reset_index()
    mean_df.insert(0, 'fold', 'mean')
    mean_df.insert(0, 'model', model_name)

    sd_df = summary.xs('std', axis=1, level=1).reset_index()
    sd_df.insert(0, 'fold', 'sd')
    sd_df.insert(0, 'model', model_name)

    # 'val' before 'test' in sort order
    final_df = pd.concat(
        [res_all, mean_df, sd_df, mean_sd_df],
        ignore_index=True,
    ).sort_values('datasplit', ascending=False).reset_index(drop=True)

    return final_df


def save_cv_summary(model_name, output_base, n_folds=5):
    """Generate CV summary and save to <output_base>/cv_summary.csv."""
    print(f'\n[cv_summary] Building CV summary for {model_name} ...')
    df = cv_results(model_name, output_base, n_folds)
    save_path = os.path.join(output_base, 'cv_summary.csv')
    df.to_csv(save_path, index=False)
    print(f'[cv_summary] Saved → {save_path}')
    print(df.to_string(index=False))
