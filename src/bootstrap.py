"""Paired bootstrap comparison of two models on the held-out test set."""

import numpy as np
import pandas as pd

from src.metrics import compute_metrics, fmt, METRIC_NAMES


def bootstrap_compare(y, prob1, prob2, groups=None, threshold=0.5, n_boot=1000, seed=42):
    """Paired bootstrap of model 2 vs model 1.

    Resampling is done at the group (patient) level when `groups` is given,
    otherwise at the image level. Returns (summary, raw) DataFrames.
    """
    y = np.asarray(y)
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)

    unique_ids = np.unique(groups)
    id_to_idx = {gid: np.where(groups == gid)[0] for gid in unique_ids}
    rng = np.random.default_rng(seed)

    obs1 = compute_metrics(y, prob1, (prob1 >= threshold).astype(int))
    obs2 = compute_metrics(y, prob2, (prob2 >= threshold).astype(int))

    boot1, boot2 = [], []
    for b in range(n_boot):
        sampled = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        idx = np.concatenate([id_to_idx[gid] for gid in sampled])
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        p1, p2 = prob1[idx], prob2[idx]
        boot1.append({'boot': b, **compute_metrics(y_b, p1, (p1 >= threshold).astype(int))})
        boot2.append({'boot': b, **compute_metrics(y_b, p2, (p2 >= threshold).astype(int))})

    boot1 = pd.DataFrame(boot1).set_index('boot')
    boot2 = pd.DataFrame(boot2).set_index('boot')
    diff = boot2 - boot1

    rows = []
    for m in obs1:
        d = diff[m].to_numpy()
        p = min(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0)
        rows.append({
            'metric': m,
            'model1': obs1[m],
            'model1_ci_lower': boot1[m].quantile(0.025),
            'model1_ci_upper': boot1[m].quantile(0.975),
            'model2': obs2[m],
            'model2_ci_lower': boot2[m].quantile(0.025),
            'model2_ci_upper': boot2[m].quantile(0.975),
            'diff': obs2[m] - obs1[m],
            'diff_ci_lower': diff[m].quantile(0.025),
            'diff_ci_upper': diff[m].quantile(0.975),
            'p_value': p,
        })
    summary = pd.DataFrame(rows)

    raw = pd.concat([boot1.add_prefix('model1_'),
                     boot2.add_prefix('model2_'),
                     diff.add_prefix('diff_')], axis=1).reset_index()
    return summary, raw


def format_table(summary, name1, name2):
    """Format as 'score (95% CI)' with a p-value column."""
    def ci(row, key):
        return f"{fmt(row[key])} ({fmt(row[key + '_ci_lower'])}–{fmt(row[key + '_ci_upper'])})"

    return pd.DataFrame({
        'Metric': summary['metric'].map(METRIC_NAMES),
        name1: summary.apply(ci, axis=1, key='model1'),
        name2: summary.apply(ci, axis=1, key='model2'),
        f'Difference ({name2} - {name1})': summary.apply(ci, axis=1, key='diff'),
        'p-value': summary['p_value'].map(lambda p: '<0.001' if p < 0.001 else fmt(p, 3)),
    })
