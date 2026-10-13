"""Binary classification metrics and number formatting."""

from decimal import Decimal, ROUND_HALF_UP

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, recall_score, precision_score, f1_score,
)

METRIC_NAMES = {
    'auc':         'AUROC',
    'prauc':       'PRAUC',
    'acc':         'Accuracy',
    'sensitivity': 'Sensitivity',
    'specificity': 'Specificity',
    'ppv':         'PPV',
    'npv':         'NPV',
    'f1':          'F1-score',
}


def compute_metrics(trues, probs, preds):
    """Return auc, prauc, acc, sensitivity, specificity, ppv, npv, f1 (positive class = 1)."""
    return {
        'auc':         roc_auc_score(trues, probs),
        'prauc':       average_precision_score(trues, probs),
        'acc':         accuracy_score(trues, preds),
        'sensitivity': recall_score(trues, preds, pos_label=1, zero_division=0),
        'specificity': recall_score(trues, preds, pos_label=0, zero_division=0),
        'ppv':         precision_score(trues, preds, pos_label=1, zero_division=0),
        'npv':         precision_score(trues, preds, pos_label=0, zero_division=0),
        'f1':          f1_score(trues, preds, zero_division=0),
    }


def fmt(x, digits=2):
    """Round half up (e.g. 0.125 -> 0.13) and format."""
    q = Decimal(1).scaleb(-digits)
    return format(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP), f'.{digits}f')
