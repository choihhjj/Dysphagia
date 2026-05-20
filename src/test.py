"""Inference functions for the three model variants."""

import json
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, recall_score, precision_score, f1_score,
)

import torch

from src.train_multimodal import _build_enhanced_features


def inference_image_only(model, test_loader, device, model_ckpt, mode='test'):
    """Run inference for the image-only model; saves results csv and metrics json. Returns AUROC."""
    model.load_state_dict(torch.load(os.path.join(model_ckpt, 'best_model.pt'),
                                     map_location=device))
    model.to(device)
    model.eval()

    probs, preds, trues, paths = [], [], [], []
    with torch.no_grad():
        for path, images, labels in tqdm(iter(test_loader), desc='Inference'):
            images = images.to(device)
            logit = model(images)
            probability = torch.softmax(logit, dim=1)[:, 1]

            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
            paths.extend(path)

    metrics = _compute_metrics(trues, probs, preds)
    print(f'[{mode}] AUC: {metrics["auc"]:.3f} | Acc: {metrics["acc"]:.3f}')

    results = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
    results['image_path'] = results['image_path'].str.split('/').str[-1]
    csv_name = 'test_inference.csv' if mode == 'test' else f'results_{mode}.csv'
    results.to_csv(os.path.join(model_ckpt, csv_name), index=False)

    with open(os.path.join(model_ckpt, f'metrics_{mode}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics['auc']


def inference_feature_only(fc_model, X_test, y_test, path_test, device, model_ckpt,
                            batch_size, mode='test'):
    """Run inference for the feature-only MLP; saves results csv and metrics json. Returns (auc, results_df)."""
    fc_model.load_state_dict(
        torch.load(os.path.join(model_ckpt, 'best_model_feature_only.pt'),
                   map_location=device)
    )
    fc_model.to(device)
    fc_model.eval()

    probs, preds, trues, paths = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            end = min(i + batch_size, len(X_test))
            batch_X = torch.tensor(X_test[i:end], dtype=torch.float32).to(device)
            batch_y = torch.tensor(y_test[i:end], dtype=torch.long).to(device)

            logit = fc_model(batch_X)
            probability = torch.softmax(logit, dim=1)[:, 1]

            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += batch_y.detach().cpu().numpy().tolist()
            paths += [path_test[j] for j in range(i, end)]

    metrics = _compute_metrics(trues, probs, preds)
    print(f'[{mode}] AUC: {metrics["auc"]:.3f} | Acc: {metrics["acc"]:.3f}')

    results = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
    results['image_path'] = [p.split('/')[-1] for p in results['image_path']]
    csv_name = 'test_inference.csv' if mode == 'test' else f'results_{mode}.csv'
    results.to_csv(os.path.join(model_ckpt, csv_name), index=False)

    with open(os.path.join(model_ckpt, f'metrics_{mode}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics['auc'], results


def inference_multimodal(fc_model, test_features, test_paths, test_df,
                         device, model_ckpt, batch_size, mode='test'):
    """Run inference for the multimodal model; saves results csv and metrics json. Returns (auc, results_df)."""
    fc_model.load_state_dict(
        torch.load(os.path.join(model_ckpt, 'best_model_multimodal.pt'),
                   map_location=device)
    )
    fc_model.to(device)
    fc_model.eval()

    test_labels = test_df['label'].values
    enhanced_test = _build_enhanced_features(test_features, test_paths, test_df)

    probs, preds, trues, paths = [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(enhanced_test), batch_size), desc='Inference'):
            end = min(i + batch_size, len(enhanced_test))
            batch_X = torch.tensor(enhanced_test[i:end], dtype=torch.float32).to(device)
            batch_y = torch.tensor(test_labels[i:end], dtype=torch.long).to(device)

            logit = fc_model(batch_X)
            probability = torch.softmax(logit, dim=1)[:, 1]

            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += batch_y.detach().cpu().numpy().tolist()
            paths += [test_paths[j] for j in range(i, end)]

    metrics = _compute_metrics(trues, probs, preds)
    print(f'[{mode}] AUC: {metrics["auc"]:.3f} | Acc: {metrics["acc"]:.3f}')

    results = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
    results['image_path'] = [p.split('/')[-1] for p in results['image_path']]
    csv_name = 'test_inference.csv' if mode == 'test' else f'results_{mode}.csv'
    results.to_csv(os.path.join(model_ckpt, csv_name), index=False)

    with open(os.path.join(model_ckpt, f'metrics_{mode}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics['auc'], results


def _compute_metrics(trues, probs, preds):
    """Compute binary classification metrics; returns dict with auc, prauc, acc, sensitivity, specificity, precision, f1."""
    return {
        'auc':         roc_auc_score(trues, probs),
        'prauc':       average_precision_score(trues, probs),
        'acc':         accuracy_score(trues, preds),
        'sensitivity': recall_score(trues, preds, pos_label=1, zero_division=0),
        'specificity': recall_score(trues, preds, pos_label=0, zero_division=0),
        'precision':   precision_score(trues, preds, zero_division=0),
        'f1':          f1_score(trues, preds, zero_division=0),
    }
