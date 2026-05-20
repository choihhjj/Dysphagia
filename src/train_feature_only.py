"""Training loop for the feature-only (tabular features) MLP."""

import copy
import os
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, accuracy_score,
    precision_score, f1_score,
)

import torch
import torch.nn as nn

from src.utils import calculate_class_weight_from_df


def train_feature_only(fc_model, quanti_columns,
                       train_df, val_df,
                       num_epochs, batch_size, optimizer, label_smoothing,
                       model_ckpt, device):
    """Train the feature-only MLP; returns (best_model, best_val_f1)."""
    fc_model.to(device)

    X_train = train_df[quanti_columns].values.astype(np.float32)
    y_train = train_df['label'].values
    path_train = train_df['img_dir'].values

    X_val = val_df[quanti_columns].values.astype(np.float32)
    y_val = val_df['label'].values
    path_val = val_df['img_dir'].values

    cl_list, class_weight = calculate_class_weight_from_df(train_df)
    criterion = nn.CrossEntropyLoss(
        weight=class_weight.to(device), label_smoothing=label_smoothing
    )

    with open(os.path.join(model_ckpt, 'class_weights.json'), 'w') as f:
        json.dump({i: cl_list[i] for i in range(len(cl_list))}, f, indent=4)

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_auc': [], 'train_acc': [],
        'val_auc': [], 'val_acc': [],
        'val_sensitivity': [], 'val_precision': [],
        'val_specificity': [], 'val_f1': [],
    }

    best_val_f1 = -1.0
    best_model = None

    for epoch in range(1, num_epochs + 1):
        fc_model.train()
        train_loss = []
        probs, preds, trues, paths = [], [], [], []

        # Shuffle training data each epoch for unbiased gradient estimates.
        perm = np.random.permutation(len(X_train))

        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:min(i + batch_size, len(X_train))]
            batch_X = torch.tensor(X_train[batch_idx], dtype=torch.float32).to(device)
            batch_y = torch.tensor(y_train[batch_idx], dtype=torch.long).to(device)

            optimizer.zero_grad()
            output = fc_model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            probability = torch.softmax(output, dim=1)[:, 1]
            train_loss.append(loss.item())
            probs += probability.detach().cpu().numpy().tolist()
            preds += output.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += batch_y.detach().cpu().numpy().tolist()
            paths += [path_train[idx] for idx in batch_idx]

        _train_loss = np.mean(train_loss)
        _train_auc = roc_auc_score(trues, probs)
        _train_prauc = average_precision_score(trues, probs)
        _train_acc = accuracy_score(trues, preds)

        (_val_loss, _val_auc, _val_acc,
         _val_sensitivity, _val_precision,
         _val_specificity, _val_f1, _val_result) = _validate_feature_only(
            fc_model, criterion, X_val, y_val, path_val, device, batch_size
        )
        _val_prauc = average_precision_score(_val_result['label'], _val_result['prob'])

        print(
            f'Epoch [{epoch}] '
            f'Train Loss: {_train_loss:.3f} | Val Loss: {_val_loss:.3f} | '
            f'Train AUC: {_train_auc:.3f} | Val AUC: {_val_auc:.3f} | '
            f'Val Sen: {_val_sensitivity:.3f} | Val Spe: {_val_specificity:.3f} | '
            f'Val F1: {_val_f1:.3f}'
        )

        history['epoch'].append(epoch)
        history['train_loss'].append(_train_loss)
        history['val_loss'].append(_val_loss)
        history['train_auc'].append(_train_auc)
        history['val_auc'].append(_val_auc)
        history['train_acc'].append(_train_acc)
        history['val_acc'].append(_val_acc)
        history['val_sensitivity'].append(_val_sensitivity)
        history['val_precision'].append(_val_precision)
        history['val_specificity'].append(_val_specificity)
        history['val_f1'].append(_val_f1)

        if _val_f1 > best_val_f1:
            best_val_f1 = _val_f1
            best_model = copy.deepcopy(fc_model)

            torch.save(fc_model.state_dict(),
                       os.path.join(model_ckpt, 'best_model_feature_only.pt'))

            best_metric = {
                'epoch': epoch,
                'train_loss': _train_loss, 'val_loss': _val_loss,
                'train_auc': _train_auc, 'val_auc': _val_auc,
                'train_prauc': _train_prauc, 'val_prauc': _val_prauc,
                'train_acc': _train_acc, 'val_acc': _val_acc,
                'val_sensitivity': _val_sensitivity, 'val_precision': _val_precision,
                'val_specificity': _val_specificity, 'val_f1': _val_f1,
            }
            with open(os.path.join(model_ckpt, 'best_model_metric_feature_only.json'), 'w') as f:
                json.dump(best_metric, f, indent=4)

            train_result = pd.DataFrame(
                {'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues}
            )
            train_result['image_path'] = [p.split('/')[-1] for p in train_result['image_path']]
            train_result.to_csv(os.path.join(model_ckpt, 'results_train.csv'), index=False)
            _val_result.to_csv(os.path.join(model_ckpt, 'results_val.csv'), index=False)

        pd.DataFrame(history).to_csv(
            os.path.join(model_ckpt, 'history_feature_only.csv'), index=False
        )

    return best_model, best_val_f1


def _validate_feature_only(fc_model, criterion, X_val, y_val, path_val, device, batch_size):
    """Run one full pass over the validation set and return all metrics."""
    fc_model.eval()
    val_loss = []
    probs, preds, trues, paths = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            end = min(i + batch_size, len(X_val))
            batch_X = torch.tensor(X_val[i:end], dtype=torch.float32).to(device)
            batch_y = torch.tensor(y_val[i:end], dtype=torch.long).to(device)

            logit = fc_model(batch_X)
            loss = criterion(logit, batch_y)
            probability = torch.softmax(logit, dim=1)[:, 1]

            val_loss.append(loss.item())
            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += batch_y.detach().cpu().numpy().tolist()
            paths += list(path_val[i:end])

    _val_loss = np.mean(val_loss)
    _val_auc = roc_auc_score(trues, probs)
    _val_acc = accuracy_score(trues, preds)
    _val_sensitivity = recall_score(trues, preds, zero_division=0)
    _val_precision = precision_score(trues, preds, zero_division=0)
    _val_specificity = recall_score(trues, preds, pos_label=0, zero_division=0)
    _val_f1 = f1_score(trues, preds, zero_division=0)

    val_result = pd.DataFrame(
        {'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues}
    )
    val_result['image_path'] = [p.split('/')[-1] for p in val_result['image_path']]

    return (_val_loss, _val_auc, _val_acc,
            _val_sensitivity, _val_precision,
            _val_specificity, _val_f1, val_result)
