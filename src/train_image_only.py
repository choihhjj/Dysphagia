"""Training loop for the image-only CNN classifier."""

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

from src.utils import calculate_class_weight_from_loader


def train(model, num_epochs, optimizer, train_loader, val_loader, scheduler,
          label_smoothing, device, model_ckpt):
    """Train the image-only CNN; returns (best_model, best_val_f1)."""
    model.to(device)

    cl_list, class_weight = calculate_class_weight_from_loader(train_loader)
    print('Class weights:', class_weight)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device),
                                    label_smoothing=label_smoothing)

    with open(os.path.join(model_ckpt, 'class_weights.json'), 'w') as f:
        json.dump({i: cl_list[i] for i in range(len(cl_list))}, f, indent=4)

    best_val_f1 = -1.0
    best_model = None
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_auc': [], 'train_acc': [],
        'val_auc': [], 'val_acc': [],
        'val_sensitivity': [], 'val_precision': [],
        'val_specificity': [], 'val_f1': [],
    }

    model_path        = os.path.join(model_ckpt, 'best_model.pt')
    metric_path       = os.path.join(model_ckpt, 'best_model_metric.json')
    train_result_path = os.path.join(model_ckpt, 'results_train.csv')
    val_result_path   = os.path.join(model_ckpt, 'results_val.csv')
    history_path      = os.path.join(model_ckpt, 'history.csv')

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = []
        probs, preds, trues, paths = [], [], [], []

        for path, images, labels in tqdm(iter(train_loader), desc=f'Epoch {epoch}'):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            probability = torch.softmax(output, dim=1)[:, 1]
            train_loss.append(loss.item())
            probs += probability.detach().cpu().numpy().tolist()
            preds += output.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
            paths.extend(path)

        _train_loss = np.mean(train_loss)
        _train_auc = roc_auc_score(trues, probs)
        _train_prauc = average_precision_score(trues, probs)
        _train_acc = accuracy_score(trues, preds)

        (_val_loss, _val_auc, _val_acc,
         _val_sensitivity, _val_precision,
         _val_specificity, _val_f1, _val_result) = _validate_image_only(
            model, criterion, val_loader, device
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

        if scheduler is not None:
            scheduler.step(_val_f1)

        if _val_f1 > best_val_f1:
            best_val_f1 = _val_f1
            best_model = copy.deepcopy(model)

            torch.save(model.state_dict(), model_path)

            best_model_metric = {
                'epoch': epoch,
                'train_loss': _train_loss, 'val_loss': _val_loss,
                'train_auc': _train_auc, 'val_auc': _val_auc,
                'train_prauc': _train_prauc, 'val_prauc': _val_prauc,
                'train_acc': _train_acc, 'val_acc': _val_acc,
                'val_sensitivity': _val_sensitivity, 'val_precision': _val_precision,
                'val_specificity': _val_specificity, 'val_f1': _val_f1,
            }
            with open(metric_path, 'w') as f:
                json.dump(best_model_metric, f, indent=4)

            train_result = pd.DataFrame(
                {'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues}
            )
            train_result['image_path'] = train_result['image_path'].str.split('/').str[-1]
            train_result.to_csv(train_result_path, index=False)
            _val_result.to_csv(val_result_path, index=False)

        pd.DataFrame(history).to_csv(history_path, index=False)

    return best_model, best_val_f1


def _validate_image_only(model, criterion, val_loader, device):
    """Run one full pass over the validation loader and return all metrics."""
    model.eval()
    val_loss = []
    probs, preds, trues, paths = [], [], [], []

    with torch.no_grad():
        for path, images, labels in tqdm(iter(val_loader), desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)

            logit = model(images)
            loss = criterion(logit, labels)
            probability = torch.softmax(logit, dim=1)[:, 1]

            val_loss.append(loss.item())
            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
            paths.extend(path)

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
    val_result['image_path'] = val_result['image_path'].str.split('/').str[-1]

    return (_val_loss, _val_auc, _val_acc,
            _val_sensitivity, _val_precision,
            _val_specificity, _val_f1, val_result)
