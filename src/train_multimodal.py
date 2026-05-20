"""Training loop for the multimodal model (image features + quantitative measurements).

run_image_only.py must be run first; its checkpoint is used to extract image features.
"""

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


QUANTI_COLUMNS = [
    'QMD_sup2', 'QMD_sup1', 'QMD_ref',
    'QMD_inf1', 'QMD_inf2', 'QMD_inf3',
    'QMD_inf4', 'QMD_inf5', 'QMD_inf6', 'QMD_inf7',
]


def extract_features(model, data_loader, checkpoint_path, device):
    """Extract 512-dim feature vectors from the pretrained encoder (head removed); returns (features, paths, labels)."""
    model.load_state_dict(torch.load(checkpoint_path + '/best_model.pt',
                                     map_location=device))
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)

    features_list, paths_list, labels_list = [], [], []

    with torch.no_grad():
        for path, images, labels in tqdm(iter(data_loader), desc='Extracting features'):
            images = images.to(device)
            feats = feature_extractor(images)
            feats = feats.squeeze(-1).squeeze(-1)  # (B, 512, 1, 1) -> (B, 512)
            features_list.append(feats.cpu())
            paths_list.extend(path)
            labels_list.extend(labels.numpy())

    return torch.cat(features_list, dim=0), paths_list, labels_list


def _build_enhanced_features(image_features, paths_list, df):
    """Concatenate 512-dim image features with quantitative measurements; returns float32 array of shape (N, 522)."""
    assert len(image_features) == len(paths_list), (
        f"image_features length ({len(image_features)}) != paths_list length ({len(paths_list)})"
    )
    quanti_lookup = df.set_index('img_dir')[QUANTI_COLUMNS]
    enhanced = []
    for feat, path in zip(image_features, paths_list):
        quanti = quanti_lookup.loc[path].values.astype(np.float32)
        enhanced.append(np.concatenate([feat.numpy(), quanti]))
    return np.array(enhanced, dtype=np.float32)


def train_multimodal(fc_model,
                     train_loader, train_features, train_paths, train_labels, train_df,
                     val_features, val_paths, val_labels, val_df,
                     num_epochs, batch_size, optimizer, scheduler, label_smoothing,
                     model_ckpt, device):
    """Train the multimodal MLP; returns (best_model, best_val_f1)."""
    fc_model.to(device)

    train_labels_arr = np.array(train_labels)
    val_labels_arr = np.array(val_labels)

    enhanced_train = _build_enhanced_features(train_features, train_paths, train_df)
    enhanced_val = _build_enhanced_features(val_features, val_paths, val_df)

    cl_list, class_weight = calculate_class_weight_from_loader(train_loader)
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
        perm = np.random.permutation(len(enhanced_train))

        for i in range(0, len(enhanced_train), batch_size):
            batch_idx = perm[i:min(i + batch_size, len(enhanced_train))]
            batch_X = torch.tensor(
                enhanced_train[batch_idx], dtype=torch.float32
            ).to(device)
            batch_y = torch.tensor(
                train_labels_arr[batch_idx], dtype=torch.long
            ).to(device)

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
            paths += [train_paths[idx] for idx in batch_idx]

        _train_loss = np.mean(train_loss)
        _train_auc = roc_auc_score(trues, probs)
        _train_prauc = average_precision_score(trues, probs)
        _train_acc = accuracy_score(trues, preds)

        (_val_loss, _val_auc, _val_acc,
         _val_sensitivity, _val_precision,
         _val_specificity, _val_f1, _val_result) = _validate_multimodal(
            fc_model, criterion, enhanced_val, val_labels_arr, val_paths, device, batch_size
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
            best_model = copy.deepcopy(fc_model)

            torch.save(fc_model.state_dict(),
                       os.path.join(model_ckpt, 'best_model_multimodal.pt'))

            best_metric = {
                'epoch': epoch,
                'train_loss': _train_loss, 'val_loss': _val_loss,
                'train_auc': _train_auc, 'val_auc': _val_auc,
                'train_prauc': _train_prauc, 'val_prauc': _val_prauc,
                'train_acc': _train_acc, 'val_acc': _val_acc,
                'val_sensitivity': _val_sensitivity, 'val_precision': _val_precision,
                'val_specificity': _val_specificity, 'val_f1': _val_f1,
            }
            with open(os.path.join(model_ckpt, 'best_model_metric_multimodal.json'), 'w') as f:
                json.dump(best_metric, f, indent=4)

            train_result = pd.DataFrame(
                {'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues}
            )
            train_result['image_path'] = [p.split('/')[-1] for p in train_result['image_path']]
            train_result.to_csv(os.path.join(model_ckpt, 'results_train.csv'), index=False)
            _val_result.to_csv(os.path.join(model_ckpt, 'results_val.csv'), index=False)

        pd.DataFrame(history).to_csv(
            os.path.join(model_ckpt, 'history_multimodal.csv'), index=False
        )

    return best_model, best_val_f1


def _validate_multimodal(fc_model, criterion, val_features, val_labels, val_paths,
                          device, batch_size):
    """Run one full pass over the validation set and return all metrics."""
    fc_model.eval()
    val_loss = []
    probs, preds, trues, paths = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(val_features), batch_size):
            end = min(i + batch_size, len(val_features))
            batch_X = torch.tensor(val_features[i:end], dtype=torch.float32).to(device)
            batch_y = torch.tensor(val_labels[i:end], dtype=torch.long).to(device)

            logit = fc_model(batch_X)
            loss = criterion(logit, batch_y)
            probability = torch.softmax(logit, dim=1)[:, 1]

            val_loss.append(loss.item())
            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += batch_y.detach().cpu().numpy().tolist()
            paths += [val_paths[j] for j in range(i, end)]

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
