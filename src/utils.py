"""Class weight utilities using complement-frequency weighting: w_c = 1 - n_c/N."""

import torch


def calculate_class_weight_from_loader(train_loader, num_classes=2):
    """Compute per-class complement-frequency weights from a DataLoader; returns (list, tensor)."""
    class_counts = [0] * num_classes
    total = 0
    for _, _, labels in train_loader:
        for label in labels.view(-1):
            class_counts[label.item()] += 1
            total += 1
    weights = [1 - (count / total) for count in class_counts]
    return weights, torch.FloatTensor(weights)


def calculate_class_weight_from_df(train_df, num_classes=2):
    """Compute per-class complement-frequency weights from a DataFrame label column; returns (list, tensor)."""
    class_counts = (
        train_df['label']
        .value_counts()
        .sort_index()
        .reindex(range(num_classes), fill_value=0)
    )
    total = class_counts.sum()
    weights = [1 - (class_counts[i] / total) for i in range(num_classes)]
    return weights, torch.FloatTensor(weights)
