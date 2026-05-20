"""Dataset with per-channel normalization anchored to training split statistics."""

import numpy as np
import cv2
import os
import json
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    """Image dataset with lazy loading and per-channel normalization.

    If label_list is None, __getitem__ returns (path, image) without a label.
    Set train=True on the training split to compute and save normalization stats.
    """

    def __init__(self, image_path_list, label_list, input_size, train=False, model_ckpt=None):
        self.image_path_list = image_path_list
        self.label_list = label_list
        self.input_size = input_size

        if train:
            self.mean, self.std = self._calculate_dataset_statistics(model_ckpt)
        else:
            self.mean, self.std = self._load_dataset_statistics(model_ckpt)

    def _calculate_dataset_statistics(self, model_ckpt=None):
        """Compute per-channel mean and std over the training set and save them."""
        means, stds = [], []
        for path in tqdm(self.image_path_list, desc='Computing dataset statistics'):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_size, self.input_size))
            img = img.astype(np.float32) / 255.0
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))

        mean = np.mean(means, axis=0)
        std = np.mean(stds, axis=0)

        if model_ckpt is not None:
            stats = {'mean': mean.tolist(), 'std': std.tolist(), 'input_size': self.input_size}
            with open(os.path.join(model_ckpt, 'dataset_statistics.json'), 'w') as f:
                json.dump(stats, f, indent=4)

        return mean, std

    def _load_dataset_statistics(self, model_ckpt):
        """Load training-set statistics to normalise non-training splits."""
        with open(os.path.join(model_ckpt, 'dataset_statistics.json'), 'r') as f:
            stats = json.load(f)
        return np.array(stats['mean']), np.array(stats['std'])

    def _get_image(self, path):
        """Read, resize, and normalise a single image to a CHW tensor."""
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = transforms.Normalize(mean=self.mean, std=self.std)(img)
        return img

    def __getitem__(self, index):
        image = self._get_image(self.image_path_list[index])
        if self.label_list is not None:
            return self.image_path_list[index], image, self.label_list[index]
        return self.image_path_list[index], image

    def __len__(self):
        return len(self.image_path_list)
