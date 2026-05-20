"""5-fold cross-validation for the image-only CNN classifier.

Usage: python scripts/run_image_only.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.config import seed_everything, seed_worker, device
from src.dataset import CustomDataset
from src.model import get_model
from src.train_image_only import train
from src.test import inference_image_only
from src.cv_summary import save_cv_summary
from src.cv_ensemble import save_ensemble

# ── paths (edit before running) ──────────────────────────────────────────────
DATASET_CSV = 'path/to/final_dataset.csv'   # must have columns: filename, img_dir, label, fold1-fold5
OUTPUT_BASE = 'outputs/image_only'
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    'EPOCHS':           500,
    'LEARNING_RATE':    1e-5,
    'BATCH_SIZE':       8,
    'WEIGHT_DECAY':     1e-3,
    'LABEL_SMOOTHING':  0.08,
    'USE_CLASS_WEIGHT': True,
    'SEED':             42,
    'MODEL_NAME':       'resnet18',
    'MODEL_PT':         True,
}


def run_fold(fold):
    """Train and evaluate the image-only model for a single fold."""
    model_ckpt = os.path.join(OUTPUT_BASE, f'fold{fold}')
    os.makedirs(model_ckpt, exist_ok=True)

    seed_everything(CFG['SEED'])

    filelist = pd.read_csv(DATASET_CSV).sort_values('filename').reset_index(drop=True)
    train_df = filelist[filelist[f'fold{fold}'] == 'train'].reset_index(drop=True)
    val_df   = filelist[filelist[f'fold{fold}'] == 'val'].reset_index(drop=True)
    test_df  = filelist[filelist[f'fold{fold}'] == 'test'].reset_index(drop=True)

    model, img_size = get_model(
        model_name=CFG['MODEL_NAME'],
        num_classes=2,
        pt=CFG['MODEL_PT'],
    )
    print(f'[fold {fold}] model={model.__class__.__name__}, img_size={img_size}')

    # training split: compute and save per-channel normalization stats
    train_dataset = CustomDataset(
        train_df['img_dir'].values, train_df['label'].values,
        img_size, train=True, model_ckpt=model_ckpt,
    )
    val_dataset = CustomDataset(
        val_df['img_dir'].values, val_df['label'].values,
        img_size, train=False, model_ckpt=model_ckpt,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CFG['BATCH_SIZE'],
        shuffle=True, drop_last=True, num_workers=0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(CFG['SEED']),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG['BATCH_SIZE'],
        shuffle=False, num_workers=0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(CFG['SEED']),
    )

    fold_cfg = {**CFG, 'MODEL_CKPT': model_ckpt, 'IMG_SIZE': img_size}
    with open(os.path.join(model_ckpt, 'config.json'), 'w') as f:
        json.dump(fold_cfg, f, indent=4)

    model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=CFG['LEARNING_RATE'], weight_decay=CFG['WEIGHT_DECAY'])

    best_model, best_val_f1 = train(
        model, CFG['EPOCHS'], optimizer, train_loader, val_loader,
        None, CFG['LABEL_SMOOTHING'], device, model_ckpt,
    )

    # build test DataLoader after training so it uses the saved normalization stats
    test_dataset = CustomDataset(
        test_df['img_dir'].values, test_df['label'].values,
        img_size, train=False, model_ckpt=model_ckpt,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CFG['BATCH_SIZE'],
        shuffle=False, num_workers=0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(CFG['SEED']),
    )

    inference_image_only(best_model, test_loader, device, model_ckpt, mode='test')


if __name__ == '__main__':
    for fold in range(1, 6):
        print(f'\n{"="*40}\nFold {fold}/5\n{"="*40}')
        run_fold(fold)

    save_cv_summary('image_only', OUTPUT_BASE)
    save_ensemble(OUTPUT_BASE)
