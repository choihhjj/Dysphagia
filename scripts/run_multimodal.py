"""5-fold cross-validation for the multimodal model.

Requires run_image_only.py to be run first.
Usage: python scripts/run_multimodal.py
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
from src.model import get_model, create_fc_model
from src.train_multimodal import extract_features, train_multimodal
from src.test import inference_multimodal
from src.cv_summary import save_cv_summary
from src.cv_ensemble import save_ensemble

# ── paths (edit before running) ──────────────────────────────────────────────
DATASET_CSV        = 'path/to/final_dataset.csv'   # columns: filename, img_dir, label, fold1-fold5
OUTPUT_BASE        = 'outputs/multimodal'
IO_CHECKPOINT_BASE = 'outputs/image_only'          # directory containing fold1…fold5 from run_image_only.py
# ─────────────────────────────────────────────────────────────────────────────

QUANTI_COLUMNS = [
    'QMD_sup2', 'QMD_sup1', 'QMD_ref',
    'QMD_inf1', 'QMD_inf2', 'QMD_inf3',
    'QMD_inf4', 'QMD_inf5', 'QMD_inf6', 'QMD_inf7',
]
IMAGE_FEATURE_DIM = 512   # ResNet18 penultimate-layer dimension

CFG = {
    'EPOCHS':           50,
    'LEARNING_RATE':    1e-5,
    'BATCH_SIZE':       8,
    'WEIGHT_DECAY':     1e-3,
    'LABEL_SMOOTHING':  0.08,
    'USE_CLASS_WEIGHT': True,
    'SEED':             42,
    'IMAGE_ENCODER':    'resnet18',
    'N_LAYERS':         2,
    'FIRST_HIDDEN':     384,
    'HIDDEN_SIZE':      [32],
    'USE_DROPOUT':      True,
    'DROPOUT_RATE':     0.1,
}


def run_fold(fold):
    """Train and evaluate the multimodal model for a single fold."""
    model_ckpt = os.path.join(OUTPUT_BASE, f'fold{fold}')
    io_ckpt    = os.path.join(IO_CHECKPOINT_BASE, f'fold{fold}')
    os.makedirs(model_ckpt, exist_ok=True)

    seed_everything(CFG['SEED'])

    filelist = pd.read_csv(DATASET_CSV).sort_values('filename').reset_index(drop=True)
    train_df = filelist[filelist[f'fold{fold}'] == 'train'].reset_index(drop=True)
    val_df   = filelist[filelist[f'fold{fold}'] == 'val'].reset_index(drop=True)
    test_df  = filelist[filelist[f'fold{fold}'] == 'test'].reset_index(drop=True)

    # load pretrained image encoder for this fold
    model, img_size = get_model(CFG['IMAGE_ENCODER'], num_classes=2)
    model.to(device)

    # training split: compute and save per-channel normalization stats
    train_dataset = CustomDataset(
        train_df['img_dir'].values, train_df['label'].values,
        img_size, train=True, model_ckpt=model_ckpt,
    )
    val_dataset = CustomDataset(
        val_df['img_dir'].values, val_df['label'].values,
        img_size, train=False, model_ckpt=model_ckpt,
    )
    test_dataset = CustomDataset(
        test_df['img_dir'].values, test_df['label'].values,
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
    test_loader = DataLoader(
        test_dataset, batch_size=CFG['BATCH_SIZE'],
        shuffle=False, num_workers=0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(CFG['SEED']),
    )

    fold_cfg = {**CFG, 'MODEL_CKPT': model_ckpt, 'IO_CKPT': io_ckpt,
                'QUANTI_COLUMNS': QUANTI_COLUMNS}
    with open(os.path.join(model_ckpt, 'config.json'), 'w') as f:
        json.dump(fold_cfg, f, indent=4)

    # extract fixed image features using the pretrained image-only encoder
    train_feats, train_paths, train_labels = extract_features(
        model, train_loader, io_ckpt, device
    )
    val_feats,   val_paths,   val_labels   = extract_features(
        model, val_loader, io_ckpt, device
    )
    test_feats,  test_paths,  _            = extract_features(
        model, test_loader, io_ckpt, device
    )

    seed_everything(CFG['SEED'])   # re-seed before MLP weight init

    fc_model = create_fc_model(
        input_size=IMAGE_FEATURE_DIM + len(QUANTI_COLUMNS),
        n_layers=CFG['N_LAYERS'],
        first_hidden=CFG['FIRST_HIDDEN'],
        hidden_sizes=CFG['HIDDEN_SIZE'],
        use_dropout=CFG['USE_DROPOUT'],
        dropout_rate=CFG['DROPOUT_RATE'],
    ).to(device)

    optimizer = optim.Adam(fc_model.parameters(),
                           lr=CFG['LEARNING_RATE'], weight_decay=CFG['WEIGHT_DECAY'])

    best_model, best_val_f1 = train_multimodal(
        fc_model=fc_model,
        train_loader=train_loader,
        train_features=train_feats, train_paths=train_paths,
        train_labels=train_labels,  train_df=train_df,
        val_features=val_feats,     val_paths=val_paths,
        val_labels=val_labels,      val_df=val_df,
        num_epochs=CFG['EPOCHS'],
        batch_size=CFG['BATCH_SIZE'],
        optimizer=optimizer,
        scheduler=None,
        label_smoothing=CFG['LABEL_SMOOTHING'],
        model_ckpt=model_ckpt,
        device=device,
    )

    inference_multimodal(
        best_model, test_feats, test_paths, test_df,
        device, model_ckpt, CFG['BATCH_SIZE'], mode='test',
    )


if __name__ == '__main__':
    for fold in range(1, 6):
        print(f'\n{"="*40}\nFold {fold}/5\n{"="*40}')
        run_fold(fold)

    save_cv_summary('multimodal', OUTPUT_BASE)
    save_ensemble(OUTPUT_BASE)
    