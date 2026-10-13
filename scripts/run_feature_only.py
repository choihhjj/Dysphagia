"""5-fold cross-validation for the feature-only (tabular features) MLP.

Usage: python scripts/run_feature_only.py [--config configs/feature_only.yaml]
                                          [--dataset_csv PATH] [--output_base DIR]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import torch.optim as optim

from configs.config import seed_everything, device, parse_config
from src.model import create_fc_model
from src.train_feature_only import train_feature_only
from src.test import inference_feature_only
from src.cv_summary import save_cv_summary
from src.cv_ensemble import save_ensemble

# Hyperparameters, QMD columns and paths are defined in configs/feature_only.yaml.
CFG = None   # loaded in __main__


def run_fold(fold):
    """Train and evaluate the feature-only MLP for a single fold."""
    model_ckpt = os.path.join(CFG['OUTPUT_BASE'], f'fold{fold}')
    os.makedirs(model_ckpt, exist_ok=True)

    seed_everything(CFG['SEED'])

    quanti_columns = CFG['QUANTI_COLUMNS']
    filelist = pd.read_csv(CFG['DATASET_CSV']).sort_values('filename').reset_index(drop=True)
    train_df = filelist[filelist[f'fold{fold}'] == 'train'].reset_index(drop=True)
    val_df   = filelist[filelist[f'fold{fold}'] == 'val'].reset_index(drop=True)
    test_df  = filelist[filelist[f'fold{fold}'] == 'test'].reset_index(drop=True)

    # clamp intermediate hidden size to at most half of first_hidden
    hidden_sizes = ([] if CFG['HIDDEN_SIZE'] is None
                    else [min(CFG['HIDDEN_SIZE'], CFG['FIRST_HIDDEN'] // 2)])

    fc_model = create_fc_model(
        input_size=len(quanti_columns),
        n_layers=CFG['N_LAYERS'],
        first_hidden=CFG['FIRST_HIDDEN'],
        hidden_sizes=hidden_sizes,
        use_dropout=CFG['USE_DROPOUT'],
        dropout_rate=CFG['DROPOUT_RATE'],
    ).to(device)

    optimizer = optim.Adam(fc_model.parameters(),
                           lr=CFG['LEARNING_RATE'], weight_decay=CFG['WEIGHT_DECAY'])

    fold_cfg = {**CFG, 'MODEL_CKPT': model_ckpt}
    with open(os.path.join(model_ckpt, 'config.json'), 'w') as f:
        json.dump(fold_cfg, f, indent=4)

    best_model, best_val_f1 = train_feature_only(
        fc_model=fc_model,
        quanti_columns=quanti_columns,
        train_df=train_df,
        val_df=val_df,
        num_epochs=CFG['EPOCHS'],
        batch_size=CFG['BATCH_SIZE'],
        optimizer=optimizer,
        label_smoothing=CFG['LABEL_SMOOTHING'],
        model_ckpt=model_ckpt,
        device=device,
    )

    X_test    = test_df[quanti_columns].values.astype(np.float32)
    y_test    = test_df['label'].values
    path_test = test_df['img_dir'].values

    inference_feature_only(
        fc_model=best_model,
        X_test=X_test,
        y_test=y_test,
        path_test=path_test,
        device=device,
        model_ckpt=model_ckpt,
        batch_size=CFG['BATCH_SIZE'],
        mode='test',
    )


if __name__ == '__main__':
    CFG = parse_config('feature_only.yaml', __doc__)
    n_folds = CFG['N_FOLDS']
    for fold in range(1, n_folds + 1):
        print(f'\n{"="*40}\nFold {fold}/{n_folds}\n{"="*40}')
        run_fold(fold)

    save_cv_summary('feature_only', CFG['OUTPUT_BASE'], n_folds)
    save_ensemble(CFG['OUTPUT_BASE'], n_folds)
