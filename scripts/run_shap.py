"""SHAP analysis of the trained multimodal model for one fold (test set).

Requires run_image_only.py and run_multimodal.py to be run first.
Usage: python scripts/run_shap.py [--config configs/multimodal.yaml] [--fold 2] [--output_dir outputs/shap]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs.config import CONFIG_DIR, load_config, seed_worker, device
from src.dataset import CustomDataset
from src.figures import set_style
from src.model import get_model, create_fc_model
from src.train_multimodal import extract_features, _build_enhanced_features
from src.shap_analysis import shap_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(CONFIG_DIR, 'multimodal.yaml'))
    parser.add_argument('--dataset_csv', default=None)
    parser.add_argument('--output_base', default=None, help='multimodal output directory')
    parser.add_argument('--io_checkpoint_base', default=None)
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--background', type=int, default=100)
    parser.add_argument('--output_dir', default='outputs/shap')
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key, value in [('DATASET_CSV', args.dataset_csv),
                       ('OUTPUT_BASE', args.output_base),
                       ('IO_CHECKPOINT_BASE', args.io_checkpoint_base)]:
        if value is not None:
            cfg[key] = value

    fold = args.fold
    model_ckpt = os.path.join(cfg['OUTPUT_BASE'], f'fold{fold}')
    io_ckpt = os.path.join(cfg['IO_CHECKPOINT_BASE'], f'fold{fold}')

    filelist = pd.read_csv(cfg['DATASET_CSV']).sort_values('filename').reset_index(drop=True)
    train_df = filelist[filelist[f'fold{fold}'] == 'train'].reset_index(drop=True)
    test_df  = filelist[filelist[f'fold{fold}'] == 'test'].reset_index(drop=True)

    model, img_size = get_model(cfg['IMAGE_ENCODER'], num_classes=2)

    # same loaders as in training; normalization stats are loaded from the multimodal fold directory
    def make_loader(df, train):
        dataset = CustomDataset(df['img_dir'].values, df['label'].values, img_size,
                                train=False, model_ckpt=model_ckpt)
        return DataLoader(dataset, batch_size=cfg['BATCH_SIZE'],
                          shuffle=train, drop_last=train, num_workers=cfg['NUM_WORKERS'],
                          worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(cfg['SEED']))

    train_feats, train_paths, _ = extract_features(model, make_loader(train_df, True), io_ckpt, device)
    test_feats, test_paths, _ = extract_features(model, make_loader(test_df, False), io_ckpt, device)
    train_x = _build_enhanced_features(train_feats, train_paths, train_df, cfg['QUANTI_COLUMNS'])
    test_x = _build_enhanced_features(test_feats, test_paths, test_df, cfg['QUANTI_COLUMNS'])

    fc_model = create_fc_model(
        input_size=cfg['IMAGE_FEATURE_DIM'] + len(cfg['QUANTI_COLUMNS']),
        n_layers=cfg['N_LAYERS'],
        first_hidden=cfg['FIRST_HIDDEN'],
        hidden_sizes=cfg['HIDDEN_SIZE'],
        use_dropout=cfg['USE_DROPOUT'],
        dropout_rate=cfg['DROPOUT_RATE'],
    ).to(device)
    fc_model.load_state_dict(torch.load(os.path.join(model_ckpt, 'best_model_multimodal.pt'),
                                        map_location=device))

    set_style()
    shap_analysis(fc_model, train_x, test_x, cfg['QUANTI_COLUMNS'],
                  background_samples=args.background, device=device,
                  save_path=args.output_dir, suffix=f'_fold{fold}', seed=cfg['SEED'])
    print(f'saved -> {args.output_dir}')


if __name__ == '__main__':
    main()
