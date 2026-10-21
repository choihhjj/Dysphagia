"""Inference with the five fold-specific models and their ensemble.

Input CSV: filename (+ QMD_* columns for feature-only / multimodal). Images are read from
<image_dir>/<filename> (600 x 268 swallowing images, `.jpg` is appended if missing).

Usage: python scripts/run_inference.py --csv data/testset_sample_qmd.csv
                                       [--model multimodal] [--checkpoint_dir checkpoints]

Samples are processed in batches of 8 by default, the same as the test-set inference of the training scripts.

Checkpoint layout (same as the training outputs):
    <checkpoint_dir>/image_only/fold{k}/best_model.pt, dataset_statistics.json
    <checkpoint_dir>/feature_only/fold{k}/best_model_feature_only.pt
    <checkpoint_dir>/multimodal/fold{k}/best_model_multimodal.pt, dataset_statistics.json
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs.config import CONFIG_DIR, load_config, seed_everything, device
from src.dataset import CustomDataset
from src.metrics import compute_metrics
from src.model import get_model, create_fc_model
from src.train_multimodal import extract_features, _build_enhanced_features


def image_loader(paths, img_size, stats_dir, batch_size):
    dataset = CustomDataset(paths, np.zeros(len(paths), dtype=int), img_size,
                            train=False, model_ckpt=stats_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def predict(fc_model, x, batch_size):
    probs = []
    fc_model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            logit = fc_model(torch.tensor(x[i:i + batch_size], dtype=torch.float32).to(device))
            probs.append(torch.softmax(logit, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs).astype(np.float64)   # saved like the training scripts (.tolist())


def predict_image_only(cfg, ckpt_dir, paths, batch_size):
    model, img_size = get_model(cfg['MODEL_NAME'], num_classes=cfg['NUM_CLASSES'], pt=False)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location=device))
    model.to(device).eval()

    probs = []
    with torch.no_grad():
        for _, images, _ in image_loader(paths, img_size, ckpt_dir, batch_size):
            probs.append(torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs).astype(np.float64)   # saved like the training scripts (.tolist())


def predict_feature_only(cfg, ckpt_dir, df, batch_size):
    hidden_sizes = ([] if cfg['HIDDEN_SIZE'] is None
                    else [min(cfg['HIDDEN_SIZE'], cfg['FIRST_HIDDEN'] // 2)])
    fc_model = create_fc_model(
        input_size=len(cfg['QUANTI_COLUMNS']),
        n_layers=cfg['N_LAYERS'],
        first_hidden=cfg['FIRST_HIDDEN'],
        hidden_sizes=hidden_sizes,
        use_dropout=cfg['USE_DROPOUT'],
        dropout_rate=cfg['DROPOUT_RATE'],
    ).to(device)
    fc_model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_model_feature_only.pt'),
                                        map_location=device))
    x = df[cfg['QUANTI_COLUMNS']].values.astype(np.float32)
    return predict(fc_model, x, batch_size)


def predict_multimodal(cfg, ckpt_dir, io_ckpt_dir, df, paths, batch_size):
    encoder, img_size = get_model(cfg['IMAGE_ENCODER'], num_classes=2, pt=False)
    loader = image_loader(paths, img_size, ckpt_dir, batch_size)
    feats, feat_paths, _ = extract_features(encoder, loader, io_ckpt_dir, device)
    x = _build_enhanced_features(feats, feat_paths, df.assign(img_dir=paths), cfg['QUANTI_COLUMNS'])

    fc_model = create_fc_model(
        input_size=cfg['IMAGE_FEATURE_DIM'] + len(cfg['QUANTI_COLUMNS']),
        n_layers=cfg['N_LAYERS'],
        first_hidden=cfg['FIRST_HIDDEN'],
        hidden_sizes=cfg['HIDDEN_SIZE'],
        use_dropout=cfg['USE_DROPOUT'],
        dropout_rate=cfg['DROPOUT_RATE'],
    ).to(device)
    fc_model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_model_multimodal.pt'),
                                        map_location=device))
    return predict(fc_model, x, batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--image_dir', default=None, help='default: directory of --csv')
    parser.add_argument('--model', default='multimodal',
                        choices=['image_only', 'feature_only', 'multimodal'])
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', default=None,
                        help='default: <csv without _qmd>_inference_<model>.csv')
    args = parser.parse_args()

    cfg = load_config(os.path.join(CONFIG_DIR, f'{args.model}.yaml'))
    seed_everything(cfg['SEED'])   # same cuDNN settings as training (deterministic, no benchmark)
    df = pd.read_csv(args.csv)
    image_dir = args.image_dir or os.path.dirname(args.csv)
    paths = np.array([
        os.path.join(image_dir, f if os.path.splitext(f)[1] else f'{f}.jpg')
        for f in df['filename'].astype(str)
    ])
    if args.model != 'feature_only':
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f'{len(missing)} images not found, e.g. {missing[0]}')

    out = df[['filename']].copy()
    probs, preds = {}, {}
    for k in range(1, args.n_folds + 1):
        ckpt_dir = os.path.join(args.checkpoint_dir, args.model, f'fold{k}')
        if args.model == 'image_only':
            prob = predict_image_only(cfg, ckpt_dir, paths, args.batch_size)
        elif args.model == 'feature_only':
            prob = predict_feature_only(cfg, ckpt_dir, df, args.batch_size)
        else:
            io_ckpt_dir = os.path.join(args.checkpoint_dir, 'image_only', f'fold{k}')
            prob = predict_multimodal(cfg, ckpt_dir, io_ckpt_dir, df, paths, args.batch_size)
        probs[f'prob_fold{k}'] = prob
        preds[f'pred_fold{k}'] = (prob > 0.5).astype(int)   # argmax of the 2-class softmax

    out = out.assign(**probs, **preds)
    out['prob_ensemble'] = out[list(probs)].mean(axis=1)
    out['pred_ensemble'] = (out['prob_ensemble'] >= args.threshold).astype(int)

    stem = os.path.splitext(args.csv)[0]
    stem = stem[:-len('_qmd')] if stem.endswith('_qmd') else stem
    output = args.output or f'{stem}_inference_{args.model}.csv'
    out.to_csv(output, index=False)
    print(f'[{args.model}] {len(out)} samples -> {output}')

    if 'label' in df.columns:
        metrics = compute_metrics(df['label'], out['prob_ensemble'], out['pred_ensemble'])
        print('ensemble metrics:', {k: round(v, 4) for k, v in metrics.items()})


if __name__ == '__main__':
    main()
