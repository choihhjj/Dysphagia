"""Batch QMD extraction from the cropped (334 x 600) swallowing images.

Usage: python scripts/run_qmd.py [--config configs/qmd.yaml]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib
matplotlib.use('Agg')
import pandas as pd
from tqdm import tqdm

from configs.config import CONFIG_DIR, load_config
from qmd.qmd import process_muscle_layer, get_qmd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=os.path.join(CONFIG_DIR, 'qmd.yaml'),
                        help='YAML config file (default: configs/qmd.yaml)')
    parser.add_argument('--input_csv', default=None, help='override INPUT_CSV')
    parser.add_argument('--data_dir', default=None, help='override DATA_DIR')
    parser.add_argument('--output_csv', default=None, help='override OUTPUT_CSV')
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key, value in [('INPUT_CSV', args.input_csv),
                       ('DATA_DIR', args.data_dir),
                       ('OUTPUT_CSV', args.output_csv)]:
        if value is not None:
            cfg[key] = value
    return cfg


def main():
    cfg = parse_args()
    input_df = pd.read_csv(cfg['INPUT_CSV'])

    rows = []
    for filename, image_dura in tqdm(zip(input_df['filename'], input_df['image_dura']),
                                     total=len(input_df), desc='QMD extraction'):
        filename = str(filename)
        process_muscle_layer(filename, data_dir=cfg['DATA_DIR'])
        qmd = get_qmd(filename, image_dura, data_dir=cfg['DATA_DIR'], show=False)
        rows.append({'filename': filename, **qmd})

    out_dir = os.path.dirname(cfg['OUTPUT_CSV'])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(cfg['OUTPUT_CSV'], index=False)
    print(f'[qmd] Saved {len(rows)} rows -> {cfg["OUTPUT_CSV"]}')


if __name__ == '__main__':
    main()
