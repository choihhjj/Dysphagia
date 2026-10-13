"""Crop the swallowing window from raw M-mode images.

Usage:
    # all images listed in ANNOTATION_CSV
    python scripts/run_swallowing_extraction.py [--config configs/swallowing.yaml]

    # a single image
    python scripts/run_swallowing_extraction.py --image path/to/image.jpg --start 150 --end 400
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import cv2
import pandas as pd
from tqdm import tqdm

from configs.config import CONFIG_DIR, load_config


def crop_swallowing(img, start, cfg):
    """Return (600 x 334 image for QMD, 600 x 268 image for the CNN)."""
    cut_start = start - cfg['PRE_ONSET_PX']
    cut_end = cut_start + cfg['WINDOW_PX']
    if cut_start < 0 or cut_end > img.shape[1]:
        raise ValueError(f'crop window [{cut_start}, {cut_end}) outside image width {img.shape[1]}')

    y0, y1 = cfg['Y_RANGE']
    img_cut = img[y0:y1, cut_start:cut_end, :]
    upper = int(img_cut.shape[0] * cfg['UPPER_CUT_RATIO'])
    return img_cut, img_cut[upper:, :, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(CONFIG_DIR, 'swallowing.yaml'))
    parser.add_argument('--image', default=None, help='single raw image (instead of ANNOTATION_CSV)')
    parser.add_argument('--start', type=int, default=None, help='swallowing onset x (px), with --image')
    parser.add_argument('--end', type=int, default=None, help='swallowing offset x (px), with --image')
    parser.add_argument('--output_dir', default=None, help='override OUTPUT_DIR')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg['OUTPUT_DIR'] = args.output_dir

    if args.image is not None:
        if args.start is None or args.end is None:
            parser.error('--image requires --start and --end')
        filename = os.path.splitext(os.path.basename(args.image))[0]
        jobs = [(filename, args.image, args.start, args.end)]
    else:
        ann = pd.read_csv(cfg['ANNOTATION_CSV'])
        jobs = [(str(r.filename), os.path.join(cfg['RAW_DIR'], f'{r.filename}.jpg'),
                 int(r.swallowing_start), int(r.swallowing_end))
                for r in ann.itertuples(index=False)]

    cut_dir      = os.path.join(cfg['OUTPUT_DIR'], 'cut')
    uppercut_dir = os.path.join(cfg['OUTPUT_DIR'], 'uppercut')
    os.makedirs(cut_dir, exist_ok=True)
    os.makedirs(uppercut_dir, exist_ok=True)

    rows = []
    for filename, img_path, start, end in tqdm(jobs):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        try:
            img_cut, img_uppercut = crop_swallowing(img, start, cfg)
        except ValueError as e:
            raise ValueError(f'{filename}: {e}') from None

        uppercut_path = os.path.join(uppercut_dir, f'{filename}.jpg')
        cv2.imwrite(os.path.join(cut_dir, f'{filename}.jpg'), img_cut)
        cv2.imwrite(uppercut_path, img_uppercut)
        rows.append({
            'filename': filename,
            'image_dura': end - start,
            'img_dir': uppercut_path.replace(os.sep, '/'),
        })

    rows = pd.DataFrame(rows)
    rows[['filename', 'image_dura']].to_csv(
        os.path.join(cfg['OUTPUT_DIR'], 'qmd_input.csv'), index=False)
    rows[['filename', 'img_dir']].to_csv(
        os.path.join(cfg['OUTPUT_DIR'], 'image_list.csv'), index=False)

    print(f'{len(rows)} images -> {cfg["OUTPUT_DIR"]}')


if __name__ == '__main__':
    main()
