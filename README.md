# Deep Learning-Based Classification of Dysphagia Severity Using M-Mode Ultrasound Imaging

Binary classification of ultrasound images using image-only, feature-only, and multimodal fusion models, evaluated with 5-fold cross-validation.

## Data Format

Prepare a single CSV containing:
- `filename`, `img_dir`, `label` (0 or 1)
- `fold1` ~ `fold5`: split assignment per fold (`train`, `val`, or `test`)
- `QMD_*` columns: quantitative measurements (for feature-only and multimodal)

## Usage

Set `DATASET_CSV` and `OUTPUT_BASE` at the top of each script, then run from the project root.

```bash
# Image-only (ResNet18)
python scripts/run_image_only.py

# Feature-only (MLP on QMD columns)
python scripts/run_feature_only.py

# Multimodal (frozen ResNet18 + QMD, requires image-only checkpoints)
# Set IO_CHECKPOINT_BASE to the image-only output directory.
python scripts/run_multimodal.py
```

CV summary and ensemble outputs are generated automatically after all folds finish. To regenerate them separately:

```bash
python scripts/cv_summary.py
python scripts/cv_ensemble.py
```

## Repository Structure

```
.
├── configs/
│   └── config.py
├── scripts/
│   ├── run_image_only.py
│   ├── run_feature_only.py
│   ├── run_multimodal.py
│   ├── cv_summary.py
│   └── cv_ensemble.py
└── src/
    ├── dataset.py
    ├── model.py
    ├── train_image_only.py
    ├── train_feature_only.py
    ├── train_multimodal.py
    ├── test.py
    ├── cv_summary.py
    ├── cv_ensemble.py
    └── utils.py
```

## Requirements

* Python 3.8+
* PyTorch 1.12+
* torchvision, scikit-learn, pandas, numpy, opencv-python, tqdm

## Citation

```
TBD
```
