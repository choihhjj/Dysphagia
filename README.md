# Deep learning-based classification of dysphagia severity using M-mode ultrasound imaging

Binary classification of dysphagia severity from M-mode ultrasound images using image-only,
feature-only (QMD), and multimodal models with 5-fold cross-validation.

## Environment

- Python 3.11
- PyTorch 2.0.1 / torchvision 0.15.2 (CUDA 11.7)
- NVIDIA GeForce RTX 3090
- Package versions: `requirements.txt`

```bash
conda create -n dysphagia python=3.11 -y
conda activate dysphagia
pip install -r requirements.txt
pip install --no-deps opencv-python==4.12.0.88
```

`opencv-python` is installed with `--no-deps` because 4.12.0.88 declares `numpy>=2`, while PyTorch 2.0.1
requires NumPy 1.x. This matches the environment used in the study.

Results may differ slightly on other GPUs or CUDA versions despite fixed seeds.

All commands are run from the repository root.

## Pipeline

### 1. Swallowing process extraction

```bash
python scripts/run_swallowing_extraction.py --config configs/swallowing.yaml

# a single image
python scripts/run_swallowing_extraction.py --image path/to/image.jpg --start <onset_x> --end <offset_x>
```

Input: raw M-mode images and an annotation CSV with the x coordinates (px) of the swallowing onset
and offset marked by an expert sonographer.

| Column | Description |
|---|---|
| `filename` | image name without extension (`<RAW_DIR>/<filename>.jpg`) |
| `swallowing_start` | swallowing onset (x, px) |
| `swallowing_end` | swallowing offset (x, px) |

Each image is cropped to a 10-s window starting 1 s before the onset (60 px = 1 s) and to rows
10–343 (600 × 334). The upper 1/5 is then removed (600 × 268).

| Output | Size | Used as |
|---|---|---|
| `cut/<filename>.jpg` | 600 × 334 | QMD input |
| `uppercut/<filename>.jpg` | 600 × 268 | CNN input |
| `qmd_input.csv` | | `filename`, `image_dura` (= offset − onset, px) |
| `image_list.csv` | | `filename`, `img_dir` |

The annotation CSV should contain only images that passed quality control (complete swallowing
process, preparation time ≥ 1 s, swallowing duration ≤ 9 s).

### 2. QMD parameter extraction

```bash
python scripts/run_qmd.py --config configs/qmd.yaml
```

`qmd/qmd.py` identifies the suprahyoid muscle region (`get_roi_simple`) and tracks the muscle boundary
bidirectionally (`process_muscle_layer`) on the 600 × 334 image. The boundary mask is then cropped by
the upper 1/5 to the 600 × 268 region, where the 10 QMD parameters are measured within the swallowing
window (`get_qmd`). See `qmd/qmd_structure.md` for the function hierarchy.

Output: `qmd_features.csv` (`filename`, `QMD_sup2`, `QMD_sup1`, `QMD_ref`, `QMD_inf1`–`QMD_inf7`).

### 3. Dataset CSV

Merge `image_list.csv`, `qmd_features.csv`, labels and fold assignments on `filename`
(one row per image):

| Column | Description |
|---|---|
| `filename`, `img_dir` | from step 1 |
| `patient_id` | patient identifier (patient-level bootstrap) |
| `label` | 0 = mild, 1 = severe |
| `fold1`–`fold5` | `train`, `val`, or `test` for each fold |
| `QMD_*` | from step 2 |

- The dataset is split into training and test sets (7:3) at the patient level with stratified sampling,
  so all examinations of a patient belong to the same set. The test set is held out and shared by all
  folds; the training set is further split into 5 folds for cross-validation.
- All images are used for training, whereas only the first image per patient-examination date is used
  for validation and test.
- The anonymized split assignment used in the study is provided as a supplementary table of the paper.

### 4. Training and evaluation

```bash
python scripts/run_image_only.py   --dataset_csv path/to/final_dataset.csv
python scripts/run_feature_only.py --dataset_csv path/to/final_dataset.csv
python scripts/run_multimodal.py   --dataset_csv path/to/final_dataset.csv --io_checkpoint_base outputs/image_only
```

The multimodal model uses the image-only checkpoint of the same fold as a frozen feature extractor,
so `run_image_only.py` must be run first.

Images are resized to 224 × 224 by bilinear interpolation (no cropping or padding), scaled to [0, 1],
and normalized with the per-channel mean/SD of the training set of each fold. No augmentation is used.

The image encoder can be changed with `MODEL_NAME` in `configs/image_only.yaml` (`resnet18`,
`mobilenet_v2`, `efficientnet_b0`, `densenet121`, `inceptionv3`); ResNet-18 was selected for the study.

Each script trains all folds and then writes the CV summary and the 5-fold ensemble results on the
test set. They can also be regenerated with `scripts/cv_summary.py` and `scripts/cv_ensemble.py`.

### 5. Bootstrap, figures, and SHAP

```bash
# patient-level paired bootstrap (1,000 iterations, 95% CI, p-value) of the ensemble predictions
python scripts/run_bootstrap.py --dataset_csv path/to/final_dataset.csv --model1 outputs/image_only --model2 outputs/multimodal

# ROC curves (mean ± SD over folds) + confusion matrices (ensemble predictions)
python scripts/plot_roc_cm.py

# SHAP analysis of the multimodal model
python scripts/run_shap.py --dataset_csv path/to/final_dataset.csv --fold 2
```

- Bootstrap: patients (`patient_id`) are resampled as clusters, with all examinations of each resampled
  patient retained together.
- SHAP: Kernel SHAP with 100 background samples from the training set, explaining all test samples.
  The representative fold (fold 2) is the fold with the median validation F1-score of the image-only
  model.

## Model configurations

Hyperparameters (selected by grid search on the mean validation F1-score across the five folds) are in
`configs/*.yaml`.

| | Image-only | Feature-only | Multimodal |
|---|---|---|---|
| Input | image (224 × 224) | 10 QMD | 512-d image feature + 10 QMD |
| Model | ResNet-18 (ImageNet pretrained) | MLP 10–16–2 | MLP 522–384–32–2 |
| Dropout | – | 0.2 | 0.1 |
| Optimizer | Adam | Adam | Adam |
| Learning rate | 1e-5 | 1e-3 | 1e-5 |
| Weight decay | 1e-3 | 1e-4 | 1e-3 |
| Batch size | 8 | 8 | 8 |
| Epochs | 500 | 50 | 50 |
| Label smoothing | 0.08 | 0 | 0.08 |

- Loss: weighted cross-entropy with class weights `w_c = 1 − n_c / N` from the training set of each
  fold (inversely proportional to class frequency for two classes)
- Checkpoint selection: epoch with the highest validation F1-score
- Classification threshold: 0.5 (not optimized)
- Ensemble prediction: mean of the predicted probabilities of the five fold-specific models

## Reproducibility

All runs use seed 42. `seed_everything()` in `configs/config.py` fixes the Python, NumPy and PyTorch
(CPU/CUDA) seeds and enables deterministic cuDNN. DataLoaders use a seeded generator. The configuration
used for each fold is saved as `config.json` in its output directory.

## Metrics

AUROC, PRAUC, accuracy, sensitivity, specificity, PPV, NPV, and F1-score (positive class = severe
dysphagia). Per-fold metrics are summarized as mean ± SD; confusion matrices and bootstrap comparisons
use the ensemble predictions.

## Outputs

```
outputs/<model>/
├── fold{k}/
│   ├── config.json
│   ├── dataset_statistics.json
│   ├── class_weights.json
│   ├── best_model*.pt
│   ├── best_model_metric*.json
│   ├── history*.csv
│   ├── results_train.csv, results_val.csv, test_inference.csv
│   └── metrics_test.json
├── cv_summary.csv                 # per-fold, mean, SD
├── test_inference_ensemble.csv
└── metrics_test_ensemble.json

outputs/swallowing/                # cut/, uppercut/, qmd_input.csv, image_list.csv
outputs/qmd/                       # qmd_features.csv
outputs/bootstrap/                 # bootstrap_<model1>_vs_<model2>{,_table,_raw}.csv
outputs/figures/                   # roc_cm.png / .tif
outputs/shap/                      # SHAP importance CSVs and summary plots
```

## Repository structure

```
.
├── configs/
│   ├── config.py
│   ├── swallowing.yaml
│   ├── qmd.yaml
│   ├── image_only.yaml
│   ├── feature_only.yaml
│   └── multimodal.yaml
├── qmd/
│   ├── qmd.py
│   └── qmd_structure.md
├── scripts/
│   ├── run_swallowing_extraction.py
│   ├── run_qmd.py
│   ├── run_image_only.py
│   ├── run_feature_only.py
│   ├── run_multimodal.py
│   ├── cv_summary.py
│   ├── cv_ensemble.py
│   ├── run_bootstrap.py
│   ├── plot_roc_cm.py
│   └── run_shap.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train_image_only.py
│   ├── train_feature_only.py
│   ├── train_multimodal.py
│   ├── test.py
│   ├── metrics.py
│   ├── cv_summary.py
│   ├── cv_ensemble.py
│   ├── bootstrap.py
│   ├── figures.py
│   ├── shap_analysis.py
│   └── utils.py
└── requirements.txt
```

## Data availability

The ultrasound data cannot be shared publicly due to ethical restrictions and patient privacy.
De-identified derived data are available from the corresponding author upon reasonable request,
subject to institutional and ethical approval. The anonymized data split is provided as a supplementary
table of the paper.

## Citation

```
TBD
```
