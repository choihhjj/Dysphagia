"""ROC curves (mean ± SD over folds) and confusion matrices (5-fold ensemble)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from src.metrics import fmt

COLORS = {'Image-only': 'royalblue', 'Feature-only': 'gray', 'Multimodal': 'darkred'}


def set_style(font='Arial'):
    """Use `font` if installed, otherwise keep the matplotlib default."""
    plt.rcParams['font.size'] = 12
    if font in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams['font.family'] = font


def save_figure(fig, save_path):
    fig.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(save_path + '.tif', dpi=300, bbox_inches='tight', format='tiff',
                facecolor='white', edgecolor='none', pil_kwargs={'compression': 'tiff_lzw'})


def plot_roc_cm(model_dfs, save_path, n_folds=5, class_names=('Mild', 'Severe')):
    """model_dfs: {model name: test_inference_ensemble DataFrame}."""
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 1 + len(model_dfs), wspace=0.4)
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_cms = [fig.add_subplot(gs[0, i]) for i in range(1, 1 + len(model_dfs))]

    # ROC: per-fold curves interpolated on a common FPR grid
    mean_fpr = np.linspace(0, 1, 100)
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='lightgray')

    for name, df in model_dfs.items():
        y_true = df['label'].values
        tprs, aucs = [], []
        for k in range(1, n_folds + 1):
            prob = df[f'prob_fold{k}'].values
            fpr, tpr, _ = roc_curve(y_true, prob)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_score(y_true, prob))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        auc_mean, auc_sd = np.mean(aucs), np.std(aucs, ddof=1)

        color = COLORS.get(name)
        ax_roc.plot(mean_fpr, mean_tpr, color=color, linewidth=2,
                    label=f'{name} ({fmt(auc_mean)} ± {fmt(auc_sd)})')
        ax_roc.fill_between(mean_fpr,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            color=color, alpha=0.15)

    ax_roc.set_xlabel('False positive rate', fontsize=14)
    ax_roc.set_ylabel('True positive rate', fontsize=14)
    ax_roc.set_title('ROC curves', fontsize=16, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10)
    ax_roc.set_aspect('equal')
    ax_roc.tick_params(axis='both', labelsize=12)
    ax_roc.set_xlim(-0.01, 1.01)
    ax_roc.set_ylim(-0.01, 1.01)
    ax_roc.set_xticks(np.arange(0, 1.01, 0.2))
    ax_roc.set_yticks(np.arange(0, 1.01, 0.2))
    ax_roc.grid(True, alpha=0.3)

    # confusion matrices of the ensemble predictions
    for ax, (name, df) in zip(ax_cms, model_dfs.items()):
        cm = confusion_matrix(df['label'].values, df['pred_ensemble'].values, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names))
        disp.plot(cmap='Blues', ax=ax, colorbar=False)
        ax.set_title(name, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('Label', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        for text in disp.text_.ravel():
            text.set_fontsize(18)
            text.set_fontweight('bold')

    save_figure(fig, save_path)
    plt.close(fig)
