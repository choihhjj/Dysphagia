"""SHAP analysis of the multimodal classifier (512 ResNet-18 features + 10 QMD features)."""

import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.figures import save_figure

FEATURE_DIM = 512
N_TOP_RESNET = 10
QMD_LABEL = 'QMD parameters'
RESNET_LABEL = f'ResNet features (top {N_TOP_RESNET})'


def _style_summary_axis(ax, y_limits):
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('center')
        label.set_x(-0.3)

    fig = ax.figure
    pos = ax.get_position()
    for cbar_ax in fig.axes:
        if cbar_ax is not ax and cbar_ax.get_position().x0 > pos.x1:
            cbar_ax.tick_params(labelsize=12)
            cbar_ax.set_ylabel('Feature value', fontsize=14)

    ax.set_ylim(y_limits)
    ax.set_xlabel('SHAP value', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)


def shap_analysis(fc_model, train_features, test_features, qmd_columns,
                  background_samples=100, num_samples=None, device='cpu',
                  save_path=None, suffix='', seed=42):
    """Kernel SHAP on the fused features; returns (mean_abs_shap, selected_df, all_df, shap_values)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    fc_model.eval()
    qmd_dim = len(qmd_columns)
    assert train_features.shape[1] == FEATURE_DIM + qmd_dim

    def model_predict(x):
        with torch.no_grad():
            out = fc_model(torch.tensor(x, dtype=torch.float32).to(device))
            return torch.softmax(out, dim=1).cpu().numpy()

    bg_idx = np.random.choice(len(train_features), background_samples, replace=False)
    explainer = shap.KernelExplainer(model_predict, train_features[bg_idx])

    if num_samples is not None and len(test_features) > num_samples:
        sample_idx = np.random.choice(len(test_features), num_samples, replace=False)
        sample_data = test_features[sample_idx]
    else:
        sample_data = test_features

    shap_values = explainer.shap_values(sample_data)
    shap_pos = shap_values[1]                       # positive class
    mean_shap = np.abs(shap_pos).mean(axis=0)
    resnet_shap = mean_shap[:FEATURE_DIM]
    qmd_shap = mean_shap[FEATURE_DIM:FEATURE_DIM + qmd_dim]
    resnet_top_idx = np.argsort(resnet_shap)[-N_TOP_RESNET:]

    resnet_all_df = pd.DataFrame({
        'Feature': [f'ResNet_{i}' for i in range(FEATURE_DIM)],
        'Importance': resnet_shap,
        'Category': RESNET_LABEL,
        'Original_Index': range(FEATURE_DIM),
    })
    qmd_df = pd.DataFrame({
        'Feature': qmd_columns,
        'Importance': qmd_shap,
        'Category': QMD_LABEL,
        'Original_Index': range(FEATURE_DIM, FEATURE_DIM + qmd_dim),
    })
    selected_df = (pd.concat([resnet_all_df.iloc[resnet_top_idx], qmd_df], ignore_index=True)
                   .sort_values('Importance', ascending=False).reset_index(drop=True))
    all_df = (pd.concat([resnet_all_df, qmd_df], ignore_index=True)
              .sort_values('Importance', ascending=False).reset_index(drop=True))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        selected_df.to_csv(os.path.join(save_path, f'mm_shap_importance_top20{suffix}.csv'))
        all_df.to_csv(os.path.join(save_path, f'mm_shap_importance_all{suffix}.csv'))

    # Figure 1: mean |SHAP| bar plot (left) + summary plot of the same 20 features (right)
    n_feat = len(selected_df)
    y_limits = (-0.5, n_feat - 0.5)
    fig = plt.figure()

    ax = plt.subplot(1, 2, 1)
    reversed_df = selected_df.assign(Importance=-selected_df['Importance'])
    sns.barplot(y='Feature', x='Importance', hue='Category', data=reversed_df,
                palette={RESNET_LABEL: 'gray', QMD_LABEL: 'forestgreen'},
                hue_order=[QMD_LABEL, RESNET_LABEL], dodge=False, orient='h', ax=ax)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{abs(t):.2f}' for t in ticks], fontsize=12)
    ax.legend(title='Feature types', loc='lower left', fontsize=10)

    max_imp = selected_df['Importance'].max()
    for i, value in enumerate(selected_df['Importance']):
        ax.text(-value - max_imp * 0.01, i, f'+{value:.2f}', va='center', ha='right', fontsize=10)

    ax.set_ylabel(' ', fontsize=14)
    ax.set_xlabel('Mean absolute SHAP value', fontsize=14)
    ax.set_xlim(-max_imp - 0.02, 0)
    ax.grid(True, alpha=0.2, axis='x')
    ax.set_ylim(y_limits)
    ax.invert_yaxis()
    ax.set_yticklabels([])
    ax.tick_params(left=False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.subplot(1, 2, 2)
    order = selected_df['Original_Index'].values
    shap.summary_plot(shap_pos[:, order], sample_data[:, order],
                      feature_names=selected_df['Feature'].tolist(),
                      show=False, max_display=n_feat)
    _style_summary_axis(plt.gca(), y_limits)

    plt.tight_layout()
    fig.set_size_inches(10, 6, forward=True)
    if save_path:
        save_figure(fig, os.path.join(save_path, f'mm_shap_combined_summary_bar{suffix}'))
    plt.close(fig)

    # Figure 2: summary plots of the QMD features (left) and the top ResNet features (right)
    y_limits = (-0.5, N_TOP_RESNET - 0.5)
    fig = plt.figure()

    plt.subplot(1, 2, 1)
    qmd_slice = slice(FEATURE_DIM, FEATURE_DIM + qmd_dim)
    shap.summary_plot(shap_pos[:, qmd_slice], sample_data[:, qmd_slice],
                      feature_names=list(qmd_columns), show=False, max_display=qmd_dim)
    _style_summary_axis(plt.gca(), y_limits)

    plt.subplot(1, 2, 2)
    shap.summary_plot(shap_pos[:, resnet_top_idx], sample_data[:, resnet_top_idx],
                      feature_names=[f'ResNet_{i}' for i in resnet_top_idx],
                      show=False, max_display=N_TOP_RESNET)
    _style_summary_axis(plt.gca(), y_limits)

    plt.tight_layout()
    if save_path:
        save_figure(fig, os.path.join(save_path, f'qmd_resnet_summary{suffix}'))
    plt.close(fig)

    return mean_shap, selected_df, all_df, shap_values
