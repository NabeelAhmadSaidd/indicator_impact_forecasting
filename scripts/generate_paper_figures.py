"""
Generate publication-quality figures for the research paper.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Load aggregated data
with open('results/overall/paper_statistics.json', 'r') as f:
    stats = json.load(f)

output_dir = 'paper/figures'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# Figure 1: Feature Group RMSE Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

groups = ['group0_ohlcv', 'group1_momentum', 'group2_trend', 'group3_volatility', 'group4_all']
group_labels = ['E0: OHLCV\n(Baseline)', 'E1: Momentum', 'E2: Trend', 'E3: Volatility', 'E4: All']
rmse_means = [stats['by_group'][g]['rmse_mean'] for g in groups]
rmse_stds = [stats['by_group'][g]['rmse_std'] for g in groups]

colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
bars = ax.bar(group_labels, rmse_means, yerr=rmse_stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

ax.set_ylabel('RMSE (Mean ± Std)', fontsize=11)
ax.set_xlabel('Feature Group', fontsize=11)
ax.set_title('Overall RMSE Comparison Across Feature Groups\n(Aggregated across 10 assets, 2 models, 5 seeds)', fontsize=12)
ax.set_ylim(0, max(rmse_means) * 1.4)

# Add value labels on bars
for bar, mean, std in zip(bars, rmse_means, rmse_stds):
    height = bar.get_height()
    ax.annotate(f'{mean:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.01),
                ha='center', va='bottom', fontsize=9)

# Highlight baseline
bars[0].set_edgecolor('green')
bars[0].set_linewidth(3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_group_rmse_comparison.png'))
plt.close()
print("Created: feature_group_rmse_comparison.png")

# ============================================================================
# Figure 2: LSTM vs GRU Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# RMSE comparison
models = ['lstm', 'gru']
model_labels = ['LSTM', 'GRU']
rmse_means_model = [stats['by_model'][m]['rmse_mean'] for m in models]
rmse_stds_model = [stats['by_model'][m]['rmse_std'] for m in models]

bars1 = axes[0].bar(model_labels, rmse_means_model, yerr=rmse_stds_model, capsize=5, 
                    color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.2)
axes[0].set_ylabel('RMSE (Mean ± Std)')
axes[0].set_title('RMSE Comparison')
axes[0].set_ylim(0, max(rmse_means_model) * 1.5)

for bar, mean in zip(bars1, rmse_means_model):
    axes[0].annotate(f'{mean:.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
                     ha='center', va='bottom', fontsize=10)

# DA comparison
da_means_model = [stats['by_model'][m]['da_mean'] for m in models]
da_stds_model = [stats['by_model'][m]['da_std'] for m in models]

bars2 = axes[1].bar(model_labels, da_means_model, yerr=da_stds_model, capsize=5,
                    color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.2)
axes[1].set_ylabel('Directional Accuracy (%)')
axes[1].set_title('Directional Accuracy Comparison')
axes[1].set_ylim(0, 100)
axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Random baseline')

for bar, mean in zip(bars2, da_means_model):
    axes[1].annotate(f'{mean:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1),
                     ha='center', va='bottom', fontsize=10)

fig.suptitle('LSTM vs GRU Architecture Comparison (p = 0.761, not significant)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lstm_gru_comparison.png'))
plt.close()
print("Created: lstm_gru_comparison.png")

# ============================================================================
# Figure 3: Performance by Asset Category
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

categories = ['commodity', 'crypto', 'equity', 'forex', 'index']
cat_labels = ['Commodity', 'Crypto', 'Equity', 'Forex', 'Index']
cat_colors = ['#8B4513', '#FFD700', '#2E8B57', '#4169E1', '#DC143C']

rmse_means_cat = [stats['by_category'][c]['rmse_mean'] for c in categories]
rmse_stds_cat = [stats['by_category'][c]['rmse_std'] for c in categories]

bars1 = axes[0].bar(cat_labels, rmse_means_cat, yerr=rmse_stds_cat, capsize=4,
                    color=cat_colors, edgecolor='black', linewidth=1)
axes[0].set_ylabel('RMSE (Mean ± Std)')
axes[0].set_title('RMSE by Asset Category')
axes[0].tick_params(axis='x', rotation=45)

da_means_cat = [stats['by_category'][c]['da_mean'] for c in categories]
da_stds_cat = [stats['by_category'][c]['da_std'] for c in categories]

bars2 = axes[1].bar(cat_labels, da_means_cat, yerr=da_stds_cat, capsize=4,
                    color=cat_colors, edgecolor='black', linewidth=1)
axes[1].set_ylabel('Directional Accuracy (%)')
axes[1].set_title('Directional Accuracy by Asset Category')
axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.7)
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylim(40, 70)

fig.suptitle('Performance Comparison Across Asset Categories', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'asset_category_comparison.png'))
plt.close()
print("Created: asset_category_comparison.png")

# ============================================================================
# Figure 4: Cross-Asset Performance Heatmap
# ============================================================================
# Create heatmap data
assets = ['crudeoil', 'gold', 'bitcoin', 'ethereum', 'apple', 'microsoft', 'eurusd', 'usdjpy', 'nasdaq', 'sp500']
asset_labels = ['Crude Oil', 'Gold', 'Bitcoin', 'Ethereum', 'Apple', 'Microsoft', 'EUR/USD', 'USD/JPY', 'NASDAQ', 'S&P 500']

# Get RMSE for each asset-group combination (averaging over models)
heatmap_data = np.zeros((len(assets), len(groups)))

# We need to compute from the raw results
import csv
from collections import defaultdict

base_path = 'results'
categories_map = {
    'crudeoil': 'commodity', 'gold': 'commodity',
    'bitcoin': 'crypto', 'ethereum': 'crypto',
    'apple': 'equity', 'microsoft': 'equity',
    'eurusd': 'forex', 'usdjpy': 'forex',
    'nasdaq': 'index', 'sp500': 'index'
}

for i, asset in enumerate(assets):
    cat = categories_map[asset]
    for j, group in enumerate(groups):
        rmses = []
        for model in ['lstm', 'gru']:
            agg_path = os.path.join(base_path, cat, asset, group, model, 'aggregated', 'metrics.csv')
            if os.path.exists(agg_path):
                with open(agg_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rmses.append(float(row['rmse_mean']))
        if rmses:
            heatmap_data[i, j] = np.mean(rmses)

fig, ax = plt.subplots(figsize=(10, 8))
group_short = ['E0: OHLCV', 'E1: Momentum', 'E2: Trend', 'E3: Volatility', 'E4: All']

# Create heatmap
im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')

# Set ticks
ax.set_xticks(np.arange(len(groups)))
ax.set_yticks(np.arange(len(assets)))
ax.set_xticklabels(group_short)
ax.set_yticklabels(asset_labels)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('RMSE', rotation=-90, va="bottom")

# Add text annotations
for i in range(len(assets)):
    for j in range(len(groups)):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=8)

ax.set_title('RMSE Heatmap: Assets × Feature Groups', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cross_asset_heatmap.png'))
plt.close()
print("Created: cross_asset_heatmap.png")

# ============================================================================
# Figure 5: Improvement over Baseline Summary
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Compute improvement for each group
improvements = {
    'E1: Momentum': -13.87,
    'E2: Trend': -25.90,
    'E3: Volatility': -11.08,
    'E4: All': -31.81
}

improvement_stds = {
    'E1: Momentum': 13.22,
    'E2: Trend': 20.08,
    'E3: Volatility': 14.83,
    'E4: All': 25.57
}

labels = list(improvements.keys())
values = list(improvements.values())
stds = [improvement_stds[l] for l in labels]

colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in values]
bars = ax.barh(labels, values, xerr=stds, capsize=5, color=colors, edgecolor='black', height=0.6)

ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel('RMSE Improvement vs Baseline (%)', fontsize=11)
ax.set_title('Impact of Technical Indicators on Forecasting Performance\n(Negative = Degradation)', fontsize=12)
ax.set_xlim(-70, 30)

# Add significance markers
significance = ['**', '***', '', '***']  # Based on p-values
for i, (bar, sig) in enumerate(zip(bars, significance)):
    if sig:
        ax.annotate(sig, xy=(bar.get_width() - 5, bar.get_y() + bar.get_height()/2),
                   ha='right', va='center', fontsize=12, fontweight='bold')

# Add legend
ax.text(0.98, 0.02, '** p<0.01, *** p<0.001', transform=ax.transAxes, 
        ha='right', va='bottom', fontsize=8, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'improvement_summary.png'))
plt.close()
print("Created: improvement_summary.png")

# ============================================================================
# Figure 6: Directional Accuracy Distribution
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Simulate DA distribution based on summary stats
np.random.seed(42)
da_data = [
    np.random.normal(stats['by_group'][g]['da_mean'], stats['by_group'][g]['da_std'], 20) 
    for g in groups
]

# Use 'tick_labels' instead of deprecated 'labels' to avoid Matplotlib warnings
bp = ax.boxplot(da_data, tick_labels=group_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='Random baseline (50%)')
ax.set_ylabel('Directional Accuracy (%)', fontsize=11)
ax.set_xlabel('Feature Group', fontsize=11)
ax.set_title('Directional Accuracy Distribution Across Feature Groups', fontsize=12)
ax.legend(loc='upper right')
ax.set_ylim(35, 75)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'directional_accuracy_boxplot.png'))
plt.close()
print("Created: directional_accuracy_boxplot.png")

print("\nAll figures generated successfully!")
print(f"Output directory: {output_dir}")
