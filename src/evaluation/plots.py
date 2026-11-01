"""
Plotting Module

This module provides visualization utilities for model evaluation,
including prediction plots, metric comparisons, and training history.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#3498db', '#e74c3c'])


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    xlabel: str = "Time",
    ylabel: str = "Price",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot predictions against actual values.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size.
        save_path: Path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(y_true))
    
    ax.plot(x, y_true, label='Actual', alpha=0.8, linewidth=1.5)
    ax.plot(x, y_pred, label='Predicted', alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a scatter plot of predictions vs actual values.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='#ff7f0e', linestyle='--', 
            label='Perfect Prediction', linewidth=2)
    
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ['loss', 'val_loss'],
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:

    # --------------------------------
    # Build train/val metric pairs first
    # --------------------------------
    metric_pairs = []
    plotted = set()

    for metric in metrics:
        if metric in plotted:
            continue

        if metric.startswith('val_'):
            base = metric[4:]
            if base in history:
                metric_pairs.append((base, metric))
                plotted.update([base, metric])
        else:
            val_metric = f"val_{metric}"
            if val_metric in history:
                metric_pairs.append((metric, val_metric))
                plotted.update([metric, val_metric])
            else:
                metric_pairs.append((metric, None))
                plotted.add(metric)

    # --------------------------------
    # Create correct number of subplots
    # --------------------------------
    fig, axes = plt.subplots(1, len(metric_pairs), figsize=figsize)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # --------------------------------
    # Plot
    # --------------------------------
    for ax, (train_metric, val_metric) in zip(axes, metric_pairs):

        epochs = range(1, len(history.get(train_metric, [])) + 1)

        if train_metric in history:
            ax.plot(
                epochs,
                history[train_metric],
                label=f"Train {train_metric}",
                linewidth=2
            )

        if val_metric and val_metric in history:
            ax.plot(
                epochs,
                history[val_metric],
                label=f"Val {train_metric}",
                linewidth=2
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(train_metric.upper())
        ax.set_title(f"{train_metric.upper()} Over Epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

def plot_metrics_comparison(
    results_df: pd.DataFrame,
    metric: str = 'rmse',
    group_by: str = 'group',
    hue: Optional[str] = 'model',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of metrics across groups.
    
    Args:
        results_df: DataFrame with results.
        metric: Metric to compare.
        group_by: Column to group by on x-axis.
        hue: Column for color grouping.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.barplot(
        data=results_df,
        x=group_by,
        y=metric,
        hue=hue,
        ax=ax,
        capsize=0.1,
        errwidth=1.5
    )
    
    ax.set_xlabel(group_by.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title or f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
    ax.legend(title=hue.replace('_', ' ').title() if hue else None, fontsize=10)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_boxplots(
    results_df: pd.DataFrame,
    metric: str = 'rmse',
    group_by: str = 'group',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create boxplots for metric comparison.
    
    Args:
        results_df: DataFrame with results.
        metric: Metric to plot.
        group_by: Column to group by.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(
        data=results_df,
        x=group_by,
        y=metric,
        ax=ax,
        palette='husl'
    )
    
    ax.set_xlabel(group_by.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title or f'{metric.upper()} Distribution by {group_by.title()}', 
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_heatmap(
    pivot_table: pd.DataFrame,
    title: str = "Performance Heatmap",
    cmap: str = "RdYlGn_r",
    annot: bool = True,
    fmt: str = ".4f",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap visualization.
    
    Args:
        pivot_table: Pivot table with values to plot.
        title: Plot title.
        cmap: Colormap name.
        annot: Whether to annotate cells.
        fmt: Format string for annotations.
        figsize: Figure size.
        save_path: Path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        pivot_table,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': 'Value'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction Error Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of prediction errors.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    errors = y_pred - y_true
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='#ff7f0e', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].axvline(x=np.mean(errors), color='#1f77b4', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    axes[0].set_xlabel('Error', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Error Histogram', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # QQ plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals vs Predicted",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot residuals against predicted values.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax.axhline(0, color='#ff7f0e', linestyle='--', linewidth=1.5, label='Zero Residual Line')

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_results_summary_plot(
    results_df: pd.DataFrame,
    metrics: List[str] = ['rmse', 'directional_accuracy'],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive summary plot of results.
    
    Args:
        results_df: DataFrame with all results.
        metrics: Metrics to include.
        figsize: Figure size.
        save_path: Path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 2, figsize=figsize)
    
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    
    for i, metric in enumerate(metrics):
        # Boxplot by group
        sns.boxplot(
            data=results_df,
            x='group',
            y=metric,
            hue='model',
            ax=axes[i, 0]
        )
        axes[i, 0].set_title(f'{metric.upper()} by Feature Group', fontsize=12)
        axes[i, 0].set_xlabel('Feature Group', fontsize=10)
        axes[i, 0].set_ylabel(metric.upper(), fontsize=10)
        axes[i, 0].tick_params(axis='x', rotation=45)
        axes[i, 0].legend(title='Model', fontsize=8)
        
        # Boxplot by asset
        sns.boxplot(
            data=results_df,
            x='asset',
            y=metric,
            hue='model',
            ax=axes[i, 1]
        )
        axes[i, 1].set_title(f'{metric.upper()} by Asset', fontsize=12)
        axes[i, 1].set_xlabel('Asset', fontsize=10)
        axes[i, 1].set_ylabel(metric.upper(), fontsize=10)
        axes[i, 1].tick_params(axis='x', rotation=45)
        axes[i, 1].legend(title='Model', fontsize=8)
    
    plt.suptitle('Results Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Test plotting functions
    np.random.seed(42)
    
    # Create sample data
    n = 100
    y_true = 100 + np.random.randn(n).cumsum()
    y_pred = y_true + np.random.randn(n) * 2
    
    # Test prediction plot
    fig = plot_predictions(y_true, y_pred)
    plt.show()
    
    # Test scatter plot
    fig = plot_prediction_scatter(y_true, y_pred)
    plt.show()
    
    # Test error distribution
    fig = plot_error_distribution(y_true, y_pred)
    plt.show()
