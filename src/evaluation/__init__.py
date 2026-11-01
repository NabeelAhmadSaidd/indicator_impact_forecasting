"""
Evaluation Package Initialization

This package provides evaluation utilities including metrics calculation,
visualization, and model explainability.
"""

from src.evaluation.metrics import (
    rmse,
    mae,
    mape,
    smape,
    mse,
    r_squared,
    directional_accuracy,
    directional_accuracy_from_changes,
    calculate_all_metrics,
    calculate_primary_metrics,
    calculate_secondary_metrics,
    MetricsCalculator
)

from src.evaluation.plots import (
    plot_predictions,
    plot_prediction_scatter,
    plot_training_history,
    plot_metrics_comparison,
    plot_boxplots,
    plot_heatmap,
    plot_error_distribution,
    create_results_summary_plot
)

from src.evaluation.explainability import (
    SHAPExplainer,
    permutation_importance,
    plot_feature_importance,
    analyze_feature_contributions
)

__all__ = [
    # Metrics
    'rmse',
    'mae',
    'mape',
    'smape',
    'mse',
    'r_squared',
    'directional_accuracy',
    'directional_accuracy_from_changes',
    'calculate_all_metrics',
    'calculate_primary_metrics',
    'calculate_secondary_metrics',
    'MetricsCalculator',
    
    # Plots
    'plot_predictions',
    'plot_prediction_scatter',
    'plot_training_history',
    'plot_metrics_comparison',
    'plot_boxplots',
    'plot_heatmap',
    'plot_error_distribution',
    'create_results_summary_plot',
    
    # Explainability
    'SHAPExplainer',
    'permutation_importance',
    'plot_feature_importance',
    'analyze_feature_contributions'
]
