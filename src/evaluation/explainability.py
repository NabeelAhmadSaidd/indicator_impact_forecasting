"""
Explainability Module

This module provides model explainability utilities including
SHAP analysis and feature importance calculations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import tensorflow as tf


logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainer for deep learning models.
    
    Provides feature importance analysis using SHAP values.
    """
    
    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_background_samples: int = 100
    ):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained model (Keras model or forecaster).
            background_data: Background dataset for SHAP.
            feature_names: Names of features.
            n_background_samples: Number of background samples to use.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        
        # Get prediction function
        if hasattr(model, 'predict'):
            self.predict_fn = model.predict
        else:
            self.predict_fn = model
        
        # Sample background data if too large
        if len(background_data) > n_background_samples:
            indices = np.random.choice(
                len(background_data),
                n_background_samples,
                replace=False
            )
            background_data = background_data[indices]
        
        self.background_data = background_data
        self.explainer = None
        self.shap_values = None
    
    def _create_explainer(self) -> None:
        """Create the SHAP explainer."""
        # For deep learning models, use DeepExplainer or GradientExplainer
        try:
            self.explainer = shap.DeepExplainer(
                self.model if hasattr(self.model, 'model') else self.model,
                self.background_data
            )
        except Exception as e:
            logger.warning(f"DeepExplainer failed: {e}. Trying KernelExplainer.")
            self.explainer = shap.KernelExplainer(
                self.predict_fn,
                self.background_data
            )
    
    def explain(
        self,
        X: np.ndarray,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate SHAP values for the given data.
        
        Args:
            X: Data to explain.
            n_samples: Number of samples to explain.
            
        Returns:
            SHAP values array.
        """
        if self.explainer is None:
            self._create_explainer()
        
        if n_samples is not None and len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
        
        logger.info(f"Calculating SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(X)
        
        return self.shap_values
    
    def get_feature_importance(
        self,
        X: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Get feature importance based on SHAP values.
        
        Args:
            X: Data to analyze. If None, uses previously computed values.
            
        Returns:
            DataFrame with feature importance.
        """
        if X is not None:
            self.explain(X)
        
        if self.shap_values is None:
            raise ValueError("No SHAP values computed. Call explain() first.")
        
        # Handle different SHAP value formats
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        # For 3D data (sequences), aggregate over time dimension
        if shap_vals.ndim == 3:
            # Shape: (samples, timesteps, features)
            # Average importance across timesteps
            importance = np.abs(shap_vals).mean(axis=(0, 1))
        else:
            importance = np.abs(shap_vals).mean(axis=0)
        
        # Create DataFrame
        if self.feature_names is not None:
            feature_names = self.feature_names[:len(importance)]
        else:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def plot_summary(
        self,
        X: np.ndarray,
        max_display: int = 10,
        plot_type: str = "bar",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create SHAP summary plot.
        
        Args:
            X: Data to explain.
            max_display: Maximum features to display.
            plot_type: Type of plot ('bar', 'dot', 'violin').
            save_path: Path to save the figure.
            
        Returns:
            Matplotlib figure.
        """
        if self.shap_values is None:
            self.explain(X)
        
        shap_vals = self.shap_values[0] if isinstance(self.shap_values, list) else self.shap_values
        
        # For 3D data, reshape for plotting
        if shap_vals.ndim == 3:
            # Average over timesteps
            shap_vals = shap_vals.mean(axis=1)
            X_plot = X.mean(axis=1)
        else:
            X_plot = X
        
        fig = plt.figure(figsize=(10, 8))
        
        shap.summary_plot(
            shap_vals,
            X_plot,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type=plot_type,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate permutation importance for features.
    
    Args:
        model: Trained model with predict method.
        X: Input features.
        y: True targets.
        feature_names: Names of features.
        n_repeats: Number of permutation repeats.
        random_state: Random seed.
        
    Returns:
        DataFrame with feature importance.
    """
    from src.evaluation.metrics import rmse
    
    np.random.seed(random_state)
    
    # Get baseline score
    y_pred = model.predict(X).flatten()
    baseline_score = rmse(y, y_pred)
    
    n_features = X.shape[-1]
    importance_scores = np.zeros((n_repeats, n_features))
    
    for repeat in range(n_repeats):
        for feature_idx in range(n_features):
            # Create permuted data
            X_permuted = X.copy()
            
            # For 3D data (sequences), permute the feature across all timesteps
            if X.ndim == 3:
                permuted_values = X[:, :, feature_idx].flatten()
                np.random.shuffle(permuted_values)
                X_permuted[:, :, feature_idx] = permuted_values.reshape(X.shape[0], X.shape[1])
            else:
                np.random.shuffle(X_permuted[:, feature_idx])
            
            # Calculate score with permuted feature
            y_pred_permuted = model.predict(X_permuted).flatten()
            permuted_score = rmse(y, y_pred_permuted)
            
            # Importance is the increase in error
            importance_scores[repeat, feature_idx] = permuted_score - baseline_score
    
    # Calculate mean and std
    mean_importance = importance_scores.mean(axis=0)
    std_importance = importance_scores.std(axis=0)
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    df = pd.DataFrame({
        'feature': feature_names[:n_features],
        'importance_mean': mean_importance,
        'importance_std': std_importance
    })
    
    return df.sort_values('importance_mean', ascending=False).reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance.
        top_n: Number of top features to show.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df = importance_df.head(top_n)
    
    # Check if we have error bars
    has_std = 'importance_std' in df.columns
    
    if has_std:
        importance_col = 'importance_mean'
        ax.barh(
            df['feature'],
            df[importance_col],
            xerr=df['importance_std'],
            capsize=3,
            color='steelblue',
            alpha=0.8
        )
    else:
        importance_col = 'importance'
        ax.barh(
            df['feature'],
            df[importance_col],
            color='steelblue',
            alpha=0.8
        )
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_feature_contributions(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = 'permutation',
    n_samples: int = 100,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive feature contribution analysis.
    
    Args:
        model: Trained model.
        X: Input features.
        y: True targets.
        feature_names: Names of features.
        method: Analysis method ('permutation', 'shap', or 'both').
        n_samples: Number of samples for analysis.
        save_dir: Directory to save results.
        
    Returns:
        Dictionary with analysis results.
    """
    results = {}
    
    # Subsample if needed
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Permutation importance
    if method in ['permutation', 'both']:
        logger.info("Calculating permutation importance...")
        perm_importance = permutation_importance(
            model, X_sample, y_sample, feature_names
        )
        results['permutation_importance'] = perm_importance
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            perm_importance.to_csv(save_path / 'permutation_importance.csv', index=False)
            
            fig = plot_feature_importance(
                perm_importance,
                title='Permutation Feature Importance',
                save_path=str(save_path / 'permutation_importance.png')
            )
            plt.close(fig)
    
    # SHAP analysis
    if method in ['shap', 'both'] and SHAP_AVAILABLE:
        logger.info("Calculating SHAP values...")
        try:
            explainer = SHAPExplainer(
                model, X_sample, feature_names
            )
            shap_importance = explainer.get_feature_importance(X_sample)
            results['shap_importance'] = shap_importance
            
            if save_dir:
                save_path = Path(save_dir)
                shap_importance.to_csv(save_path / 'shap_importance.csv', index=False)
                
                fig = plot_feature_importance(
                    shap_importance,
                    title='SHAP Feature Importance',
                    save_path=str(save_path / 'shap_importance.png')
                )
                plt.close(fig)
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            results['shap_error'] = str(e)
    
    return results


if __name__ == "__main__":
    # Test explainability functions
    import numpy as np
    
    np.random.seed(42)
    
    # Create sample data and simple model for testing
    n_samples = 200
    n_timesteps = 10
    n_features = 5
    
    X = np.random.randn(n_samples, n_timesteps, n_features)
    y = np.random.randn(n_samples)
    
    # Create feature names
    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Test permutation importance with a simple model
    class SimpleModel:
        def predict(self, X):
            return X[:, -1, 3:4]  # Just return last Close value
    
    model = SimpleModel()
    
    importance = permutation_importance(
        model, X, y, feature_names, n_repeats=5
    )
    
    print("Permutation Importance:")
    print(importance)
    
    # Plot
    fig = plot_feature_importance(importance, title='Test Feature Importance')
    plt.show()
