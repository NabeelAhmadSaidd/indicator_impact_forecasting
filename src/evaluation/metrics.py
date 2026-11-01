"""
Evaluation Metrics Module

This module provides functions for calculating evaluation metrics
for financial time series forecasting models.

Metrics include:
- Primary: RMSE, Directional Accuracy
- Secondary: MAE, MAPE
"""

import numpy as np
from typing import Dict, Tuple, Union, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        RMSE value.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        MAE value.
    """
    return mean_absolute_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        epsilon: Small value to avoid division by zero.
        
    Returns:
        MAPE value as percentage.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    
    if not np.any(mask):
        return 0.0
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Directional Accuracy.
    
    Measures the percentage of times the model correctly predicts
    the direction of price movement.
    
    Args:
        y_true: True values (sequence of prices).
        y_pred: Predicted values (sequence of prices).
        
    Returns:
        Directional accuracy as percentage.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) < 2:
        return 0.0
    
    # Calculate actual direction (1 for up, -1 for down, 0 for no change)
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Calculate accuracy
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)
    
    return (correct / total) * 100 if total > 0 else 0.0


def directional_accuracy_from_changes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Directional Accuracy from predicted vs actual changes.
    
    This version compares predicted direction from previous value
    to actual direction from previous value.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        y_prev: Previous values (for calculating direction).
        
    Returns:
        Directional accuracy as percentage.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if y_prev is None:
        # Use first value as previous
        if len(y_true) < 2:
            return 0.0
        y_prev = np.roll(y_true, 1)
        y_prev[0] = y_true[0]
        y_true = y_true[1:]
        y_pred = y_pred[1:]
        y_prev = y_prev[1:]
    else:
        y_prev = np.array(y_prev).flatten()
    
    # Calculate directions
    true_direction = np.sign(y_true - y_prev)
    pred_direction = np.sign(y_pred - y_prev)
    
    # Calculate accuracy
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)
    
    return (correct / total) * 100 if total > 0 else 0.0


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        epsilon: Small value to avoid division by zero.
        
    Returns:
        SMAPE value as percentage.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    
    return np.mean(numerator / denominator) * 100


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        R-squared value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        MSE value.
    """
    return mean_squared_error(y_true, y_pred)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        Dictionary of all metrics.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'directional_accuracy': directional_accuracy(y_true, y_pred),
        'r_squared': r_squared(y_true, y_pred),
        'mse': mse(y_true, y_pred)
    }


def calculate_primary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate primary evaluation metrics (RMSE and Directional Accuracy).
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        Dictionary with RMSE and directional accuracy.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    return {
        'rmse': rmse(y_true, y_pred),
        'directional_accuracy': directional_accuracy(y_true, y_pred)
    }


def calculate_secondary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate secondary evaluation metrics (MAE and MAPE).
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        Dictionary with MAE and MAPE.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    return {
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred)
    }


class MetricsCalculator:
    """
    Class for calculating and storing metrics across multiple evaluations.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.results = []
    
    def add_result(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Add a result and calculate its metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            metadata: Optional metadata for this result.
            
        Returns:
            Calculated metrics.
        """
        metrics = calculate_all_metrics(y_true, y_pred)
        
        result = {
            'metrics': metrics,
            'metadata': metadata or {}
        }
        self.results.append(result)
        
        return metrics
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics of all results.
        
        Returns:
            Dictionary with mean, std, min, max for each metric.
        """
        if not self.results:
            return {}
        
        # Collect all metrics
        metric_values = {}
        for result in self.results:
            for metric, value in result['metrics'].items():
                if metric not in metric_values:
                    metric_values[metric] = []
                metric_values[metric].append(value)
        
        # Calculate summary statistics
        summary = {}
        for metric, values in metric_values.items():
            values = np.array(values)
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return summary
    
    def clear(self) -> None:
        """Clear all stored results."""
        self.results = []


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Create sample data
    n = 100
    y_true = 100 + np.random.randn(n).cumsum()  # Random walk
    noise = np.random.randn(n) * 2
    y_pred = y_true + noise
    
    # Calculate all metrics
    metrics = calculate_all_metrics(y_true, y_pred)
    
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test directional accuracy
    print(f"\nDirectional Accuracy: {directional_accuracy(y_true, y_pred):.2f}%")
