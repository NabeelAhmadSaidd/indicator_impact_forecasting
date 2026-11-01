"""
Logging Utility Module

This module provides logging utilities for the indicator impact forecasting project.
Supports both file and console logging with configurable formats and levels.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import os

from src.utils.config import get_project_root, Config


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log messages for console output.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: Union[str, int] = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
    colored: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and/or console handlers.
    
    Args:
        name: Name of the logger.
        log_file: Path to the log file. If None, uses default based on name.
        level: Logging level (e.g., 'DEBUG', 'INFO', logging.DEBUG).
        log_to_console: Whether to log to console.
        log_to_file: Whether to log to file.
        colored: Whether to use colored output for console.
        
    Returns:
        Configured logger instance.
    """
    # Convert string level to int if necessary
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    file_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    console_format = '%(asctime)s | %(levelname)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    file_formatter = logging.Formatter(file_format, datefmt=date_format)
    
    if colored and sys.platform != 'win32':
        console_formatter = ColoredFormatter(console_format, datefmt=date_format)
    else:
        console_formatter = logging.Formatter(console_format, datefmt=date_format)
    
    # Add file handler
    if log_to_file:
        if log_file is None:
            log_dir = get_project_root() / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f'{name}.log'
        else:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(console_format, datefmt=date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Name of the logger.
        
    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class ExperimentLogger:
    """
    Logger specifically designed for tracking experiments.
    
    Logs experiment configurations, metrics, and results in a structured format.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: Optional[Union[str, Path]] = None,
        level: Union[str, int] = logging.INFO
    ):
        """
        Initialize the experiment logger.
        
        Args:
            experiment_name: Name of the experiment.
            log_dir: Directory for log files.
            level: Logging level.
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        if log_dir is None:
            log_dir = get_project_root() / 'logs'
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'{experiment_name}_{timestamp}.log'
        
        self.logger = setup_logger(
            name=f'experiment.{experiment_name}',
            log_file=log_file,
            level=level
        )
    
    def log_config(self, config: dict) -> None:
        """Log experiment configuration."""
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT CONFIGURATION")
        self.logger.info("=" * 60)
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60)
    
    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log metrics for a training epoch."""
        metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch:4d} | {metrics_str}")
    
    def log_evaluation(self, split: str, metrics: dict) -> None:
        """Log evaluation metrics."""
        self.logger.info(f"--- {split.upper()} EVALUATION ---")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.6f}")
    
    def log_result(self, result: dict) -> None:
        """Log final experiment result."""
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT RESULTS")
        self.logger.info("=" * 60)
        for key, value in result.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        duration = datetime.now() - self.start_time
        self.logger.info(f"  Total Duration: {duration}")
        self.logger.info("=" * 60)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)


# Module-level loggers for common use cases
preprocessing_logger = None
training_logger = None
evaluation_logger = None


def get_preprocessing_logger() -> logging.Logger:
    """Get the preprocessing logger."""
    global preprocessing_logger
    if preprocessing_logger is None:
        preprocessing_logger = setup_logger('preprocessing')
    return preprocessing_logger


def get_training_logger() -> logging.Logger:
    """Get the training logger."""
    global training_logger
    if training_logger is None:
        training_logger = setup_logger('training')
    return training_logger


def get_evaluation_logger() -> logging.Logger:
    """Get the evaluation logger."""
    global evaluation_logger
    if evaluation_logger is None:
        evaluation_logger = setup_logger('evaluation')
    return evaluation_logger


if __name__ == "__main__":
    # Test logging functionality
    logger = setup_logger('test', level='DEBUG')
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test experiment logger
    exp_logger = ExperimentLogger('test_experiment')
    exp_logger.log_config({'model': 'lstm', 'asset': 'BTCUSD', 'seed': 42})
    exp_logger.log_epoch(1, {'loss': 0.001234, 'val_loss': 0.001567})
    exp_logger.log_evaluation('test', {'rmse': 0.0234, 'mae': 0.0156})
