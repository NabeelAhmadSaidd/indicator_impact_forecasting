"""
Seed Utility Module

This module provides utilities for setting random seeds to ensure
reproducibility across experiments.
"""

import os
import random
from typing import Optional
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - TensorFlow
    - PYTHONHASHSEED environment variable
    
    Args:
        seed: The random seed value to use.
    """
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PYTHONHASHSEED for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set TensorFlow seeds
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        # For older TensorFlow versions
        try:
            tf.compat.v1.set_random_seed(seed)
        except AttributeError:
            pass
        
        # Ensure deterministic operations (may impact performance)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
    except ImportError:
        pass


def set_tensorflow_deterministic() -> None:
    """
    Configure TensorFlow for deterministic behavior.
    
    This function sets environment variables and configurations
    to make TensorFlow operations as deterministic as possible.
    
    Note: This may impact performance due to disabling certain
    optimizations.
    """
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    try:
        import tensorflow as tf
        
        # Enable deterministic operations
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass
        
        # Disable GPU memory growth to ensure consistent behavior
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
                    
    except ImportError:
        pass


def get_seed_from_run_id(base_seed: int, run_id: int) -> int:
    """
    Generate a deterministic seed from a base seed and run ID.
    
    This allows for reproducible multiple runs with different seeds.
    
    Args:
        base_seed: The base seed value.
        run_id: The run identifier (1-indexed).
        
    Returns:
        A deterministic seed value.
    """
    return base_seed * run_id + run_id


def save_seed(seed: int, filepath: str) -> None:
    """
    Save the seed value to a file for reproducibility tracking.
    
    Args:
        seed: The seed value to save.
        filepath: Path to save the seed file.
    """
    from pathlib import Path
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(str(seed))


def load_seed(filepath: str) -> int:
    """
    Load a seed value from a file.
    
    Args:
        filepath: Path to the seed file.
        
    Returns:
        The loaded seed value.
        
    Raises:
        FileNotFoundError: If the seed file doesn't exist.
    """
    from pathlib import Path
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Seed file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return int(f.read().strip())


class SeedManager:
    """
    Manager class for handling seeds across multiple experiments.
    
    Attributes:
        base_seed: The base seed for the experiment set.
        seeds: List of seeds for multiple runs.
        current_run: The current run index.
    """
    
    def __init__(self, base_seed: int = 42, num_runs: int = 5):
        """
        Initialize the seed manager.
        
        Args:
            base_seed: The base seed value.
            num_runs: Number of experimental runs.
        """
        self.base_seed = base_seed
        self.num_runs = num_runs
        self.seeds = self._generate_seeds()
        self.current_run = 0
    
    def _generate_seeds(self) -> list:
        """Generate seeds for all runs."""
        # Use a deterministic sequence based on base seed
        rng = np.random.RandomState(self.base_seed)
        return list(rng.randint(0, 2**31 - 1, size=self.num_runs))
    
    def get_seed(self, run_id: Optional[int] = None) -> int:
        """
        Get the seed for a specific run.
        
        Args:
            run_id: Run identifier (0-indexed). If None, uses current run.
            
        Returns:
            The seed for the specified run.
        """
        if run_id is None:
            run_id = self.current_run
        
        if run_id >= len(self.seeds):
            raise ValueError(f"Run ID {run_id} exceeds number of runs {self.num_runs}")
        
        return self.seeds[run_id]
    
    def set_seed_for_run(self, run_id: Optional[int] = None) -> int:
        """
        Set the random seed for a specific run.
        
        Args:
            run_id: Run identifier (0-indexed). If None, uses current run.
            
        Returns:
            The seed that was set.
        """
        seed = self.get_seed(run_id)
        set_seed(seed)
        return seed
    
    def next_run(self) -> int:
        """
        Advance to the next run and set its seed.
        
        Returns:
            The seed for the new run.
        """
        self.current_run += 1
        if self.current_run >= self.num_runs:
            raise ValueError("All runs completed")
        return self.set_seed_for_run()
    
    def reset(self) -> None:
        """Reset to the first run."""
        self.current_run = 0
    
    def __iter__(self):
        """Iterate over all seeds."""
        self.current_run = 0
        return self
    
    def __next__(self) -> int:
        """Get the next seed in iteration."""
        if self.current_run >= self.num_runs:
            raise StopIteration
        seed = self.set_seed_for_run()
        self.current_run += 1
        return seed


if __name__ == "__main__":
    # Test seed functionality
    print("Testing seed utilities...")
    
    # Test basic seed setting
    set_seed(42)
    print(f"Random number after seed 42: {random.random()}")
    print(f"NumPy random after seed 42: {np.random.random()}")
    
    # Test reproducibility
    set_seed(42)
    r1 = random.random()
    n1 = np.random.random()
    
    set_seed(42)
    r2 = random.random()
    n2 = np.random.random()
    
    assert r1 == r2, "Random reproducibility failed"
    assert n1 == n2, "NumPy reproducibility failed"
    print("Reproducibility test passed!")
    
    # Test seed manager
    manager = SeedManager(base_seed=42, num_runs=5)
    print(f"Seeds for 5 runs: {manager.seeds}")
    
    for i, seed in enumerate(manager):
        print(f"Run {i}: seed = {seed}")
