"""
Main Pipeline Entry Point

This module serves as the main entry point for the indicator impact forecasting
pipeline. It orchestrates the complete workflow from data preprocessing to
model training and evaluation.

Usage:
    python -m src.main --asset BTCUSD --group group5_all --model lstm --run 1
    python -m src.main --mode preprocess --asset all
    python -m src.main --mode train --asset BTCUSD --group all --model all
    python -m src.main --mode evaluate --asset BTCUSD
    python -m src.main --mode ablation --runs 5
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config, get_project_root
from src.utils.logger import setup_logger, ExperimentLogger
from src.utils.seed import set_seed
from src.data.loader import DataLoader, load_asset_data
from src.data.preprocessing import preprocess_asset, DataPreprocessor
from src.data.window_generator import DatasetGenerator
from src.training.trainer import Trainer, ExperimentRunner, train_single_model
from src.evaluation.metrics import calculate_all_metrics


logger = setup_logger('main')


def preprocess_data(
    assets: Optional[List[str]] = None,
    config: Optional[Config] = None
) -> None:
    """
    Preprocess raw data and add technical indicators.
    
    Args:
        assets: List of assets to preprocess. If None, preprocess all.
        config: Configuration object.
    """
    if config is None:
        config = Config.load()
    
    if assets is None or 'all' in assets:
        assets = config.get_all_assets()
    
    logger.info(f"Preprocessing {len(assets)} assets...")
    
    for asset in assets:
        try:
            logger.info(f"Processing {asset}...")
            preprocess_asset(asset, config, save_interim=True)
        except Exception as e:
            logger.error(f"Failed to preprocess {asset}: {e}")


def build_datasets(
    assets: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    config: Optional[Config] = None
) -> None:
    """
    Build windowed datasets for training.
    
    Args:
        assets: List of assets.
        groups: List of feature groups.
        config: Configuration object.
    """
    if config is None:
        config = Config.load()
    
    if assets is None or 'all' in assets:
        assets = config.get_all_assets()
    
    if groups is None or 'all' in groups:
        groups = list(config.experiments['groups'].keys())
    
    generator = DatasetGenerator(config)
    
    total = len(assets) * len(groups)
    current = 0
    
    for asset in assets:
        for group in groups:
            current += 1
            logger.info(f"[{current}/{total}] Building dataset: {asset}/{group}")
            
            try:
                generator.create_dataset(asset, group, save=True)
            except Exception as e:
                logger.error(f"Failed to build dataset for {asset}/{group}: {e}")


def train_model(
    asset: str,
    group: str,
    model_type: str,
    run_id: int = 1,
    config: Optional[Config] = None
) -> dict:
    """
    Train a single model.
    
    Args:
        asset: Asset name.
        group: Feature group name.
        model_type: Model type ('lstm' or 'gru').
        run_id: Run identifier.
        config: Configuration object.
        
    Returns:
        Training results dictionary.
    """
    if config is None:
        config = Config.load(model_type=model_type)
    
    runner = ExperimentRunner(config)
    
    return runner.run_experiment(
        asset_name=asset,
        group_name=group,
        model_type=model_type,
        run_id=run_id
    )


def run_ablation_study(
    assets: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    n_runs: int = 5,
    config: Optional[Config] = None
) -> None:
    """
    Run complete ablation study.
    
    Args:
        assets: List of assets.
        groups: List of feature groups.
        models: List of model types.
        n_runs: Number of runs per configuration.
        config: Configuration object.
    """
    if config is None:
        config = Config.load()
    
    if assets is None or 'all' in assets:
        assets = config.get_all_assets()
    
    if groups is None or 'all' in groups:
        groups = list(config.experiments['groups'].keys())
    
    if models is None or 'all' in models:
        models = config.experiments.get('models', ['lstm', 'gru'])
    
    runner = ExperimentRunner(config)
    
    logger.info("Starting ablation study...")
    logger.info(f"Assets: {assets}")
    logger.info(f"Groups: {groups}")
    logger.info(f"Models: {models}")
    logger.info(f"Runs per config: {n_runs}")
    
    results_df = runner.run_ablation_study(
        assets=assets,
        groups=groups,
        models=models,
        n_runs=n_runs
    )
    
    # Save results
    results_dir = config.get_path('results') / 'tables'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(results_dir / 'ablation_results.csv', index=False)
    
    # Save summary
    summary = runner.get_summary()
    summary.to_csv(results_dir / 'ablation_summary.csv')
    
    logger.info(f"Ablation study complete. Results saved to {results_dir}")


def evaluate_model(
    asset: str,
    group: str,
    model_type: str,
    run_id: int = 1,
    config: Optional[Config] = None
) -> dict:
    """
    Evaluate a trained model.
    
    Args:
        asset: Asset name.
        group: Feature group name.
        model_type: Model type.
        run_id: Run identifier.
        config: Configuration object.
        
    Returns:
        Evaluation metrics dictionary.
    """
    from src.data.loader import load_processed_data
    from src.models import get_model
    import numpy as np
    
    if config is None:
        config = Config.load(model_type=model_type)
    
    # Load model
    model_path = (
        config.get_path('checkpoints') /
        asset /
        model_type /
        f"run_{run_id:03d}" /
        f"{asset}_{group}_{model_type}_run{run_id:03d}_best.keras"
    )
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = get_model(model_type)
    model.load(str(model_path))
    
    # Load test data
    data = load_processed_data(asset, group, config)
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_test, y_pred)
    
    logger.info(f"Evaluation results for {asset}/{group}/{model_type}/run_{run_id}:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.6f}")
    
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Indicator Impact Forecasting Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess all assets
  python -m src.main --mode preprocess --asset all
  
  # Build datasets for specific asset and all groups
  python -m src.main --mode build --asset BTCUSD --group all
  
  # Train a single model
  python -m src.main --mode train --asset BTCUSD --group group5_all --model lstm --run 1
  
  # Run full ablation study
  python -m src.main --mode ablation --runs 5
  
  # Evaluate a model
  python -m src.main --mode evaluate --asset BTCUSD --group group5_all --model lstm --run 1
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['preprocess', 'build', 'train', 'evaluate', 'ablation', 'full'],
        default='full',
        help='Pipeline mode to run'
    )
    
    parser.add_argument(
        '--asset',
        type=str,
        nargs='+',
        default=['all'],
        help='Asset(s) to process'
    )
    
    parser.add_argument(
        '--group',
        type=str,
        nargs='+',
        default=['all'],
        help='Feature group(s) to use'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        nargs='+',
        default=['all'],
        help='Model type(s) to train'
    )
    
    parser.add_argument(
        '--run',
        type=int,
        default=1,
        help='Run ID for single training'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=5,
        help='Number of runs for ablation study'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level'
    )
    
    args = parser.parse_args()
    
    # Set global seed
    set_seed(42)
    
    # Load config
    config = Config.load()
    
    # Process assets argument
    assets = args.asset if 'all' not in args.asset else None
    groups = args.group if 'all' not in args.group else None
    models = args.model if 'all' not in args.model else None
    
    try:
        if args.mode == 'preprocess':
            preprocess_data(assets, config)
        
        elif args.mode == 'build':
            build_datasets(assets, groups, config)
        
        elif args.mode == 'train':
            if assets is None or groups is None or models is None:
                logger.error("For train mode, specify --asset, --group, and --model")
                return 1
            
            for asset in assets:
                for group in groups:
                    for model_type in models:
                        train_model(asset, group, model_type, args.run, config)
        
        elif args.mode == 'evaluate':
            if assets is None or groups is None or models is None:
                logger.error("For evaluate mode, specify --asset, --group, and --model")
                return 1
            
            for asset in assets:
                for group in groups:
                    for model_type in models:
                        evaluate_model(asset, group, model_type, args.run, config)
        
        elif args.mode == 'ablation':
            run_ablation_study(assets, groups, models, args.runs, config)
        
        elif args.mode == 'full':
            logger.info("Running full pipeline...")
            
            # Step 1: Preprocess
            logger.info("Step 1: Preprocessing data...")
            preprocess_data(assets, config)
            
            # Step 2: Build datasets
            logger.info("Step 2: Building datasets...")
            build_datasets(assets, groups, config)
            
            # Step 3: Run ablation study
            logger.info("Step 3: Running ablation study...")
            run_ablation_study(assets, groups, models, args.runs, config)
            
            logger.info("Full pipeline complete!")
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    sys.exit(main())
