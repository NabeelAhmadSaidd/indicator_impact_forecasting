# Do Technical Indicators Improve Deep Learning Forecasts?

This repository contains the code, data, and results for the research paper "Do Technical Indicators Improve Deep Learning Forecasts? An Empirical Ablation Study Across Asset Classes". The study evaluates the impact of technical indicators on deep learning-based financial time series forecasting across multiple asset classes.

## Code and Experiments

The repository includes Python code and Jupyter notebooks to reproduce the experiments conducted in the paper. The experiments use LSTM and GRU models to forecast next-day prices for various financial assets, comparing performance with and without technical indicators.

## Data

The study uses historical OHLCV (Open, High, Low, Close, Volume) data for 10 assets across 5 categories: commodities, cryptocurrencies, equities, forex, and indices. Raw data must be downloaded separately and placed in the `data/raw/` directory. Processed data and features are generated during the experiment reproduction.

## Reproduce Results

To reproduce the results:

1. Ensure all dependencies are installed (see Usage).
2. Download and place raw data in `data/raw/`.
3. Run the notebooks in order:
   - `notebooks/03_dataset_creation.ipynb` - Creates datasets and features.
   - `notebooks/04_model_training.ipynb` - Trains the models.
   - `notebooks/05_results_visualization.ipynb` - Generates visualizations and summaries.

Alternatively, use the main script: `python src/main.py`.

## Usage

1. Clone the repository.
2. Create and activate the conda environment: `conda env create -f environment.yml` and `conda activate timeseries`.
3. Install additional dependencies if needed: `pip install -r requirements.txt` (if present).
4. Run scripts or notebooks as described above.

## Results

The `results/` directory contains experiment outputs, including model predictions, performance metrics, statistical analyses, and visualizations. Key outputs include:
- Performance summaries for each asset and configuration.
- Tables comparing LSTM vs. GRU and different feature groups.
- Figures showing volatility effects and asset category performance.

## Project Structure

- `configs/` - Configuration files for experiments.
- `data/` - Raw and processed data directories.
- `logs/` - Training logs.
- `models/` - Saved model checkpoints.
- `notebooks/` - Jupyter notebooks for EDA, analysis, dataset creation, training, and visualization.
- `paper/` - LaTeX source for the research paper.
- `results/` - Experiment results and summaries.
- `scripts/` - Utility scripts for aggregation and figure generation.
- `src/` - Source code modules (data, features, models, training, evaluation, utils).

## Citation

If you use this code or findings in your work, please cite the paper:

```
Ahmad, N. (2026). Do Technical Indicators Improve Deep Learning Forecasts? An Empirical Ablation Study Across Asset Classes.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.