"""
Aggregate all experimental results for paper generation.
"""
import os
import json
import csv
from collections import defaultdict
import numpy as np
from scipy import stats

# Define structure
categories = ['commodity', 'crypto', 'equity', 'forex', 'index']
assets_by_category = {
    'commodity': ['crudeoil', 'gold'],
    'crypto': ['bitcoin', 'ethereum'],
    'equity': ['apple', 'microsoft'],
    'forex': ['eurusd', 'usdjpy'],
    'index': ['nasdaq', 'sp500']
}
groups = ['group0_ohlcv', 'group1_momentum', 'group2_trend', 'group3_volatility', 'group4_all']
group_names = {
    'group0_ohlcv': 'E0: OHLCV Only',
    'group1_momentum': 'E1: OHLCV + Momentum',
    'group2_trend': 'E2: OHLCV + Trend',
    'group3_volatility': 'E3: OHLCV + Volatility',
    'group4_all': 'E4: OHLCV + All'
}
models = ['lstm', 'gru']
seeds = ['42', '123', '456', '739', '1126']

# Collect all metrics
aggregated_results = []
individual_results = []
base_path = 'results'

for category in categories:
    for asset in assets_by_category[category]:
        for group in groups:
            for model in models:
                # Get aggregated metrics
                agg_path = os.path.join(base_path, category, asset, group, model, 'aggregated', 'metrics.csv')
                if os.path.exists(agg_path):
                    with open(agg_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            result = {
                                'category': category,
                                'asset': asset,
                                'group': group,
                                'model': model,
                                'rmse_mean': float(row['rmse_mean']),
                                'rmse_std': float(row['rmse_std']),
                                'mae_mean': float(row['mae_mean']),
                                'mae_std': float(row['mae_std']),
                                'mape_mean': float(row['mape_mean']),
                                'mape_std': float(row['mape_std']),
                                'da_mean': float(row['directional_accuracy_mean']),
                                'da_std': float(row['directional_accuracy_std'])
                            }
                            aggregated_results.append(result)
                
                # Collect individual seed results for statistical tests
                for seed in seeds:
                    seed_path = os.path.join(base_path, category, asset, group, model, seed, 'metrics.csv')
                    if os.path.exists(seed_path):
                        with open(seed_path, 'r') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                result = {
                                    'category': category,
                                    'asset': asset,
                                    'group': group,
                                    'model': model,
                                    'seed': seed,
                                    'rmse': float(row['rmse']),
                                    'mae': float(row['mae']),
                                    'mape': float(row['mape']),
                                    'da': float(row['directional_accuracy'])
                                }
                                individual_results.append(result)

print("=" * 80)
print("COMPREHENSIVE EXPERIMENTAL RESULTS ANALYSIS")
print("=" * 80)
print()
print("Total aggregated configurations:", len(aggregated_results))
print("Total individual experiments:", len(individual_results))
print()

# ============================================================================
# OVERALL FEATURE GROUP PERFORMANCE
# ============================================================================
print("=" * 80)
print("TABLE 1: OVERALL FEATURE GROUP PERFORMANCE")
print("=" * 80)

by_group = defaultdict(list)
for r in aggregated_results:
    by_group[r['group']].append(r)

print("\nGroup\t\t\t\tRMSE\t\t\tMAE\t\t\tMAPE\t\t\tDA")
print("-" * 100)
for group in groups:
    rmses = [r['rmse_mean'] for r in by_group[group]]
    maes = [r['mae_mean'] for r in by_group[group]]
    mapes = [r['mape_mean'] for r in by_group[group]]
    das = [r['da_mean'] for r in by_group[group]]
    
    print("{:<24}\t{:.4f}±{:.4f}\t\t{:.4f}±{:.4f}\t\t{:.2f}±{:.2f}\t\t{:.2f}±{:.2f}".format(
        group_names[group],
        np.mean(rmses), np.std(rmses),
        np.mean(maes), np.std(maes),
        np.mean(mapes), np.std(mapes),
        np.mean(das), np.std(das)
    ))

# ============================================================================
# LSTM vs GRU COMPARISON
# ============================================================================
print()
print("=" * 80)
print("TABLE 2: LSTM vs GRU COMPARISON")
print("=" * 80)

by_model = defaultdict(list)
for r in aggregated_results:
    by_model[r['model']].append(r)

print("\nModel\t\tRMSE\t\t\tMAE\t\t\tMAPE\t\t\tDA")
print("-" * 80)
for model in models:
    rmses = [r['rmse_mean'] for r in by_model[model]]
    maes = [r['mae_mean'] for r in by_model[model]]
    mapes = [r['mape_mean'] for r in by_model[model]]
    das = [r['da_mean'] for r in by_model[model]]
    
    print("{:<8}\t{:.4f}±{:.4f}\t\t{:.4f}±{:.4f}\t\t{:.2f}±{:.2f}\t\t{:.2f}±{:.2f}".format(
        model.upper(),
        np.mean(rmses), np.std(rmses),
        np.mean(maes), np.std(maes),
        np.mean(mapes), np.std(mapes),
        np.mean(das), np.std(das)
    ))

# Statistical test for LSTM vs GRU
lstm_rmses = [r['rmse'] for r in individual_results if r['model'] == 'lstm']
gru_rmses = [r['rmse'] for r in individual_results if r['model'] == 'gru']
t_stat, p_val = stats.ttest_ind(lstm_rmses, gru_rmses)
print("\nPaired t-test (LSTM vs GRU RMSE):")
print("  t-statistic: {:.4f}".format(t_stat))
print("  p-value: {:.6f}".format(p_val))
print("  Significant at alpha=0.05:", "Yes" if p_val < 0.05 else "No")

# Cohen's d effect size
cohens_d = (np.mean(lstm_rmses) - np.mean(gru_rmses)) / np.sqrt((np.std(lstm_rmses)**2 + np.std(gru_rmses)**2) / 2)
print("  Cohen's d: {:.4f}".format(cohens_d))

# ============================================================================
# BY ASSET CATEGORY
# ============================================================================
print()
print("=" * 80)
print("TABLE 3: PERFORMANCE BY ASSET CATEGORY")
print("=" * 80)

by_category = defaultdict(list)
for r in aggregated_results:
    by_category[r['category']].append(r)

print("\nCategory\tRMSE\t\t\tMAE\t\t\tMAPE\t\t\tDA")
print("-" * 80)
for cat in categories:
    rmses = [r['rmse_mean'] for r in by_category[cat]]
    maes = [r['mae_mean'] for r in by_category[cat]]
    mapes = [r['mape_mean'] for r in by_category[cat]]
    das = [r['da_mean'] for r in by_category[cat]]
    
    print("{:<12}\t{:.4f}±{:.4f}\t\t{:.4f}±{:.4f}\t\t{:.2f}±{:.2f}\t\t{:.2f}±{:.2f}".format(
        cat.capitalize(),
        np.mean(rmses), np.std(rmses),
        np.mean(maes), np.std(maes),
        np.mean(mapes), np.std(mapes),
        np.mean(das), np.std(das)
    ))

# ============================================================================
# HIGH VS LOW VOLATILITY COMPARISON
# ============================================================================
print()
print("=" * 80)
print("HIGH VOLATILITY (Crypto, Commodity) vs LOW VOLATILITY (Equity, Forex, Index)")
print("=" * 80)

high_vol = [r for r in aggregated_results if r['category'] in ['crypto', 'commodity']]
low_vol = [r for r in aggregated_results if r['category'] in ['equity', 'forex', 'index']]

high_rmses = [r['rmse_mean'] for r in high_vol]
low_rmses = [r['rmse_mean'] for r in low_vol]
high_das = [r['da_mean'] for r in high_vol]
low_das = [r['da_mean'] for r in low_vol]

print("\nHigh Volatility Assets:")
print("  RMSE: {:.4f} ± {:.4f}".format(np.mean(high_rmses), np.std(high_rmses)))
print("  DA: {:.2f}% ± {:.2f}%".format(np.mean(high_das), np.std(high_das)))
print("\nLow Volatility Assets:")
print("  RMSE: {:.4f} ± {:.4f}".format(np.mean(low_rmses), np.std(low_rmses)))
print("  DA: {:.2f}% ± {:.2f}%".format(np.mean(low_das), np.std(low_das)))

# ============================================================================
# BASELINE VS INDICATOR GROUPS
# ============================================================================
print()
print("=" * 80)
print("FEATURE GROUP ABLATION: IMPROVEMENT OVER BASELINE (Group 0)")
print("=" * 80)

# For each asset and model, compute improvement
improvements = defaultdict(list)
for asset in [a for cat in categories for a in assets_by_category[cat]]:
    for model in models:
        baseline_results = [r for r in aggregated_results 
                          if r['asset'] == asset and r['model'] == model and r['group'] == 'group0_ohlcv']
        if baseline_results:
            baseline_rmse = baseline_results[0]['rmse_mean']
            for group in groups[1:]:
                group_results = [r for r in aggregated_results 
                               if r['asset'] == asset and r['model'] == model and r['group'] == group]
                if group_results:
                    group_rmse = group_results[0]['rmse_mean']
                    improvement = ((baseline_rmse - group_rmse) / baseline_rmse) * 100
                    improvements[group].append(improvement)

print("\nGroup\t\t\t\tMean Improvement (%)\tPositive Cases\tNegative Cases")
print("-" * 80)
for group in groups[1:]:
    imps = improvements[group]
    positive = sum(1 for i in imps if i > 0)
    negative = sum(1 for i in imps if i < 0)
    print("{:<24}\t{:+.2f}% ± {:.2f}%\t\t{}\t\t{}".format(
        group_names[group],
        np.mean(imps), np.std(imps),
        positive, negative
    ))

# ============================================================================
# STATISTICAL SIGNIFICANCE TESTS
# ============================================================================
print()
print("=" * 80)
print("STATISTICAL SIGNIFICANCE: INDICATOR GROUPS vs BASELINE")
print("=" * 80)

# Collect baseline RMSEs per asset-model combination
baseline_individual = {}
for r in individual_results:
    if r['group'] == 'group0_ohlcv':
        key = (r['asset'], r['model'], r['seed'])
        baseline_individual[key] = r['rmse']

for group in groups[1:]:
    group_rmses = []
    baseline_rmses = []
    for r in individual_results:
        if r['group'] == group:
            key = (r['asset'], r['model'], r['seed'])
            if key in baseline_individual:
                group_rmses.append(r['rmse'])
                baseline_rmses.append(baseline_individual[key])
    
    if len(group_rmses) > 0:
        t_stat, p_val = stats.ttest_rel(baseline_rmses, group_rmses)
        cohens_d = (np.mean(baseline_rmses) - np.mean(group_rmses)) / np.sqrt((np.std(baseline_rmses)**2 + np.std(group_rmses)**2) / 2)
        
        print("\n{} vs Baseline:".format(group_names[group]))
        print("  Paired t-test: t={:.4f}, p={:.6f}".format(t_stat, p_val))
        print("  Significant at alpha=0.05:", "Yes" if p_val < 0.05 else "No")
        print("  Cohen's d: {:.4f}".format(cohens_d))
        print("  Improvement: {:.2f}%".format((np.mean(baseline_rmses) - np.mean(group_rmses)) / np.mean(baseline_rmses) * 100))

# ============================================================================
# BEST CONFIGURATIONS
# ============================================================================
print()
print("=" * 80)
print("BEST CONFIGURATIONS BY ASSET")
print("=" * 80)

for category in categories:
    print("\n{}:".format(category.upper()))
    for asset in assets_by_category[category]:
        asset_results = [r for r in aggregated_results if r['asset'] == asset]
        best = min(asset_results, key=lambda x: x['rmse_mean'])
        print("  {}: {} + {} (RMSE={:.4f}, DA={:.2f}%)".format(
            asset.capitalize(), best['model'].upper(), group_names[best['group']], 
            best['rmse_mean'], best['da_mean']
        ))

# ============================================================================
# VOLATILITY INDICATOR EFFECT BY CATEGORY
# ============================================================================
print()
print("=" * 80)
print("HYPOTHESIS H3: VOLATILITY INDICATORS BENEFIT CRYPTO MORE")
print("=" * 80)

for cat in categories:
    cat_results = [r for r in aggregated_results if r['category'] == cat]
    baseline = [r['rmse_mean'] for r in cat_results if r['group'] == 'group0_ohlcv']
    volatility = [r['rmse_mean'] for r in cat_results if r['group'] == 'group3_volatility']
    improvement = (np.mean(baseline) - np.mean(volatility)) / np.mean(baseline) * 100
    print("{}: Volatility indicator improvement = {:+.2f}%".format(cat.capitalize(), improvement))

# ============================================================================
# SUMMARY STATISTICS FOR PAPER
# ============================================================================
print()
print("=" * 80)
print("SUMMARY STATISTICS FOR PAPER")
print("=" * 80)

total_experiments = len(individual_results)
unique_configs = len(aggregated_results)
print("Total individual experiments:", total_experiments)
print("Unique configurations:", unique_configs)
print("Seeds per configuration:", 5)
print("Assets:", 10)
print("Feature groups:", 5)
print("Model architectures:", 2)

# Overall best and worst
best_overall = min(aggregated_results, key=lambda x: x['rmse_mean'])
worst_overall = max(aggregated_results, key=lambda x: x['rmse_mean'])
print("\nBest Configuration:")
print("  {} - {} + {} (RMSE={:.4f})".format(
    best_overall['asset'], best_overall['model'].upper(), 
    group_names[best_overall['group']], best_overall['rmse_mean']))
print("\nWorst Configuration:")
print("  {} - {} + {} (RMSE={:.4f})".format(
    worst_overall['asset'], worst_overall['model'].upper(), 
    group_names[worst_overall['group']], worst_overall['rmse_mean']))

# RMSE range
all_rmses = [r['rmse_mean'] for r in aggregated_results]
print("\nRMSE Range: {:.4f} - {:.4f}".format(min(all_rmses), max(all_rmses)))
print("Mean RMSE: {:.4f} ± {:.4f}".format(np.mean(all_rmses), np.std(all_rmses)))

# DA range
all_das = [r['da_mean'] for r in aggregated_results]
print("\nDA Range: {:.2f}% - {:.2f}%".format(min(all_das), max(all_das)))
print("Mean DA: {:.2f}% ± {:.2f}%".format(np.mean(all_das), np.std(all_das)))

# Save comprehensive results to JSON for paper tables
output = {
    'summary': {
        'total_experiments': total_experiments,
        'unique_configs': unique_configs,
        'mean_rmse': float(np.mean(all_rmses)),
        'std_rmse': float(np.std(all_rmses)),
        'mean_da': float(np.mean(all_das)),
        'std_da': float(np.std(all_das)),
        'rmse_range': [float(min(all_rmses)), float(max(all_rmses))],
        'da_range': [float(min(all_das)), float(max(all_das))]
    },
    'by_group': {},
    'by_model': {},
    'by_category': {},
    'best_configs': {}
}

for group in groups:
    rmses = [r['rmse_mean'] for r in by_group[group]]
    maes = [r['mae_mean'] for r in by_group[group]]
    mapes = [r['mape_mean'] for r in by_group[group]]
    das = [r['da_mean'] for r in by_group[group]]
    output['by_group'][group] = {
        'rmse_mean': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
        'mae_mean': float(np.mean(maes)),
        'mae_std': float(np.std(maes)),
        'mape_mean': float(np.mean(mapes)),
        'mape_std': float(np.std(mapes)),
        'da_mean': float(np.mean(das)),
        'da_std': float(np.std(das))
    }

for model in models:
    rmses = [r['rmse_mean'] for r in by_model[model]]
    das = [r['da_mean'] for r in by_model[model]]
    output['by_model'][model] = {
        'rmse_mean': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
        'da_mean': float(np.mean(das)),
        'da_std': float(np.std(das))
    }

for cat in categories:
    rmses = [r['rmse_mean'] for r in by_category[cat]]
    das = [r['da_mean'] for r in by_category[cat]]
    output['by_category'][cat] = {
        'rmse_mean': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
        'da_mean': float(np.mean(das)),
        'da_std': float(np.std(das))
    }

for cat in categories:
    for asset in assets_by_category[cat]:
        asset_results = [r for r in aggregated_results if r['asset'] == asset]
        best = min(asset_results, key=lambda x: x['rmse_mean'])
        output['best_configs'][asset] = {
            'category': cat,
            'model': best['model'],
            'group': best['group'],
            'rmse': float(best['rmse_mean']),
            'da': float(best['da_mean'])
        }

with open('results/overall/paper_statistics.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved to results/overall/paper_statistics.json")
