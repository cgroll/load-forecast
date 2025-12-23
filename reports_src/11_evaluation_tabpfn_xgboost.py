# %%
"""Evaluate Multi-Horizon TabPFN vs XGBoost Quarterly Model Training Results.

This report evaluates multi-horizon models (forecasting 1-8 periods ahead, 15min to 2h)
comparing TabPFN and XGBoost approaches. Models were trained on all quarters' training data
and tested on each quarter's test period separately.

The report:
1. Loads training results from models/tabpfn_xgboost_quarterly/training_results.json
2. Creates comprehensive visualizations:
   - Overall metrics by horizon (all test data combined)
   - Performance degradation across horizons
   - Per-quarter metrics by horizon
   - Horizon comparison plots
   - TabPFN vs XGBoost comparison
3. Exports metrics to metrics/tabpfn_xgboost_quarterly_evaluation.json for DVC tracking

This script is designed to be run via generate_report.sh to produce HTML and Markdown outputs.
Uses Jupyter cell blocks (# %%) for interactive execution.
"""

# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import centralized paths
from load_forecast import Paths

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)

# %%
# Configuration
EXPERIMENT_NAME = "tabpfn_xgboost_quarterly"

print("="*70)
print("MULTI-HORIZON QUARTERLY MODEL EVALUATION: TabPFN vs XGBoost")
print("="*70)
print(f"Experiment: {EXPERIMENT_NAME}")

# %%
# Load training results
experiment_dir = Paths.MODELS / EXPERIMENT_NAME
results_file = experiment_dir / 'training_results.json'

print(f"\nLoading results from: {results_file}")

with open(results_file, 'r') as f:
    training_data = json.load(f)

overall_horizon_results = training_data['overall_horizon_results']
per_quarter_per_horizon_results = training_data['per_quarter_per_horizon_results']

print(f"Loaded results for multi-horizon models")
print(f"Training timestamp: {training_data['timestamp']}")
print(f"Description: {training_data.get('description', 'N/A')}")
print(f"Number of horizons: {len(overall_horizon_results)}")
print(f"Quarters evaluated: {len(per_quarter_per_horizon_results)}")

# %%
# Extract configuration
config = training_data['config']

print("\n" + "="*70)
print("EXPERIMENT CONFIGURATION")
print("="*70)
print(f"Number of horizons: {config['num_horizons']} (15min to {config['num_horizons']*15}min ahead)")
print(f"Test days per quarter: {config['test_days']}")
print(f"Minimum data coverage: {config['min_data_coverage']*100}%")
print(f"Random seed: {config['random_seed']}")
print(f"Time-based features (known ahead): {config['num_known_ahead_features']}")
print(f"Forecast features (weather/market/lags): {config['num_forecast_features']}")
print(f"Other features: {config.get('num_other_features', 0)}")
print(f"Total features: {config['total_features']}")

# %%
# SECTION 1: OVERALL METRICS BY HORIZON
print("\n" + "="*70)
print("OVERALL METRICS BY HORIZON (ALL TEST DATA COMBINED)")
print("="*70)

print(f"\n{'Horizon':<10} {'Minutes':<10} {'Model':<18} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-"*80)

for h_result in overall_horizon_results:
    h = h_result['horizon']
    h_min = h_result['horizon_minutes']

    for model in ['tabpfn', 'direct_xgb']:
        m = h_result['metrics'][model]
        print(f"{h:<10} {h_min:<10} {model:<18} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")
    print("-"*80)

# %%
# Visualization 1: Metrics by horizon (line plots)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

horizons = [h['horizon'] for h in overall_horizon_results]
horizon_minutes = [h['horizon_minutes'] for h in overall_horizon_results]

models = ['tabpfn', 'direct_xgb']
model_labels = ['TabPFN', 'Direct XGB']
colors = ['purple', 'blue']
markers = ['o', 's']

# RMSE by horizon
for model, label, color, marker in zip(models, model_labels, colors, markers):
    rmse_vals = [h['metrics'][model]['rmse'] for h in overall_horizon_results]
    axes[0].plot(horizon_minutes, rmse_vals, label=label, color=color, marker=marker,
                 linewidth=2.5, markersize=8, alpha=0.8)

axes[0].set_xlabel('Forecast Horizon (minutes)', fontsize=12)
axes[0].set_ylabel('RMSE', fontsize=12)
axes[0].set_title('RMSE vs Forecast Horizon', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(horizon_minutes)

# MAE by horizon
for model, label, color, marker in zip(models, model_labels, colors, markers):
    mae_vals = [h['metrics'][model]['mae'] for h in overall_horizon_results]
    axes[1].plot(horizon_minutes, mae_vals, label=label, color=color, marker=marker,
                 linewidth=2.5, markersize=8, alpha=0.8)

axes[1].set_xlabel('Forecast Horizon (minutes)', fontsize=12)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('MAE vs Forecast Horizon', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(horizon_minutes)

# R² by horizon
for model, label, color, marker in zip(models, model_labels, colors, markers):
    r2_vals = [h['metrics'][model]['r2'] for h in overall_horizon_results]
    axes[2].plot(horizon_minutes, r2_vals, label=label, color=color, marker=marker,
                 linewidth=2.5, markersize=8, alpha=0.8)

axes[2].set_xlabel('Forecast Horizon (minutes)', fontsize=12)
axes[2].set_ylabel('R² Score', fontsize=12)
axes[2].set_title('R² Score vs Forecast Horizon', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(horizon_minutes)

plt.tight_layout()
plt.show()

# %%
# Visualization 2: Performance degradation (relative to horizon 1)
print("\n" + "="*70)
print("PERFORMANCE DEGRADATION ANALYSIS")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for model, label, color, marker in zip(models, model_labels, colors, markers):
    # RMSE increase (%)
    rmse_vals = [h['metrics'][model]['rmse'] for h in overall_horizon_results]
    rmse_h1 = rmse_vals[0]
    rmse_increase = [(r - rmse_h1) / rmse_h1 * 100 for r in rmse_vals]

    axes[0].plot(horizon_minutes, rmse_increase, label=label, color=color, marker=marker,
                 linewidth=2.5, markersize=8, alpha=0.8)

    # R² decrease
    r2_vals = [h['metrics'][model]['r2'] for h in overall_horizon_results]
    r2_h1 = r2_vals[0]
    r2_decrease = [(r2_h1 - r) for r in r2_vals]

    axes[1].plot(horizon_minutes, r2_decrease, label=label, color=color, marker=marker,
                 linewidth=2.5, markersize=8, alpha=0.8)

axes[0].set_xlabel('Forecast Horizon (minutes)', fontsize=12)
axes[0].set_ylabel('RMSE Increase (%)', fontsize=12)
axes[0].set_title('RMSE Degradation from Horizon 1', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(horizon_minutes)
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

axes[1].set_xlabel('Forecast Horizon (minutes)', fontsize=12)
axes[1].set_ylabel('R² Decrease', fontsize=12)
axes[1].set_title('R² Degradation from Horizon 1', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(horizon_minutes)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.show()

print("\nDegradation Summary (Horizon 1 → Horizon 8):")
print("-"*70)
for model, label in zip(models, model_labels):
    rmse_vals = [h['metrics'][model]['rmse'] for h in overall_horizon_results]
    rmse_increase_pct = (rmse_vals[-1] - rmse_vals[0]) / rmse_vals[0] * 100

    r2_vals = [h['metrics'][model]['r2'] for h in overall_horizon_results]
    r2_decrease = r2_vals[0] - r2_vals[-1]

    print(f"{label:<18} RMSE: +{rmse_increase_pct:.1f}%  |  R²: -{r2_decrease:.4f}")

# %%
# Visualization 3: Model comparison (TabPFN vs XGBoost)
print("\n" + "="*70)
print("TABPFN VS XGBOOST COMPARISON")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# RMSE difference (TabPFN - XGBoost)
rmse_diff = []
for h_result in overall_horizon_results:
    diff = h_result['metrics']['tabpfn']['rmse'] - h_result['metrics']['direct_xgb']['rmse']
    rmse_diff.append(diff)

axes[0, 0].bar(horizon_minutes, rmse_diff, color=['green' if d < 0 else 'red' for d in rmse_diff], alpha=0.7)
axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0, 0].set_xlabel('Forecast Horizon (minutes)', fontsize=11)
axes[0, 0].set_ylabel('RMSE Difference', fontsize=11)
axes[0, 0].set_title('RMSE Difference (TabPFN - XGBoost)\nNegative = TabPFN better', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].set_xticks(horizon_minutes)

# MAE difference (TabPFN - XGBoost)
mae_diff = []
for h_result in overall_horizon_results:
    diff = h_result['metrics']['tabpfn']['mae'] - h_result['metrics']['direct_xgb']['mae']
    mae_diff.append(diff)

axes[0, 1].bar(horizon_minutes, mae_diff, color=['green' if d < 0 else 'red' for d in mae_diff], alpha=0.7)
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0, 1].set_xlabel('Forecast Horizon (minutes)', fontsize=11)
axes[0, 1].set_ylabel('MAE Difference', fontsize=11)
axes[0, 1].set_title('MAE Difference (TabPFN - XGBoost)\nNegative = TabPFN better', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].set_xticks(horizon_minutes)

# R² difference (TabPFN - XGBoost)
r2_diff = []
for h_result in overall_horizon_results:
    diff = h_result['metrics']['tabpfn']['r2'] - h_result['metrics']['direct_xgb']['r2']
    r2_diff.append(diff)

axes[1, 0].bar(horizon_minutes, r2_diff, color=['green' if d > 0 else 'red' for d in r2_diff], alpha=0.7)
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1, 0].set_xlabel('Forecast Horizon (minutes)', fontsize=11)
axes[1, 0].set_ylabel('R² Difference', fontsize=11)
axes[1, 0].set_title('R² Difference (TabPFN - XGBoost)\nPositive = TabPFN better', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].set_xticks(horizon_minutes)

# Win count
wins_tabpfn = sum(1 for d in rmse_diff if d < 0)
wins_xgb = sum(1 for d in rmse_diff if d >= 0)

axes[1, 1].bar(['TabPFN', 'XGBoost'], [wins_tabpfn, wins_xgb], color=['purple', 'blue'], alpha=0.7)
axes[1, 1].set_ylabel('Number of Horizons', fontsize=11)
axes[1, 1].set_title('Model Performance Wins\n(Based on RMSE)', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, (label, count) in enumerate([('TabPFN', wins_tabpfn), ('XGBoost', wins_xgb)]):
    axes[1, 1].text(i, count + 0.1, f'{count}/{len(horizon_minutes)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nModel Performance Summary:")
print(f"  TabPFN wins: {wins_tabpfn}/{len(horizon_minutes)} horizons (lower RMSE)")
print(f"  XGBoost wins: {wins_xgb}/{len(horizon_minutes)} horizons (lower RMSE)")

# %%
# SECTION 2: PER-QUARTER METRICS BY HORIZON
print("\n" + "="*70)
print("PER-QUARTER METRICS BY HORIZON")
print("="*70)

# Create a comprehensive table
print(f"\n{'Quarter':<15} {'Horizon':<10} {'Minutes':<10} {'Model':<18} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-"*95)

for q_result in per_quarter_per_horizon_results:
    qlabel = q_result['quarter_label']

    # Show first and last horizon only (to keep output manageable)
    for h_idx in [0, -1]:  # First and last horizon
        h_result = q_result['horizon_results'][h_idx]
        h = h_result['horizon']
        h_min = h_result['horizon_minutes']

        for model in ['tabpfn', 'direct_xgb']:
            m = h_result['metrics'][model]
            print(f"{qlabel:<15} {h:<10} {h_min:<10} {model:<18} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")

    print("-"*95)

# %%
# Visualization 4: Per-quarter performance by horizon
print("\n" + "="*70)
print("PER-QUARTER PERFORMANCE ACROSS HORIZONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = axes.flatten()

for idx, q_result in enumerate(per_quarter_per_horizon_results):
    ax = axes[idx]
    qlabel = q_result['quarter_label']

    h_mins = [h['horizon_minutes'] for h in q_result['horizon_results']]

    # Plot metrics for each model
    for model, label, color, marker in zip(models, model_labels, colors, markers):
        rmse_vals = [h['metrics'][model]['rmse'] for h in q_result['horizon_results']]
        ax.plot(h_mins, rmse_vals, label=label, color=color, marker=marker,
                linewidth=2, markersize=6, alpha=0.8)

    ax.set_xlabel('Forecast Horizon (minutes)', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title(f'{qlabel}: RMSE by Horizon', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(h_mins)

plt.tight_layout()
plt.show()

# %%
# Visualization 5: Horizon comparison heatmaps
print("\n" + "="*70)
print("HEATMAPS: RMSE BY QUARTER AND HORIZON")
print("="*70)

# Prepare data for heatmaps
quarter_labels = [q['quarter_label'] for q in per_quarter_per_horizon_results]
horizon_labels = [f"H{h['horizon']}\n({h['horizon_minutes']}min)"
                  for h in per_quarter_per_horizon_results[0]['horizon_results']]

fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# TabPFN heatmap
rmse_matrix_tabpfn = []
for q_result in per_quarter_per_horizon_results:
    rmse_row = [h['metrics']['tabpfn']['rmse'] for h in q_result['horizon_results']]
    rmse_matrix_tabpfn.append(rmse_row)

rmse_df_tabpfn = pd.DataFrame(rmse_matrix_tabpfn, index=quarter_labels, columns=horizon_labels)

sns.heatmap(rmse_df_tabpfn, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0],
            cbar_kws={'label': 'RMSE'}, linewidths=0.5)
axes[0].set_title('TabPFN RMSE by Quarter and Horizon', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Forecast Horizon', fontsize=12)
axes[0].set_ylabel('Quarter', fontsize=12)

# XGBoost heatmap
rmse_matrix_xgb = []
for q_result in per_quarter_per_horizon_results:
    rmse_row = [h['metrics']['direct_xgb']['rmse'] for h in q_result['horizon_results']]
    rmse_matrix_xgb.append(rmse_row)

rmse_df_xgb = pd.DataFrame(rmse_matrix_xgb, index=quarter_labels, columns=horizon_labels)

sns.heatmap(rmse_df_xgb, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1],
            cbar_kws={'label': 'RMSE'}, linewidths=0.5)
axes[1].set_title('Direct XGBoost RMSE by Quarter and Horizon', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Forecast Horizon', fontsize=12)
axes[1].set_ylabel('Quarter', fontsize=12)

plt.tight_layout()
plt.show()

# %%
# Visualization 6: Model comparison heatmap (difference)
print("\n" + "="*70)
print("HEATMAP: RMSE DIFFERENCE (TabPFN - XGBoost)")
print("="*70)

# Create difference matrix
rmse_diff_matrix = []
for q_result in per_quarter_per_horizon_results:
    diff_row = [
        h['metrics']['tabpfn']['rmse'] - h['metrics']['direct_xgb']['rmse']
        for h in q_result['horizon_results']
    ]
    rmse_diff_matrix.append(diff_row)

rmse_diff_df = pd.DataFrame(rmse_diff_matrix, index=quarter_labels, columns=horizon_labels)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(rmse_diff_df, annot=True, fmt='.4f', cmap='RdYlGn_r', center=0, ax=ax,
            cbar_kws={'label': 'RMSE Difference'}, linewidths=0.5)
ax.set_title('RMSE Difference: TabPFN - XGBoost\n(Negative = TabPFN better, Positive = XGBoost better)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Forecast Horizon', fontsize=12)
ax.set_ylabel('Quarter', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Export metrics for DVC tracking
print("\n" + "="*70)
print("EXPORTING METRICS FOR DVC TRACKING")
print("="*70)

metrics_output = {
    'experiment': EXPERIMENT_NAME,
    'overall_horizon_metrics': {}
}

# Overall metrics by horizon
for h_result in overall_horizon_results:
    h_key = f"h{h_result['horizon']}"
    metrics_output['overall_horizon_metrics'][h_key] = {
        'horizon': h_result['horizon'],
        'horizon_minutes': h_result['horizon_minutes'],
        'metrics': h_result['metrics']
    }

# Per-quarter metrics (summary: first and last horizon only to keep file small)
metrics_output['per_quarter_summary'] = {}
for q_result in per_quarter_per_horizon_results:
    q_key = f"q{q_result['quarter_num']}_{q_result['year']}"
    metrics_output['per_quarter_summary'][q_key] = {
        'quarter_label': q_result['quarter_label'],
        'h1_metrics': q_result['horizon_results'][0]['metrics'],
        'h8_metrics': q_result['horizon_results'][-1]['metrics']
    }

# Performance degradation summary
metrics_output['degradation_summary'] = {}
for model, label in zip(models, model_labels):
    rmse_vals = [h['metrics'][model]['rmse'] for h in overall_horizon_results]
    rmse_h1 = rmse_vals[0]
    rmse_h8 = rmse_vals[-1]
    rmse_increase_pct = (rmse_h8 - rmse_h1) / rmse_h1 * 100

    r2_vals = [h['metrics'][model]['r2'] for h in overall_horizon_results]
    r2_decrease = r2_vals[0] - r2_vals[-1]

    metrics_output['degradation_summary'][model] = {
        'rmse_h1': float(rmse_h1),
        'rmse_h8': float(rmse_h8),
        'rmse_increase_pct': float(rmse_increase_pct),
        'r2_h1': float(r2_vals[0]),
        'r2_h8': float(r2_vals[-1]),
        'r2_decrease': float(r2_decrease)
    }

# Model comparison summary
metrics_output['model_comparison'] = {
    'rmse_differences': {
        f'h{i+1}': float(d) for i, d in enumerate(rmse_diff)
    },
    'tabpfn_wins': int(wins_tabpfn),
    'xgb_wins': int(wins_xgb)
}

# Save metrics
metrics_dir = Path('metrics')
metrics_dir.mkdir(exist_ok=True)
metrics_file = metrics_dir / f'{EXPERIMENT_NAME}_evaluation.json'

with open(metrics_file, 'w') as f:
    json.dump(metrics_output, f, indent=2)

print(f"✓ Metrics saved to: {metrics_file}")

# %%
# Summary statistics
print("\n" + "="*70)
print("SUMMARY: BEST AND WORST PERFORMING HORIZONS")
print("="*70)

for model, label in zip(models, model_labels):
    print(f"\n{label}:")
    rmse_vals = [h['metrics'][model]['rmse'] for h in overall_horizon_results]

    best_h_idx = np.argmin(rmse_vals)
    worst_h_idx = np.argmax(rmse_vals)

    best_h = overall_horizon_results[best_h_idx]
    worst_h = overall_horizon_results[worst_h_idx]

    print(f"  Best:  Horizon {best_h['horizon']} ({best_h['horizon_minutes']}min) - RMSE: {rmse_vals[best_h_idx]:.4f}")
    print(f"  Worst: Horizon {worst_h['horizon']} ({worst_h['horizon_minutes']}min) - RMSE: {rmse_vals[worst_h_idx]:.4f}")
    print(f"  Range: {rmse_vals[worst_h_idx] - rmse_vals[best_h_idx]:.4f} ({(rmse_vals[worst_h_idx] / rmse_vals[best_h_idx] - 1)*100:.1f}% increase)")

# %%
print("\n" + "="*70)
print("EVALUATION REPORT COMPLETE")
print("="*70)
print(f"\n✓ Analyzed {config['num_horizons']} forecast horizons")
print(f"✓ Evaluated {len(per_quarter_per_horizon_results)} quarters")
print(f"✓ Compared 2 models: TabPFN and Direct XGBoost")
print(f"✓ Metrics exported to: {metrics_file}")

print("\nKey Findings:")
for model, label in zip(models, model_labels):
    rmse_h1 = overall_horizon_results[0]['metrics'][model]['rmse']
    rmse_h8 = overall_horizon_results[-1]['metrics'][model]['rmse']
    increase_pct = (rmse_h8 - rmse_h1) / rmse_h1 * 100
    print(f"  {label}: RMSE increases {increase_pct:.1f}% from H1 to H8 (15min → 120min)")

print("\nModel Comparison:")
print(f"  TabPFN wins: {wins_tabpfn}/{len(horizon_minutes)} horizons")
print(f"  XGBoost wins: {wins_xgb}/{len(horizon_minutes)} horizons")

# Calculate average RMSE improvement
avg_rmse_improvement = np.mean(rmse_diff)
if avg_rmse_improvement < 0:
    print(f"  Average RMSE improvement: TabPFN is {abs(avg_rmse_improvement):.4f} better on average")
else:
    print(f"  Average RMSE improvement: XGBoost is {avg_rmse_improvement:.4f} better on average")
