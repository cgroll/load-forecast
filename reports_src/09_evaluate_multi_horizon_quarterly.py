# %%
"""Evaluate Multi-Horizon Quarterly Model Training Results.

This report evaluates multi-horizon models (forecasting 1-8 periods ahead, 15min to 2h)
trained on all quarters' training data and tested on each quarter's test period separately.

The report:
1. Loads training results from models/multi_horizon_quarterly/training_results.json
2. Creates comprehensive visualizations:
   - Overall metrics by horizon (all test data combined)
   - Performance degradation across horizons
   - Per-quarter metrics by horizon
   - Horizon comparison plots
3. Exports metrics to metrics/multi_horizon_quarterly_evaluation.json for DVC tracking

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
EXPERIMENT_NAME = "multi_horizon_quarterly"

print("="*70)
print("MULTI-HORIZON QUARTERLY MODEL EVALUATION REPORT")
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

    for model in ['baseline', 'direct_xgb']:
        m = h_result['metrics'][model]
        print(f"{h:<10} {h_min:<10} {model:<18} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")
    print("-"*80)

# %%
# Visualization 1: Metrics by horizon (line plots)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

horizons = [h['horizon'] for h in overall_horizon_results]
horizon_minutes = [h['horizon_minutes'] for h in overall_horizon_results]

models = ['baseline', 'direct_xgb']
model_labels = ['Baseline', 'Direct XGB']
colors = ['gray', 'blue']
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

        for model in ['baseline', 'direct_xgb']:
            m = h_result['metrics'][model]
            print(f"{qlabel:<15} {h:<10} {h_min:<10} {model:<18} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")

    print("-"*95)

# %%
# Visualization 3: Per-quarter performance by horizon
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
# Visualization 4: Horizon comparison heatmap (Direct XGB RMSE)
print("\n" + "="*70)
print("HEATMAP: RMSE BY QUARTER AND HORIZON (Direct XGBoost)")
print("="*70)

# Prepare data for heatmap
quarter_labels = [q['quarter_label'] for q in per_quarter_per_horizon_results]
horizon_labels = [f"H{h['horizon']}\n({h['horizon_minutes']}min)"
                  for h in per_quarter_per_horizon_results[0]['horizon_results']]

# Create matrix of RMSE values (Direct XGB)
rmse_matrix = []
for q_result in per_quarter_per_horizon_results:
    rmse_row = [h['metrics']['direct_xgb']['rmse'] for h in q_result['horizon_results']]
    rmse_matrix.append(rmse_row)

rmse_df = pd.DataFrame(rmse_matrix, index=quarter_labels, columns=horizon_labels)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(rmse_df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'RMSE'}, linewidths=0.5)
ax.set_title('Direct XGBoost RMSE by Quarter and Horizon', fontsize=14, fontweight='bold')
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
print(f"✓ Compared 2 models: Baseline and Direct XGBoost")
print(f"✓ Metrics exported to: {metrics_file}")

print("\nKey Findings:")
for model, label in zip(models, model_labels):
    rmse_h1 = overall_horizon_results[0]['metrics'][model]['rmse']
    rmse_h8 = overall_horizon_results[-1]['metrics'][model]['rmse']
    increase_pct = (rmse_h8 - rmse_h1) / rmse_h1 * 100
    print(f"  {label}: RMSE increases {increase_pct:.1f}% from H1 to H8 (15min → 120min)")
