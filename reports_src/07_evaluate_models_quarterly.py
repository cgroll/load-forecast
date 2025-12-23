# %%
"""Evaluate Combined Quarterly Model Training Results.

This report evaluates a single combined model trained on all quarters' training data
and tested on each quarter's test period separately.

The report:
1. Loads training results from models/quarterly_split/training_results.json
2. Recreates predictions for visualization
3. Creates comprehensive visualizations:
   - Overall metrics (all test data combined)
   - Time series plots for each quarter's test period
   - Scatter plots for overall performance
   - Per-quarter metrics comparison
4. Exports metrics to metrics/quarterly_split_evaluation.json for DVC tracking

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
EXPERIMENT_NAME = "quarterly_split"

print("="*70)
print("QUARTERLY MODEL EVALUATION REPORT")
print("="*70)
print(f"Experiment: {EXPERIMENT_NAME}")

# %%
# Load training results
experiment_dir = Paths.MODELS / EXPERIMENT_NAME
results_file = experiment_dir / 'training_results.json'

print(f"\nLoading results from: {results_file}")

with open(results_file, 'r') as f:
    training_data = json.load(f)

results = training_data['results']
per_quarter_metrics = results['per_quarter_metrics']
overall_metrics = results['overall_metrics']

print(f"Loaded results for combined model")
print(f"Training timestamp: {training_data['timestamp']}")
print(f"Description: {training_data.get('description', 'N/A')}")
print(f"Quarters evaluated: {len(per_quarter_metrics)}")

# %%
# Extract configuration and results
config = training_data['config']

print("\n" + "="*70)
print("EXPERIMENT CONFIGURATION")
print("="*70)
print(f"Test days per quarter: {config['test_days']}")
print(f"Minimum data coverage: {config['min_data_coverage']*100}%")
print(f"Random seed: {config['random_seed']}")
print(f"Training period: {results['train_start']} to {results['train_end']}")
print(f"Test period: {results['test_start']} to {results['test_end']}")

# %%
# Load original data to recreate predictions
print("\n" + "="*70)
print("LOADING DATA FOR VISUALIZATION")
print("="*70)

data_with_features = pd.read_csv(
    Paths.DATA_WITH_FEATURES,
    index_col=0,
    parse_dates=True
)

# Remove timezone info if present (simplifies date comparisons)
if hasattr(data_with_features.index, 'tz') and data_with_features.index.tz is not None:
    data_with_features.index = data_with_features.index.tz_localize(None)

# Keep full data for time series visualization
data_full = data_with_features.copy()

# Clean data for model training
data_clean = data_with_features.dropna()

print(f"Loaded data shape: {data_clean.shape}")
print(f"Full data shape (with NaN): {data_full.shape}")

# %%
# Recreate combined model predictions for visualization
print("\n" + "="*70)
print("RECREATING COMBINED MODEL PREDICTIONS")
print("="*70)
print("Training one model on ALL quarters, testing on each quarter separately")

from xgboost import XGBRegressor
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.model.model_creator import ModelCreator
from openstef.model_selection.model_selection import split_data_train_validation_test

# Set random seed
np.random.seed(config['random_seed'])

def split_into_calendar_quarters(df):
    """Split DataFrame into calendar quarters."""
    df_with_quarter = df.copy()
    df_with_quarter['year'] = df_with_quarter.index.year
    df_with_quarter['quarter'] = df_with_quarter.index.quarter

    unique_quarters = df_with_quarter[['year', 'quarter']].drop_duplicates().sort_values(['year', 'quarter'])

    quarters = []
    quarter_info = []

    for _, row in unique_quarters.iterrows():
        year = int(row['year'])
        quarter_num = int(row['quarter'])
        mask = (df_with_quarter['year'] == year) & (df_with_quarter['quarter'] == quarter_num)
        quarter_data = df.loc[mask]

        if len(quarter_data) > 0:
            quarters.append(quarter_data)
            quarter_info.append((year, quarter_num))

    return quarters, quarter_info

def find_test_period(df, target_col, test_days, min_coverage):
    """Find test period with sufficient data coverage."""
    df_with_date = df.copy()
    df_with_date['date'] = df_with_date.index.date
    expected_obs_per_day = 96

    daily_counts = df_with_date.groupby('date')[target_col].count()
    daily_coverage = daily_counts / expected_obs_per_day
    valid_days = daily_coverage[daily_coverage >= min_coverage].index

    test_days_to_use = min(len(valid_days), test_days)
    test_dates = sorted(valid_days)[-test_days_to_use:]
    test_df = df_with_date[df_with_date['date'].isin(test_dates)].drop('date', axis=1)

    if len(test_dates) > 0:
        first_test_date = pd.Timestamp(test_dates[0])
        cutoff_date = first_test_date - pd.Timedelta(days=1)
    else:
        cutoff_date = df.index.max()

    return cutoff_date, test_df, test_days_to_use

def prepare_train_test_split(quarter_df, target_col='load', test_days=14, min_coverage=0.95):
    """Split a quarter into train and test sets."""
    feature_cols = [col for col in quarter_df.columns if col != target_col]
    cutoff_date, test_df, actual_test_days = find_test_period(quarter_df, target_col, test_days, min_coverage)
    train_df = quarter_df[quarter_df.index <= cutoff_date]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, test_df.index

# Prepare quarterly splits
quarters, quarter_info = split_into_calendar_quarters(data_clean)

# Collect train/test data from each quarter
all_X_train = []
all_y_train = []
all_X_test = []
all_y_test = []
all_test_indices = []

print(f"\nPreparing data splits for {len(quarters)} quarters...")
for i, (quarter, (year, quarter_num)) in enumerate(zip(quarters, quarter_info), 1):
    X_train, X_test, y_train, y_test, test_index = prepare_train_test_split(
        quarter, test_days=config['test_days'], min_coverage=config['min_data_coverage']
    )

    all_X_train.append(X_train)
    all_y_train.append(y_train)
    all_X_test.append(X_test)
    all_y_test.append(y_test)
    all_test_indices.append(test_index)

    print(f"  Q{quarter_num} {year}: {len(X_train)} train, {len(X_test)} test")

# Combine all training data
X_train_combined = pd.concat(all_X_train, axis=0)
y_train_combined = pd.concat(all_y_train, axis=0)

print(f"\nCombined training data: {len(X_train_combined)} rows")
print(f"Training combined models...")

# Train Baseline model (persistence)
print("  - Baseline...")

# Train Direct XGBoost
print("  - Direct XGBoost...")
direct_model_combined = XGBRegressor(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=config['random_seed'], n_jobs=-1,
    objective='reg:squarederror'
)
direct_model_combined.fit(X_train_combined, y_train_combined)

# Train OpenSTEF XGBoost
print("  - OpenSTEF XGBoost...")
pj_dict_combined = dict(
    id=999, model="xgb", quantiles=[0.5], forecast_type="demand",
    lat=52.0, lon=5.0, horizon_minutes=15, resolution_minutes=15,
    name="combined_all_quarters", hyper_params={},
    feature_names=None, default_modelspecs=None,
)
pj_combined = PredictionJobDataClass(**pj_dict_combined)

train_data_full_combined = pd.concat([
    quarter[quarter.index.isin(X_train.index)].copy()
    for quarter, X_train in zip(quarters, all_X_train)
], axis=0)
train_data_full_combined['horizon'] = pj_combined['horizon_minutes'] / 60
cols_ordered = ['load'] + [col for col in train_data_full_combined.columns if col not in ['load', 'horizon']] + ['horizon']
train_data_openstef_combined = train_data_full_combined[cols_ordered].copy()

train_split_combined, validation_split_combined, _, _ = split_data_train_validation_test(
    train_data_openstef_combined, test_fraction=0.0, back_test=False
)

openstef_model_combined = ModelCreator.create_model(pj_combined["model"], quantiles=pj_combined["quantiles"])
train_x_combined = train_split_combined.iloc[:, 1:-1]
train_y_combined = train_split_combined.iloc[:, 0]
validation_x_combined = validation_split_combined.iloc[:, 1:-1]
validation_y_combined = validation_split_combined.iloc[:, 0]

eval_set_combined = [(train_x_combined, train_y_combined), (validation_x_combined, validation_y_combined)]
openstef_model_combined.set_params(early_stopping_rounds=10, random_state=config['random_seed'])
openstef_model_combined.fit(train_x_combined, train_y_combined, eval_set=eval_set_combined)

# Generate predictions for each quarter's test set
print(f"\nGenerating predictions for each quarter's test set...")
visualization_data = []

for i, ((year, quarter_num), X_test_q, y_test_q, test_idx_q) in enumerate(zip(quarter_info, all_X_test, all_y_test, all_test_indices)):
    print(f"  Q{quarter_num} {year}...")

    # Baseline predictions
    y_pred_baseline_q = np.zeros(len(y_test_q))
    train_before_q = y_train_combined[y_train_combined.index < test_idx_q.min()]
    if len(train_before_q) > 0:
        y_pred_baseline_q[0] = train_before_q.iloc[-1]
    else:
        y_pred_baseline_q[0] = y_train_combined.iloc[-1]

    for j in range(1, len(y_test_q)):
        y_pred_baseline_q[j] = y_test_q.iloc[j-1]

    # XGBoost predictions
    y_pred_direct_q = direct_model_combined.predict(X_test_q)
    y_pred_openstef_q = openstef_model_combined.predict(X_test_q)

    visualization_data.append({
        'quarter_num': quarter_num,
        'year': year,
        'quarter_label': f"Q{quarter_num} {year}",
        'test_index': test_idx_q,
        'y_test': y_test_q,
        'y_pred_baseline': y_pred_baseline_q,
        'y_pred_direct': y_pred_direct_q,
        'y_pred_openstef': y_pred_openstef_q,
    })

print("Predictions recreated for all quarters")

# %%
# Organize metrics into by_quarter structure
print("\n" + "="*70)
print("ORGANIZING METRICS")
print("="*70)

# Create by_quarter structure from per_quarter_metrics
by_quarter = {}
for quarter_key, quarter_data in per_quarter_metrics.items():
    by_quarter[quarter_key] = {
        'quarter_label': quarter_data['quarter_label'],
        'metrics': quarter_data['metrics']
    }

print(f"Organized metrics for {len(by_quarter)} quarters")

# %%
# SECTION 1: OVERALL METRICS ACROSS ALL QUARTERS
print("\n" + "="*70)
print("OVERALL METRICS (ALL TEST DATA COMBINED)")
print("="*70)

if overall_metrics:
    print(f"\n{'Model':<18} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-"*54)
    for model in ['baseline', 'direct_xgb', 'openstef_xgb']:
        m = overall_metrics[model]
        print(f"{model:<18} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")
else:
    print("No overall metrics available")

# %%
# Visualization 1: Overall metrics comparison (all test data combined)
if overall_metrics:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    models = ['Baseline', 'Direct XGB', 'OpenSTEF XGB']
    model_keys = ['baseline', 'direct_xgb', 'openstef_xgb']
    colors = ['gray', 'blue', 'red']

    # RMSE
    rmse_vals = [overall_metrics[m]['rmse'] for m in model_keys]
    axes[0].bar(models, rmse_vals, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].set_title('Overall RMSE (All Test Data)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for i, val in enumerate(rmse_vals):
        axes[0].text(i, val + 0.01*max(rmse_vals), f'{val:.4f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # MAE
    mae_vals = [overall_metrics[m]['mae'] for m in model_keys]
    axes[1].bar(models, mae_vals, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Overall MAE (All Test Data)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for i, val in enumerate(mae_vals):
        axes[1].text(i, val + 0.01*max(mae_vals), f'{val:.4f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # R²
    r2_vals = [overall_metrics[m]['r2'] for m in model_keys]
    axes[2].bar(models, r2_vals, color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('R² Score', fontsize=12)
    axes[2].set_title('Overall R² Score (All Test Data)', fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    for i, val in enumerate(r2_vals):
        axes[2].text(i, val + 0.01, f'{val:.4f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

# %%
# SECTION 2: TIME SERIES VISUALIZATIONS
print("\n" + "="*70)
print("TIME SERIES VISUALIZATIONS")
print("="*70)

# Visualization 2: Time series - Line plots comparing all models per quarter
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = axes.flatten()

for i, vdata in enumerate(visualization_data):
    ax = axes[i]

    test_index = vdata['test_index']
    y_test = vdata['y_test']

    # Get metrics from training data
    quarter_key = f"q{vdata['quarter_num']}_{vdata['year']}"
    metrics = by_quarter[quarter_key]['metrics']

    # Get full data for this quarter's test period (including NaN values)
    quarter_full = data_full[(data_full.index >= test_index.min()) & (data_full.index <= test_index.max())]

    # Plot true load from full dataset
    ax.plot(quarter_full.index, quarter_full['load'],
            label='True Load', linewidth=2.5, alpha=0.8, color='black', marker='o', markersize=3)

    # Plot baseline
    ax.plot(test_index, vdata['y_pred_baseline'],
            label=f"Baseline (RMSE={metrics['baseline']['rmse']:.4f})",
            linewidth=2, alpha=0.6, color='gray', linestyle=':', marker='x', markersize=2)

    # Plot Direct XGBoost
    ax.plot(test_index, vdata['y_pred_direct'],
            label=f"Direct XGB (RMSE={metrics['direct_xgb']['rmse']:.4f})",
            linewidth=2, alpha=0.7, color='blue', linestyle='--', marker='s', markersize=2)

    # Plot OpenSTEF XGBoost
    ax.plot(test_index, vdata['y_pred_openstef'],
            label=f"OpenSTEF XGB (RMSE={metrics['openstef_xgb']['rmse']:.4f})",
            linewidth=2, alpha=0.7, color='red', linestyle='-.', marker='^', markersize=2)

    ax.set_title(f"{vdata['quarter_label']}: Model Comparison",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Load', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %%
# SECTION 3: SCATTER PLOTS (Overall - All Test Data Combined)
print("\n" + "="*70)
print("SCATTER PLOTS (OVERALL - ALL TEST DATA)")
print("="*70)

if overall_metrics:
    # Combine all test data
    y_test_all = np.concatenate([vd['y_test'].values for vd in visualization_data])
    y_pred_baseline_all = np.concatenate([vd['y_pred_baseline'] for vd in visualization_data])
    y_pred_direct_all = np.concatenate([vd['y_pred_direct'] for vd in visualization_data])
    y_pred_openstef_all = np.concatenate([vd['y_pred_openstef'] for vd in visualization_data])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Baseline
    axes[0].scatter(y_test_all, y_pred_baseline_all, alpha=0.4, s=20,
                    edgecolors='k', linewidth=0.3, color='gray')
    min_val = min(y_test_all.min(), y_pred_baseline_all.min())
    max_val = max(y_test_all.max(), y_pred_baseline_all.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('True Load', fontsize=11)
    axes[0].set_ylabel('Predicted Load', fontsize=11)
    axes[0].set_title(f"Baseline\nRMSE={overall_metrics['baseline']['rmse']:.4f}, R²={overall_metrics['baseline']['r2']:.4f}",
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='box')

    # Direct XGB
    axes[1].scatter(y_test_all, y_pred_direct_all, alpha=0.4, s=20,
                    edgecolors='k', linewidth=0.3, color='blue')
    min_val = min(y_test_all.min(), y_pred_direct_all.min())
    max_val = max(y_test_all.max(), y_pred_direct_all.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('True Load', fontsize=11)
    axes[1].set_ylabel('Predicted Load', fontsize=11)
    axes[1].set_title(f"Direct XGBoost\nRMSE={overall_metrics['direct_xgb']['rmse']:.4f}, R²={overall_metrics['direct_xgb']['r2']:.4f}",
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')

    # OpenSTEF XGB
    axes[2].scatter(y_test_all, y_pred_openstef_all, alpha=0.4, s=20,
                    edgecolors='k', linewidth=0.3, color='red')
    min_val = min(y_test_all.min(), y_pred_openstef_all.min())
    max_val = max(y_test_all.max(), y_pred_openstef_all.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
    axes[2].set_xlabel('True Load', fontsize=11)
    axes[2].set_ylabel('Predicted Load', fontsize=11)
    axes[2].set_title(f"OpenSTEF XGBoost\nRMSE={overall_metrics['openstef_xgb']['rmse']:.4f}, R²={overall_metrics['openstef_xgb']['r2']:.4f}",
                      fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal', adjustable='box')

    fig.suptitle("Overall Performance: Scatter Plots (All Test Data)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# %%
# SECTION 4: PER-QUARTER METRICS
print("\n" + "="*70)
print("PER-QUARTER METRICS BREAKDOWN")
print("="*70)

print(f"\n{'Quarter':<15} {'Model':<18} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-"*75)

for quarter_key in sorted(by_quarter.keys()):
    q = by_quarter[quarter_key]
    qlabel = q['quarter_label']
    for model in ['baseline', 'direct_xgb', 'openstef_xgb']:
        m = q['metrics'][model]
        print(f"{qlabel:<15} {model:<18} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")
    print("-"*75)

# %%
# Visualization 4: Metrics comparison across quarters (bar charts by quarter)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

quarter_labels = [v['quarter_label'] for v in visualization_data]
x_pos = np.arange(len(quarter_labels))

# Prepare data
rmse_baseline = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]['metrics']['baseline']['rmse'] for v in visualization_data]
rmse_direct = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]['metrics']['direct_xgb']['rmse'] for v in visualization_data]
rmse_openstef = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]['metrics']['openstef_xgb']['rmse'] for v in visualization_data]

mae_baseline = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]['metrics']['baseline']['mae'] for v in visualization_data]
mae_direct = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]['metrics']['direct_xgb']['mae'] for v in visualization_data]
mae_openstef = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]['metrics']['openstef_xgb']['mae'] for v in visualization_data]

r2_baseline = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]['metrics']['baseline']['r2'] for v in visualization_data]
r2_direct = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]['metrics']['direct_xgb']['r2'] for v in visualization_data]
r2_openstef = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]['metrics']['openstef_xgb']['r2'] for v in visualization_data]

width = 0.25

# RMSE
axes[0].bar(x_pos - width, rmse_baseline, width, label='Baseline', color='gray', alpha=0.7, edgecolor='black')
axes[0].bar(x_pos, rmse_direct, width, label='Direct XGB', color='blue', alpha=0.7, edgecolor='black')
axes[0].bar(x_pos + width, rmse_openstef, width, label='OpenSTEF XGB', color='red', alpha=0.7, edgecolor='black')
axes[0].set_ylabel('RMSE', fontsize=12)
axes[0].set_title('RMSE Comparison Across Quarters', fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(quarter_labels, rotation=45, ha='right')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# MAE
axes[1].bar(x_pos - width, mae_baseline, width, label='Baseline', color='gray', alpha=0.7, edgecolor='black')
axes[1].bar(x_pos, mae_direct, width, label='Direct XGB', color='blue', alpha=0.7, edgecolor='black')
axes[1].bar(x_pos + width, mae_openstef, width, label='OpenSTEF XGB', color='red', alpha=0.7, edgecolor='black')
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('MAE Comparison Across Quarters', fontsize=14, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(quarter_labels, rotation=45, ha='right')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

# R²
axes[2].bar(x_pos - width, r2_baseline, width, label='Baseline', color='gray', alpha=0.7, edgecolor='black')
axes[2].bar(x_pos, r2_direct, width, label='Direct XGB', color='blue', alpha=0.7, edgecolor='black')
axes[2].bar(x_pos + width, r2_openstef, width, label='OpenSTEF XGB', color='red', alpha=0.7, edgecolor='black')
axes[2].set_ylabel('R² Score', fontsize=12)
axes[2].set_title('R² Score Comparison Across Quarters', fontsize=14, fontweight='bold')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(quarter_labels, rotation=45, ha='right')
axes[2].legend(fontsize=10)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Export metrics for DVC tracking
print("\n" + "="*70)
print("EXPORTING METRICS FOR DVC TRACKING")
print("="*70)

metrics_output = {
    'experiment': EXPERIMENT_NAME,
    'overall_metrics': overall_metrics,
    'per_quarter_metrics': {}
}

for quarter_key, quarter_data in per_quarter_metrics.items():
    metrics_output['per_quarter_metrics'][quarter_key] = quarter_data['metrics']

# Save metrics
metrics_dir = Path('metrics')
metrics_dir.mkdir(exist_ok=True)
metrics_file = metrics_dir / f'{EXPERIMENT_NAME}_evaluation.json'

with open(metrics_file, 'w') as f:
    json.dump(metrics_output, f, indent=2)

print(f"✓ Metrics saved to: {metrics_file}")

# %%
print("\n" + "="*70)
print("EVALUATION REPORT COMPLETE")
print("="*70)
print(f"\n✓ Analyzed {len(visualization_data)} quarters")
print(f"✓ Compared 3 models: Baseline, Direct XGBoost, OpenSTEF XGBoost")
print(f"✓ Metrics exported to: {metrics_file}")

if overall_metrics:
    print("\nOverall Results (All Test Data Combined):")
    for model in ['baseline', 'direct_xgb', 'openstef_xgb']:
        m = overall_metrics[model]
        print(f"  {model.upper()}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R²={m['r2']:.4f}")
