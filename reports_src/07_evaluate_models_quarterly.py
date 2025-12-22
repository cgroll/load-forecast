# %%
"""Evaluate Quarterly Model Training Results.

This report:
1. Loads training results from models/quarterly_split/training_results.json
2. Loads the original data to get actual load values and predictions
3. Creates comprehensive visualizations:
   - Line plots comparing all three models per quarter
   - Scatter plots for each model per quarter
   - Error distribution histograms
   - Metrics comparison bar charts across quarters
   - Aggregated metrics summary
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

print(f"Loaded results for {len(training_data['by_quarter'])} quarters")
print(f"Training timestamp: {training_data['timestamp']}")

# %%
# Extract configuration and results
config = training_data['config']
by_quarter = training_data['by_quarter']
aggregated = training_data['aggregated']

print("\n" + "="*70)
print("EXPERIMENT CONFIGURATION")
print("="*70)
print(f"Test days per quarter: {config['test_days']}")
print(f"Minimum data coverage: {config['min_data_coverage']*100}%")
print(f"Random seed: {config['random_seed']}")

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
# Recreate predictions for visualization
# We need to re-run predictions since we didn't save them
print("\n" + "="*70)
print("RECREATING PREDICTIONS FOR VISUALIZATION")
print("="*70)

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

quarters, quarter_info = split_into_calendar_quarters(data_clean)

# Recreate predictions
visualization_data = []

for i, (quarter, (year, quarter_num)) in enumerate(zip(quarters, quarter_info), 1):
    print(f"Processing Q{quarter_num} {year}...")

    X_train, X_test, y_train, y_test, test_index = prepare_train_test_split(
        quarter, test_days=config['test_days'], min_coverage=config['min_data_coverage']
    )

    # Baseline
    y_pred_baseline = np.zeros(len(y_test))
    y_pred_baseline[0] = y_train.iloc[-1]
    for j in range(1, len(y_test)):
        y_pred_baseline[j] = y_test.iloc[j-1]

    # Direct XGBoost
    direct_model = XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=config['random_seed'], n_jobs=-1,
        objective='reg:squarederror'
    )
    direct_model.fit(X_train, y_train)
    y_pred_direct = direct_model.predict(X_test)

    # OpenSTEF XGBoost
    pj_dict = dict(
        id=i, model="xgb", quantiles=[0.5], forecast_type="demand",
        lat=52.0, lon=5.0, horizon_minutes=15, resolution_minutes=15,
        name=f"Q{quarter_num}_{year}", hyper_params={},
        feature_names=None, default_modelspecs=None,
    )
    pj = PredictionJobDataClass(**pj_dict)

    train_data_full = quarter[quarter.index.isin(X_train.index)].copy()
    train_data_full['horizon'] = pj['horizon_minutes'] / 60
    cols_ordered = ['load'] + [col for col in train_data_full.columns if col not in ['load', 'horizon']] + ['horizon']
    train_data_openstef = train_data_full[cols_ordered].copy()

    train_split, validation_split, _, _ = split_data_train_validation_test(
        train_data_openstef, test_fraction=0.0, back_test=False
    )

    openstef_model = ModelCreator.create_model(pj["model"], quantiles=pj["quantiles"])
    train_x = train_split.iloc[:, 1:-1]
    train_y = train_split.iloc[:, 0]
    validation_x = validation_split.iloc[:, 1:-1]
    validation_y = validation_split.iloc[:, 0]

    eval_set = [(train_x, train_y), (validation_x, validation_y)]
    openstef_model.set_params(early_stopping_rounds=10, random_state=config['random_seed'])
    openstef_model.fit(train_x, train_y, eval_set=eval_set)
    y_pred_openstef = openstef_model.predict(X_test)

    visualization_data.append({
        'quarter_num': quarter_num,
        'year': year,
        'quarter_label': f"Q{quarter_num} {year}",
        'test_index': test_index,
        'y_test': y_test,
        'y_pred_baseline': y_pred_baseline,
        'y_pred_direct': y_pred_direct,
        'y_pred_openstef': y_pred_openstef,
    })

print("Predictions recreated for all quarters")

# %%
# Export metrics for DVC tracking
print("\n" + "="*70)
print("EXPORTING METRICS FOR DVC TRACKING")
print("="*70)

metrics_output = {
    'experiment': EXPERIMENT_NAME,
    'by_quarter': {},
    'aggregated': aggregated
}

for quarter_key, quarter_data in by_quarter.items():
    metrics_output['by_quarter'][quarter_key] = quarter_data['metrics']

# Save metrics
metrics_dir = Path('metrics')
metrics_dir.mkdir(exist_ok=True)
metrics_file = metrics_dir / f'{EXPERIMENT_NAME}_evaluation.json'

with open(metrics_file, 'w') as f:
    json.dump(metrics_output, f, indent=2)

print(f"✓ Metrics saved to: {metrics_file}")

# %%
# Summary table
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

quarter_labels = [v['quarter_label'] for v in by_quarter.values()]
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
# Aggregated metrics
print("\n" + "="*70)
print("AGGREGATED METRICS (MEAN ± STD)")
print("="*70)

for model in ['baseline', 'direct_xgb', 'openstef_xgb']:
    print(f"\n{model.upper()}:")
    for metric in ['rmse', 'mae', 'r2']:
        mean_val = aggregated[model][f'{metric}_mean']
        std_val = aggregated[model][f'{metric}_std']
        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

# %%
# Visualization 1: Aggregated metrics comparison (mean ± std across quarters)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

models = ['Baseline', 'Direct XGB', 'OpenSTEF XGB']
model_keys = ['baseline', 'direct_xgb', 'openstef_xgb']
colors = ['gray', 'blue', 'red']

# RMSE
rmse_means = [aggregated[m]['rmse_mean'] for m in model_keys]
rmse_stds = [aggregated[m]['rmse_std'] for m in model_keys]
axes[0].bar(models, rmse_means, yerr=rmse_stds, color=colors, alpha=0.7, edgecolor='black', capsize=10)
axes[0].set_ylabel('RMSE', fontsize=12)
axes[0].set_title('Average RMSE (Mean ± Std)', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, (m, s) in enumerate(zip(rmse_means, rmse_stds)):
    axes[0].text(i, m + s + 0.01*max(rmse_means), f'{m:.4f}±{s:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# MAE
mae_means = [aggregated[m]['mae_mean'] for m in model_keys]
mae_stds = [aggregated[m]['mae_std'] for m in model_keys]
axes[1].bar(models, mae_means, yerr=mae_stds, color=colors, alpha=0.7, edgecolor='black', capsize=10)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('Average MAE (Mean ± Std)', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for i, (m, s) in enumerate(zip(mae_means, mae_stds)):
    axes[1].text(i, m + s + 0.01*max(mae_means), f'{m:.4f}±{s:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# R²
r2_means = [aggregated[m]['r2_mean'] for m in model_keys]
r2_stds = [aggregated[m]['r2_std'] for m in model_keys]
axes[2].bar(models, r2_means, yerr=r2_stds, color=colors, alpha=0.7, edgecolor='black', capsize=10)
axes[2].set_ylabel('R² Score', fontsize=12)
axes[2].set_title('Average R² Score (Mean ± Std)', fontsize=14, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)
for i, (m, s) in enumerate(zip(r2_means, r2_stds)):
    axes[2].text(i, m + s + 0.01, f'{m:.4f}±{s:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Visualization 2: Metrics comparison across quarters (bar charts by quarter)
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
# Visualization 3: Time series - Line plots comparing all models per quarter
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
# Visualization 4: Scatter plots for each quarter (3 models per quarter)
for vdata in visualization_data:
    quarter_key = f"q{vdata['quarter_num']}_{vdata['year']}"
    metrics = by_quarter[quarter_key]['metrics']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    y_test = vdata['y_test'].values

    # Baseline
    axes[0].scatter(y_test, vdata['y_pred_baseline'], alpha=0.6, s=30,
                    edgecolors='k', linewidth=0.5, color='gray')
    min_val = min(y_test.min(), vdata['y_pred_baseline'].min())
    max_val = max(y_test.max(), vdata['y_pred_baseline'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('True Load', fontsize=11)
    axes[0].set_ylabel('Predicted Load', fontsize=11)
    axes[0].set_title(f"Baseline\nRMSE={metrics['baseline']['rmse']:.4f}, R²={metrics['baseline']['r2']:.4f}",
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='box')

    # Direct XGB
    axes[1].scatter(y_test, vdata['y_pred_direct'], alpha=0.6, s=30,
                    edgecolors='k', linewidth=0.5, color='blue')
    min_val = min(y_test.min(), vdata['y_pred_direct'].min())
    max_val = max(y_test.max(), vdata['y_pred_direct'].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('True Load', fontsize=11)
    axes[1].set_ylabel('Predicted Load', fontsize=11)
    axes[1].set_title(f"Direct XGBoost\nRMSE={metrics['direct_xgb']['rmse']:.4f}, R²={metrics['direct_xgb']['r2']:.4f}",
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')

    # OpenSTEF XGB
    axes[2].scatter(y_test, vdata['y_pred_openstef'], alpha=0.6, s=30,
                    edgecolors='k', linewidth=0.5, color='red')
    min_val = min(y_test.min(), vdata['y_pred_openstef'].min())
    max_val = max(y_test.max(), vdata['y_pred_openstef'].max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
    axes[2].set_xlabel('True Load', fontsize=11)
    axes[2].set_ylabel('Predicted Load', fontsize=11)
    axes[2].set_title(f"OpenSTEF XGBoost\nRMSE={metrics['openstef_xgb']['rmse']:.4f}, R²={metrics['openstef_xgb']['r2']:.4f}",
                      fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal', adjustable='box')

    fig.suptitle(f"{vdata['quarter_label']}: Scatter Plots", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# %%
# Visualization 5: Error distributions per quarter
for vdata in visualization_data:
    quarter_key = f"q{vdata['quarter_num']}_{vdata['year']}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    y_test = vdata['y_test'].values
    errors_baseline = y_test - vdata['y_pred_baseline']
    errors_direct = y_test - vdata['y_pred_direct']
    errors_openstef = y_test - vdata['y_pred_openstef']

    # Baseline
    axes[0].hist(errors_baseline, bins=40, alpha=0.7, color='gray', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0].axvline(errors_baseline.mean(), color='green', linestyle='--', linewidth=2,
                    label=f'Mean: {errors_baseline.mean():.4f}')
    axes[0].set_xlabel('Prediction Error', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f"Baseline (Std: {errors_baseline.std():.4f})", fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Direct XGB
    axes[1].hist(errors_direct, bins=40, alpha=0.7, color='blue', edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[1].axvline(errors_direct.mean(), color='green', linestyle='--', linewidth=2,
                    label=f'Mean: {errors_direct.mean():.4f}')
    axes[1].set_xlabel('Prediction Error', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f"Direct XGBoost (Std: {errors_direct.std():.4f})", fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    # OpenSTEF XGB
    axes[2].hist(errors_openstef, bins=40, alpha=0.7, color='red', edgecolor='black')
    axes[2].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[2].axvline(errors_openstef.mean(), color='green', linestyle='--', linewidth=2,
                    label=f'Mean: {errors_openstef.mean():.4f}')
    axes[2].set_xlabel('Prediction Error', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title(f"OpenSTEF XGBoost (Std: {errors_openstef.std():.4f})", fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"{vdata['quarter_label']}: Error Distributions", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# %%
print("\n" + "="*70)
print("EVALUATION REPORT COMPLETE")
print("="*70)
print(f"\n✓ Analyzed {len(visualization_data)} quarters")
print(f"✓ Compared 3 models: Baseline, Direct XGBoost, OpenSTEF XGBoost")
print(f"✓ Metrics exported to: {metrics_file}")
print("\nKey Findings:")
print(f"  - Best average RMSE: {min(rmse_means):.4f} ({models[rmse_means.index(min(rmse_means))]})")
print(f"  - Best average MAE: {min(mae_means):.4f} ({models[mae_means.index(min(mae_means))]})")
print(f"  - Best average R²: {max(r2_means):.4f} ({models[r2_means.index(max(r2_means))]})")
