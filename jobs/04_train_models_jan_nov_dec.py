# %%
"""Train and Evaluate Models using Jan-Nov/Dec split.

This script combines training and evaluation:
1. Loads preprocessed feature-enriched data from data/processed/data_with_features.csv
2. Splits data: Jan-Nov for training, December for testing
3. Trains three models:
   a) Baseline (Persistence): Uses last known load value as prediction
   b) Direct XGBoost with manual configuration
   c) OpenSTEF XGBOpenstfRegressor with OpenSTEF's training approach
4. Evaluates models on December test period
5. Creates comprehensive visualizations:
   - Metrics comparison bar charts
   - Time series plots
   - Scatter plots for each model
   - Error distribution histograms
6. Saves training results to models/jan_nov_dec/training_results.json
7. Exports metrics to metrics/jan_nov_dec_evaluation.json for DVC tracking

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
from xgboost import XGBRegressor
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import centralized paths
from load_forecast import Paths

# OpenSTEF imports
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.model.model_creator import ModelCreator
from openstef.model_selection.model_selection import split_data_train_validation_test

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 8)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
# Configuration
EXPERIMENT_NAME = "jan_nov_dec"

print("="*70)
print("JAN-NOV vs DECEMBER MODEL TRAINING AND EVALUATION")
print("="*70)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Random seed: {RANDOM_SEED}")

# %%
# Load preprocessed feature-enriched data
print("\n" + "="*70)
print("LOADING FEATURE-ENRICHED DATA")
print("="*70)
print(f"Loading from: {Paths.DATA_WITH_FEATURES}")

# Ensure output directories exist
Paths.ensure_dirs()

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

print(f"Loaded data shape: {data_with_features.shape}")
print(f"Date range: {data_with_features.index.min()} to {data_with_features.index.max()}")
print(f"Number of features: {data_with_features.shape[1]}")

# %%
# Drop rows with missing values
print("\n" + "="*70)
print("CLEANING DATA - REMOVING ROWS WITH MISSING VALUES")
print("="*70)

print(f"Shape before cleaning: {data_with_features.shape}")
data_clean = data_with_features.dropna()
print(f"Shape after dropping NaN: {data_clean.shape}")
print(f"Date range: {data_clean.index.min()} to {data_clean.index.max()}")

# %%
# Split data: Jan-Nov for training, December for testing
print("\n" + "="*70)
print("SPLITTING DATA: JAN-NOV TRAINING, DECEMBER TESTING")
print("="*70)

# Identify the year
years_in_data = data_clean.index.year.unique()
print(f"Years in data: {years_in_data}")

# Use the last year for December testing
target_year = years_in_data[-1]
print(f"Using year {target_year} for test split")

# Split: Train on Jan-Nov, Test on December
train_mask = (data_clean.index.year == target_year) & (data_clean.index.month < 12)
test_mask = (data_clean.index.year == target_year) & (data_clean.index.month == 12)

# If we have multiple years, include all previous years in training
if len(years_in_data) > 1:
    prev_years_mask = data_clean.index.year < target_year
    train_mask = train_mask | prev_years_mask

train_data_full = data_clean[train_mask].copy()
test_data_full = data_clean[test_mask].copy()

print(f"\nTrain data: {len(train_data_full)} rows")
print(f"  Date range: {train_data_full.index.min()} to {train_data_full.index.max()}")
print(f"  Months included: {sorted(train_data_full.index.month.unique())}")

print(f"\nTest data (December): {len(test_data_full)} rows")
print(f"  Date range: {test_data_full.index.min()} to {test_data_full.index.max()}")

# %%
# Prepare data for models
print("\n" + "="*70)
print("PREPARING DATA FOR MODELS")
print("="*70)

target_col = 'load'
feature_cols = [col for col in data_clean.columns if col != target_col]

X_train = train_data_full[feature_cols]
y_train = train_data_full[target_col]
X_test = test_data_full[feature_cols]
y_test = test_data_full[target_col]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# %%
# 1. BASELINE MODEL (Persistence)
print("\n" + "="*70)
print("TRAINING BASELINE MODEL (PERSISTENCE)")
print("="*70)

y_pred_baseline = np.zeros(len(y_test))
y_pred_baseline[0] = y_train.iloc[-1]

for i in range(1, len(y_test)):
    y_pred_baseline[i] = y_test.iloc[i-1]

rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

print(f"  RMSE: {rmse_baseline:.4f}")
print(f"  MAE:  {mae_baseline:.4f}")
print(f"  R²:   {r2_baseline:.4f}")

# %%
# 2. DIRECT XGBOOST MODEL
print("\n" + "="*70)
print("TRAINING DIRECT XGBOOST MODEL")
print("="*70)

direct_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    objective='reg:squarederror'
)

direct_model.fit(X_train, y_train)
y_pred_direct = direct_model.predict(X_test)

rmse_direct = np.sqrt(mean_squared_error(y_test, y_pred_direct))
mae_direct = mean_absolute_error(y_test, y_pred_direct)
r2_direct = r2_score(y_test, y_pred_direct)

print(f"  RMSE: {rmse_direct:.4f}")
print(f"  MAE:  {mae_direct:.4f}")
print(f"  R²:   {r2_direct:.4f}")

# %%
# 3. OPENSTEF XGBOOST MODEL
print("\n" + "="*70)
print("TRAINING OPENSTEF XGBOOST MODEL")
print("="*70)

# Create prediction job configuration
pj_dict = dict(
    id=1,
    model="xgb",
    quantiles=[0.5],
    forecast_type="demand",
    lat=52.0,
    lon=5.0,
    horizon_minutes=15,
    resolution_minutes=15,
    name="JanNovDec",
    hyper_params={},
    feature_names=None,
    default_modelspecs=None,
)
pj = PredictionJobDataClass(**pj_dict)

# Prepare data in OpenSTEF format
horizon_value = pj['horizon_minutes'] / 60
train_data_full['horizon'] = horizon_value
test_data_full['horizon'] = horizon_value

cols_ordered = ['load'] + [col for col in train_data_full.columns if col not in ['load', 'horizon']] + ['horizon']
train_data_openstef = train_data_full[cols_ordered].copy()
test_data_openstef = test_data_full[cols_ordered].copy()

# Split train data into train/validation using OpenSTEF's method
train_split, validation_split, _, _ = split_data_train_validation_test(
    train_data_openstef,
    test_fraction=0.0,
    back_test=False,
)

# Create and train OpenSTEF model
openstef_model = ModelCreator.create_model(
    pj["model"],
    quantiles=pj["quantiles"],
)

train_x = train_split.iloc[:, 1:-1]
train_y = train_split.iloc[:, 0]
validation_x = validation_split.iloc[:, 1:-1]
validation_y = validation_split.iloc[:, 0]

eval_set = [(train_x, train_y), (validation_x, validation_y)]
openstef_model.set_params(early_stopping_rounds=10, random_state=RANDOM_SEED)

openstef_model.fit(train_x, train_y, eval_set=eval_set)

# Make predictions
test_x_openstef = test_data_openstef.iloc[:, 1:-1]
y_test_openstef = test_data_openstef.iloc[:, 0]
y_pred_openstef = openstef_model.predict(test_x_openstef)

rmse_openstef = np.sqrt(mean_squared_error(y_test_openstef, y_pred_openstef))
mae_openstef = mean_absolute_error(y_test_openstef, y_pred_openstef)
r2_openstef = r2_score(y_test_openstef, y_pred_openstef)

print(f"  RMSE: {rmse_openstef:.4f}")
print(f"  MAE:  {mae_openstef:.4f}")
print(f"  R²:   {r2_openstef:.4f}")

# %%
# Save training results
print("\n" + "="*70)
print("SAVING TRAINING RESULTS")
print("="*70)

# Create experiment output directory
experiment_dir = Paths.MODELS / EXPERIMENT_NAME
experiment_dir.mkdir(parents=True, exist_ok=True)

metrics = {
    'baseline': {
        'rmse': float(rmse_baseline),
        'mae': float(mae_baseline),
        'r2': float(r2_baseline)
    },
    'direct_xgb': {
        'rmse': float(rmse_direct),
        'mae': float(mae_direct),
        'r2': float(r2_direct)
    },
    'openstef_xgb': {
        'rmse': float(rmse_openstef),
        'mae': float(mae_openstef),
        'r2': float(r2_openstef)
    }
}

output_data = {
    'experiment': EXPERIMENT_NAME,
    'timestamp': datetime.now().isoformat(),
    'config': {
        'test_year': int(target_year),
        'test_month': 12,
        'random_seed': RANDOM_SEED
    },
    'hyperparameters': {
        'direct_xgb': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'openstef_xgb': {
            'early_stopping_rounds': 10
        }
    },
    'data_split': {
        'train_size': len(train_data_full),
        'test_size': len(test_data_full),
        'train_start': str(train_data_full.index.min()),
        'train_end': str(train_data_full.index.max()),
        'test_start': str(test_data_full.index.min()),
        'test_end': str(test_data_full.index.max())
    },
    'metrics': metrics
}

output_file = experiment_dir / 'training_results.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Training results saved to: {output_file}")

# %%
# Print training summary
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"\n{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-"*60)
print(f"{'Baseline':<20} {rmse_baseline:<12.4f} {mae_baseline:<12.4f} {r2_baseline:<12.4f}")
print(f"{'Direct XGBoost':<20} {rmse_direct:<12.4f} {mae_direct:<12.4f} {r2_direct:<12.4f}")
print(f"{'OpenSTEF XGBoost':<20} {rmse_openstef:<12.4f} {mae_openstef:<12.4f} {r2_openstef:<12.4f}")

# %%
# ============================================================
# EVALUATION AND VISUALIZATION
# ============================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# %%
# Visualization 1: Metrics comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = ['Baseline\nPersistence', 'Direct\nXGBoost', 'OpenSTEF\nXGBoost']
x_pos = np.arange(len(models))

# RMSE comparison
rmse_vals = [metrics['baseline']['rmse'], metrics['direct_xgb']['rmse'], metrics['openstef_xgb']['rmse']]
bars = axes[0].bar(x_pos, rmse_vals, color=['gray', 'blue', 'red'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('RMSE', fontsize=12)
axes[0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(models)
axes[0].grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, rmse_vals)):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01*max(rmse_vals),
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# MAE comparison
mae_vals = [metrics['baseline']['mae'], metrics['direct_xgb']['mae'], metrics['openstef_xgb']['mae']]
bars = axes[1].bar(x_pos, mae_vals, color=['gray', 'blue', 'red'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('MAE Comparison', fontsize=14, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(models)
axes[1].grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, mae_vals)):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.01*max(mae_vals),
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# R² comparison
r2_vals = [metrics['baseline']['r2'], metrics['direct_xgb']['r2'], metrics['openstef_xgb']['r2']]
bars = axes[2].bar(x_pos, r2_vals, color=['gray', 'blue', 'red'], alpha=0.7, edgecolor='black')
axes[2].set_ylabel('R² Score', fontsize=12)
axes[2].set_title('R² Comparison', fontsize=14, fontweight='bold')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(models)
min_r2 = min(r2_vals)
axes[2].set_ylim([min(0, min_r2 - 0.1), 1])
axes[2].grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, r2_vals)):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Visualization 2: Time series comparison using full data (including NaN values)
fig, ax = plt.subplots(figsize=(18, 8))

# Get December data from full dataset (with NaN values)
test_mask_full = (data_full.index.year == target_year) & (data_full.index.month == 12)
december_full = data_full[test_mask_full]

# Plot true load values from full dataset
ax.plot(december_full.index, december_full['load'],
        label='True Load', linewidth=2.5, alpha=0.8, color='black', marker='o', markersize=3)

# Plot predictions only where we have them (on clean data indices)
december_indices = test_data_full.index
ax.plot(december_indices, y_pred_baseline,
        label=f'Baseline - Persistence (RMSE={metrics["baseline"]["rmse"]:.4f}, MAE={metrics["baseline"]["mae"]:.4f})',
        linewidth=2, alpha=0.6, color='gray', linestyle=':', marker='x', markersize=2)

ax.plot(december_indices, y_pred_direct,
        label=f'Direct XGBoost (RMSE={metrics["direct_xgb"]["rmse"]:.4f}, MAE={metrics["direct_xgb"]["mae"]:.4f})',
        linewidth=2, alpha=0.7, color='blue', linestyle='--', marker='s', markersize=2)

ax.plot(december_indices, y_pred_openstef,
        label=f'OpenSTEF XGBoost (RMSE={metrics["openstef_xgb"]["rmse"]:.4f}, MAE={metrics["openstef_xgb"]["mae"]:.4f})',
        linewidth=2, alpha=0.7, color='red', linestyle='-.', marker='^', markersize=2)

ax.set_title('December Forecast Comparison: Baseline vs Direct XGBoost vs OpenSTEF XGBoost\n(Jan-Nov Training, December Testing)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=13)
ax.set_ylabel('Load', fontsize=13)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# Visualization 3: Scatter plots comparing predictions
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Baseline scatter
axes[0].scatter(y_test, y_pred_baseline, alpha=0.6, s=30,
                edgecolors='k', linewidth=0.5, color='gray')
min_val = min(y_test.min(), y_pred_baseline.min())
max_val = max(y_test.max(), y_pred_baseline.max())
axes[0].plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Perfect prediction', alpha=0.7)
axes[0].set_xlabel('True Load', fontsize=12)
axes[0].set_ylabel('Predicted Load', fontsize=12)
axes[0].set_title(f'Baseline - Persistence\n(RMSE={metrics["baseline"]["rmse"]:.4f}, R²={metrics["baseline"]["r2"]:.4f})',
                  fontsize=13, fontweight='bold')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal', adjustable='box')

# Direct XGBoost scatter
axes[1].scatter(y_test, y_pred_direct, alpha=0.6, s=30,
                edgecolors='k', linewidth=0.5, color='blue')
min_val = min(y_test.min(), y_pred_direct.min())
max_val = max(y_test.max(), y_pred_direct.max())
axes[1].plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Perfect prediction', alpha=0.7)
axes[1].set_xlabel('True Load', fontsize=12)
axes[1].set_ylabel('Predicted Load', fontsize=12)
axes[1].set_title(f'Direct XGBoost\n(RMSE={metrics["direct_xgb"]["rmse"]:.4f}, R²={metrics["direct_xgb"]["r2"]:.4f})',
                  fontsize=13, fontweight='bold')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal', adjustable='box')

# OpenSTEF scatter
axes[2].scatter(y_test, y_pred_openstef, alpha=0.6, s=30,
                edgecolors='k', linewidth=0.5, color='red')
min_val = min(y_test.min(), y_pred_openstef.min())
max_val = max(y_test.max(), y_pred_openstef.max())
axes[2].plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Perfect prediction', alpha=0.7)
axes[2].set_xlabel('True Load', fontsize=12)
axes[2].set_ylabel('Predicted Load', fontsize=12)
axes[2].set_title(f'OpenSTEF XGBoost\n(RMSE={metrics["openstef_xgb"]["rmse"]:.4f}, R²={metrics["openstef_xgb"]["r2"]:.4f})',
                  fontsize=13, fontweight='bold')
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)
axes[2].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

# %%
# Visualization 4: Error distribution comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Calculate errors
errors_baseline = y_test - y_pred_baseline
errors_direct = y_test - y_pred_direct
errors_openstef = y_test - y_pred_openstef

# Baseline error distribution
axes[0].hist(errors_baseline, bins=50, alpha=0.7, color='gray', edgecolor='black')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
axes[0].axvline(errors_baseline.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean error: {errors_baseline.mean():.4f}')
axes[0].set_xlabel('Prediction Error (True - Predicted)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'Baseline - Persistence Error Distribution\n(Std: {errors_baseline.std():.4f})',
                  fontsize=13, fontweight='bold')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3, axis='y')

# Direct XGBoost error distribution
axes[1].hist(errors_direct, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
axes[1].axvline(errors_direct.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean error: {errors_direct.mean():.4f}')
axes[1].set_xlabel('Prediction Error (True - Predicted)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'Direct XGBoost Error Distribution\n(Std: {errors_direct.std():.4f})',
                  fontsize=13, fontweight='bold')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3, axis='y')

# OpenSTEF error distribution
axes[2].hist(errors_openstef, bins=50, alpha=0.7, color='red', edgecolor='black')
axes[2].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
axes[2].axvline(errors_openstef.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean error: {errors_openstef.mean():.4f}')
axes[2].set_xlabel('Prediction Error (True - Predicted)', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].set_title(f'OpenSTEF XGBoost Error Distribution\n(Std: {errors_openstef.std():.4f})',
                  fontsize=13, fontweight='bold')
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %%
# Export metrics for DVC tracking
print("\n" + "="*70)
print("EXPORTING METRICS FOR DVC TRACKING")
print("="*70)

metrics_output = {
    'experiment': EXPERIMENT_NAME,
    'metrics': metrics
}

# Save metrics
metrics_dir = Path('metrics')
metrics_dir.mkdir(exist_ok=True)
metrics_file = metrics_dir / f'{EXPERIMENT_NAME}_evaluation.json'

with open(metrics_file, 'w') as f:
    json.dump(metrics_output, f, indent=2)

print(f"Metrics saved to: {metrics_file}")

# %%
print("\n" + "="*70)
print("TRAINING AND EVALUATION COMPLETE")
print("="*70)
print(f"\nAnalyzed December {target_year} test period")
print(f"Compared 3 models: Baseline, Direct XGBoost, OpenSTEF XGBoost")
print(f"Training results saved to: {output_file}")
print(f"Metrics exported to: {metrics_file}")

# Calculate improvements
rmse_improvement_direct = ((metrics['baseline']['rmse'] - metrics['direct_xgb']['rmse']) / metrics['baseline']['rmse']) * 100
rmse_improvement_openstef = ((metrics['baseline']['rmse'] - metrics['openstef_xgb']['rmse']) / metrics['baseline']['rmse']) * 100

print("\nKey Findings:")
print(f"  - Best RMSE: {min(rmse_vals):.4f} ({models[rmse_vals.index(min(rmse_vals))].replace(chr(10), ' ')})")
print(f"  - Direct XGBoost RMSE improvement over baseline: {rmse_improvement_direct:+.2f}%")
print(f"  - OpenSTEF XGBoost RMSE improvement over baseline: {rmse_improvement_openstef:+.2f}%")
