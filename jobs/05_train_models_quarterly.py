# %%
"""Train and Evaluate Models using Quarterly Splits.

This script combines training and evaluation:
1. Loads preprocessed feature-enriched data from data/processed/data_with_features.csv
2. Splits data into 4 calendar quarters (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)
3. For each quarter, uses first part for training and last 14 days (with >95% coverage) for testing
4. Trains three models (combined across all quarters):
   a) Baseline (Persistence): Uses last known load value as prediction
   b) Direct XGBoost with manual configuration
   c) OpenSTEF XGBOpenstfRegressor with OpenSTEF's training approach
5. Evaluates models on:
   - Overall test data (all quarters combined)
   - Individual quarterly test periods
6. Creates comprehensive visualizations
7. Saves training results to models/quarterly_split/training_results.json
8. Exports metrics to metrics/quarterly_split_evaluation.json for DVC tracking

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

# Import centralized paths and data splitting utilities
from load_forecast import (
    Paths,
    split_quarters_train_test,
)

# OpenSTEF imports
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.model.model_creator import ModelCreator
from openstef.model_selection.model_selection import split_data_train_validation_test

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
# Configuration
EXPERIMENT_NAME = "quarterly_split"
TEST_DAYS = 14
MIN_DATA_COVERAGE = 0.95  # 95% non-missing values per day
TARGET_COL = 'load'

print("="*70)
print("QUARTERLY MODEL TRAINING AND EVALUATION")
print("="*70)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Test period: Last {TEST_DAYS} days per quarter")
print(f"Minimum data coverage: {MIN_DATA_COVERAGE*100}%")
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
# Split data into train/test using the centralized function
print("\n" + "="*70)
print("SPLITTING DATA INTO TRAIN/TEST BY QUARTERS")
print("="*70)

train_data, test_data, quarter_info = split_quarters_train_test(
    data_clean,
    target_col=TARGET_COL,
    test_days=TEST_DAYS,
    min_coverage=MIN_DATA_COVERAGE
)

print(f"\nCombined train size: {len(train_data)} rows (across {len(quarter_info)} quarters)")
print(f"Combined test size: {len(test_data)} rows")
print(f"Train period: {train_data.index.min()} to {train_data.index.max()}")
print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")

print("\nQuarter breakdown:")
for qinfo in quarter_info:
    print(f"  {qinfo['quarter_label']}: Train {qinfo['train_size']} rows, Test {qinfo['test_size']} rows")

# %%
# Create experiment output directory
experiment_dir = Paths.MODELS / EXPERIMENT_NAME
experiment_dir.mkdir(parents=True, exist_ok=True)

print(f"\nExperiment directory: {experiment_dir}")

# %%
# ============================================================
# MODEL 1: BASELINE (PERSISTENCE)
# ============================================================
print("\n" + "="*70)
print("MODEL 1: BASELINE (PERSISTENCE)")
print("="*70)
print("Strategy: Predict t using load value from t-1")

# Combine train and test data, sort by time
all_data = pd.concat([train_data, test_data], axis=0).sort_index()

# Create baseline predictions: shift load by 1 period
all_data_baseline = all_data[[TARGET_COL]].copy()
all_data_baseline['baseline_pred'] = all_data_baseline[TARGET_COL].shift(1)

# Extract predictions for test period
baseline_test = all_data_baseline.loc[test_data.index].copy()
baseline_test = baseline_test.dropna()  # Remove first prediction which has no prior value

y_test_baseline = baseline_test[TARGET_COL]
y_pred_baseline = baseline_test['baseline_pred']

# Calculate metrics
rmse_baseline = np.sqrt(mean_squared_error(y_test_baseline, y_pred_baseline))
mae_baseline = mean_absolute_error(y_test_baseline, y_pred_baseline)
r2_baseline = r2_score(y_test_baseline, y_pred_baseline)

print(f"Overall metrics: RMSE={rmse_baseline:.4f}, MAE={mae_baseline:.4f}, R²={r2_baseline:.4f}")

# %%
# ============================================================
# MODEL 2: DIRECT XGBOOST
# ============================================================
print("\n" + "="*70)
print("MODEL 2: DIRECT XGBOOST")
print("="*70)
print("Strategy: Train XGBoost with manual hyperparameters")

# Prepare X and y for XGBoost
feature_cols = [col for col in train_data.columns if col != TARGET_COL]

X_train = train_data[feature_cols]
y_train = train_data[TARGET_COL]
X_test = test_data[feature_cols]
y_test = test_data[TARGET_COL]

print(f"Train: {len(X_train)} samples, {len(feature_cols)} features")
print(f"Test: {len(X_test)} samples")

# Train model
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

# Calculate metrics
rmse_direct = np.sqrt(mean_squared_error(y_test, y_pred_direct))
mae_direct = mean_absolute_error(y_test, y_pred_direct)
r2_direct = r2_score(y_test, y_pred_direct)

print(f"Overall metrics: RMSE={rmse_direct:.4f}, MAE={mae_direct:.4f}, R²={r2_direct:.4f}")

# %%
# ============================================================
# MODEL 3: OPENSTEF XGBOOST
# ============================================================
print("\n" + "="*70)
print("MODEL 3: OPENSTEF XGBOOST")
print("="*70)
print("Strategy: Use OpenSTEF's training approach with train/validation split")

# Create prediction job configuration
pj_dict = dict(
    id=999,
    model="xgb",
    quantiles=[0.5],
    forecast_type="demand",
    lat=52.0,
    lon=5.0,
    horizon_minutes=15,
    resolution_minutes=15,
    name="quarterly_combined",
    hyper_params={},
    feature_names=None,
    default_modelspecs=None,
)
pj = PredictionJobDataClass(**pj_dict)

# Prepare data in OpenSTEF format: add horizon column
train_data_openstef = train_data.copy()
horizon_value = pj['horizon_minutes'] / 60
train_data_openstef['horizon'] = horizon_value

# Reorder columns: load first, features, horizon last
cols_ordered = [TARGET_COL] + [col for col in train_data_openstef.columns if col not in [TARGET_COL, 'horizon']] + ['horizon']
train_data_openstef = train_data_openstef[cols_ordered]

print(f"Training data shape (with horizon): {train_data_openstef.shape}")

# Split into train/validation using OpenSTEF's method
train_split, validation_split, _, _ = split_data_train_validation_test(
    train_data_openstef,
    test_fraction=0.0,
    back_test=False,
)

print(f"Train split: {len(train_split)} samples")
print(f"Validation split: {len(validation_split)} samples")

# Create and train OpenSTEF model
openstef_model = ModelCreator.create_model(
    pj["model"],
    quantiles=pj["quantiles"],
)

train_x = train_split.iloc[:, 1:-1]  # Exclude first col (load) and last col (horizon)
train_y = train_split.iloc[:, 0]     # First col is load
validation_x = validation_split.iloc[:, 1:-1]
validation_y = validation_split.iloc[:, 0]

eval_set = [(train_x, train_y), (validation_x, validation_y)]
openstef_model.set_params(early_stopping_rounds=10, random_state=RANDOM_SEED)

openstef_model.fit(train_x, train_y, eval_set=eval_set)

# Make predictions on test set
y_pred_openstef = openstef_model.predict(X_test)

# Calculate metrics
rmse_openstef = np.sqrt(mean_squared_error(y_test, y_pred_openstef))
mae_openstef = mean_absolute_error(y_test, y_pred_openstef)
r2_openstef = r2_score(y_test, y_pred_openstef)

print(f"Overall metrics: RMSE={rmse_openstef:.4f}, MAE={mae_openstef:.4f}, R²={r2_openstef:.4f}")

# %%
# ============================================================
# EVALUATION BY QUARTER
# ============================================================
print("\n" + "="*70)
print("EVALUATING ON INDIVIDUAL QUARTERLY TEST SETS")
print("="*70)

# Create a dataframe with all predictions
test_predictions = pd.DataFrame({
    'load': y_test,
    'baseline': y_pred_baseline.reindex(y_test.index),
    'direct_xgb': y_pred_direct,
    'openstef_xgb': y_pred_openstef,
}, index=y_test.index)

# Add quarter information to test predictions
test_predictions['year'] = test_predictions.index.year
test_predictions['quarter'] = test_predictions.index.quarter

per_quarter_results = []
visualization_data = []

for qinfo in quarter_info:
    # Filter predictions for this quarter
    quarter_mask = (
        (test_predictions['year'] == qinfo['year']) &
        (test_predictions['quarter'] == qinfo['quarter_num'])
    )
    quarter_preds = test_predictions[quarter_mask].copy()

    if len(quarter_preds) == 0:
        continue

    # Drop rows with missing predictions (e.g., baseline first prediction)
    quarter_preds = quarter_preds.dropna()

    y_test_q = quarter_preds['load']
    y_pred_baseline_q = quarter_preds['baseline']
    y_pred_direct_q = quarter_preds['direct_xgb']
    y_pred_openstef_q = quarter_preds['openstef_xgb']

    # Calculate metrics for this quarter
    metrics_q = {
        'baseline': {
            'rmse': float(np.sqrt(mean_squared_error(y_test_q, y_pred_baseline_q))),
            'mae': float(mean_absolute_error(y_test_q, y_pred_baseline_q)),
            'r2': float(r2_score(y_test_q, y_pred_baseline_q))
        },
        'direct_xgb': {
            'rmse': float(np.sqrt(mean_squared_error(y_test_q, y_pred_direct_q))),
            'mae': float(mean_absolute_error(y_test_q, y_pred_direct_q)),
            'r2': float(r2_score(y_test_q, y_pred_direct_q))
        },
        'openstef_xgb': {
            'rmse': float(np.sqrt(mean_squared_error(y_test_q, y_pred_openstef_q))),
            'mae': float(mean_absolute_error(y_test_q, y_pred_openstef_q)),
            'r2': float(r2_score(y_test_q, y_pred_openstef_q))
        }
    }

    print(f"\n{qinfo['quarter_label']} ({len(y_test_q)} predictions):")
    print(f"  Baseline    - RMSE: {metrics_q['baseline']['rmse']:.4f}, MAE: {metrics_q['baseline']['mae']:.4f}, R²: {metrics_q['baseline']['r2']:.4f}")
    print(f"  Direct XGB  - RMSE: {metrics_q['direct_xgb']['rmse']:.4f}, MAE: {metrics_q['direct_xgb']['mae']:.4f}, R²: {metrics_q['direct_xgb']['r2']:.4f}")
    print(f"  OpenSTEF XGB- RMSE: {metrics_q['openstef_xgb']['rmse']:.4f}, MAE: {metrics_q['openstef_xgb']['mae']:.4f}, R²: {metrics_q['openstef_xgb']['r2']:.4f}")

    per_quarter_results.append({
        **qinfo,
        'metrics': metrics_q
    })

    # Store visualization data
    visualization_data.append({
        'quarter_num': qinfo['quarter_num'],
        'year': qinfo['year'],
        'quarter_label': qinfo['quarter_label'],
        'test_index': y_test_q.index,
        'y_test': y_test_q,
        'y_pred_baseline': y_pred_baseline_q.values,
        'y_pred_direct': y_pred_direct_q,
        'y_pred_openstef': y_pred_openstef_q,
    })

# %%
# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "="*70)
print("SAVING TRAINING RESULTS")
print("="*70)

# Store overall combined results
combined_result = {
    'train_size': len(train_data),
    'test_size': len(test_data),
    'train_start': str(train_data.index.min()),
    'train_end': str(train_data.index.max()),
    'test_start': str(test_data.index.min()),
    'test_end': str(test_data.index.max()),
    'overall_metrics': {
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
    },
    'per_quarter_metrics': {f"q{r['quarter_num']}_{r['year']}": r for r in per_quarter_results}
}

output_data = {
    'experiment': EXPERIMENT_NAME,
    'timestamp': datetime.now().isoformat(),
    'description': 'Combined model trained on all quarters, evaluated on each quarter separately',
    'config': {
        'test_days': TEST_DAYS,
        'min_data_coverage': MIN_DATA_COVERAGE,
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
    'results': combined_result
}

output_file = experiment_dir / 'training_results.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Training results saved to: {output_file}")

# %%
# Print summary table
print("\n" + "="*70)
print("SUMMARY: OVERALL PERFORMANCE")
print("="*70)
print(f"\nCombined model trained on all {len(quarter_info)} quarters, tested on all test data:")
print(f"{'Model':<18} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-"*60)
for model in ['baseline', 'direct_xgb', 'openstef_xgb']:
    m = combined_result['overall_metrics'][model]
    print(f"{model:<18} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")

print("\n" + "="*70)
print("SUMMARY: PER-QUARTER PERFORMANCE")
print("="*70)
print(f"\n{'Quarter':<15} {'Model':<18} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-"*75)

for res in per_quarter_results:
    qlabel = res['quarter_label']
    for model in ['baseline', 'direct_xgb', 'openstef_xgb']:
        m = res['metrics'][model]
        print(f"{qlabel:<15} {model:<18} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")
    print("-"*75)

# %%
# ============================================================
# VISUALIZATIONS
# ============================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Organize metrics
overall_metrics = combined_result['overall_metrics']
per_quarter_metrics = combined_result['per_quarter_metrics']

by_quarter = {}
for quarter_key, quarter_data in per_quarter_metrics.items():
    by_quarter[quarter_key] = {
        'quarter_label': quarter_data['quarter_label'],
        'metrics': quarter_data['metrics']
    }

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
rmse_baseline = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]["metrics"]['baseline']['rmse'] for v in visualization_data]
rmse_direct = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]["metrics"]['direct_xgb']['rmse'] for v in visualization_data]
rmse_openstef = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]["metrics"]['openstef_xgb']['rmse'] for v in visualization_data]

mae_baseline = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]["metrics"]['baseline']['mae'] for v in visualization_data]
mae_direct = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]["metrics"]['direct_xgb']['mae'] for v in visualization_data]
mae_openstef = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]["metrics"]['openstef_xgb']['mae'] for v in visualization_data]

r2_baseline = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]["metrics"]['baseline']['r2'] for v in visualization_data]
r2_direct = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]["metrics"]['direct_xgb']['r2'] for v in visualization_data]
r2_openstef = [by_quarter[f"q{v['quarter_num']}_{v['year']}"]["metrics"]['openstef_xgb']['r2'] for v in visualization_data]

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

print(f"Metrics saved to: {metrics_file}")

# %%
print("\n" + "="*70)
print("TRAINING AND EVALUATION COMPLETE")
print("="*70)
print(f"\nTrained 1 combined model on all {len(quarter_info)} quarters")
print(f"Evaluated on overall test set + {len(per_quarter_results)} individual quarterly test sets")
print(f"Training results saved to: {output_file}")
print(f"Metrics exported to: {metrics_file}")

if overall_metrics:
    print("\nOverall Results (All Test Data Combined):")
    for model in ['baseline', 'direct_xgb', 'openstef_xgb']:
        m = overall_metrics[model]
        print(f"  {model.upper()}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R²={m['r2']:.4f}")
