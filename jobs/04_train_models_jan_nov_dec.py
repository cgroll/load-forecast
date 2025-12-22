# %%
"""Train Baseline, Direct XGBoost, and OpenSTEF XGBoost models using Jan-Nov/Dec split.

This script:
1. Loads preprocessed feature-enriched data from data/processed/data_with_features.csv
2. Splits data: Jan-Nov for training, December for testing
3. Trains three models:
   a) Baseline (Persistence): Uses last known load value as prediction
   b) Direct XGBoost with manual configuration
   c) OpenSTEF XGBOpenstfRegressor with OpenSTEF's training approach
4. Saves training metadata and results to models/jan_nov_dec/

Uses Jupyter cell blocks (# %%) for interactive execution.
"""

# %%
# Imports
import pandas as pd
import numpy as np
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

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
# Configuration
EXPERIMENT_NAME = "jan_nov_dec"

print("="*70)
print("JAN-NOV vs DECEMBER MODEL TRAINING")
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
# Save results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Create experiment output directory
experiment_dir = Paths.MODELS / EXPERIMENT_NAME
experiment_dir.mkdir(parents=True, exist_ok=True)

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
    'metrics': {
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
}

output_file = experiment_dir / 'training_results.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to: {output_file}")

# %%
# Print summary
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"\n{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-"*60)
print(f"{'Baseline':<20} {rmse_baseline:<12.4f} {mae_baseline:<12.4f} {r2_baseline:<12.4f}")
print(f"{'Direct XGBoost':<20} {rmse_direct:<12.4f} {mae_direct:<12.4f} {r2_direct:<12.4f}")
print(f"{'OpenSTEF XGBoost':<20} {rmse_openstef:<12.4f} {mae_openstef:<12.4f} {r2_openstef:<12.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nResults saved to: {output_file}")
print("\nNext step: Run the evaluation report to generate visualizations")
