# %%
"""Train Baseline, Direct XGBoost, and OpenSTEF XGBoost models using quarterly splits.

This script:
1. Loads preprocessed feature-enriched data from data/processed/data_with_features.csv
2. Splits data into 4 calendar quarters (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)
3. For each quarter:
   - Uses first part for training
   - Uses last 14 days (with >95% non-missing values) for testing
4. Trains three models per quarter:
   a) Baseline (Persistence): Uses last known load value as prediction
   b) Direct XGBoost with manual configuration
   c) OpenSTEF XGBOpenstfRegressor with OpenSTEF's training approach
5. Saves training metadata and results to models/quarterly_split/

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
EXPERIMENT_NAME = "quarterly_split"
TEST_DAYS = 14
MIN_DATA_COVERAGE = 0.95  # 95% non-missing values per day

print("="*70)
print("QUARTERLY MODEL TRAINING")
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
# Split data into calendar quarters
def split_into_calendar_quarters(df: pd.DataFrame):
    """
    Split DataFrame into calendar quarters (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec).
    Returns a list of DataFrames and quarter info.
    """
    df_with_quarter = df.copy()
    df_with_quarter['year'] = df_with_quarter.index.year
    df_with_quarter['quarter'] = df_with_quarter.index.quarter

    unique_quarters = df_with_quarter[['year', 'quarter']].drop_duplicates().sort_values(['year', 'quarter'])

    quarters = []
    quarter_info = []

    print("\nSplitting data into calendar quarters:")
    print("="*70)

    for _, row in unique_quarters.iterrows():
        year = int(row['year'])
        quarter_num = int(row['quarter'])

        mask = (df_with_quarter['year'] == year) & (df_with_quarter['quarter'] == quarter_num)
        quarter_data = df.loc[mask]

        if len(quarter_data) > 0:
            quarters.append(quarter_data)
            quarter_info.append((year, quarter_num))

            month_ranges = {1: "Jan-Mar", 2: "Apr-Jun", 3: "Jul-Sep", 4: "Oct-Dec"}

            print(f"Q{quarter_num} {year} ({month_ranges[quarter_num]}): "
                  f"{len(quarter_data)} rows, from {quarter_data.index.min()} to {quarter_data.index.max()}")

    print("="*70)
    print(f"Total quarters found: {len(quarters)}\n")

    return quarters, quarter_info

quarters, quarter_info = split_into_calendar_quarters(data_clean)

# %%
# Function to find test period with sufficient data coverage
def find_test_period(df: pd.DataFrame, target_col: str, test_days: int, min_coverage: float):
    """
    Find the last test_days from the end of the quarter where each day has
    at least min_coverage non-missing values.

    Returns:
        cutoff_date: The date that separates train and test
        test_df: The test DataFrame
        days_found: Number of valid test days found
    """
    # Group by date to check coverage per day
    df_with_date = df.copy()
    df_with_date['date'] = df_with_date.index.date

    # Count observations per day and check coverage
    # Assuming 15-min intervals: 96 observations per day (24 * 4)
    expected_obs_per_day = 96

    daily_counts = df_with_date.groupby('date')[target_col].count()
    daily_coverage = daily_counts / expected_obs_per_day

    # Find valid days (with sufficient coverage)
    valid_days = daily_coverage[daily_coverage >= min_coverage].index

    if len(valid_days) < test_days:
        print(f"Warning: Only {len(valid_days)} days with >{min_coverage*100}% coverage, needed {test_days}")
        # Use what we have
        test_days_to_use = len(valid_days)
    else:
        test_days_to_use = test_days

    # Take the last N valid days
    test_dates = sorted(valid_days)[-test_days_to_use:]

    # Create test set from these dates
    test_df = df_with_date[df_with_date['date'].isin(test_dates)].drop('date', axis=1)

    # Cutoff is the day before the first test date
    if len(test_dates) > 0:
        first_test_date = pd.Timestamp(test_dates[0])
        cutoff_date = first_test_date - pd.Timedelta(days=1)
    else:
        cutoff_date = df.index.max()

    return cutoff_date, test_df, test_days_to_use

# %%
# Function to prepare train/test split for a quarter
def prepare_train_test_split(quarter_df: pd.DataFrame, target_col: str = 'load',
                             test_days: int = 14, min_coverage: float = 0.95):
    """
    Split a quarter into train and test sets.
    Test set: last N days with sufficient data coverage.
    Train set: everything before test set.
    """
    feature_cols = [col for col in quarter_df.columns if col != target_col]

    # Find test period with sufficient coverage
    cutoff_date, test_df, actual_test_days = find_test_period(
        quarter_df, target_col, test_days, min_coverage
    )

    # Create train set (everything up to cutoff)
    train_df = quarter_df[quarter_df.index <= cutoff_date]

    # Prepare X and y
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, test_df.index, actual_test_days

# %%
# Create experiment output directory
experiment_dir = Paths.MODELS / EXPERIMENT_NAME
experiment_dir.mkdir(parents=True, exist_ok=True)

print(f"\nExperiment directory: {experiment_dir}")

# %%
# Prepare train/test splits for each quarter
print(f"\n{'='*70}")
print("PREPARING TRAIN/TEST SPLITS FOR EACH QUARTER")
print(f"{'='*70}")

all_X_train = []
all_y_train = []
all_X_test = []
all_y_test = []
all_test_indices = []
quarter_split_info = []

for i, (quarter, (year, quarter_num)) in enumerate(zip(quarters, quarter_info), 1):
    print(f"\nQ{quarter_num} {year}:")

    # Prepare train/test split
    X_train, X_test, y_train, y_test, test_index, actual_test_days = prepare_train_test_split(
        quarter, test_days=TEST_DAYS, min_coverage=MIN_DATA_COVERAGE
    )

    print(f"  Train: {len(X_train)} rows ({X_train.index.min()} to {X_train.index.max()})")
    print(f"  Test: {len(X_test)} rows, {actual_test_days} days ({test_index.min()} to {test_index.max()})")
    print(f"  Features: {X_train.shape[1]}")

    # Collect data for combined model
    all_X_train.append(X_train)
    all_y_train.append(y_train)
    all_X_test.append(X_test)
    all_y_test.append(y_test)
    all_test_indices.append(test_index)

    quarter_split_info.append({
        'quarter_num': quarter_num,
        'year': year,
        'quarter_label': f"Q{quarter_num} {year}",
        'train_size': len(X_train),
        'test_size': len(X_test),
        'test_days': actual_test_days,
        'train_start': str(X_train.index.min()),
        'train_end': str(X_train.index.max()),
        'test_start': str(test_index.min()),
        'test_end': str(test_index.max()),
    })

# %%
# Train models on ALL quarterly data combined
print(f"\n{'='*70}")
print("TRAINING MODELS ON ALL QUARTERS COMBINED")
print(f"{'='*70}")
print(f"Strategy: Train on data from all {len(quarters)} quarters, test on each quarter's test period")

# Combine all training and test data
X_train_combined = pd.concat(all_X_train, axis=0)
y_train_combined = pd.concat(all_y_train, axis=0)
X_test_combined = pd.concat(all_X_test, axis=0)
y_test_combined = pd.concat(all_y_test, axis=0)
test_index_combined = pd.concat([pd.Series(idx) for idx in all_test_indices]).sort_values()

print(f"Combined train size: {len(X_train_combined)} rows (across {len(quarters)} quarters)")
print(f"Combined test size: {len(X_test_combined)} rows")
print(f"Train period: {X_train_combined.index.min()} to {X_train_combined.index.max()}")
print(f"Test period: {test_index_combined.min()} to {test_index_combined.max()}")

# ----------------------------------------------------------------
# 1. BASELINE MODEL (Persistence) - Combined
# ----------------------------------------------------------------
print(f"\n[COMBINED] Training Baseline (Persistence) Model...")

y_pred_baseline_combined = np.zeros(len(y_test_combined))
# For combined model, we need to handle boundaries between quarters
# Use simple approach: predict using previous actual value
for i in range(len(y_test_combined)):
    if i == 0:
        # First prediction uses last training value
        y_pred_baseline_combined[i] = y_train_combined.iloc[-1]
    else:
        # Use previous test value
        y_pred_baseline_combined[i] = y_test_combined.iloc[i-1]

rmse_baseline_combined = np.sqrt(mean_squared_error(y_test_combined, y_pred_baseline_combined))
mae_baseline_combined = mean_absolute_error(y_test_combined, y_pred_baseline_combined)
r2_baseline_combined = r2_score(y_test_combined, y_pred_baseline_combined)

print(f"  RMSE: {rmse_baseline_combined:.4f}, MAE: {mae_baseline_combined:.4f}, R²: {r2_baseline_combined:.4f}")

# ----------------------------------------------------------------
# 2. DIRECT XGBOOST MODEL - Combined
# ----------------------------------------------------------------
print(f"\n[COMBINED] Training Direct XGBoost Model...")

direct_model_combined = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    objective='reg:squarederror'
)

direct_model_combined.fit(X_train_combined, y_train_combined)
y_pred_direct_combined = direct_model_combined.predict(X_test_combined)

rmse_direct_combined = np.sqrt(mean_squared_error(y_test_combined, y_pred_direct_combined))
mae_direct_combined = mean_absolute_error(y_test_combined, y_pred_direct_combined)
r2_direct_combined = r2_score(y_test_combined, y_pred_direct_combined)

print(f"  RMSE: {rmse_direct_combined:.4f}, MAE: {mae_direct_combined:.4f}, R²: {r2_direct_combined:.4f}")

# ----------------------------------------------------------------
# 3. OPENSTEF XGBOOST MODEL - Combined
# ----------------------------------------------------------------
print(f"\n[COMBINED] Training OpenSTEF XGBoost Model...")

# Create prediction job configuration
pj_dict_combined = dict(
    id=999,
    model="xgb",
    quantiles=[0.5],
    forecast_type="demand",
    lat=52.0,
    lon=5.0,
    horizon_minutes=15,
    resolution_minutes=15,
    name="combined_all_quarters",
    hyper_params={},
    feature_names=None,
    default_modelspecs=None,
)
pj_combined = PredictionJobDataClass(**pj_dict_combined)

# Prepare combined training data in OpenSTEF format
train_data_full_combined = pd.concat([
    quarter[quarter.index.isin(X_train.index)].copy()
    for quarter, X_train in zip(quarters, all_X_train)
], axis=0)
horizon_value = pj_combined['horizon_minutes'] / 60
train_data_full_combined['horizon'] = horizon_value

# Reorder columns: load first, features, horizon last
cols_ordered = ['load'] + [col for col in train_data_full_combined.columns if col not in ['load', 'horizon']] + ['horizon']
train_data_openstef_combined = train_data_full_combined[cols_ordered].copy()

# Split into train/validation using OpenSTEF's method
train_split_combined, validation_split_combined, _, _ = split_data_train_validation_test(
    train_data_openstef_combined,
    test_fraction=0.0,
    back_test=False,
)

# Create and train OpenSTEF model
openstef_model_combined = ModelCreator.create_model(
    pj_combined["model"],
    quantiles=pj_combined["quantiles"],
)

train_x_combined = train_split_combined.iloc[:, 1:-1]
train_y_combined = train_split_combined.iloc[:, 0]
validation_x_combined = validation_split_combined.iloc[:, 1:-1]
validation_y_combined = validation_split_combined.iloc[:, 0]

eval_set_combined = [(train_x_combined, train_y_combined), (validation_x_combined, validation_y_combined)]
openstef_model_combined.set_params(early_stopping_rounds=10, random_state=RANDOM_SEED)

openstef_model_combined.fit(train_x_combined, train_y_combined, eval_set=eval_set_combined)

# Make predictions
y_pred_openstef_combined = openstef_model_combined.predict(X_test_combined)

rmse_openstef_combined = np.sqrt(mean_squared_error(y_test_combined, y_pred_openstef_combined))
mae_openstef_combined = mean_absolute_error(y_test_combined, y_pred_openstef_combined)
r2_openstef_combined = r2_score(y_test_combined, y_pred_openstef_combined)

print(f"  RMSE: {rmse_openstef_combined:.4f}, MAE: {mae_openstef_combined:.4f}, R²: {r2_openstef_combined:.4f}")

# %%
# Evaluate combined models on each quarter's test data separately
print(f"\n{'='*70}")
print("EVALUATING ON INDIVIDUAL QUARTERLY TEST SETS")
print(f"{'='*70}")

per_quarter_results = []

for i, (X_test_q, y_test_q, test_idx_q, qinfo) in enumerate(zip(all_X_test, all_y_test, all_test_indices, quarter_split_info)):
    print(f"\n{qinfo['quarter_label']}:")

    # Baseline predictions for this quarter
    y_pred_baseline_q = np.zeros(len(y_test_q))
    # Find the last training value before this quarter's test period
    train_before_q = y_train_combined[y_train_combined.index < test_idx_q.min()]
    if len(train_before_q) > 0:
        y_pred_baseline_q[0] = train_before_q.iloc[-1]
    else:
        y_pred_baseline_q[0] = y_train_combined.iloc[-1]

    for j in range(1, len(y_test_q)):
        y_pred_baseline_q[j] = y_test_q.iloc[j-1]

    # XGBoost predictions for this quarter
    y_pred_direct_q = direct_model_combined.predict(X_test_q)
    y_pred_openstef_q = openstef_model_combined.predict(X_test_q)

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

    print(f"  Baseline    - RMSE: {metrics_q['baseline']['rmse']:.4f}, MAE: {metrics_q['baseline']['mae']:.4f}, R²: {metrics_q['baseline']['r2']:.4f}")
    print(f"  Direct XGB  - RMSE: {metrics_q['direct_xgb']['rmse']:.4f}, MAE: {metrics_q['direct_xgb']['mae']:.4f}, R²: {metrics_q['direct_xgb']['r2']:.4f}")
    print(f"  OpenSTEF XGB- RMSE: {metrics_q['openstef_xgb']['rmse']:.4f}, MAE: {metrics_q['openstef_xgb']['mae']:.4f}, R²: {metrics_q['openstef_xgb']['r2']:.4f}")

    per_quarter_results.append({
        **qinfo,
        'metrics': metrics_q
    })

# Store overall combined results
combined_result = {
    'train_size': len(X_train_combined),
    'test_size': len(X_test_combined),
    'train_start': str(X_train_combined.index.min()),
    'train_end': str(X_train_combined.index.max()),
    'test_start': str(test_index_combined.min()),
    'test_end': str(test_index_combined.max()),
    'overall_metrics': {
        'baseline': {
            'rmse': float(rmse_baseline_combined),
            'mae': float(mae_baseline_combined),
            'r2': float(r2_baseline_combined)
        },
        'direct_xgb': {
            'rmse': float(rmse_direct_combined),
            'mae': float(mae_direct_combined),
            'r2': float(r2_direct_combined)
        },
        'openstef_xgb': {
            'rmse': float(rmse_openstef_combined),
            'mae': float(mae_openstef_combined),
            'r2': float(r2_openstef_combined)
        }
    },
    'per_quarter_metrics': {f"q{r['quarter_num']}_{r['year']}": r for r in per_quarter_results}
}

# %%
# Save results to JSON
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

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

print(f"Results saved to: {output_file}")

# %%
# Print summary table
print("\n" + "="*70)
print("SUMMARY: OVERALL PERFORMANCE")
print("="*70)
print(f"\nCombined model trained on all {len(quarters)} quarters, tested on all test data:")
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

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nTrained 1 combined model on all {len(quarters)} quarters")
print(f"Evaluated on overall test set + {len(per_quarter_results)} individual quarterly test sets")
print(f"Results saved to: {output_file}")
print("\nNext step: Run the evaluation report to generate visualizations")
