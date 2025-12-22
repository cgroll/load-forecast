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
# Train models for each quarter
results = []

for i, (quarter, (year, quarter_num)) in enumerate(zip(quarters, quarter_info), 1):
    print(f"\n{'='*70}")
    print(f"QUARTER {i}: Q{quarter_num} {year}")
    print(f"{'='*70}")

    # Prepare train/test split
    X_train, X_test, y_train, y_test, test_index, actual_test_days = prepare_train_test_split(
        quarter, test_days=TEST_DAYS, min_coverage=MIN_DATA_COVERAGE
    )

    print(f"Train size: {len(X_train)} rows")
    print(f"Test size: {len(X_test)} rows ({actual_test_days} days)")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Train period: {X_train.index.min()} to {X_train.index.max()}")
    print(f"Test period: {test_index.min()} to {test_index.max()}")

    # ----------------------------------------------------------------
    # 1. BASELINE MODEL (Persistence)
    # ----------------------------------------------------------------
    print(f"\n[Q{quarter_num} {year}] Training Baseline (Persistence) Model...")

    y_pred_baseline = np.zeros(len(y_test))
    y_pred_baseline[0] = y_train.iloc[-1]

    for j in range(1, len(y_test)):
        y_pred_baseline[j] = y_test.iloc[j-1]

    rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    r2_baseline = r2_score(y_test, y_pred_baseline)

    print(f"  RMSE: {rmse_baseline:.4f}, MAE: {mae_baseline:.4f}, R²: {r2_baseline:.4f}")

    # ----------------------------------------------------------------
    # 2. DIRECT XGBOOST MODEL
    # ----------------------------------------------------------------
    print(f"\n[Q{quarter_num} {year}] Training Direct XGBoost Model...")

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

    print(f"  RMSE: {rmse_direct:.4f}, MAE: {mae_direct:.4f}, R²: {r2_direct:.4f}")

    # ----------------------------------------------------------------
    # 3. OPENSTEF XGBOOST MODEL
    # ----------------------------------------------------------------
    print(f"\n[Q{quarter_num} {year}] Training OpenSTEF XGBoost Model...")

    # Create prediction job configuration
    pj_dict = dict(
        id=i,
        model="xgb",
        quantiles=[0.5],
        forecast_type="demand",
        lat=52.0,
        lon=5.0,
        horizon_minutes=15,
        resolution_minutes=15,
        name=f"Q{quarter_num}_{year}",
        hyper_params={},
        feature_names=None,
        default_modelspecs=None,
    )
    pj = PredictionJobDataClass(**pj_dict)

    # Prepare data in OpenSTEF format
    train_data_full = quarter[quarter.index.isin(X_train.index)].copy()
    horizon_value = pj['horizon_minutes'] / 60
    train_data_full['horizon'] = horizon_value

    # Reorder columns: load first, features, horizon last
    cols_ordered = ['load'] + [col for col in train_data_full.columns if col not in ['load', 'horizon']] + ['horizon']
    train_data_openstef = train_data_full[cols_ordered].copy()

    # Split into train/validation using OpenSTEF's method
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
    y_pred_openstef = openstef_model.predict(X_test)

    rmse_openstef = np.sqrt(mean_squared_error(y_test, y_pred_openstef))
    mae_openstef = mean_absolute_error(y_test, y_pred_openstef)
    r2_openstef = r2_score(y_test, y_pred_openstef)

    print(f"  RMSE: {rmse_openstef:.4f}, MAE: {mae_openstef:.4f}, R²: {r2_openstef:.4f}")

    # ----------------------------------------------------------------
    # Store results for this quarter
    # ----------------------------------------------------------------
    quarter_result = {
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

    results.append(quarter_result)

# %%
# Calculate aggregated metrics
print("\n" + "="*70)
print("CALCULATING AGGREGATED METRICS")
print("="*70)

def aggregate_metrics(results):
    """Calculate mean and std for each model across quarters."""
    models = ['baseline', 'direct_xgb', 'openstef_xgb']
    metrics = ['rmse', 'mae', 'r2']

    aggregated = {}

    for model in models:
        aggregated[model] = {}
        for metric in metrics:
            values = [r['metrics'][model][metric] for r in results]
            aggregated[model][f'{metric}_mean'] = float(np.mean(values))
            aggregated[model][f'{metric}_std'] = float(np.std(values))

    return aggregated

aggregated_metrics = aggregate_metrics(results)

# Print aggregated results
print("\nAggregated Metrics (Mean ± Std):")
print("-"*70)
for model in ['baseline', 'direct_xgb', 'openstef_xgb']:
    print(f"\n{model.upper()}:")
    for metric in ['rmse', 'mae', 'r2']:
        mean_val = aggregated_metrics[model][f'{metric}_mean']
        std_val = aggregated_metrics[model][f'{metric}_std']
        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

# %%
# Save results to JSON
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output_data = {
    'experiment': EXPERIMENT_NAME,
    'timestamp': datetime.now().isoformat(),
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
    'by_quarter': {f"q{r['quarter_num']}_{r['year']}": r for r in results},
    'aggregated': aggregated_metrics
}

output_file = experiment_dir / 'training_results.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to: {output_file}")

# %%
# Print summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"\n{'Quarter':<15} {'Model':<15} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-"*70)

for res in results:
    qlabel = res['quarter_label']
    for model in ['baseline', 'direct_xgb', 'openstef_xgb']:
        m = res['metrics'][model]
        print(f"{qlabel:<15} {model:<15} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")
    print("-"*70)

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nTrained models for {len(results)} quarters")
print(f"Results saved to: {output_file}")
print("\nNext step: Run the evaluation report to generate visualizations")
