# %%
"""Train Multi-Horizon TabPFN and Direct XGBoost models using quarterly splits.

This script:
1. Loads preprocessed feature-enriched data from data/processed/data_with_features.csv
2. Splits data into 4 calendar quarters (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)
3. For each quarter:
   - Uses first part for training
   - Uses last 14 days (with >95% non-missing values) for testing
4. Trains models for 8 forecast horizons (15 min, 30 min, ..., 2 hours ahead)
5. Uses ONLY features available at forecast time:
   - Time-based cyclic features (sin/cos)
   - Time-based categorical features (month, day, hour, etc.)
   - Holiday features
   - Load profile features
   - Weekday features
   EXCLUDES weather, market data, and lag features (not available multi-period ahead)
6. Compares two model approaches:
   - TabPFN: Foundation model for tabular data (transformer-based)
   - Direct XGBoost: Traditional gradient boosting
7. Saves training metadata and results to models/tabpfn_xgboost_quarterly/

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
from tabpfn import TabPFNRegressor
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import centralized paths
from load_forecast import Paths

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
# Configuration
EXPERIMENT_NAME = "tabpfn_xgboost_quarterly"
TEST_DAYS = 14
MIN_DATA_COVERAGE = 0.95  # 95% non-missing values per day
NUM_HORIZONS = 8  # Forecast 1-8 periods ahead (15min to 2h)

print("="*70)
print("MULTI-HORIZON QUARTERLY MODEL TRAINING: TabPFN vs XGBoost")
print("="*70)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Forecast horizons: {NUM_HORIZONS} periods (15 min to {NUM_HORIZONS * 15} min)")
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
# Identify features by availability type
print("\n" + "="*70)
print("CATEGORIZING FEATURES BY AVAILABILITY")
print("="*70)

all_features = [col for col in data_with_features.columns if col != 'load']

# Features that ARE known multi-period ahead (no shifting needed beyond time alignment)
known_ahead_patterns = [
    # Time-based cyclic features
    'sin', 'cos',
    # Time-based categorical features
    'month', 'day', 'hour', 'week', 'quarter', 'year',
    # Holiday features
    'holiday', 'bridge',
    # Load profile features (if present)
    'profile', 'pattern',
]

# Features from forecasts (weather, market, lags) - need shifting by h-1
forecast_patterns = [
    # Weather features
    'windspeed', 'windpower', 'radiation', 'temperature', 'pressure',
    'humidity', 'saturation', 'vapour', 'dewpoint', 'air_density', 'dni', 'gti',
    # Market data
    'APX', 'price',
    # Lag features
    'T-', 'lag',
]

# Categorize features
known_ahead_features = []
forecast_features = []
other_features = []

for feature in all_features:
    is_known_ahead = any(pattern.lower() in feature.lower() for pattern in known_ahead_patterns)
    is_forecast = any(pattern.lower() in feature.lower() for pattern in forecast_patterns)

    if is_known_ahead:
        known_ahead_features.append(feature)
    elif is_forecast:
        forecast_features.append(feature)
    else:
        other_features.append(feature)

print(f"\nFeatures known multi-period ahead (time-based): {len(known_ahead_features)}")
print(f"Features from 1-period forecasts (need shifting): {len(forecast_features)}")
print(f"Other features: {len(other_features)}")

print("\n" + "-"*70)
print("KNOWN AHEAD FEATURES:")
print("-"*70)
for i, feat in enumerate(sorted(known_ahead_features), 1):
    print(f"{i:3d}. {feat}")

print("\n" + "-"*70)
print("FORECAST FEATURES (first 20):")
print("-"*70)
for i, feat in enumerate(sorted(forecast_features)[:20], 1):
    print(f"{i:3d}. {feat}")
if len(forecast_features) > 20:
    print(f"... and {len(forecast_features) - 20} more")

if len(other_features) > 0:
    print("\n" + "-"*70)
    print("OTHER FEATURES:")
    print("-"*70)
    for i, feat in enumerate(sorted(other_features), 1):
        print(f"{i:3d}. {feat}")

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
# Function to create multi-horizon datasets
def create_multi_horizon_datasets(df: pd.DataFrame, known_ahead_cols: list, forecast_cols: list,
                                   target_col: str = 'load', num_horizons: int = 8):
    """
    Create multiple datasets for different forecast horizons.

    IMPORTANT: The original dataset's forecast features are already 1-period-ahead forecasts.
    This means row i contains:
    - load[i]: actual load at time t+1
    - time features[i]: time features at time t+1
    - forecast features[i]: 1-period forecasts made at time t (for time t+1)

    For horizon h, we want to predict load at time t+h using data available at time t:
    - Horizon 1 (t+1): Use row[i] directly (forecasts from t, time features at t+1, load at t+1)
    - Horizon 2 (t+2): Use row[i] for load and time features, but row[i-1] for forecast features
      (forecasts from t-1 are our last available, predicting t+2 from t)
    - Horizon h: Use row[i] for load and time features, but row[i-(h-1)] for forecast features

    Args:
        df: DataFrame with all features and target
        known_ahead_cols: Features known multi-period ahead (time-based)
        forecast_cols: Features from 1-period forecasts (already in dataset)
        target_col: Target column name
        num_horizons: Number of forecast horizons

    Returns:
        List of tuples (horizon, X, y) where X and y are aligned for that horizon
    """
    datasets = []

    for horizon in range(1, num_horizons + 1):
        # Target and known-ahead features: use row[i] directly
        # (these are at time t+h when we want to predict t+h)
        y_horizon = df[target_col].copy()
        X_known = df[known_ahead_cols].copy()

        # Forecast features: shift BACKWARD by (horizon - 1)
        # For horizon 1: no shift (use row[i])
        # For horizon 2: shift by 1 (use row[i-1])
        # For horizon h: shift by h-1 (use row[i-(h-1)])
        if horizon == 1:
            X_forecast = df[forecast_cols].copy()
        else:
            X_forecast = df[forecast_cols].shift(horizon - 1)

        # Combine features
        X_combined = pd.concat([X_known, X_forecast], axis=1)

        # Remove rows where any data is NaN (from shifting)
        valid_mask = ~(y_horizon.isna() | X_combined.isna().any(axis=1))

        X_horizon = X_combined[valid_mask].copy()
        y_horizon_clean = y_horizon[valid_mask].copy()

        datasets.append((horizon, X_horizon, y_horizon_clean))

    return datasets

# %%
# Function to prepare train/test split for a quarter with multi-horizon
def prepare_train_test_split_multi_horizon(quarter_df: pd.DataFrame, known_ahead_cols: list, forecast_cols: list,
                                           target_col: str = 'load', test_days: int = 14, min_coverage: float = 0.95,
                                           num_horizons: int = 8):
    """
    Split a quarter into train and test sets for multiple horizons.
    Test set: last N days with sufficient data coverage.
    Train set: everything before test set (minus num_horizons to avoid data leakage).
    """
    # Find test period with sufficient coverage
    cutoff_date, test_df_full, actual_test_days = find_test_period(
        quarter_df, target_col, test_days, min_coverage
    )

    # Create train set (everything up to cutoff, minus buffer for horizons)
    # We need to ensure train data doesn't include targets that overlap with test period
    train_df = quarter_df[quarter_df.index <= cutoff_date]

    # Create multi-horizon datasets
    train_datasets = create_multi_horizon_datasets(train_df, known_ahead_cols, forecast_cols, target_col, num_horizons)
    test_datasets = create_multi_horizon_datasets(test_df_full, known_ahead_cols, forecast_cols, target_col, num_horizons)

    return train_datasets, test_datasets, actual_test_days

# %%
# Create experiment output directory
experiment_dir = Paths.MODELS / EXPERIMENT_NAME
experiment_dir.mkdir(parents=True, exist_ok=True)

print(f"\nExperiment directory: {experiment_dir}")

# %%
# Prepare train/test splits for each quarter and horizon
print(f"\n{'='*70}")
print("PREPARING MULTI-HORIZON TRAIN/TEST SPLITS FOR EACH QUARTER")
print(f"{'='*70}")

all_train_datasets = []  # List of lists: [quarter][horizon] = (horizon, X_train, y_train)
all_test_datasets = []   # List of lists: [quarter][horizon] = (horizon, X_test, y_test)
quarter_split_info = []

for i, (quarter, (year, quarter_num)) in enumerate(zip(quarters, quarter_info), 1):
    print(f"\nQ{quarter_num} {year}:")

    # Prepare train/test split with multi-horizon
    train_datasets_q, test_datasets_q, actual_test_days = prepare_train_test_split_multi_horizon(
        quarter, known_ahead_features, forecast_features, test_days=TEST_DAYS, min_coverage=MIN_DATA_COVERAGE, num_horizons=NUM_HORIZONS
    )

    # Store datasets
    all_train_datasets.append(train_datasets_q)
    all_test_datasets.append(test_datasets_q)

    # Print info for first horizon as summary
    _, X_train_h1, y_train_h1 = train_datasets_q[0]
    _, X_test_h1, y_test_h1 = test_datasets_q[0]

    print(f"  Train: {len(X_train_h1)} rows ({X_train_h1.index.min()} to {X_train_h1.index.max()})")
    print(f"  Test: {len(X_test_h1)} rows, {actual_test_days} days ({X_test_h1.index.min()} to {X_test_h1.index.max()})")
    print(f"  Available features: {X_train_h1.shape[1]}")
    print(f"  Horizons: {NUM_HORIZONS} (15min to {NUM_HORIZONS*15}min ahead)")

    quarter_split_info.append({
        'quarter_num': quarter_num,
        'year': year,
        'quarter_label': f"Q{quarter_num} {year}",
        'train_size': len(X_train_h1),
        'test_size': len(X_test_h1),
        'test_days': actual_test_days,
        'train_start': str(X_train_h1.index.min()),
        'train_end': str(X_train_h1.index.max()),
        'test_start': str(X_test_h1.index.min()),
        'test_end': str(X_test_h1.index.max()),
    })

# %%
# Train models for each horizon on ALL quarterly data combined
print(f"\n{'='*70}")
print("TRAINING MULTI-HORIZON MODELS ON ALL QUARTERS COMBINED")
print(f"{'='*70}")
print(f"Strategy: For each horizon, train on combined data from all {len(quarters)} quarters")

# Store models and results for each horizon
horizon_models = {
    'tabpfn': [],
    'direct_xgb': []
}

horizon_results = []

for horizon in range(1, NUM_HORIZONS + 1):
    print(f"\n{'='*70}")
    print(f"HORIZON {horizon} ({horizon * 15} minutes ahead)")
    print(f"{'='*70}")

    # Combine training data across all quarters for this horizon
    X_train_combined = pd.concat([train_datasets_q[horizon-1][1] for train_datasets_q in all_train_datasets], axis=0)
    y_train_combined = pd.concat([train_datasets_q[horizon-1][2] for train_datasets_q in all_train_datasets], axis=0)

    # Combine test data across all quarters for this horizon
    X_test_combined = pd.concat([test_datasets_q[horizon-1][1] for test_datasets_q in all_test_datasets], axis=0)
    y_test_combined = pd.concat([test_datasets_q[horizon-1][2] for test_datasets_q in all_test_datasets], axis=0)

    print(f"Combined train size: {len(X_train_combined)} rows")
    print(f"Combined test size: {len(X_test_combined)} rows")

    # ----------------------------------------------------------------
    # 1. TABPFN MODEL
    # ----------------------------------------------------------------
    print(f"\n[H{horizon}] Training TabPFN Model...")

    tabpfn_model = TabPFNRegressor(device='cpu', random_state=RANDOM_SEED)

    print(f"  Fitting TabPFN model (foundation model for tabular data)...")
    tabpfn_model.fit(X_train_combined, y_train_combined)

    print(f"  Making predictions...")
    y_pred_tabpfn = tabpfn_model.predict(X_test_combined)

    rmse_tabpfn = np.sqrt(mean_squared_error(y_test_combined, y_pred_tabpfn))
    mae_tabpfn = mean_absolute_error(y_test_combined, y_pred_tabpfn)
    r2_tabpfn = r2_score(y_test_combined, y_pred_tabpfn)

    print(f"  RMSE: {rmse_tabpfn:.4f}, MAE: {mae_tabpfn:.4f}, R²: {r2_tabpfn:.4f}")

    # ----------------------------------------------------------------
    # 2. DIRECT XGBOOST MODEL
    # ----------------------------------------------------------------
    print(f"\n[H{horizon}] Training Direct XGBoost Model...")

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

    direct_model.fit(X_train_combined, y_train_combined)
    y_pred_direct = direct_model.predict(X_test_combined)

    rmse_direct = np.sqrt(mean_squared_error(y_test_combined, y_pred_direct))
    mae_direct = mean_absolute_error(y_test_combined, y_pred_direct)
    r2_direct = r2_score(y_test_combined, y_pred_direct)

    print(f"  RMSE: {rmse_direct:.4f}, MAE: {mae_direct:.4f}, R²: {r2_direct:.4f}")

    # Store models and results
    horizon_models['tabpfn'].append(tabpfn_model)
    horizon_models['direct_xgb'].append(direct_model)

    horizon_results.append({
        'horizon': horizon,
        'horizon_minutes': horizon * 15,
        'train_size': len(X_train_combined),
        'test_size': len(X_test_combined),
        'metrics': {
            'tabpfn': {
                'rmse': float(rmse_tabpfn),
                'mae': float(mae_tabpfn),
                'r2': float(r2_tabpfn)
            },
            'direct_xgb': {
                'rmse': float(rmse_direct),
                'mae': float(mae_direct),
                'r2': float(r2_direct)
            }
        }
    })

# %%
# Evaluate models on each quarter's test data separately for each horizon
print(f"\n{'='*70}")
print("EVALUATING ON INDIVIDUAL QUARTERLY TEST SETS (ALL HORIZONS)")
print(f"{'='*70}")

per_quarter_per_horizon_results = []

for q_idx, (test_datasets_q, qinfo) in enumerate(zip(all_test_datasets, quarter_split_info)):
    print(f"\n{qinfo['quarter_label']}:")

    quarter_horizon_results = []

    for horizon in range(1, NUM_HORIZONS + 1):
        _, X_test_q, y_test_q = test_datasets_q[horizon - 1]

        # TabPFN predictions for this quarter and horizon
        tabpfn_model_h = horizon_models['tabpfn'][horizon - 1]
        y_pred_tabpfn_q = tabpfn_model_h.predict(X_test_q)

        # XGBoost predictions for this quarter and horizon
        direct_model_h = horizon_models['direct_xgb'][horizon - 1]
        y_pred_direct_q = direct_model_h.predict(X_test_q)

        # Calculate metrics
        metrics_q_h = {
            'tabpfn': {
                'rmse': float(np.sqrt(mean_squared_error(y_test_q, y_pred_tabpfn_q))),
                'mae': float(mean_absolute_error(y_test_q, y_pred_tabpfn_q)),
                'r2': float(r2_score(y_test_q, y_pred_tabpfn_q))
            },
            'direct_xgb': {
                'rmse': float(np.sqrt(mean_squared_error(y_test_q, y_pred_direct_q))),
                'mae': float(mean_absolute_error(y_test_q, y_pred_direct_q)),
                'r2': float(r2_score(y_test_q, y_pred_direct_q))
            }
        }

        quarter_horizon_results.append({
            'horizon': horizon,
            'horizon_minutes': horizon * 15,
            'metrics': metrics_q_h
        })

    # Print summary for this quarter (show first and last horizon)
    print(f"  Horizon 1 (15min) - TabPFN RMSE: {quarter_horizon_results[0]['metrics']['tabpfn']['rmse']:.4f}, "
          f"Direct XGB RMSE: {quarter_horizon_results[0]['metrics']['direct_xgb']['rmse']:.4f}")
    print(f"  Horizon {NUM_HORIZONS} ({NUM_HORIZONS*15}min) - TabPFN RMSE: {quarter_horizon_results[-1]['metrics']['tabpfn']['rmse']:.4f}, "
          f"Direct XGB RMSE: {quarter_horizon_results[-1]['metrics']['direct_xgb']['rmse']:.4f}")

    per_quarter_per_horizon_results.append({
        **qinfo,
        'horizon_results': quarter_horizon_results
    })

# %%
# Save results to JSON
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output_data = {
    'experiment': EXPERIMENT_NAME,
    'timestamp': datetime.now().isoformat(),
    'description': 'Multi-horizon models (1-8 periods ahead) comparing TabPFN and Direct XGBoost. Trained on all quarters combined, evaluated per quarter. For horizon h: uses time-based features at t+h and forecast features from t (which are already 1-period forecasts in the dataset), shifted back by h-1 periods.',
    'config': {
        'num_horizons': NUM_HORIZONS,
        'test_days': TEST_DAYS,
        'min_data_coverage': MIN_DATA_COVERAGE,
        'random_seed': RANDOM_SEED,
        'num_known_ahead_features': len(known_ahead_features),
        'num_forecast_features': len(forecast_features),
        'num_other_features': len(other_features),
        'total_features': len(known_ahead_features) + len(forecast_features) + len(other_features)
    },
    'known_ahead_features': known_ahead_features,
    'forecast_features': forecast_features[:50],  # Save first 50 to keep file size reasonable
    'other_features': other_features,
    'hyperparameters': {
        'tabpfn': {
            'device': 'cpu',
            'random_state': RANDOM_SEED
        },
        'direct_xgb': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    },
    'overall_horizon_results': horizon_results,
    'per_quarter_per_horizon_results': per_quarter_per_horizon_results
}

output_file = experiment_dir / 'training_results.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to: {output_file}")

# %%
# Print summary tables
print("\n" + "="*70)
print("SUMMARY: OVERALL PERFORMANCE BY HORIZON")
print("="*70)

print(f"\n{'Horizon':<10} {'Minutes':<10} {'Model':<18} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-"*80)

for h_result in horizon_results:
    h = h_result['horizon']
    h_min = h_result['horizon_minutes']

    for model in ['tabpfn', 'direct_xgb']:
        m = h_result['metrics'][model]
        print(f"{h:<10} {h_min:<10} {model:<18} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")
    print("-"*80)

print("\n" + "="*70)
print("SUMMARY: COMPARISON ACROSS HORIZONS")
print("="*70)
print(f"\n{'Horizon':<10} {'Minutes':<10} {'Model':<18} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-"*80)

for h_result in horizon_results:
    h = h_result['horizon']
    h_min = h_result['horizon_minutes']

    for model in ['tabpfn', 'direct_xgb']:
        m = h_result['metrics'][model]
        print(f"{h:<10} {h_min:<10} {model:<18} {m['rmse']:<12.4f} {m['mae']:<12.4f} {m['r2']:<12.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nTrained {NUM_HORIZONS} horizons (15min to {NUM_HORIZONS*15}min ahead)")
print(f"Models trained on all {len(quarters)} quarters combined")
print(f"Evaluated on {len(per_quarter_per_horizon_results)} individual quarterly test sets")
print(f"Using {len(known_ahead_features)} time-based features + {len(forecast_features)} forecast features")
print(f"Results saved to: {output_file}")
print("\nNext step: Run the evaluation report to generate visualizations")
