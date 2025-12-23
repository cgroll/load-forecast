# %%
"""Train and Evaluate Multi-Horizon Models using Quarterly Splits.

This script trains and evaluates multi-horizon models (forecasting 1-8 periods ahead, 15min to 2h):
1. Loads preprocessed feature-enriched data from data/processed/data_with_features.csv
2. Splits data into train/test using quarterly splits (centralized function)
3. For each horizon (1-8 periods ahead):
   - Uses features known ahead: time-based (cyclic, categorical, holidays, etc.)
   - Uses forecast features: shifted appropriately for each horizon
   - Trains TabPFN and direct XGBoost models
4. Evaluates on overall test data and individual quarterly test periods
5. Creates comprehensive visualizations and analysis
6. Saves training results and exports metrics for DVC tracking

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

# Import centralized paths and data splitting utilities
from load_forecast import (
    Paths,
    split_quarters_train_test,
)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
# Configuration
EXPERIMENT_NAME = "multi_horizon_quarterly"
TEST_DAYS = 14
MIN_DATA_COVERAGE = 0.95  # 95% non-missing values per day
NUM_HORIZONS = 8  # Forecast 1-8 periods ahead (15min to 2h)
SELECTED_HORIZONS = [1, 4, 8]  # Only train models for these specific horizons
TARGET_COL = 'load'

print("="*70)
print("MULTI-HORIZON QUARTERLY MODEL TRAINING AND EVALUATION")
print("="*70)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Forecast horizons: {SELECTED_HORIZONS} (15 min to {max(SELECTED_HORIZONS) * 15} min)")
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
# Identify features by availability type
print("\n" + "="*70)
print("CATEGORIZING FEATURES BY AVAILABILITY")
print("="*70)

all_features = [col for col in data_clean.columns if col != TARGET_COL]

# Features that ARE known multi-period ahead (no shifting needed beyond time alignment)
known_ahead_patterns = [
    "season_sine",
    "season_cosine",
    "day0fweek_sine",
    "day0fweek_cosine",
    "month_sine",
    "month_cosine",
    "time0fday_sine",
    "time0fday_cosine",

    "IsWeekendDay",
    "IsWeekDay",
    "IsSunday",
    "Month",
    "Quarter",
    "is_national_holiday",
    "is_bridgeday",
    "is_schoolholiday",
    "daylight_continuous",

    "is_nieuwjaarsdag",
    "is_goede_vrijdag",
    "is_eerste_paasdag",
    "is_tweede_paasdag",
    "is_koningsdag",
    "is_hemelvaart",
    "is_eerste_pinksterdag",
    "is_tweede_pinksterdag",
    "is_eerste_kerstdag",
    "is_tweede_kerstdag",
    "is_bevrijdingsdag",
    "is_herfstvakantiemidden",
    "is_kerstvakantie",
    "is_voorjaarsvakantiemidden",
]

# Categorize features
known_ahead_features = []
forecast_features = []

for feature in all_features:
    is_known_ahead = any(pattern.lower() in feature.lower() for pattern in known_ahead_patterns)

    if is_known_ahead:
        known_ahead_features.append(feature)
    else:
        forecast_features.append(feature)

print(f"\nFeatures known multi-period ahead (time-based): {len(known_ahead_features)}")
print(f"Features from 1-period forecasts (need shifting): {len(forecast_features)}")

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
# Combine train and test data with type tracking
print("\n" + "="*70)
print("PREPARING MULTI-HORIZON DATASETS")
print("="*70)

# Select only features we're using (known ahead + forecast features + target)
features_to_keep = known_ahead_features + forecast_features + [TARGET_COL]

# Filter data to only include these features
train_data_filtered = train_data[features_to_keep].copy()
test_data_filtered = test_data[features_to_keep].copy()

# Create separate tracking for split type (not included in features)
train_indices = train_data_filtered.index
test_indices = test_data_filtered.index

# Combine and sort by time
all_data = pd.concat([train_data_filtered, test_data_filtered], axis=0).sort_index()

# Create split type series aligned with all_data
split_type = pd.Series('test', index=all_data.index)
split_type.loc[train_indices] = 'train'

print(f"Combined data shape: {all_data.shape}")
print(f"Train rows: {(split_type == 'train').sum()}")
print(f"Test rows: {(split_type == 'test').sum()}")

# %%
# Function to create multi-horizon datasets
def create_multi_horizon_datasets(df: pd.DataFrame, split_type_series: pd.Series, known_ahead_cols: list,
                                   forecast_cols: list, target_col: str = 'load', num_horizons: int = 8):
    """
    Create multiple datasets for different forecast horizons.

    The original dataset's forecast features are already 1-period-ahead forecasts.
    For horizon h, we want to predict load at time t+h using data available at time t:
    - Horizon 1 (t+1): Use row[i] directly (forecasts from t, time features at t+1, load at t+1)
    - Horizon 2 (t+2): Use row[i] for load and time features, but row[i-1] for forecast features
    - Horizon h: Use row[i] for load and time features, but row[i-(h-1)] for forecast features

    Args:
        df: DataFrame with all features and target
        split_type_series: Series indicating 'train' or 'test' for each row
        known_ahead_cols: Features known multi-period ahead (time-based)
        forecast_cols: Features from 1-period forecasts (already in dataset)
        target_col: Target column name
        num_horizons: Number of forecast horizons

    Returns:
        List of tuples (horizon, X, y, split_type) where X, y, and split_type are aligned for that horizon
    """
    datasets = []

    for horizon in range(1, num_horizons + 1):
        # Target and known-ahead features: use row[i] directly
        y_horizon = df[target_col].copy()
        X_known = df[known_ahead_cols].copy()
        split_type_h = split_type_series.copy()

        # Forecast features: shift BACKWARD by (horizon - 1)
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
        split_type_clean = split_type_h[valid_mask].copy()

        # Convert any object columns to numeric (this can happen after shifting)
        for col in X_horizon.columns:
            if X_horizon[col].dtype == 'object':
                X_horizon[col] = pd.to_numeric(X_horizon[col], errors='coerce')

        datasets.append((horizon, X_horizon, y_horizon_clean, split_type_clean))

    return datasets

print(f"Creating multi-horizon datasets for horizons: {SELECTED_HORIZONS}...")
all_horizon_datasets_full = create_multi_horizon_datasets(
    all_data, split_type, known_ahead_features, forecast_features, TARGET_COL, NUM_HORIZONS
)

# Filter to only selected horizons
all_horizon_datasets = [ds for ds in all_horizon_datasets_full if ds[0] in SELECTED_HORIZONS]

print("\nDataset sizes by horizon:")
for horizon, X, y, split_type in all_horizon_datasets:
    train_size = (split_type == 'train').sum()
    test_size = (split_type == 'test').sum()
    print(f"  Horizon {horizon} ({horizon*15}min): Train {train_size}, Test {test_size}, Features {X.shape[1]}")

# %%
# Create experiment output directory
experiment_dir = Paths.MODELS / EXPERIMENT_NAME
experiment_dir.mkdir(parents=True, exist_ok=True)

print(f"\nExperiment directory: {experiment_dir}")

# %%
# Train models for each horizon
print(f"\n{'='*70}")
print("TRAINING MULTI-HORIZON MODELS")
print(f"{'='*70}")

horizon_models = {
    'tabpfn': [],
    'direct_xgb': []
}

horizon_results = []

for horizon, X_all, y_all, split_type_all in all_horizon_datasets:
    print(f"\n{'='*70}")
    print(f"HORIZON {horizon} ({horizon * 15} minutes ahead)")
    print(f"{'='*70}")

    # Split into train and test
    X_train = X_all[split_type_all == 'train']
    y_train = y_all[split_type_all == 'train']
    X_test = X_all[split_type_all == 'test']
    y_test = y_all[split_type_all == 'test']

    print(f"Train size: {len(X_train)} rows")
    print(f"Test size: {len(X_test)} rows")

    # ----------------------------------------------------------------
    # 1. TABPFN MODEL
    # ----------------------------------------------------------------
    print(f"\n[H{horizon}] Training TabPFN Model...")

    tabpfn_model = TabPFNRegressor(device='cuda', random_state=RANDOM_SEED)

    print(f"  Fitting TabPFN model (foundation model for tabular data)...")
    tabpfn_model.fit(X_train, y_train)

    print(f"  Making predictions...")
    y_pred_tabpfn = tabpfn_model.predict(X_test)

    rmse_tabpfn = np.sqrt(mean_squared_error(y_test, y_pred_tabpfn))
    mae_tabpfn = mean_absolute_error(y_test, y_pred_tabpfn)
    r2_tabpfn = r2_score(y_test, y_pred_tabpfn)

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

    direct_model.fit(X_train, y_train)
    y_pred_direct = direct_model.predict(X_test)

    rmse_direct = np.sqrt(mean_squared_error(y_test, y_pred_direct))
    mae_direct = mean_absolute_error(y_test, y_pred_direct)
    r2_direct = r2_score(y_test, y_pred_direct)

    print(f"  RMSE: {rmse_direct:.4f}, MAE: {mae_direct:.4f}, R²: {r2_direct:.4f}")

    # Store models and results
    horizon_models['tabpfn'].append(tabpfn_model)
    horizon_models['direct_xgb'].append(direct_model)

    horizon_results.append({
        'horizon': horizon,
        'horizon_minutes': horizon * 15,
        'train_size': len(X_train),
        'test_size': len(X_test),
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

for qinfo in quarter_info:
    print(f"\n{qinfo['quarter_label']}:")

    quarter_horizon_results = []

    for horizon, X_all, y_all, split_type_all in all_horizon_datasets:
        # Filter for this quarter's test data
        # We need to match the test indices from this quarter
        test_start = pd.Timestamp(qinfo['test_start'])
        test_end = pd.Timestamp(qinfo['test_end'])

        # Get test data for this quarter and horizon
        quarter_mask = (y_all.index >= test_start) & (y_all.index <= test_end) & (split_type_all == 'test')

        if quarter_mask.sum() == 0:
            continue

        X_test_q = X_all[quarter_mask]
        y_test_q = y_all[quarter_mask]

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
    if len(quarter_horizon_results) > 0:
        h_first = quarter_horizon_results[0]
        h_last = quarter_horizon_results[-1]
        print(f"  Horizon {h_first['horizon']} ({h_first['horizon_minutes']}min) - TabPFN RMSE: {h_first['metrics']['tabpfn']['rmse']:.4f}, "
              f"Direct XGB RMSE: {h_first['metrics']['direct_xgb']['rmse']:.4f}")
        print(f"  Horizon {h_last['horizon']} ({h_last['horizon_minutes']}min) - TabPFN RMSE: {h_last['metrics']['tabpfn']['rmse']:.4f}, "
              f"Direct XGB RMSE: {h_last['metrics']['direct_xgb']['rmse']:.4f}")

    per_quarter_per_horizon_results.append({
        **qinfo,
        'horizon_results': quarter_horizon_results
    })

# %%
# Save results to JSON
print("\n" + "="*70)
print("SAVING TRAINING RESULTS")
print("="*70)

output_data = {
    'experiment': EXPERIMENT_NAME,
    'timestamp': datetime.now().isoformat(),
    'description': f'Multi-horizon models (horizons {SELECTED_HORIZONS}) comparing TabPFN and Direct XGBoost. Trained on all quarters combined, evaluated per quarter. For horizon h: uses time-based features at t+h and forecast features from t (which are already 1-period forecasts in the dataset), shifted back by h-1 periods.',
    'config': {
        'num_horizons': NUM_HORIZONS,
        'selected_horizons': SELECTED_HORIZONS,
        'test_days': TEST_DAYS,
        'min_data_coverage': MIN_DATA_COVERAGE,
        'random_seed': RANDOM_SEED,
        'num_known_ahead_features': len(known_ahead_features),
        'num_forecast_features': len(forecast_features),
        'total_features': len(known_ahead_features) + len(forecast_features)
    },
    'known_ahead_features': known_ahead_features,
    'forecast_features': forecast_features[:50],  # Save first 50 to keep file size reasonable
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
# ============================================================
# SUMMARY TABLES AND METRICS EXPORT
# ============================================================

print("\n" + "="*70)
print("SUMMARY TABLES")
print("="*70)

# %%
# SECTION 1: OVERALL METRICS BY HORIZON
print("\n" + "="*70)
print("OVERALL METRICS BY HORIZON (ALL TEST DATA COMBINED)")
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

# %%
# SECTION 2: PERFORMANCE DEGRADATION ANALYSIS
print("\n" + "="*70)
print("PERFORMANCE DEGRADATION ANALYSIS")
print("="*70)

models = ['tabpfn', 'direct_xgb']
model_labels = ['TabPFN', 'Direct XGB']

print(f"\nDegradation Summary (Horizon {SELECTED_HORIZONS[0]} → Horizon {SELECTED_HORIZONS[-1]}):")
print("-"*70)
for model, label in zip(models, model_labels):
    rmse_vals = [h['metrics'][model]['rmse'] for h in horizon_results]
    rmse_increase_pct = (rmse_vals[-1] - rmse_vals[0]) / rmse_vals[0] * 100

    r2_vals = [h['metrics'][model]['r2'] for h in horizon_results]
    r2_decrease = r2_vals[0] - r2_vals[-1]

    print(f"{label:<18} RMSE: +{rmse_increase_pct:.1f}%  |  R²: -{r2_decrease:.4f}")

# %%
# SECTION 3: PER-QUARTER METRICS BY HORIZON
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
# Export metrics for DVC tracking
print("\n" + "="*70)
print("EXPORTING METRICS FOR DVC TRACKING")
print("="*70)

metrics_output = {
    'experiment': EXPERIMENT_NAME,
    'overall_horizon_metrics': {}
}

# Overall metrics by horizon
for h_result in horizon_results:
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
        f'h{SELECTED_HORIZONS[0]}_metrics': q_result['horizon_results'][0]['metrics'],
        f'h{SELECTED_HORIZONS[-1]}_metrics': q_result['horizon_results'][-1]['metrics']
    }

# Performance degradation summary
metrics_output['degradation_summary'] = {}
for model, label in zip(models, model_labels):
    rmse_vals = [h['metrics'][model]['rmse'] for h in horizon_results]
    rmse_h1 = rmse_vals[0]
    rmse_h8 = rmse_vals[-1]
    rmse_increase_pct = (rmse_h8 - rmse_h1) / rmse_h1 * 100

    r2_vals = [h['metrics'][model]['r2'] for h in horizon_results]
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

print(f"Metrics saved to: {metrics_file}")

# %%
# Summary statistics
print("\n" + "="*70)
print("SUMMARY: BEST AND WORST PERFORMING HORIZONS")
print("="*70)

for model, label in zip(models, model_labels):
    print(f"\n{label}:")
    rmse_vals = [h['metrics'][model]['rmse'] for h in horizon_results]

    best_h_idx = np.argmin(rmse_vals)
    worst_h_idx = np.argmax(rmse_vals)

    best_h = horizon_results[best_h_idx]
    worst_h = horizon_results[worst_h_idx]

    print(f"  Best:  Horizon {best_h['horizon']} ({best_h['horizon_minutes']}min) - RMSE: {rmse_vals[best_h_idx]:.4f}")
    print(f"  Worst: Horizon {worst_h['horizon']} ({worst_h['horizon_minutes']}min) - RMSE: {rmse_vals[worst_h_idx]:.4f}")
    print(f"  Range: {rmse_vals[worst_h_idx] - rmse_vals[best_h_idx]:.4f} ({(rmse_vals[worst_h_idx] / rmse_vals[best_h_idx] - 1)*100:.1f}% increase)")

# %%
print("\n" + "="*70)
print("TRAINING AND EVALUATION COMPLETE")
print("="*70)
print(f"\nTrained {len(SELECTED_HORIZONS)} horizons: {SELECTED_HORIZONS} (15min to {max(SELECTED_HORIZONS)*15}min ahead)")
print(f"Models trained on all {len(quarter_info)} quarters combined")
print(f"Evaluated on {len(per_quarter_per_horizon_results)} individual quarterly test sets")
print(f"Using {len(known_ahead_features)} time-based features + {len(forecast_features)} forecast features")
print(f"Training results saved to: {output_file}")
print(f"Metrics exported to: {metrics_file}")

print("\nKey Findings:")
for model, label in zip(models, model_labels):
    rmse_first = horizon_results[0]['metrics'][model]['rmse']
    rmse_last = horizon_results[-1]['metrics'][model]['rmse']
    increase_pct = (rmse_last - rmse_first) / rmse_first * 100
    h_first = horizon_results[0]['horizon']
    h_last = horizon_results[-1]['horizon']
    print(f"  {label}: RMSE increases {increase_pct:.1f}% from H{h_first} to H{h_last} ({h_first*15}min → {h_last*15}min)")
