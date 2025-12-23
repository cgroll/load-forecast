# %%
"""Analyze Feature Importance Across Multi-Horizon Models.

This report analyzes how feature importance changes across different forecast horizons
using the same setup as jobs/08_multi_horizon_quarterly.py. The analysis:

1. Loads preprocessed feature-enriched data from data/processed/data_with_features.csv
2. Splits data into calendar quarters (same as experiment 08)
3. For each of 8 forecast horizons (15 min to 2 hours ahead):
   - Trains an XGBoost model on combined quarterly data
   - Extracts feature importance scores (gain, weight, cover)
4. Analyzes how feature importance evolves across horizons:
   - Which features become more/less important as horizon increases
   - Top features per horizon
   - Feature importance heatmaps
5. Categorizes features by type (time-based, weather, lags, etc.)
6. Saves results to models/feature_importance_analysis/

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
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Import centralized paths
from load_forecast import Paths

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
# Configuration
EXPERIMENT_NAME = "feature_importance_analysis"
TEST_DAYS = 14
MIN_DATA_COVERAGE = 0.95  # 95% non-missing values per day
NUM_HORIZONS = 8  # Forecast 1-8 periods ahead (15min to 2h)

print("="*70)
print("MULTI-HORIZON FEATURE IMPORTANCE ANALYSIS")
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

# %%
# Create a more detailed categorization for visualization
def categorize_feature_detailed(feature_name: str) -> str:
    """Categorize a feature into a detailed type for analysis."""
    fname_lower = feature_name.lower()

    # Time-based cyclic
    if any(p in fname_lower for p in ['sin', 'cos']):
        return 'Time Cyclic'

    # Time-based categorical
    if any(p in fname_lower for p in ['month', 'day', 'hour', 'week', 'quarter', 'year']):
        return 'Time Categorical'

    # Holiday
    if any(p in fname_lower for p in ['holiday', 'bridge']):
        return 'Holiday'

    # Weather - temperature related
    if any(p in fname_lower for p in ['temperature', 'dewpoint']):
        return 'Weather: Temperature'

    # Weather - solar/radiation
    if any(p in fname_lower for p in ['radiation', 'dni', 'gti']):
        return 'Weather: Solar'

    # Weather - wind
    if any(p in fname_lower for p in ['windspeed', 'windpower']):
        return 'Weather: Wind'

    # Weather - other
    if any(p in fname_lower for p in ['pressure', 'humidity', 'saturation', 'vapour', 'air_density']):
        return 'Weather: Other'

    # Market
    if any(p in fname_lower for p in ['apx', 'price']):
        return 'Market'

    # Lag features
    if any(p in fname_lower for p in ['t-', 'lag']):
        return 'Lag Features'

    # Profile
    if any(p in fname_lower for p in ['profile', 'pattern']):
        return 'Load Profile'

    return 'Other'

# Categorize all features
feature_categories = {feat: categorize_feature_detailed(feat) for feat in all_features}

# Count by category
category_counts = pd.Series(feature_categories).value_counts()
print("\n" + "="*70)
print("FEATURE CATEGORIES")
print("="*70)
for cat, count in category_counts.items():
    print(f"{cat:<25} {count:>3} features")

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
# Split data into calendar quarters (same function as in 08_multi_horizon_quarterly.py)
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
# Function to find test period with sufficient data coverage (same as in 08)
def find_test_period(df: pd.DataFrame, target_col: str, test_days: int, min_coverage: float):
    """
    Find the last test_days from the end of the quarter where each day has
    at least min_coverage non-missing values.
    """
    df_with_date = df.copy()
    df_with_date['date'] = df_with_date.index.date

    expected_obs_per_day = 96

    daily_counts = df_with_date.groupby('date')[target_col].count()
    daily_coverage = daily_counts / expected_obs_per_day

    valid_days = daily_coverage[daily_coverage >= min_coverage].index

    if len(valid_days) < test_days:
        test_days_to_use = len(valid_days)
    else:
        test_days_to_use = test_days

    test_dates = sorted(valid_days)[-test_days_to_use:]
    test_df = df_with_date[df_with_date['date'].isin(test_dates)].drop('date', axis=1)

    if len(test_dates) > 0:
        first_test_date = pd.Timestamp(test_dates[0])
        cutoff_date = first_test_date - pd.Timedelta(days=1)
    else:
        cutoff_date = df.index.max()

    return cutoff_date, test_df, test_days_to_use

# %%
# Function to create multi-horizon datasets (same as in 08)
def create_multi_horizon_datasets(df: pd.DataFrame, known_ahead_cols: list, forecast_cols: list,
                                   target_col: str = 'load', num_horizons: int = 8):
    """
    Create multiple datasets for different forecast horizons.
    """
    datasets = []

    for horizon in range(1, num_horizons + 1):
        y_horizon = df[target_col].copy()
        X_known = df[known_ahead_cols].copy()

        if horizon == 1:
            X_forecast = df[forecast_cols].copy()
        else:
            X_forecast = df[forecast_cols].shift(horizon - 1)

        X_combined = pd.concat([X_known, X_forecast], axis=1)

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
    """
    cutoff_date, test_df_full, actual_test_days = find_test_period(
        quarter_df, target_col, test_days, min_coverage
    )

    train_df = quarter_df[quarter_df.index <= cutoff_date]

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

all_train_datasets = []
all_test_datasets = []

for i, (quarter, (year, quarter_num)) in enumerate(zip(quarters, quarter_info), 1):
    print(f"\nQ{quarter_num} {year}:")

    train_datasets_q, test_datasets_q, actual_test_days = prepare_train_test_split_multi_horizon(
        quarter, known_ahead_features, forecast_features, test_days=TEST_DAYS, min_coverage=MIN_DATA_COVERAGE, num_horizons=NUM_HORIZONS
    )

    all_train_datasets.append(train_datasets_q)
    all_test_datasets.append(test_datasets_q)

    _, X_train_h1, y_train_h1 = train_datasets_q[0]
    _, X_test_h1, y_test_h1 = test_datasets_q[0]

    print(f"  Train: {len(X_train_h1)} rows ({X_train_h1.index.min()} to {X_train_h1.index.max()})")
    print(f"  Test: {len(X_test_h1)} rows, {actual_test_days} days ({X_test_h1.index.min()} to {X_test_h1.index.max()})")
    print(f"  Available features: {X_train_h1.shape[1]}")

# %%
# Train XGBoost models for each horizon and extract feature importance
print(f"\n{'='*70}")
print("TRAINING MODELS AND EXTRACTING FEATURE IMPORTANCE")
print(f"{'='*70}")
print(f"Training XGBoost models for {NUM_HORIZONS} horizons...")

# Store feature importance for each horizon
horizon_feature_importance = []

for horizon in range(1, NUM_HORIZONS + 1):
    print(f"\n{'='*70}")
    print(f"HORIZON {horizon} ({horizon * 15} minutes ahead)")
    print(f"{'='*70}")

    # Combine training data across all quarters for this horizon
    X_train_combined = pd.concat([train_datasets_q[horizon-1][1] for train_datasets_q in all_train_datasets], axis=0)
    y_train_combined = pd.concat([train_datasets_q[horizon-1][2] for train_datasets_q in all_train_datasets], axis=0)

    print(f"Combined train size: {len(X_train_combined)} rows")
    print(f"Number of features: {X_train_combined.shape[1]}")

    # Train XGBoost model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        objective='reg:squarederror'
    )

    print("  Training model...")
    model.fit(X_train_combined, y_train_combined)
    print("  Training complete!")

    # Extract feature importance using the Booster's get_score method
    # This returns a dict with feature names as keys
    booster = model.get_booster()

    # Get feature importance scores (using booster's internal feature names)
    importance_gain = booster.get_score(importance_type='gain')
    importance_weight = booster.get_score(importance_type='weight')
    importance_cover = booster.get_score(importance_type='cover')

    # XGBoost uses f0, f1, f2... as internal feature names
    # We need to map them back to actual column names
    feature_names = X_train_combined.columns.tolist()

    # Create importance data with proper mapping
    importance_data = []
    for i, feat_name in enumerate(feature_names):
        # XGBoost internal name is f{index}
        xgb_internal_name = f'f{i}'

        importance_data.append({
            'feature': feat_name,
            'gain': float(importance_gain.get(xgb_internal_name, 0.0)),
            'weight': float(importance_weight.get(xgb_internal_name, 0.0)),
            'cover': float(importance_cover.get(xgb_internal_name, 0.0)),
            'category': feature_categories.get(feat_name, 'Other')
        })

    print(f"  Extracted importance for {len(importance_data)} features")
    print(f"  Non-zero gain features: {sum(1 for d in importance_data if d['gain'] > 0)}")

    importance_df = pd.DataFrame(importance_data)

    # Normalize importance scores to sum to 1 (for easier comparison across horizons)
    if importance_df['gain'].sum() > 0:
        importance_df['gain_normalized'] = importance_df['gain'] / importance_df['gain'].sum()
    else:
        importance_df['gain_normalized'] = 0

    if importance_df['weight'].sum() > 0:
        importance_df['weight_normalized'] = importance_df['weight'] / importance_df['weight'].sum()
    else:
        importance_df['weight_normalized'] = 0

    if importance_df['cover'].sum() > 0:
        importance_df['cover_normalized'] = importance_df['cover'] / importance_df['cover'].sum()
    else:
        importance_df['cover_normalized'] = 0

    # Sort by gain (most common importance metric)
    importance_df = importance_df.sort_values('gain', ascending=False)

    # Store results
    horizon_feature_importance.append({
        'horizon': horizon,
        'horizon_minutes': horizon * 15,
        'importance_df': importance_df,
        'top_10_features': importance_df.head(10)['feature'].tolist(),
        'top_10_gain': importance_df.head(10)['gain'].tolist()
    })

    # Print top 10 features for this horizon
    print(f"\n  Top 10 features by gain:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:<40} Gain: {row['gain']:>10.4f} ({row['category']})")

# %%
# VISUALIZATION 1: Top features by horizon (heatmap)
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Get union of top N features across all horizons
TOP_N = 20
all_top_features = set()
for h_result in horizon_feature_importance:
    all_top_features.update(h_result['importance_df'].head(TOP_N)['feature'].tolist())

all_top_features = sorted(all_top_features)

print(f"\nTotal unique top-{TOP_N} features across all horizons: {len(all_top_features)}")

# Create matrix for heatmap: features x horizons
importance_matrix_gain = []
importance_matrix_weight = []

for feat in all_top_features:
    gain_row = []
    weight_row = []
    for h_result in horizon_feature_importance:
        feat_importance = h_result['importance_df'][h_result['importance_df']['feature'] == feat]
        if len(feat_importance) > 0:
            gain_row.append(feat_importance.iloc[0]['gain_normalized'])
            weight_row.append(feat_importance.iloc[0]['weight_normalized'])
        else:
            gain_row.append(0.0)
            weight_row.append(0.0)
    importance_matrix_gain.append(gain_row)
    importance_matrix_weight.append(weight_row)

horizon_labels = [f"H{h['horizon']}\n({h['horizon_minutes']}min)" for h in horizon_feature_importance]

# Create DataFrame for heatmap
importance_df_gain = pd.DataFrame(
    importance_matrix_gain,
    index=all_top_features,
    columns=horizon_labels
)

# Sort by total importance across all horizons
importance_df_gain['total'] = importance_df_gain.sum(axis=1)
importance_df_gain = importance_df_gain.sort_values('total', ascending=False).drop('total', axis=1)

# Plot heatmap
fig, ax = plt.subplots(figsize=(14, max(10, len(all_top_features) * 0.3)))
sns.heatmap(importance_df_gain, cmap='YlOrRd', annot=False, fmt='.3f',
            cbar_kws={'label': 'Normalized Gain'}, linewidths=0.5, ax=ax)
ax.set_title(f'Feature Importance (Gain) Across Horizons\nTop {TOP_N} Features by Total Importance',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Forecast Horizon', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# VISUALIZATION 2: Feature importance evolution for top 10 features
print("\nPlotting feature importance evolution...")

# Get overall top 10 features (by average gain across horizons)
avg_importance = {}
for feat in all_top_features:
    gains = [h['importance_df'][h['importance_df']['feature'] == feat]['gain_normalized'].values[0]
             if len(h['importance_df'][h['importance_df']['feature'] == feat]) > 0 else 0
             for h in horizon_feature_importance]
    avg_importance[feat] = np.mean(gains)

top_10_overall = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
top_10_features = [feat for feat, _ in top_10_overall]

fig, ax = plt.subplots(figsize=(16, 8))

horizon_numbers = [h['horizon'] for h in horizon_feature_importance]
horizon_minutes = [h['horizon_minutes'] for h in horizon_feature_importance]

colors = plt.cm.tab10(np.linspace(0, 1, 10))

for idx, feat in enumerate(top_10_features):
    gains = []
    for h_result in horizon_feature_importance:
        feat_imp = h_result['importance_df'][h_result['importance_df']['feature'] == feat]
        if len(feat_imp) > 0:
            gains.append(feat_imp.iloc[0]['gain_normalized'])
        else:
            gains.append(0.0)

    ax.plot(horizon_minutes, gains, marker='o', linewidth=2.5, markersize=8,
            label=feat, color=colors[idx], alpha=0.8)

ax.set_xlabel('Forecast Horizon (minutes)', fontsize=12)
ax.set_ylabel('Normalized Gain', fontsize=12)
ax.set_title('Feature Importance Evolution Across Horizons\nTop 10 Features by Average Importance',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(horizon_minutes)
plt.tight_layout()
plt.show()

# %%
# VISUALIZATION 3: Category-level importance across horizons
print("\nAnalyzing importance by feature category...")

# Aggregate importance by category for each horizon
category_importance = []

for h_result in horizon_feature_importance:
    h = h_result['horizon']
    h_min = h_result['horizon_minutes']

    # Group by category and sum normalized gain
    cat_imp = h_result['importance_df'].groupby('category')['gain_normalized'].sum().to_dict()

    category_importance.append({
        'horizon': h,
        'horizon_minutes': h_min,
        **cat_imp
    })

category_df = pd.DataFrame(category_importance)

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(16, 8))

categories = [col for col in category_df.columns if col not in ['horizon', 'horizon_minutes']]
categories_sorted = sorted(categories, key=lambda c: -category_df[c].sum())

bottom = np.zeros(len(category_df))
colors_cat = plt.cm.Set3(np.linspace(0, 1, len(categories_sorted)))

for idx, cat in enumerate(categories_sorted):
    if cat in category_df.columns:
        ax.bar(category_df['horizon_minutes'], category_df[cat], bottom=bottom,
               label=cat, color=colors_cat[idx], alpha=0.9)
        bottom += category_df[cat].values

ax.set_xlabel('Forecast Horizon (minutes)', fontsize=12)
ax.set_ylabel('Total Normalized Gain', fontsize=12)
ax.set_title('Feature Category Importance Across Horizons\nStacked by Category',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax.set_xticks(horizon_minutes)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %%
# VISUALIZATION 4: Individual category trends
print("\nPlotting category importance trends...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

# Plot top 4 categories
top_categories = categories_sorted[:4]

for idx, cat in enumerate(top_categories):
    ax = axes[idx]
    if cat in category_df.columns:
        ax.plot(category_df['horizon_minutes'], category_df[cat],
                marker='o', linewidth=2.5, markersize=8, color=colors_cat[idx], alpha=0.8)
        ax.set_xlabel('Forecast Horizon (minutes)', fontsize=11)
        ax.set_ylabel('Normalized Gain', fontsize=11)
        ax.set_title(f'{cat} Importance', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(horizon_minutes)

plt.tight_layout()
plt.show()

# %%
# VISUALIZATION 5: Feature importance change (H1 vs H8)
print("\nAnalyzing feature importance change from H1 to H8...")

h1_importance = horizon_feature_importance[0]['importance_df'].set_index('feature')['gain_normalized']
h8_importance = horizon_feature_importance[-1]['importance_df'].set_index('feature')['gain_normalized']

# Calculate change
importance_change = h8_importance - h1_importance

# Sort by absolute change
importance_change_sorted = importance_change.abs().sort_values(ascending=False)

# Get top 15 features with biggest change
top_changes = importance_change_sorted.head(15)

fig, ax = plt.subplots(figsize=(12, 8))

features = top_changes.index.tolist()
changes = [importance_change[feat] for feat in features]

colors_change = ['green' if c > 0 else 'red' for c in changes]

ax.barh(range(len(features)), changes, color=colors_change, alpha=0.7)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=10)
ax.set_xlabel('Change in Normalized Gain (H8 - H1)', fontsize=12)
ax.set_title('Features with Largest Importance Change\nFrom Horizon 1 (15min) to Horizon 8 (120min)',
             fontsize=14, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\nTop features gaining importance (H1 → H8):")
gaining = importance_change.sort_values(ascending=False).head(10)
for feat, change in gaining.items():
    print(f"  {feat:<40} +{change:>8.4f}")

print("\nTop features losing importance (H1 → H8):")
losing = importance_change.sort_values(ascending=True).head(10)
for feat, change in losing.items():
    print(f"  {feat:<40} {change:>8.4f}")

# %%
# Save results to JSON
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Prepare output data
output_data = {
    'experiment': EXPERIMENT_NAME,
    'config': {
        'num_horizons': NUM_HORIZONS,
        'test_days': TEST_DAYS,
        'min_data_coverage': MIN_DATA_COVERAGE,
        'random_seed': RANDOM_SEED,
        'top_n_features': TOP_N
    },
    'feature_categories': {cat: int(count) for cat, count in category_counts.items()},
    'horizons': []
}

# Add per-horizon results
for h_result in horizon_feature_importance:
    horizon_data = {
        'horizon': h_result['horizon'],
        'horizon_minutes': h_result['horizon_minutes'],
        'top_10_features': h_result['top_10_features'],
        'top_10_gain': [float(g) for g in h_result['top_10_gain']],
        'category_importance': {}
    }

    # Add category-level importance
    cat_imp = h_result['importance_df'].groupby('category')['gain_normalized'].sum().to_dict()
    horizon_data['category_importance'] = {cat: float(imp) for cat, imp in cat_imp.items()}

    output_data['horizons'].append(horizon_data)

# Add importance change analysis
output_data['importance_change'] = {
    'h1_to_h8': {
        'gaining': {feat: float(change) for feat, change in gaining.head(10).items()},
        'losing': {feat: float(change) for feat, change in losing.head(10).items()}
    }
}

output_file = experiment_dir / 'feature_importance_results.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to: {output_file}")

# %%
# Export summary metrics for DVC
metrics_dir = Path('metrics')
metrics_dir.mkdir(exist_ok=True)
metrics_file = metrics_dir / 'feature_importance_analysis.json'

metrics_data = {
    'top_3_h1': horizon_feature_importance[0]['top_10_features'][:3],
    'top_3_h8': horizon_feature_importance[-1]['top_10_features'][:3],
    'category_importance_h1': {
        cat: float(imp)
        for cat, imp in horizon_feature_importance[0]['importance_df'].groupby('category')['gain_normalized'].sum().items()
    },
    'category_importance_h8': {
        cat: float(imp)
        for cat, imp in horizon_feature_importance[-1]['importance_df'].groupby('category')['gain_normalized'].sum().items()
    }
}

with open(metrics_file, 'w') as f:
    json.dump(metrics_data, f, indent=2)

print(f"✓ Metrics saved to: {metrics_file}")

# %%
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("="*70)
print(f"\n✓ Analyzed {NUM_HORIZONS} forecast horizons")
print(f"✓ Examined {len(all_features)} features across {len(category_counts)} categories")
print(f"✓ Identified top {TOP_N} most important features")
print(f"✓ Results saved to: {output_file}")
print(f"✓ Metrics saved to: {metrics_file}")

print("\nKey Findings:")
print(f"  Top feature (H1): {horizon_feature_importance[0]['top_10_features'][0]}")
print(f"  Top feature (H8): {horizon_feature_importance[-1]['top_10_features'][0]}")

print("\nCategory importance trends:")
for cat in categories_sorted[:5]:
    if cat in category_df.columns:
        h1_val = category_df[cat].iloc[0]
        h8_val = category_df[cat].iloc[-1]
        change = h8_val - h1_val
        print(f"  {cat:<25} H1: {h1_val:.3f} → H8: {h8_val:.3f} (Δ {change:+.3f})")
