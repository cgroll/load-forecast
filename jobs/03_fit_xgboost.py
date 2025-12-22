"""
Quarterly XGBoost forecasting analysis with feature engineering.

This script:
1. Loads Excel data with load and weather features
2. Applies feature engineering using OpenSTEF's apply_features function
3. Splits data into 4 quarters by timestamp
4. For each quarter: trains XGBoost on first 70%, predicts on next 30%
5. Evaluates RMSE and R² for each quarter
6. Creates visualizations: line plots and scatter plots with temporal coloring

Uses Jupyter cell blocks (# %%) for interactive execution.
"""

# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# OpenSTEF imports for feature engineering
from openstef.feature_engineering.apply_features import apply_features
from openstef.data_classes.prediction_job import PredictionJobDataClass

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# %%
# Load Excel data
def load_xlsx_data() -> pd.DataFrame:
    """Load the XLSX data with load and weather features."""
    data = pd.read_excel(
        "~/Downloads/input_data_sun_heavy.xlsx",
        index_col=0,
        parse_dates=True
    )
    print(f"Loaded data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"\nColumns: {list(data.columns)}")
    print(f"\nFirst few rows:")
    print(data.head())
    return data

input_data = load_xlsx_data()


# %%
# Create prediction job for feature engineering
def create_prediction_job() -> PredictionJobDataClass:
    """Create a minimal prediction job configuration for feature engineering."""
    pj_dict = dict(
        id=1,
        model="xgb",
        quantiles=[10, 30, 50, 70, 90],
        forecast_type="demand",
        lat=52.0,  # Default latitude (Netherlands)
        lon=5.0,   # Default longitude (Netherlands)
        horizon_minutes=15,  # 1 step ahead = 15 minutes
        resolution_minutes=15,
        name="QuarterlyAnalysis",
        hyper_params={},
        feature_names=None,  # Will use all available features
        default_modelspecs=None,
    )
    return PredictionJobDataClass(**pj_dict)

pj = create_prediction_job()
print(f"Created prediction job: {pj['name']}")
print(f"Horizon: {pj['horizon_minutes']} minutes")
print(f"Resolution: {pj['resolution_minutes']} minutes")

# %%
# Apply OpenSTEF feature engineering
print("\nApplying OpenSTEF feature engineering...")
print("This will add:")
print("  - Time-based features (cyclic and categorical)")
print("  - Lag features based on load history")
print("  - Weather-derived features")
print("  - Holiday features")
print("  - And more...")

data_with_features = apply_features(
    data=input_data.copy(),
    pj=pj,
    feature_names=None,  # Use all available features
    horizon=pj['horizon_minutes'] / 60,  # Convert to hours
)

print(f"\nFeatures added! New shape: {data_with_features.shape}")
print(f"Number of columns: {data_with_features.shape[1]}")
print("\nNew columns added:")
new_cols = [col for col in data_with_features.columns if col not in input_data.columns]
print(f"  Total new features: {len(new_cols)}")
print("\nSample of new features:")
for i, col in enumerate(new_cols[:10], 1):
    print(f"  {i}. {col}")
if len(new_cols) > 10:
    print(f"  ... and {len(new_cols) - 10} more features")

# %%
# Resample data to ensure 15-minute intervals
print("\n" + "="*70)
print("RESAMPLING DATA TO 15-MINUTE INTERVALS")
print("="*70)
print(f"Original data shape: {input_data.shape}")
print(f"Original date range: {input_data.index.min()} to {input_data.index.max()}")
print(f"Original time span: {(input_data.index.max() - input_data.index.min()).days} days")

# Create a complete 15-minute time index
start_time = input_data.index.min().floor('15min')
end_time = input_data.index.max().ceil('15min')
complete_index = pd.date_range(start=start_time, end=end_time, freq='15min')

print(f"\nExpected 15-min intervals: {len(complete_index):,}")
print(f"Actual data points: {len(input_data):,}")
print(f"Missing intervals: {len(complete_index) - len(input_data):,}")

# Reindex to the complete 15-minute grid
input_data_resampled = input_data.reindex(complete_index)

print(f"\nResampled data shape: {input_data_resampled.shape}")
print(f"Resampled date range: {input_data_resampled.index.min()} to {input_data_resampled.index.max()}")

# %%
# Apply feature engineering to resampled data
print("\n" + "="*70)
print("APPLYING FEATURE ENGINEERING TO RESAMPLED DATA")
print("="*70)
print("This will add:")
print("  - Time-based features (cyclic and categorical)")
print("  - Lag features based on load history")
print("  - Weather-derived features")
print("  - Holiday features")
print("  - And more...")

data_with_features = apply_features(
    data=input_data_resampled.copy(),
    pj=pj,
    feature_names=None,  # Use all available features
    horizon=pj['horizon_minutes'] / 60,  # Convert to hours
)

print(f"\nFeatures added! New shape: {data_with_features.shape}")
print(f"Number of columns: {data_with_features.shape[1]}")

# %%
# Drop rows with any missing values in features or target
# (lag features will have NaN at the beginning, and we may have missing values from original data)
print("\n" + "="*70)
print("CLEANING DATA - REMOVING ROWS WITH MISSING VALUES")
print("="*70)

print(f"Data shape before cleaning: {data_with_features.shape}")
print(f"Date range before cleaning: {data_with_features.index.min()} to {data_with_features.index.max()}")

# Check missing values in feature-engineered data
missing_after_features = data_with_features.isnull().sum()
missing_after_features = missing_after_features[missing_after_features > 0].sort_values(ascending=False)
print(f"\nColumns with missing values after feature engineering:")
if len(missing_after_features) > 0:
    print(f"  Total columns with missing values: {len(missing_after_features)}")
    print(f"\n  Top 10 columns with most missing values:")
    for col, count in missing_after_features.head(10).items():
        pct = (count / len(data_with_features)) * 100
        print(f"    {col:<30} {count:>8,} ({pct:>6.2f}%)")
else:
    print("  No missing values!")

data_clean = data_with_features.dropna()
print(f"\nData shape after dropping NaN: {data_clean.shape}")
print(f"Date range after cleaning: {data_clean.index.min()} to {data_clean.index.max()}")
print(f"Rows removed: {len(data_with_features) - len(data_clean):,} ({(len(data_with_features) - len(data_clean)) / len(data_with_features) * 100:.2f}%)")

# %%
# Split data into calendar quarters
def split_into_calendar_quarters(df: pd.DataFrame) -> list:
    """
    Split DataFrame into calendar quarters (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec).
    Returns a list of DataFrames, one per quarter found in the data.
    """
    # Add a column to identify quarter and year
    df_with_quarter = df.copy()
    df_with_quarter['year'] = df_with_quarter.index.year
    df_with_quarter['quarter'] = df_with_quarter.index.quarter

    # Get unique year-quarter combinations
    unique_quarters = df_with_quarter[['year', 'quarter']].drop_duplicates().sort_values(['year', 'quarter'])

    quarters = []
    quarter_info = []

    print("\nSplitting data into calendar quarters:")
    print("="*70)

    for _, row in unique_quarters.iterrows():
        year = row['year']
        quarter_num = row['quarter']

        # Filter data for this specific quarter
        mask = (df_with_quarter['year'] == year) & (df_with_quarter['quarter'] == quarter_num)
        quarter_data = df.loc[mask]  # Use original df without the extra columns

        if len(quarter_data) > 0:
            quarters.append(quarter_data)
            quarter_info.append((year, quarter_num))

            # Define month ranges for readability
            month_ranges = {1: "Jan-Mar", 2: "Apr-Jun", 3: "Jul-Sep", 4: "Oct-Dec"}

            print(f"Q{quarter_num} {year} ({month_ranges[quarter_num]}): "
                  f"{len(quarter_data)} rows, from {quarter_data.index.min()} to {quarter_data.index.max()}")

    print("="*70)
    print(f"Total quarters found: {len(quarters)}\n")

    return quarters, quarter_info

quarters, quarter_info = split_into_calendar_quarters(data_clean)

# %%
# Prepare data for modeling: separate features and target
def prepare_train_test_split(quarter_df: pd.DataFrame, target_col: str = 'load', test_days: int = 14):
    """
    Split a quarter into train and test sets.
    The last `test_days` days are used as the test set, the rest as training.

    Args:
        quarter_df: DataFrame for one quarter
        target_col: Name of the target column
        test_days: Number of days to use for testing (default: 14)

    Returns:
        X_train, X_test, y_train, y_test, test_index
    """
    # Identify feature columns (everything except the target)
    feature_cols = [col for col in quarter_df.columns if col != target_col]

    # Calculate the cutoff date for train/test split
    # Take the last `test_days` days as test set
    max_date = quarter_df.index.max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)

    # Split into train and test based on date
    train_df = quarter_df[quarter_df.index <= cutoff_date]
    test_df = quarter_df[quarter_df.index > cutoff_date]

    # Check if we have both train and test data
    if len(train_df) == 0 or len(test_df) == 0:
        print(f"Warning: Insufficient data for {test_days}-day test split!")
        print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, test_df.index

# %%
# Train XGBoost models for each quarter and evaluate
results = []

for i, (quarter, (year, quarter_num)) in enumerate(zip(quarters, quarter_info), 1):
    print(f"\n{'='*60}")
    print(f"Q{quarter_num} {year} (Calendar Quarter {i})")
    print(f"{'='*60}")

    # Prepare train/test split (last 14 days as test)
    X_train, X_test, y_train, y_test, test_index = prepare_train_test_split(quarter, test_days=14)

    print(f"Train size: {len(X_train)} rows")
    print(f"Test size: {len(X_test)} rows (last 14 days)")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Train period: {X_train.index.min()} to {X_train.index.max()}")
    print(f"Test period: {test_index.min()} to {test_index.max()}")

    # Train XGBoost model
    print("\nTraining XGBoost model...")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    # Get the full quarter date range for visualization
    quarter_start = quarter.index.min()
    quarter_end = quarter.index.max()

    # Extract original load data for the full quarter (before dropna)
    # This allows us to plot all load values including those with missing features
    original_load_full_quarter = input_data_resampled.loc[
        (input_data_resampled.index >= quarter_start) &
        (input_data_resampled.index <= quarter_end),
        'load'
    ]

    # Store results
    results.append({
        'quarter': i,
        'year': year,
        'quarter_num': quarter_num,
        'quarter_label': f"Q{quarter_num} {year}",
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'test_index': test_index,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'original_load_full_quarter': original_load_full_quarter,  # All load values for the quarter
        'quarter_start': quarter_start,
        'quarter_end': quarter_end,
    })

# %%
# Print summary table
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
print(f"{'Quarter':<15} {'RMSE':<15} {'MAE':<15} {'R²':<15}")
print("-"*75)
for res in results:
    print(f"{res['quarter_label']:<15} {res['rmse']:<15.4f} {res['mae']:<15.4f} {res['r2']:<15.4f}")

# %%
# Visualization: Line plots (True vs Predicted over time)
# Show only test period (last 14 days) but with ALL load values from original data
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, res in enumerate(results):
    ax = axes[i]

    # Get test period date range
    test_start = res['test_index'].min()
    test_end = res['test_index'].max()

    # Extract ALL original load values for the test period (including points with missing features)
    original_load_test_period = input_data_resampled.loc[
        (input_data_resampled.index >= test_start) &
        (input_data_resampled.index <= test_end),
        'load'
    ]

    # Plot ALL true load values for test period (even where predictions might be missing)
    ax.plot(original_load_test_period.index, original_load_test_period.values,
            label='True Load (all)', linewidth=2, alpha=0.7, color='blue', marker='o', markersize=2)

    # Plot predictions (only where we have complete features)
    ax.plot(res['test_index'], res['y_pred'],
            label='Predicted Load', linewidth=2, alpha=0.8, color='red', linestyle='--', marker='x', markersize=3)

    ax.set_title(f"{res['quarter_label']}: True vs Predicted Load (Test Period)\n(RMSE={res['rmse']:.4f}, MAE={res['mae']:.4f}, R²={res['r2']:.4f})",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Load', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('quarterly_forecasts_timeseries.png', dpi=300, bbox_inches='tight')
print("\nSaved time series plot: quarterly_forecasts_timeseries.png")
plt.show()

# %%
# Visualization: Scatter plots (True vs Predicted with temporal coloring)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, res in enumerate(results):
    ax = axes[i]

    # Create color map based on timestamp position within the quarter
    time_normalized = np.arange(len(res['y_test'])) / len(res['y_test'])

    # Scatter plot with color gradient over time
    scatter = ax.scatter(res['y_test'].values, res['y_pred'],
                        c=time_normalized, cmap='viridis',
                        alpha=0.6, s=20, edgecolors='k', linewidth=0.3)

    # Add perfect prediction line (y=x)
    min_val = min(res['y_test'].min(), res['y_pred'].min())
    max_val = max(res['y_test'].max(), res['y_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', linewidth=2, label='Perfect prediction', alpha=0.7)

    ax.set_title(f"{res['quarter_label']}: Predicted vs True Load\n(RMSE={res['rmse']:.4f}, MAE={res['mae']:.4f}, R²={res['r2']:.4f})",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('True Load', fontsize=10)
    ax.set_ylabel('Predicted Load', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time progression (early → late)', rotation=270, labelpad=20)

    # Make axes equal for better comparison
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('quarterly_forecasts_scatter.png', dpi=300, bbox_inches='tight')
print("\nSaved scatter plot: quarterly_forecasts_scatter.png")
plt.show()

# %%
# Feature importance analysis for one quarter (e.g., Quarter 1)
quarter_to_analyze = 0  # 0-indexed, so this is Quarter 1

feature_importance = results[quarter_to_analyze]['model'].feature_importances_
feature_names = results[quarter_to_analyze]['X_train'].columns

# Create DataFrame for feature importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Plot top 20 most important features
plt.figure(figsize=(10, 8))
top_n = 20
top_features = importance_df.head(top_n)

plt.barh(range(top_n), top_features['importance'].values)
plt.yticks(range(top_n), top_features['feature'].values)
plt.xlabel('Feature Importance (Gain)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title(f'Top {top_n} Most Important Features ({results[quarter_to_analyze]["quarter_label"]})',
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\nSaved feature importance plot: feature_importance.png")
plt.show()

print(f"\nTop {top_n} features:")
print(importance_df.head(top_n).to_string(index=False))

# %%
# Combined metrics comparison across quarters
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

quarter_labels = [res['quarter_label'] for res in results]
quarters_nums = list(range(len(results)))  # Use indices for bar positions
rmse_values = [res['rmse'] for res in results]
mae_values = [res['mae'] for res in results]
r2_values = [res['r2'] for res in results]

# RMSE comparison
axes[0].bar(quarters_nums, rmse_values, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Quarter', fontsize=12)
axes[0].set_ylabel('RMSE', fontsize=12)
axes[0].set_title('RMSE Comparison Across Quarters', fontsize=14, fontweight='bold')
axes[0].set_xticks(quarters_nums)
axes[0].set_xticklabels(quarter_labels, rotation=45, ha='right')
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(rmse_values):
    axes[0].text(i, v + 0.01 * max(rmse_values), f'{v:.4f}',
                ha='center', va='bottom', fontsize=10)

# MAE comparison
axes[1].bar(quarters_nums, mae_values, color='coral', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Quarter', fontsize=12)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('MAE Comparison Across Quarters', fontsize=14, fontweight='bold')
axes[1].set_xticks(quarters_nums)
axes[1].set_xticklabels(quarter_labels, rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(mae_values):
    axes[1].text(i, v + 0.01 * max(mae_values), f'{v:.4f}',
                ha='center', va='bottom', fontsize=10)

# R² comparison
axes[2].bar(quarters_nums, r2_values, color='seagreen', alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Quarter', fontsize=12)
axes[2].set_ylabel('R² Score', fontsize=12)
axes[2].set_title('R² Score Comparison Across Quarters', fontsize=14, fontweight='bold')
axes[2].set_xticks(quarters_nums)
axes[2].set_xticklabels(quarter_labels, rotation=45, ha='right')
axes[2].grid(axis='y', alpha=0.3)
axes[2].set_ylim([0, 1])

# Add value labels on bars
for i, v in enumerate(r2_values):
    axes[2].text(i, v + 0.01, f'{v:.4f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved metrics comparison plot: metrics_comparison.png")
plt.show()

# %%
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nGenerated files:")
print("  - quarterly_forecasts_timeseries.png")
print("  - quarterly_forecasts_scatter.png")
print("  - feature_importance.png")
print("  - metrics_comparison.png")
