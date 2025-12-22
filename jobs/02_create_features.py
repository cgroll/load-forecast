# %%
"""Feature Engineering for load forecast data.

This script:
1. Loads Excel data with load and weather features
2. Resamples to 15-minute intervals
3. Applies OpenSTEF feature engineering to add temporal and lag features
4. Saves the feature-enriched data to CSV
5. Tracks the output with DVC
"""

# %%
# Imports
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import centralized paths
from load_forecast import Paths

# OpenSTEF imports for feature engineering
from openstef.feature_engineering.apply_features import apply_features
from openstef.data_classes.prediction_job import PredictionJobDataClass

# %%
# Load Excel data
print("="*70)
print("LOADING DATA")
print("="*70)
print(f"Loading from: {Paths.INPUT_DATA_EXCEL}")

data = pd.read_excel(
    Paths.INPUT_DATA_EXCEL,
    index_col=0,
    parse_dates=True
)

print(f"Loaded data shape: {data.shape}")
print(f"Date range: {data.index.min()} to {data.index.max()}")
print(f"Columns: {list(data.columns)}")
print(f"\nFirst few rows:")
print(data.head())

# %%
# Resample data to ensure 15-minute intervals
print("\n" + "="*70)
print("RESAMPLING DATA TO 15-MINUTE INTERVALS")
print("="*70)
print(f"Original data shape: {data.shape}")
print(f"Original date range: {data.index.min()} to {data.index.max()}")
print(f"Original time span: {(data.index.max() - data.index.min()).days} days")

# Create a complete 15-minute time index
start_time = data.index.min().floor('15min')
end_time = data.index.max().ceil('15min')
complete_index = pd.date_range(start=start_time, end=end_time, freq='15min')

print(f"\nExpected 15-min intervals: {len(complete_index):,}")
print(f"Actual data points: {len(data):,}")
print(f"Missing intervals: {len(complete_index) - len(data):,}")

# Reindex to the complete 15-minute grid
data_resampled = data.reindex(complete_index)

print(f"\nResampled data shape: {data_resampled.shape}")
print(f"Resampled date range: {data_resampled.index.min()} to {data_resampled.index.max()}")

# %%
# Create prediction job for feature engineering
print("\n" + "="*70)
print("CREATING PREDICTION JOB CONFIGURATION")
print("="*70)

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
        name="FeatureEngineering",
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
print("\n" + "="*70)
print("APPLYING OPENSTEF FEATURE ENGINEERING")
print("="*70)
print("This will add:")
print("  - Time-based features (cyclic and categorical)")
print("  - Lag features based on load history")
print("  - Weather-derived features")
print("  - Holiday features")
print("  - And more...")

data_with_features = apply_features(
    data=data_resampled.copy(),
    pj=pj,
    feature_names=None,  # Use all available features
    horizon=pj['horizon_minutes'] / 60,  # Convert to hours
)

print(f"\nFeatures added! New shape: {data_with_features.shape}")
print(f"Original columns: {data_resampled.shape[1]}")
print(f"New columns: {data_with_features.shape[1]}")
print(f"Features added: {data_with_features.shape[1] - data_resampled.shape[1]}")

# Identify new columns
original_cols = set(data_resampled.columns)
new_cols = [col for col in data_with_features.columns if col not in original_cols]

print(f"\nNew feature columns ({len(new_cols)}):")
print("\nSample of new features:")
for i, col in enumerate(new_cols[:20], 1):
    print(f"  {i:2d}. {col}")
if len(new_cols) > 20:
    print(f"  ... and {len(new_cols) - 20} more features")

# %%
# Save processed data to CSV
print("\n" + "="*70)
print("SAVING PROCESSED DATA")
print("="*70)

# Ensure output directory exists
Paths.ensure_dirs()

# Save to CSV
output_path = Paths.DATA_WITH_FEATURES
data_with_features.to_csv(output_path)

print(f"Saved processed data to: {output_path}")
print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")

# %%
print("\n" + "="*70)
print("FEATURE ENGINEERING COMPLETE")
print("="*70)
print("\nSummary:")
print(f"  - Input data shape: {data_resampled.shape}")
print(f"  - Output data shape: {data_with_features.shape}")
print(f"  - New features added: {len(new_cols)}")
print(f"  - Output file: {output_path}")
print("\nNext steps:")
print("  1. Run 03_analysis_features.py to analyze missing values and visualize patterns")
print("  2. Proceed to model training (03_fit_xgboost.py)")
