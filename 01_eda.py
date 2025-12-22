# %%
"""Exploratory Data Analysis for load forecast data."""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

# %%
# Load the data
print("="*70)
print("LOADING DATA")
print("="*70)
df_raw = pd.read_excel("data/input_data_sun_heavy.xlsx")

# Convert first column to datetime and set as index
df_raw["Unnamed: 0"] = pd.to_datetime(df_raw["Unnamed: 0"])
df_raw = df_raw.rename(columns={"Unnamed: 0": "datetime"})
df_raw = df_raw.set_index("datetime")

print(f"Raw data loaded: {df_raw.shape}")
print(f"Raw date range: {df_raw.index.min().date()} to {df_raw.index.max().date()}")

# %%
# Resample to 15-minute intervals
print("\n" + "="*70)
print("RESAMPLING TO 15-MINUTE INTERVALS")
print("="*70)

# Create complete 15-minute time index
start_time = df_raw.index.min().floor('15min')
end_time = df_raw.index.max().ceil('15min')
complete_index = pd.date_range(start=start_time, end=end_time, freq='15min')

missing_intervals = len(complete_index) - len(df_raw)

if missing_intervals == 0:
    print("Data is already at 15-minute intervals. No resampling needed.")
    df = df_raw
else:
    print(f"Expected 15-min intervals: {len(complete_index):,}")
    print(f"Actual raw data points: {len(df_raw):,}")
    print(f"Missing intervals to fill: {missing_intervals:,}")

    # Reindex to complete 15-minute grid
    df = df_raw.reindex(complete_index)

    print(f"\nResampled data shape: {df.shape}")
    print(f"Resampled date range: {df.index.min().date()} to {df.index.max().date()}")

# %%
# Data Information Summary
print("\n" + "="*70)
print("DATA INFORMATION SUMMARY")
print("="*70)

# Basic counts
n_observations = len(df)
date_begin = df.index.min().date()
date_end = df.index.max().date()
n_days = (df.index.max() - df.index.min()).days

# Target variable
target_col = 'load'
n_missing_y = df[target_col].isna().sum()
pct_missing_y = (n_missing_y / n_observations) * 100

# Feature variables (all except load)
feature_cols = [col for col in df.columns if col != target_col]
n_x_variables = len(feature_cols)

# Missing X (any feature missing)
missing_x = df[feature_cols].isnull().any(axis=1)
n_missing_x = missing_x.sum()
pct_missing_x = (n_missing_x / n_observations) * 100

# Complete observations (both X and y present)
complete_obs = ~(df[target_col].isna() | missing_x)
n_complete = complete_obs.sum()
pct_complete = (n_complete / n_observations) * 100

print(f"\nNumber of observations: {n_observations:,}")
print(f"Date range: {date_begin} to {date_end} ({n_days} days)")
print(f"\nTarget variable: '{target_col}'")
print(f"  - Missing values: {n_missing_y:,} ({pct_missing_y:.2f}%)")
print(f"\nFeature variables (X):")
print(f"  - Number of variables: {n_x_variables}")
print(f"  - Observations with any missing feature: {n_missing_x:,} ({pct_missing_x:.2f}%)")
print(f"\nComplete observations (X + y both present): {n_complete:,} ({pct_complete:.2f}%)")

# %%
# Gap Analysis - Helper function
def find_consecutive_gaps(mask):
    """
    Find start/end indices and lengths of runs of True in a boolean Series/array.
    Returns list of (start_idx, end_idx) tuples (end is inclusive).
    """
    arr = mask.values
    n = len(arr)
    if n == 0:
        return []

    # Pad with False at both ends to handle edge cases
    padded = np.concatenate([[False], arr, [False]])
    diff = np.diff(padded.astype(int))

    gap_starts = np.where(diff == 1)[0]
    gap_ends = np.where(diff == -1)[0] - 1

    gaps = list(zip(gap_starts, gap_ends))
    return gaps

# %%
# Gap Analysis
print("\n" + "="*70)
print("MISSING DATA GAP ANALYSIS")
print("="*70)

# Analyze gaps for target, features, and overall
gap_analyses = {
    'Target (load)': df[target_col].isna(),
    'Features (X)': missing_x,
    'Overall (any)': df[target_col].isna() | missing_x
}

all_gap_results = {}

for gap_type, missing_mask in gap_analyses.items():
    print(f"\n{gap_type}:")
    print("-"*70)

    gaps = find_consecutive_gaps(missing_mask)

    if len(gaps) == 0:
        print("  No missing data gaps found!")
        all_gap_results[gap_type] = None
        continue

    # Calculate gap information
    gaps_info = []
    for start_idx, end_idx in gaps:
        gap_len = end_idx - start_idx + 1
        gap_start_time = df.index[start_idx]
        gap_end_time = df.index[end_idx]
        gap_duration_days = (gap_end_time - gap_start_time).total_seconds() / (3600 * 24)

        gaps_info.append({
            'gap_id': len(gaps_info) + 1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'n_observations_missing': gap_len,
            'start_date': gap_start_time.date(),
            'end_date': gap_end_time.date(),
            'duration_days': gap_duration_days,
        })

    gaps_df = pd.DataFrame(gaps_info)
    gaps_df_sorted = gaps_df.sort_values('duration_days', ascending=False)
    all_gap_results[gap_type] = gaps_df_sorted

    print(f"  Total gaps: {len(gaps_df)}")
    print(f"  Total missing observations: {gaps_df['n_observations_missing'].sum():,}")
    print(f"  Average gap size: {gaps_df['n_observations_missing'].mean():.1f} observations")
    print(f"  Median gap size: {gaps_df['n_observations_missing'].median():.0f} observations")
    print(f"  Largest gap: {gaps_df['n_observations_missing'].max():,} observations "
          f"({gaps_df['duration_days'].max():.2f} days)")

    # Show top 10 largest gaps
    if len(gaps_df_sorted) > 0:
        print(f"\n  Top 10 Largest Gaps:")
        print(f"  {'#':<5} {'Start Date':<15} {'End Date':<15} {'Days':<10} {'Obs':<10}")
        print("  " + "-"*60)
        for idx, (_, gap) in enumerate(gaps_df_sorted.head(10).iterrows(), 1):
            print(f"  {idx:<5} "
                  f"{str(gap['start_date']):<15} "
                  f"{str(gap['end_date']):<15} "
                  f"{gap['duration_days']:<10.2f} "
                  f"{gap['n_observations_missing']:<10}")

# %%
# Visualize missing data timeline
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

# Plot 1: Missing target variable
axes[0].fill_between(df.index, 0, df[target_col].isna().astype(int),
                      alpha=0.6, color='darkred', label='Missing load')
axes[0].set_ylim(-0.1, 1.1)
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(['Complete', 'Missing'])
axes[0].set_ylabel('Load Status', fontsize=11)
axes[0].set_title('Missing Data Timeline - Target Variable (Load)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].legend(loc='upper right')

# Plot 2: Missing features
axes[1].fill_between(df.index, 0, missing_x.astype(int),
                      alpha=0.6, color='darkorange', label='Missing features')
axes[1].set_ylim(-0.1, 1.1)
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['Complete', 'Missing'])
axes[1].set_ylabel('Features Status', fontsize=11)
axes[1].set_title('Missing Data Timeline - Features (X variables)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].legend(loc='upper right')

# Plot 3: Missing any
axes[2].fill_between(df.index, 0, (df[target_col].isna() | missing_x).astype(int),
                      alpha=0.6, color='red', label='Missing any')
axes[2].set_ylim(-0.1, 1.1)
axes[2].set_yticks([0, 1])
axes[2].set_yticklabels(['Complete', 'Missing'])
axes[2].set_xlabel('Timestamp', fontsize=12)
axes[2].set_ylabel('Overall Status', fontsize=11)
axes[2].set_title('Missing Data Timeline - Any Variable', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='x')
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()


# %%
# Categorize variables based on documentation
print("\n" + "="*70)
print("VARIABLE CATEGORIZATION")
print("="*70)

# Define variable groups based on naming patterns
load_vars = ['load']

# Weather/climate variables (temperature, humidity, pressure, wind, radiation, cloud, clearsky, mxld, snowdepth)
weather_vars = [col for col in df.columns if any(keyword in col.lower() for keyword in
    ['temperature', 'temp', 'humidity', 'pressure', 'wind', 'radiation', 'cloud', 'weather',
     'clearsky', 'mxld', 'snowdepth'])]

# Pricing variables
pricing_vars = [col for col in df.columns if any(keyword in col.lower() for keyword in
    ['price', 'epex', 'apx'])]

# Load profile variables (remaining variables not in other categories)
profile_vars = [col for col in df.columns if col not in load_vars + weather_vars + pricing_vars]

print(f"\nLoad variables ({len(load_vars)}): {load_vars}")
print(f"\nWeather/Climate variables ({len(weather_vars)}):")
for var in weather_vars[:10]:
    print(f"  - {var}")
if len(weather_vars) > 10:
    print(f"  ... and {len(weather_vars) - 10} more")

print(f"\nPricing variables ({len(pricing_vars)}): {pricing_vars if pricing_vars else 'None found'}")

print(f"\nLoad profile variables ({len(profile_vars)}):")
for var in profile_vars[:10]:
    print(f"  - {var}")
if len(profile_vars) > 10:
    print(f"  ... and {len(profile_vars) - 10} more")

# %%
# Helper function to create time series plot
def plot_timeseries(column_name, data, ax):
    """Create a time series plot for a given column."""
    ax.plot(data.index, data[column_name], linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(column_name, fontsize=12)
    ax.set_title(f"Time Series: {column_name}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics as text box (only missing values)
    missing_count = data[column_name].isna().sum()
    missing_pct = (missing_count / len(data)) * 100
    stats_text = f"Missing: {missing_count} ({missing_pct:.1f}%)"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9, family="monospace")

# %%
# Section 1: Load Variable
print("\n" + "="*70)
print("SECTION 1: LOAD VARIABLE")
print("="*70)

for column in load_vars:
    print(f"\nPlotting: {column}")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_timeseries(column, df, ax)
    plt.tight_layout()
    plt.show()

# %%
# Section 2: Weather/Climate Variables
print("\n" + "="*70)
print("SECTION 2: WEATHER/CLIMATE VARIABLES")
print("="*70)

for column in weather_vars:
    print(f"\nPlotting: {column}")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_timeseries(column, df, ax)
    plt.tight_layout()
    plt.show()

# %%
# Section 3: Pricing Variables
if pricing_vars:
    print("\n" + "="*70)
    print("SECTION 3: PRICING VARIABLES")
    print("="*70)

    for column in pricing_vars:
        print(f"\nPlotting: {column}")
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_timeseries(column, df, ax)
        plt.tight_layout()
        plt.show()

# %%
# Section 4: Load Profile Variables
print("\n" + "="*70)
print("SECTION 4: LOAD PROFILE VARIABLES")
print("="*70)

for column in profile_vars:
    print(f"\nPlotting: {column}")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_timeseries(column, df, ax)
    plt.tight_layout()
    plt.show()

# %%
print("\n" + "="*70)
print("EDA COMPLETE")
print("="*70)
