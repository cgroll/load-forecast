# %%
"""Exploratory Data Analysis - Part 2: Individual Time Series Plots."""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 3)

# %%
# Load the data
print("="*70)
print("LOADING DATA FOR TIME SERIES VISUALIZATION")
print("="*70)
df_raw = pd.read_excel("data/input_data_sun_heavy.xlsx")

# Convert first column to datetime and set as index
df_raw["Unnamed: 0"] = pd.to_datetime(df_raw["Unnamed: 0"])
df_raw = df_raw.rename(columns={"Unnamed: 0": "datetime"})
df_raw = df_raw.set_index("datetime")

# Resample to 15-minute intervals
start_time = df_raw.index.min().floor('15min')
end_time = df_raw.index.max().ceil('15min')
complete_index = pd.date_range(start=start_time, end=end_time, freq='15min')

missing_intervals = len(complete_index) - len(df_raw)

if missing_intervals == 0:
    df = df_raw
else:
    df = df_raw.reindex(complete_index)

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

# %%
# Categorize variables
target_col = 'load'
load_vars = ['load']

weather_vars = [col for col in df.columns if any(keyword in col.lower() for keyword in
    ['temperature', 'temp', 'humidity', 'pressure', 'wind', 'radiation', 'cloud', 'weather',
     'clearsky', 'mxld', 'snowdepth'])]

pricing_vars = [col for col in df.columns if any(keyword in col.lower() for keyword in
    ['price', 'epex', 'apx'])]

profile_vars = [col for col in df.columns if col not in load_vars + weather_vars + pricing_vars]

print(f"Variable counts: Load={len(load_vars)}, Weather={len(weather_vars)}, Pricing={len(pricing_vars)}, Profile={len(profile_vars)}")

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
    fig, ax = plt.subplots(figsize=(7, 3))
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
    fig, ax = plt.subplots(figsize=(7, 3))
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
        fig, ax = plt.subplots(figsize=(7, 3))
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
    fig, ax = plt.subplots(figsize=(7, 3))
    plot_timeseries(column, df, ax)
    plt.tight_layout()
    plt.show()

# %%
print("\n" + "="*70)
print("EDA PART 2 (TIME SERIES PLOTS) COMPLETE")
print("="*70)
