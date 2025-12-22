# %%
# # Feature Analysis and Visualization
#
# This script:
#
# 1. Loads the feature-enriched data
# 2. Analyzes missing value patterns
# 3. Visualizes missing data for top 20 features with most missing values
# 4. Provides summary statistics for new features

# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# %%
# Load processed data
print("="*70)
print("LOADING PROCESSED DATA")
print("="*70)

data_path = Path("data/processed/data_with_features.csv")
data_with_features = pd.read_csv(data_path, index_col=0, parse_dates=True)

print(f"Loaded data shape: {data_with_features.shape}")
print(f"Date range: {data_with_features.index.min()} to {data_with_features.index.max()}")

# Load original data to identify new features
original_data = pd.read_excel(
    "data/raw_inputs/input_data_sun_heavy.xlsx",
    index_col=0,
    parse_dates=True
)

# Identify new columns (features added by OpenSTEF)
original_cols = set(original_data.columns)
new_cols = [col for col in data_with_features.columns if col not in original_cols]

print(f"\nOriginal columns: {len(original_cols)}")
print(f"Total columns: {len(data_with_features.columns)}")
print(f"New features: {len(new_cols)}")

# %%
# Missing value analysis for new features
print("\n" + "="*70)
print("MISSING VALUE ANALYSIS - NEW FEATURES ONLY")
print("="*70)

# Calculate missing values for new features
new_features_df = data_with_features[new_cols]
missing_counts = new_features_df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

total_rows = len(data_with_features)

if len(missing_counts) == 0:
    print("\n✓ No missing values in any new features!")
else:
    print(f"\nNew features with missing values: {len(missing_counts)} out of {len(new_cols)}")
    print(f"Total observations: {total_rows:,}")

    print("\n" + "-"*70)
    print(f"{'Feature':<40} {'Missing':<12} {'Percentage':<12}")
    print("-"*70)

    for col, count in missing_counts.head(20).items():
        pct = (count / total_rows) * 100
        print(f"{col:<40} {count:>10,}   {pct:>10.2f}%")

    if len(missing_counts) > 20:
        print(f"\n... and {len(missing_counts) - 20} more features with missing values")

    print("\n" + "-"*70)
    print(f"{'Summary Statistics':<40}")
    print("-"*70)
    print(f"{'Total missing cells (new features):':<40} {missing_counts.sum():>10,}")
    print(f"{'Average missing per feature:':<40} {missing_counts.mean():>10.1f}")
    print(f"{'Median missing per feature:':<40} {missing_counts.median():>10.0f}")
    print(f"{'Max missing in any feature:':<40} {missing_counts.max():>10,}")

# %%
# Visualize missing data patterns for top 20 features
print("\n" + "="*70)
print("MISSING DATA VISUALIZATION - TOP 20 FEATURES")
print("="*70)

if len(missing_counts) > 0:
    # Select top 20 features with most missing values
    top_missing_features = missing_counts.head(20).index.tolist()

    print(f"\nVisualizing top {len(top_missing_features)} features with most missing values")

    # Create a heatmap showing missing data patterns over time
    # Sample the data if it's too large (show every nth row for visualization)
    sample_rate = max(1, len(data_with_features) // 1000)  # Max 1000 rows for visualization

    if sample_rate > 1:
        print(f"Sampling every {sample_rate}th row for visualization clarity")

    sampled_data = new_features_df.iloc[::sample_rate]
    sampled_index = data_with_features.index[::sample_rate]

    # Create binary matrix (1 = missing, 0 = present)
    missing_matrix = sampled_data[top_missing_features].isnull().astype(int)

    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot heatmap
    sns.heatmap(
        missing_matrix.T,
        cmap=['lightgreen', 'darkred'],
        cbar_kws={'label': 'Missing (1) vs Present (0)'},
        ax=ax,
        yticklabels=top_missing_features,
        xticklabels=False  # Don't show x-tick labels, we'll add dates separately
    )

    # Add custom x-axis labels with dates in yyyy-mm-dd format
    # Select evenly spaced dates for x-axis labels
    n_labels = 10  # Number of date labels to show
    label_indices = np.linspace(0, len(sampled_index) - 1, n_labels, dtype=int)
    label_positions = label_indices
    label_texts = [sampled_index[i].strftime('%Y-%m-%d') for i in label_indices]

    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_texts, rotation=45, ha='right')

    ax.set_title(f'Missing Data Pattern - Top {len(top_missing_features)} Features with Most Missing Values',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

    # Print statistics for visualized features
    print(f"\nMissing value statistics for visualized features:")
    print("-"*70)
    for i, feature in enumerate(top_missing_features, 1):
        missing = missing_counts[feature]
        pct = (missing / total_rows) * 100
        print(f"{i:2d}. {feature:<35} {missing:>8,} ({pct:>5.1f}%)")
else:
    print("\nNo missing values to visualize!")

# %%
# Summary statistics for new features
print("\n" + "="*70)
print("SUMMARY STATISTICS - NEW FEATURES")
print("="*70)

# Group features by type based on naming patterns
feature_groups = {
    'Time-based (cyclic)': [col for col in new_cols if any(x in col.lower() for x in ['sin', 'cos'])],
    'Time-based (categorical)': [col for col in new_cols if any(x in col.lower() for x in ['month', 'day', 'hour', 'week', 'quarter', 'year']) and 'sin' not in col.lower() and 'cos' not in col.lower()],
    'Lag features': [col for col in new_cols if 'lag' in col.lower() or 'T-' in col],
    'Holiday features': [col for col in new_cols if 'holiday' in col.lower() or 'bridge' in col.lower()],
    'Other features': [],
}

# Assign remaining features to 'Other'
categorized = set()
for group_features in feature_groups.values():
    categorized.update(group_features)

feature_groups['Other features'] = [col for col in new_cols if col not in categorized]

print("\nNew features grouped by type:")
print("-"*70)

for group_name, group_features in feature_groups.items():
    if len(group_features) > 0:
        print(f"\n{group_name}: {len(group_features)} features")

        # Show all features (not just first 5)
        for col in group_features:
            missing = new_features_df[col].isnull().sum()
            missing_pct = (missing / total_rows) * 100
            print(f"  - {col:<35} (missing: {missing:>6,} = {missing_pct:>5.1f}%)")

# %%
print("\n" + "="*70)
print("FEATURE ANALYSIS COMPLETE")
print("="*70)
print("\nKey insights:")
print(f"  - Total new features: {len(new_cols)}")
print(f"  - Features with missing values: {len(missing_counts)}")
if len(missing_counts) > 0:
    print(f"  - Highest missing value rate: {(missing_counts.max() / total_rows * 100):.1f}%")
    print(f"  - Most affected feature: {missing_counts.idxmax()}")
print("\nNext steps:")
print("  1. Decide on missing value handling strategy (dropna, forward-fill, etc.)")
print("  2. Proceed to model training")
