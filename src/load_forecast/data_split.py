"""Data splitting utilities for quarterly train-test splits."""

import pandas as pd
import numpy as np
from typing import List, Tuple


def split_into_calendar_quarters(df: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[Tuple[int, int]]]:
    """Split DataFrame into calendar quarters.

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        quarters: List of DataFrames, one per quarter
        quarter_info: List of (year, quarter_num) tuples
    """
    df_with_quarter = df.copy()
    df_with_quarter['year'] = df_with_quarter.index.year
    df_with_quarter['quarter'] = df_with_quarter.index.quarter

    unique_quarters = df_with_quarter[['year', 'quarter']].drop_duplicates().sort_values(['year', 'quarter'])

    quarters = []
    quarter_info = []

    for _, row in unique_quarters.iterrows():
        year = int(row['year'])
        quarter_num = int(row['quarter'])
        mask = (df_with_quarter['year'] == year) & (df_with_quarter['quarter'] == quarter_num)
        quarter_data = df.loc[mask]

        if len(quarter_data) > 0:
            quarters.append(quarter_data)
            quarter_info.append((year, quarter_num))

    return quarters, quarter_info


def find_test_period(
    df: pd.DataFrame,
    target_col: str,
    test_days: int,
    min_coverage: float
) -> Tuple[pd.Timestamp, pd.DataFrame, int]:
    """Find test period with sufficient data coverage.

    Strategy:
    - Get all valid_days (with sufficient coverage based on min_coverage threshold)
    - test_df includes at least `test_days_to_use` valid days, plus any days after the cutoff,
      even if they are not valid
    - end_test_date does not have to be a valid day

    Args:
        df: DataFrame for a single quarter
        target_col: Name of target column (e.g., 'load')
        test_days: Desired number of test days
        min_coverage: Minimum data coverage threshold (0-1)

    Returns:
        cutoff_date: Last date to include in training set
        test_df: Test set DataFrame
        test_days_to_use: Actual number of valid test days used
    """
    df_with_date = df.copy()
    df_with_date['date'] = df_with_date.index.date
    expected_obs_per_day = 96  # 15-minute intervals: 24 hours * 4

    # Compute coverage for each day
    daily_counts = df_with_date.groupby('date')[target_col].count()
    daily_coverage = daily_counts / expected_obs_per_day
    valid_days = sorted(daily_coverage[daily_coverage >= min_coverage].index)

    test_days_to_use = min(len(valid_days), test_days)
    assert test_days_to_use > 0, "No valid days found with sufficient coverage"

    # Find the earliest valid day for desired test period
    start_test_date = valid_days[-test_days_to_use]
    # Determine end_test_date: last day in the data
    end_test_date = df_with_date['date'].max()

    # Select all rows from start_test_date to end_test_date, inclusive (regardless of whether covered or not)
    test_mask = (df_with_date['date'] >= start_test_date) & (df_with_date['date'] <= end_test_date)
    test_df = df_with_date[test_mask].drop('date', axis=1)
    first_test_date = pd.Timestamp(valid_days[-test_days_to_use])
    cutoff_date = first_test_date - pd.Timedelta(days=1)

    return cutoff_date, test_df, test_days_to_use


def prepare_train_test_split(
    quarter_df: pd.DataFrame,
    target_col: str = 'load',
    test_days: int = 14,
    min_coverage: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex]:
    """Split a quarter into train and test sets with sufficient data coverage.

    Args:
        quarter_df: DataFrame for a single quarter
        target_col: Name of target column (default: 'load')
        test_days: Desired number of test days (default: 14)
        min_coverage: Minimum data coverage threshold (default: 0.95)

    Returns:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        test_index: DatetimeIndex of test set
    """
    feature_cols = [col for col in quarter_df.columns if col != target_col]
    cutoff_date, test_df, actual_test_days = find_test_period(
        quarter_df, target_col, test_days, min_coverage
    )
    train_df = quarter_df[quarter_df.index <= cutoff_date]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, test_df.index


def split_quarters_train_test(
    df: pd.DataFrame,
    target_col: str = 'load',
    test_days: int = 14,
    min_coverage: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    """Split data into calendar quarters and prepare train/test splits.

    This function:
    1. Splits the data into calendar quarters
    2. For each quarter, identifies valid test period (last N days with sufficient coverage)
    3. Combines all quarters' data into global train and test sets
    4. Returns complete dataframes (not split into X/y) for flexible model preparation

    Args:
        df: DataFrame with DatetimeIndex and features
        target_col: Name of target column (default: 'load')
        test_days: Desired number of test days per quarter (default: 14)
        min_coverage: Minimum data coverage threshold (default: 0.95)

    Returns:
        train_data: Combined training data from all quarters (includes target + features)
        test_data: Combined test data from all quarters (includes target + features)
        quarter_info: List of dicts with quarter metadata including train/test boundaries
    """
    # Split into calendar quarters
    quarters, quarter_tuples = split_into_calendar_quarters(df)

    all_train_data = []
    all_test_data = []
    quarter_metadata = []

    for quarter_df, (year, quarter_num) in zip(quarters, quarter_tuples):
        # Find the test period for this quarter
        cutoff_date, test_df, actual_test_days = find_test_period(
            quarter_df, target_col, test_days, min_coverage
        )
        train_df = quarter_df[quarter_df.index <= cutoff_date]

        all_train_data.append(train_df)
        all_test_data.append(test_df)

        quarter_metadata.append({
            'quarter_num': quarter_num,
            'year': year,
            'quarter_label': f"Q{quarter_num} {year}",
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_start': str(train_df.index.min()),
            'train_end': str(train_df.index.max()),
            'test_start': str(test_df.index.min()),
            'test_end': str(test_df.index.max()),
        })

    # Combine all quarters
    train_data = pd.concat(all_train_data, axis=0).sort_index()
    test_data = pd.concat(all_test_data, axis=0).sort_index()

    return train_data, test_data, quarter_metadata
