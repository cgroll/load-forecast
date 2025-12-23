"""Load forecast package."""

from load_forecast.paths import Paths
from load_forecast.data_split import (
    split_into_calendar_quarters,
    find_test_period,
    prepare_train_test_split,
    split_quarters_train_test,
)

__all__ = [
    "Paths",
    "split_into_calendar_quarters",
    "find_test_period",
    "prepare_train_test_split",
    "split_quarters_train_test",
]
__version__ = "0.1.0"
