"""
Centralized path definitions for the load-forecast project.

All paths are defined relative to the project root directory, making them
independent of the current working directory. This is especially useful when
running scripts with ipykernel or from different locations.

Usage:
    from paths import Paths

    # Load data
    df = pd.read_csv(Paths.DATA_WITH_FEATURES)

    # Save output
    model.save(Paths.MODELS / "my_model.pkl")
"""

from pathlib import Path


class Paths:
    """Central path configuration for the project."""

    # Project root directory (two levels up from this file: src/load_forecast/paths.py -> project root)
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

    # ============================================================
    # DATA DIRECTORIES
    # ============================================================

    # Raw input data
    DATA_RAW = PROJECT_ROOT / "data" / "raw_inputs"
    INPUT_DATA_EXCEL = DATA_RAW / "input_data_sun_heavy.xlsx"

    # Processed data
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    DATA_WITH_FEATURES = DATA_PROCESSED / "data_with_features.csv"

    # ============================================================
    # OUTPUT DIRECTORIES
    # ============================================================

    # Models
    MODELS = PROJECT_ROOT / "models"

    # Figures and plots
    FIGURES = PROJECT_ROOT / "figures"

    # Reports
    REPORTS = PROJECT_ROOT / "reports"
    REPORTS_HTML = REPORTS / "html"
    REPORTS_MARKDOWN = REPORTS / "markdown"

    # ============================================================
    # SOURCE DIRECTORIES
    # ============================================================

    # Job scripts
    JOBS = PROJECT_ROOT / "jobs"

    # Report source scripts
    REPORTS_SRC = PROJECT_ROOT / "reports_src"

    # Tools
    TOOLS = PROJECT_ROOT / "tools"

    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories if they don't exist."""
        dirs_to_create = [
            cls.DATA_RAW,
            cls.DATA_PROCESSED,
            cls.MODELS,
            cls.FIGURES,
            cls.REPORTS_HTML,
            cls.REPORTS_MARKDOWN,
        ]

        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

        print(f"Ensured all directories exist relative to: {cls.PROJECT_ROOT}")

    @classmethod
    def get_relative_path(cls, path: Path) -> Path:
        """Get path relative to project root."""
        try:
            return path.relative_to(cls.PROJECT_ROOT)
        except ValueError:
            # Path is not relative to project root
            return path

    @classmethod
    def print_config(cls):
        """Print current path configuration."""
        print("="*70)
        print("PATH CONFIGURATION")
        print("="*70)
        print(f"\nProject Root: {cls.PROJECT_ROOT}")
        print(f"\nData Paths:")
        print(f"  Raw Data:           {cls.DATA_RAW}")
        print(f"  Input Excel:        {cls.INPUT_DATA_EXCEL}")
        print(f"  Processed Data:     {cls.DATA_PROCESSED}")
        print(f"  Features CSV:       {cls.DATA_WITH_FEATURES}")
        print(f"\nOutput Paths:")
        print(f"  Models:             {cls.MODELS}")
        print(f"  Figures:            {cls.FIGURES}")
        print(f"  Reports:            {cls.REPORTS}")
        print(f"  Reports HTML:       {cls.REPORTS_HTML}")
        print(f"  Reports Markdown:   {cls.REPORTS_MARKDOWN}")
        print("="*70)


# Convenience function for quick testing
if __name__ == "__main__":
    Paths.print_config()
    print("\nChecking if key files exist:")
    print(f"  Input Excel exists:  {Paths.INPUT_DATA_EXCEL.exists()}")
    print(f"  Features CSV exists: {Paths.DATA_WITH_FEATURES.exists()}")
