# Energy Load Forecasting

Experimental project exploring energy load forecasting using typical forecasting variables including previous load, weather and climate data, market prices, consumption profiles, seasonality, and holiday indicators.

## Outputs

- **Exploratory Data Analysis**: [reports/markdown/01_eda.md](reports/markdown/01_eda.md)
- **Feature Engineering**: [reports/markdown/03_analyse_features.md](reports/markdown/03_analyse_features.md)
- **Model Training & Evaluation (Jan-Nov vs Dec)**: [reports/markdown/05_evaluate_models_jan_nov_dec.md](reports/markdown/05_evaluate_models_jan_nov_dec.md)
- **Model Training & Evaluation (Quarterly)**: [reports/markdown/07_evaluate_models_quarterly.md](reports/markdown/07_evaluate_models_quarterly.md)

## Project Structure

The project uses DVC to manage reproducible data processing and model training pipelines:

- **Data jobs**: Feature engineering and model training scripts in `jobs/`
- **Report generation**: Dual-format outputs (HTML for local viewing, Markdown + images for GitHub) in `reports_src/`
- **Pipeline management**: All stages defined in `dvc.yaml`

## Setup

Install dependencies using uv:

```bash
uv sync
```

## Reproduce Results

Run the entire pipeline:

```bash
dvc repro
```

Or run specific stages:

```bash
dvc repro report_01_eda                          # EDA report only
dvc repro report_03_analyse_features             # Feature analysis
dvc repro report_05_evaluate_models_jan_nov_dec  # Jan-Nov vs Dec evaluation
dvc repro report_07_evaluate_models_quarterly    # Quarterly evaluation
```

Reports are generated in both HTML (`reports/html/`) and Markdown (`reports/markdown/`) formats.

## References

- **OpenSTEF**: [Open Short Term Energy Forecasting platform](https://openstef.github.io/openstef/index.html)
- **OpenSTEF Examples**: [Offline forecasting examples](https://github.com/OpenSTEF/openstef-offline-example)
- **Similar Dataset**: [Liander 2024 Energy Forecasting Benchmark on HuggingFace](https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark) (not used in this project, but similar in structure)
