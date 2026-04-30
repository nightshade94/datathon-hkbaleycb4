# datathon-hkbaleycb4

A professional starter repository for a Data Science competition (Datathon), structured for reproducibility and clean MLOps practices.

## Project Structure

```
datathon-hkbaleycb4/
├── data/
│   ├── raw/          # Original, immutable data
│   └── processed/    # Cleaned and feature-engineered data
├── images/
│   ├── eda/          # Exploratory data analysis plots
│   └── model/        # Model performance charts
├── notebooks/        # Jupyter notebooks for exploration
├── report/           # Final reports and summaries
├── src/              # Python source modules
│   ├── __init__.py
│   ├── config.py
│   ├── data_prep.py
│   ├── forecasting.py
│   ├── part1_solver.py
│   └── visualization.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## How to Reproduce

### Prerequisites

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — a fast Python package and project manager:

```bash
# On macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Set Up the Environment

Clone the repository and restore the exact environment from the lock file:

```bash
git clone <repository-url>
cd datathon-hkbaleycb4
uv sync
```

This creates a `.venv/` virtual environment and installs all pinned dependencies automatically.

### Run Scripts

Use `uv run` to execute any Python script inside the managed environment without manually activating it:

```bash
uv run src/data_prep.py
uv run src/part1_solver.py
uv run src/forecasting.py
```

### Launch Jupyter

```bash
uv run jupyter notebook
```

## Baseline Modeling Quickstart

Run the baseline forecasting script (Linear Regression + Seasonal Naive):

```bash
uv run src/forecasting.py --method auto
```

Outputs:
- `data/processed/submission.csv`
- `data/processed/baseline_metrics.json`

Supporting bilingual docs:
- `report/round1_summary.md`
- `report/baseline_linear_regression.md`
- `report/baseline_time_series.md`
- `report/submission_runbook.md`

## Dependencies

| Package | Purpose |
|---|---|
| pandas | Data manipulation and analysis |
| numpy | Numerical computing |
| scikit-learn | Machine learning algorithms |
| matplotlib | Base plotting library |
| seaborn | Statistical data visualization |
| xgboost | Gradient boosting (XGBoost) |
| lightgbm | Gradient boosting (LightGBM) |
| shap | Model explainability |
| jupyter | Interactive notebooks |
