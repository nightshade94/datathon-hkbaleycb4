# Submission Runbook

## 1. Environment

```bash
uv sync
```

The repository is reproducible from `pyproject.toml` and `uv.lock`. Raw competition data must be placed locally in `data/raw/`; it is not committed.

## 2. Generate Processed Data

```bash
uv run src/data_prep.py
```

Expected result:

- Cleaned CSV snapshots under `data/processed/`
- `data/processed/daily_feature_store.csv`

## 3. Compute Part 1 Answers

```bash
uv run src/part1_solver.py
```

Current answers from `data/raw/`:

```text
Q1: C
Q2: D
Q3: B
Q4: C
Q5: C
Q6: A
Q7: C
Q8: A
Q9: A
Q10: C
```

Use `uv run src/part1_solver.py --json` to print supporting calculations.

## 4. Regenerate EDA Figures

```bash
uv run src/visualization.py
```

Expected result: refreshed PNG files in `images/eda/`.

## 5. Generate Kaggle Submission

```bash
uv run src/forecasting.py --method auto
```

Expected result:

- `data/processed/submission.csv`
- `data/processed/baseline_metrics.json`

Latest validation metrics:

| Selected method | MAE | RMSE | R2 |
|---|---:|---:|---:|
| seasonal_profile | 635,253.88 | 889,452.74 | 0.7159 |

Submission format check:

- 548 rows
- Columns exactly `Date,Revenue,COGS`
- Date order identical to `sample_submission.csv`
- No missing or negative predictions

## 6. Manual Final Submission

- Upload `data/processed/submission.csv` to Kaggle.
- Export and upload the final PDF report using the required NeurIPS template.
- Include the public or organizer-accessible GitHub repository link.
- Fill the official form with MCQ answers, Kaggle link, report PDF, student ID photos for all members, and onsite final-round availability confirmation.
