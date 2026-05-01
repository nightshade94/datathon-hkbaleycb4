# Datathon Round 1 Finalization Todo

## Repo Hygiene

- [x] Capture current repo state and preserve pre-existing notebook/image changes.
- [x] Remove the duplicate `src/feature_store-Copy1.py` after confirming it matches `src/feature_store.py`.
- [x] Keep raw data and generated CSV artifacts out of version control.

## Core Scripts

- [x] Implement `src/part1_solver.py` so all 10 MCQ answers are computed from local CSV files.
- [x] Implement `src/visualization.py` with a CLI that regenerates core EDA figures.
- [x] Fix `src/forecasting.py` so `submission.csv` is complete, non-null, non-negative, and ordered like `sample_submission.csv`.
- [x] Add a conservative leak-safe forecasting option only if validation improves over the current baseline.

## Documentation

- [x] Rewrite `README.md` with setup, data placement, execution commands, solution architecture, metrics, and reproducibility notes.
- [x] Update `report/submission_runbook.md` to match the final commands, outputs, and manual submission checklist.

## Verification

- [x] Run `uv run python -m compileall -q src`.
- [x] Run `uv run src/data_prep.py` and confirm processed tables plus `daily_feature_store.csv`.
- [x] Run `uv run src/part1_solver.py` and confirm all 10 answers are produced.
- [x] Run `uv run src/visualization.py` and confirm EDA images are regenerated.
- [x] Run `uv run src/forecasting.py --method auto` and confirm metrics plus valid `submission.csv`.
- [x] Run final `git status --short` and separate new edits from pre-existing changes.

## Review

Repository is ready for a professional Round 1 code submission.

- Source cleanup: removed tracked duplicate `src/feature_store-Copy1.py`; no raw data or generated CSV files are staged for commit.
- MCQ solver: `uv run src/part1_solver.py --json` produced all 10 answers from `data/raw`: Q1 C, Q2 D, Q3 B, Q4 C, Q5 C, Q6 A, Q7 C, Q8 A, Q9 A, Q10 C.
- EDA: `uv run src/visualization.py` generated 6 figures under `images/eda/`.
- Forecasting: `uv run src/forecasting.py --method auto` selected `seasonal_profile` with MAE 635,253.88, RMSE 889,452.74, R2 0.7159 on the temporal validation split.
- Submission validation: `data/processed/submission.csv` has 548 rows, columns exactly `Date,Revenue,COGS`, zero date-order mismatches versus `sample_submission.csv`, and zero missing/negative prediction values.
- Final git status note: pre-existing user changes remain in `notebooks/drawchart.ipynb`, `notebooks/eda_story_report.ipynb`, `notebooks/question.ipynb`, plus pre-existing untracked `images/eda/08_backtest_actual_predicted.png` and `images/eda/09_forecast_to_action.png`. New implementation edits are in README/runbook, `src/part1_solver.py`, `src/visualization.py`, `src/forecasting.py`, `tasks/todo.md`, the duplicate-file deletion, and regenerated/new EDA images.
