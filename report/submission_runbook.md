# Submission Runbook / Hướng dẫn chạy nộp bài

## VI - Chuẩn bị dữ liệu

Đảm bảo có ít nhất:

- `data/raw/sales.csv` (hoặc `data/processed/sales.csv`)
- `data/raw/sample_submission.csv` (hoặc `data/processed/sample_submission.csv`)

## VI - Chạy baseline

### Tự chọn model tốt hơn theo MAE (auto)

```bash
uv run src/forecasting.py --method auto
```

### Chạy Time-Series Cross Validation (expanding window) + auto chọn model

```bash
uv run src/forecasting.py --method auto --cv-folds 5 --cv-valid-size 30 --cv-min-train-size 180
```

### Chạy cố định Linear Regression

```bash
uv run src/forecasting.py --method linear_regression
```

### Test pipeline nhanh với baseline Linear Regression (kèm CV)

```bash
uv run src/forecasting.py --method linear_regression --cv-folds 5 --cv-valid-size 30 --cv-min-train-size 180
```

### Chạy cố định Seasonal Naive

```bash
uv run src/forecasting.py --method seasonal_naive
```

### Chạy model chính LightGBM + Time-Series CV tuning

```bash
uv run src/forecasting.py --method lightgbm --cv-folds 5 --cv-valid-size 30 --cv-min-train-size 180 --lgbm-max-trials 6
```

### Chỉ chạy CV (không sinh submission.csv)

```bash
uv run src/forecasting.py --method linear_regression --cv-only
```

## VI - Kết quả đầu ra

- `data/processed/submission.csv`
- `data/processed/baseline_metrics.json`

Upload file `submission.csv` lên Kaggle theo đúng định dạng cuộc thi.

---

## EN - Data prerequisites

Ensure at least one location contains:

- `sales.csv` in `data/raw` or `data/processed`
- `sample_submission.csv` in `data/raw` or `data/processed`

## EN - Run baseline

### Auto-select better model by validation MAE

```bash
uv run src/forecasting.py --method auto
```

### Run Time-Series Cross Validation (expanding window) + auto model selection

```bash
uv run src/forecasting.py --method auto --cv-folds 5 --cv-valid-size 30 --cv-min-train-size 180
```

### Force Linear Regression

```bash
uv run src/forecasting.py --method linear_regression
```

### Quick pipeline test with Linear Regression baseline (with CV)

```bash
uv run src/forecasting.py --method linear_regression --cv-folds 5 --cv-valid-size 30 --cv-min-train-size 180
```

### Force Seasonal Naive

```bash
uv run src/forecasting.py --method seasonal_naive
```

### Run primary LightGBM model + Time-Series CV tuning

```bash
uv run src/forecasting.py --method lightgbm --cv-folds 5 --cv-valid-size 30 --cv-min-train-size 180 --lgbm-max-trials 6
```

### Run CV only (skip submission.csv generation)

```bash
uv run src/forecasting.py --method linear_regression --cv-only
```

## EN - Outputs

- `data/processed/submission.csv`
- `data/processed/baseline_metrics.json`

`baseline_metrics.json` now includes LightGBM trial tuning results and the selected best trial by CV MAE.

Upload `submission.csv` to Kaggle with exact required format and row order.
