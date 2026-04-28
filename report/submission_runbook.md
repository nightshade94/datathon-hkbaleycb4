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

### Chạy cố định Linear Regression

```bash
uv run src/forecasting.py --method linear_regression
```

### Chạy cố định Seasonal Naive

```bash
uv run src/forecasting.py --method seasonal_naive
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

### Force Linear Regression

```bash
uv run src/forecasting.py --method linear_regression
```

### Force Seasonal Naive

```bash
uv run src/forecasting.py --method seasonal_naive
```

## EN - Outputs

- `data/processed/submission.csv`
- `data/processed/baseline_metrics.json`

Upload `submission.csv` to Kaggle with exact required format and row order.
