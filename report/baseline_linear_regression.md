# Baseline Guide: Linear Regression / Hướng dẫn baseline: Hồi quy tuyến tính

## VI - Mục tiêu

Dựng baseline đơn giản, dễ tái lập để dự báo `Revenue` theo `Date`, dùng như mốc so sánh trước khi nâng cấp mô hình.

## VI - Quy trình đề xuất

1. Dùng `sales.csv` (train) làm dữ liệu lịch sử.
2. Parse `Date`, sort tăng dần theo thời gian.
3. Tạo feature thời gian:
   - `days_since_start`
   - `day_of_week`, `month`
   - seasonal sin/cos theo tuần và năm
4. Chia validation theo thời gian (ví dụ 20% cuối).
5. Fit `LinearRegression` trên train window.
6. Đánh giá trên validation bằng MAE, RMSE, R2.
7. Refit trên toàn bộ lịch sử rồi dự báo cho các ngày trong `sample_submission.csv`.

## VI - Lưu ý chất lượng

- Không random split cho chuỗi thời gian.
- Không dùng bất kỳ cột mục tiêu test nào để tạo feature.
- Baseline này mạnh ở xu hướng tổng thể, yếu ở pattern phi tuyến/đột biến.

---

## EN - Goal

Build a reproducible, fast baseline to forecast `Revenue` from calendar features, used as a benchmark before advanced models.

## EN - Suggested workflow

1. Use `sales.csv` as historical training data.
2. Parse and sort by `Date`.
3. Engineer calendar features:
   - `days_since_start`
   - `day_of_week`, `month`
   - weekly/yearly sinusoidal terms
4. Use a temporal holdout (e.g., last 20%) for validation.
5. Train `LinearRegression` on the train window.
6. Evaluate MAE, RMSE, R2 on the holdout window.
7. Refit on full history and predict dates in `sample_submission.csv`.

## EN - Quality notes

- Never use random split for time series.
- Prevent leakage from hidden test target values.
- This baseline captures trend/seasonality but may miss nonlinear shocks.

## Run command

```bash
uv run src/forecasting.py --method linear_regression
```
