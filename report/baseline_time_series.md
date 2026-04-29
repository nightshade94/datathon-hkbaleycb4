# Baseline Guide: Time Series (Seasonal Naive) / Hướng dẫn baseline: Chuỗi thời gian (Seasonal Naive)

## VI - Mục tiêu

Tạo baseline time-series tối giản, dễ diễn giải: dự đoán doanh thu ngày hiện tại dựa trên doanh thu ngày cách đó 7 ngày.

## VI - Quy trình đề xuất

1. Dùng `sales.csv`, parse `Date`, sort theo thời gian.
2. Chia validation theo thời gian (holdout cuối chuỗi).
3. Seasonal-naive rule:
   - Dự báo cho ngày `t` bằng giá trị tại `t-7`.
   - Nếu thiếu `t-7`, fallback sang giá trị gần nhất có sẵn.
4. Đo MAE/RMSE/R2 trên validation.
5. Dự báo horizon submission theo cùng quy tắc.

## VI - Khi nào baseline này hữu ích

- Dữ liệu có chu kỳ tuần rõ.
- Cần baseline rất nhanh để kiểm tra pipeline và format submission.

## VI - Hạn chế

- Không học xu hướng dài hạn như tăng trưởng/giảm dài kỳ.
- Không tận dụng biến ngoại sinh (traffic, inventory, promo...).

---

## EN - Goal

Create a minimal and interpretable time-series baseline: predict each day from the value 7 days earlier.

## EN - Suggested workflow

1. Load `sales.csv`, parse `Date`, and sort chronologically.
2. Use a temporal validation holdout.
3. Seasonal-naive rule:
   - Forecast `t` using value at `t-7`.
   - If missing, fallback to the latest available past value.
4. Evaluate with MAE/RMSE/R2.
5. Use the same logic for submission horizon forecasting.

## EN - Where this baseline helps

- Strong weekly patterns.
- Fast pipeline sanity check and submission-format validation.

## EN - Limitations

- Does not learn long-term trend shifts.
- Does not use exogenous signals (traffic, inventory, promotions).

## Run command

```bash
uv run src/forecasting.py --method seasonal_naive
```
