# Tài liệu: Dữ liệu được train như thế nào với Time-Series model

## 1. Mục tiêu của pipeline train

Trong bài toán dự báo `Revenue`, mục tiêu là:

1. Học quy luật từ dữ liệu lịch sử (`sales.csv`)
2. Đánh giá model theo đúng thứ tự thời gian (tránh leakage)
3. Train model trên toàn bộ lịch sử
4. Dự báo cho các ngày trong `sample_submission.csv`

Output chuẩn của pipeline:

- `data/processed/submission.csv`
- `data/processed/baseline_metrics.json`

---

## 2. Luồng dữ liệu trong repository này

Pipeline hiện tại nằm ở `src/forecasting.py`:

1. Tìm file input từ `data/processed` rồi fallback sang `data/raw`
2. Chuẩn hóa dữ liệu:
   - Parse `Date` về datetime
   - Parse `Revenue` về numeric
   - Loại dòng lỗi và sort theo thời gian
3. Chạy **Time-Series CV (expanding window)**
4. So sánh baseline:
   - `linear_regression`
   - `seasonal_naive`
5. Nếu `--method auto`: chọn model có MAE trung bình CV tốt hơn
6. Train model đã chọn trên toàn bộ lịch sử để tạo dự báo test
7. Clamp `Revenue >= 0`
8. Nếu template thiếu `COGS`, ước lượng theo tỷ lệ lịch sử `COGS/Revenue` (fallback `0.1`)

---

## 3. Vì sao phải dùng Time-Series CV

Với dữ liệu chuỗi thời gian, không được random split như bài toán tabular thông thường.

Lý do:

- Dữ liệu tương lai không được xuất hiện trong train
- Cách split phải mô phỏng đúng điều kiện thực tế: luôn dự đoán tương lai từ quá khứ

Pipeline này dùng **expanding window**:

- Fold đầu: train ít, validate gần sau đó
- Mỗi fold sau: train mở rộng thêm, validate tiến về phía trước

---

## 4. Ví dụ sample (expanding window)

Giả sử có 20 ngày dữ liệu và cấu hình:

- `cv_min_train_size = 8`
- `cv_valid_size = 3`
- `cv_folds = 3`

Ta có:

| Fold | Train index | Validate index |
|---|---|---|
| 1 | `0..7` | `8..10` |
| 2 | `0..10` | `11..13` |
| 3 | `0..13` | `14..16` |

Mỗi fold đều giữ nguyên nguyên tắc: **train trước, validate sau**.

---

## 5. Model học gì từ dữ liệu ngày tháng

Với baseline `LinearRegression`, feature được tạo từ cột `Date`:

- `days_since`
- `day_of_week`
- `month`
- `week_sin`, `week_cos`
- `year_sin`, `year_cos`

Ý nghĩa:

- Bắt xu hướng tăng/giảm theo thời gian (`days_since`)
- Bắt chu kỳ tuần/năm bằng biến sin-cos

---

## 6. Đánh giá model như thế nào

Ở mỗi fold, model dự báo `Revenue` cho tập validate và tính:

- **MAE**: sai số tuyệt đối trung bình
- **RMSE**: nhạy với sai số lớn
- **R2**: mức độ giải thích phương sai

Sau đó lấy trung bình qua tất cả folds để có điểm CV ổn định.

---

## 7. Ví dụ sample kết quả metrics

Ví dụ `baseline_metrics.json` có thể chứa:

```json
{
  "selected_method": "seasonal_naive",
  "selected_metrics": {
    "mae": 2632389.13,
    "rmse": 3237935.16,
    "r2": -3.19
  },
  "cv": {
    "strategy": "expanding_window",
    "config": {
      "folds": 5,
      "valid_size": 30,
      "min_train_size": 180
    }
  },
  "submission_path": "data\\processed\\submission.csv"
}
```

---

## 8. Cách chạy thực tế (CLI)

Chạy baseline Linear Regression + CV:

```bash
uv run src/forecasting.py --method linear_regression --cv-folds 5 --cv-valid-size 30 --cv-min-train-size 180
```

Auto chọn model theo CV MAE:

```bash
uv run src/forecasting.py --method auto --cv-folds 5 --cv-valid-size 30 --cv-min-train-size 180
```

Chỉ chạy CV (không tạo `submission.csv`):

```bash
uv run src/forecasting.py --method linear_regression --cv-only
```

Chạy model chính LightGBM + tuning theo Time-Series CV:

```bash
uv run src/forecasting.py --method lightgbm --cv-folds 5 --cv-valid-size 30 --cv-min-train-size 180 --lgbm-max-trials 6
```

---

## 9. Lỗi thường gặp và cách xử lý

1. **Không đủ số dòng cho cấu hình CV**
   - Giảm `--cv-folds` hoặc `--cv-valid-size`
   - Giảm `--cv-min-train-size`

2. **Thiếu file input**
   - Kiểm tra `sales.csv` và `sample_submission.csv` có trong `data/raw` hoặc `data/processed`

3. **Date parse lỗi**
   - Kiểm tra format cột `Date` trong CSV

---

## 10. Nên đọc tiếp ở đâu

- Notebook thực hành từng dòng: `notebooks/huong_dan_train_baseline_vi.ipynb`
- Notebook giảng giải cho người mới: `notebooks/bai_giang_train_time_series_vi.ipynb`
