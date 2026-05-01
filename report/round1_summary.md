# Datathon 2026 Round 1 Summary / Tóm tắt Vòng 1

## VI - Tóm tắt nhanh

- **Phần 1 (20 điểm):** 10 câu MCQ, mỗi câu 2 điểm.
- **Phần 2 (60 điểm):** trực quan hóa + phân tích dữ liệu, chấm theo độ sâu từ Descriptive -> Prescriptive.
- **Phần 3 (20 điểm):** dự báo `Revenue` cho giai đoạn test, đánh giá bằng **MAE, RMSE, R2**.

### Ràng buộc quan trọng

1. Không dùng dữ liệu ngoài.
2. Không gây leakage (đặc biệt không dùng target test làm feature).
3. Nộp `submission.csv` đúng thứ tự dòng như `sample_submission.csv`.
4. Cần mã nguồn tái lập + giải thích mô hình (feature importance / SHAP hoặc tương đương).

### Checklist triển khai baseline

1. Làm sạch và chuẩn hóa dữ liệu.
2. Tạo baseline Linear Regression + baseline Time Series.
3. Đo MAE/RMSE/R2 trên validation theo thời gian.
4. Sinh `submission.csv` theo đúng định dạng.
5. Ghi lại runbook để tái lập nhanh.

---

## EN - Quick summary

- **Part 1 (20 pts):** 10 MCQs, 2 points each.
- **Part 2 (60 pts):** EDA + storytelling, judged across Descriptive -> Prescriptive depth.
- **Part 3 (20 pts):** forecast `Revenue` on test horizon using **MAE, RMSE, R2**.

### Critical constraints

1. No external data.
2. No target leakage (especially from hidden test targets).
3. Keep `submission.csv` in the same row order as `sample_submission.csv`.
4. Provide reproducible code + explainability section (feature importance / SHAP or equivalent).

### Baseline execution checklist

1. Clean and standardize data.
2. Build Linear Regression baseline + Time Series baseline.
3. Evaluate MAE/RMSE/R2 with temporal validation.
4. Generate correctly formatted `submission.csv`.
5. Keep a clear runbook for reproducibility.
