"""Baseline forecasting script (linear regression + seasonal naive)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _find_existing_file(base_dir: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return None


def _load_sales(processed_dir: Path, raw_dir: Path) -> pd.DataFrame:
    sales_path = _find_existing_file(processed_dir, ["sales.csv"])
    if sales_path is None:
        sales_path = _find_existing_file(raw_dir, ["sales.csv"])
    if sales_path is None:
        raise FileNotFoundError("Could not find sales.csv in data/processed or data/raw.")

    sales_df = pd.read_csv(sales_path, low_memory=False)
    required = {"Date", "Revenue"}
    missing = required.difference(sales_df.columns)
    if missing:
        raise ValueError(f"sales.csv is missing required columns: {sorted(missing)}")

    sales_df["Date"] = pd.to_datetime(sales_df["Date"], errors="coerce")
    sales_df = sales_df.dropna(subset=["Date", "Revenue"]).copy()
    sales_df = sales_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    sales_df["Revenue"] = pd.to_numeric(sales_df["Revenue"], errors="coerce")
    sales_df = sales_df.dropna(subset=["Revenue"]).reset_index(drop=True)
    return sales_df


def _load_submission_template(processed_dir: Path, raw_dir: Path) -> pd.DataFrame:
    template_path = _find_existing_file(
        processed_dir, ["sample_submission.csv", "samplesubmission.csv"]
    )
    if template_path is None:
        template_path = _find_existing_file(raw_dir, ["sample_submission.csv", "samplesubmission.csv"])
    if template_path is None:
        raise FileNotFoundError(
            "Could not find sample_submission.csv (or samplesubmission.csv) in data/processed or data/raw."
        )

    template_df = pd.read_csv(template_path, low_memory=False)
    if "Date" not in template_df.columns:
        raise ValueError("sample_submission.csv is missing required column: Date")
    template_df["Date"] = pd.to_datetime(template_df["Date"], errors="coerce")
    if template_df["Date"].isna().any():
        raise ValueError("sample_submission.csv contains invalid Date values.")
    return template_df


def _date_features(dates: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    days_since = (dates - origin).dt.days.astype(float)
    day_of_week = dates.dt.dayofweek.astype(float)
    month = dates.dt.month.astype(float)
    day_of_year = dates.dt.dayofyear.astype(float)
    week_sin = np.sin(2.0 * np.pi * day_of_week / 7.0)
    week_cos = np.cos(2.0 * np.pi * day_of_week / 7.0)
    year_sin = np.sin(2.0 * np.pi * day_of_year / 365.25)
    year_cos = np.cos(2.0 * np.pi * day_of_year / 365.25)

    return pd.DataFrame(
        {
            "days_since": days_since,
            "day_of_week": day_of_week,
            "month": month,
            "week_sin": week_sin,
            "week_cos": week_cos,
            "year_sin": year_sin,
            "year_cos": year_cos,
        }
    )


def _fit_linear_regression(train_df: pd.DataFrame, predict_dates: pd.Series) -> np.ndarray:
    origin = train_df["Date"].min()
    x_train = _date_features(train_df["Date"], origin)
    y_train = train_df["Revenue"].to_numpy()

    model = LinearRegression()
    model.fit(x_train, y_train)

    x_predict = _date_features(predict_dates, origin)
    return model.predict(x_predict)


def _seasonal_naive_predict(history_df: pd.DataFrame, predict_dates: pd.Series) -> np.ndarray:
    known = {row.Date: float(row.Revenue) for row in history_df.itertuples(index=False)}
    fallback = float(history_df["Revenue"].iloc[-1])
    predictions: list[float] = []

    for date in pd.to_datetime(predict_dates).sort_values():
        lag_date = date - pd.Timedelta(days=7)
        if lag_date in known:
            pred = known[lag_date]
        else:
            past_dates = [d for d in known if d < date]
            pred = known[max(past_dates)] if past_dates else fallback
        predictions.append(float(pred))
        known[date] = float(pred)

    ordered = pd.DataFrame({"Date": pd.to_datetime(predict_dates), "pred": predictions})
    return ordered.sort_values("Date")["pred"].to_numpy()


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except ValueError:
        r2 = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _predict_with_method(train_df: pd.DataFrame, predict_dates: pd.Series, method: str) -> np.ndarray:
    if method == "linear_regression":
        return _fit_linear_regression(train_df, predict_dates)
    if method == "seasonal_naive":
        return _seasonal_naive_predict(train_df, predict_dates)
    raise ValueError(f"Unsupported method: {method}")


def _generate_expanding_window_splits(
    sales_df: pd.DataFrame,
    *,
    n_splits: int,
    valid_size: int,
    min_train_size: int,
) -> list[tuple[int, int, int]]:
    n_rows = len(sales_df)
    if n_splits < 1:
        raise ValueError("cv_folds must be >= 1.")
    if valid_size < 1:
        raise ValueError("cv_valid_size must be >= 1.")
    if min_train_size < 1:
        raise ValueError("cv_min_train_size must be >= 1.")

    required_rows = min_train_size + n_splits * valid_size
    if n_rows < required_rows:
        raise ValueError(
            "Not enough rows in sales.csv for CV config. "
            f"Need at least {required_rows} rows but got {n_rows}. "
            "Reduce --cv-folds / --cv-valid-size or --cv-min-train-size."
        )

    fold_windows: list[tuple[int, int, int]] = []
    for fold_idx in range(n_splits):
        train_end = min_train_size + fold_idx * valid_size
        valid_start = train_end
        valid_end = valid_start + valid_size
        fold_windows.append((train_end, valid_start, valid_end))
    return fold_windows


def _cross_validate_methods(
    sales_df: pd.DataFrame,
    *,
    methods: list[str],
    n_splits: int,
    valid_size: int,
    min_train_size: int,
) -> dict[str, dict[str, object]]:
    fold_windows = _generate_expanding_window_splits(
        sales_df,
        n_splits=n_splits,
        valid_size=valid_size,
        min_train_size=min_train_size,
    )

    cv_results: dict[str, dict[str, object]] = {}
    for method in methods:
        fold_metrics: list[dict[str, float | int]] = []
        for fold_number, (train_end, valid_start, valid_end) in enumerate(fold_windows, start=1):
            train_df = sales_df.iloc[:train_end].copy()
            valid_df = sales_df.iloc[valid_start:valid_end].copy()
            y_valid = valid_df["Revenue"].to_numpy()
            valid_pred = _predict_with_method(train_df, valid_df["Date"], method)
            metrics = _evaluate(y_valid, valid_pred)
            fold_metrics.append(
                {
                    "fold": fold_number,
                    "train_rows": int(train_df.shape[0]),
                    "valid_rows": int(valid_df.shape[0]),
                    **metrics,
                }
            )

        avg_metrics = {
            "mae": float(np.mean([float(f["mae"]) for f in fold_metrics])),
            "rmse": float(np.mean([float(f["rmse"]) for f in fold_metrics])),
            "r2": float(np.mean([float(f["r2"]) for f in fold_metrics])),
        }
        cv_results[method] = {
            "n_folds": n_splits,
            "valid_size": valid_size,
            "min_train_size": min_train_size,
            "avg_metrics": avg_metrics,
            "fold_metrics": fold_metrics,
        }
    return cv_results


def _load_promotions(processed_dir: Path, raw_dir: Path) -> pd.DataFrame:
    promotions_path = _find_existing_file(processed_dir, ["promotions.csv"])
    if promotions_path is None:
        promotions_path = _find_existing_file(raw_dir, ["promotions.csv"])
    if promotions_path is None:
        return pd.DataFrame(columns=["start_date", "end_date", "promo_type"])

    promotions_df = pd.read_csv(promotions_path, low_memory=False)
    normalized_cols = {col.strip().lower().replace("_", ""): col for col in promotions_df.columns}
    start_col = normalized_cols.get("startdate")
    end_col = normalized_cols.get("enddate")
    promo_type_col = normalized_cols.get("promotype")
    if start_col is None or end_col is None:
        return pd.DataFrame(columns=["start_date", "end_date", "promo_type"])

    promotions = pd.DataFrame(
        {
            "start_date": pd.to_datetime(promotions_df[start_col], errors="coerce"),
            "end_date": pd.to_datetime(promotions_df[end_col], errors="coerce"),
            "promo_type": (
                promotions_df[promo_type_col].astype("string").str.lower().fillna("unknown")
                if promo_type_col is not None
                else "unknown"
            ),
        }
    )
    return promotions.dropna(subset=["start_date", "end_date"]).reset_index(drop=True)


def _load_daily_feature_store(processed_dir: Path) -> pd.DataFrame:
    feature_store_path = processed_dir / "daily_feature_store.csv"
    if not feature_store_path.exists():
        from src.feature_store import create_daily_feature_store

        create_daily_feature_store(processed_dir=processed_dir, output_path=feature_store_path)

    daily = pd.read_csv(feature_store_path, low_memory=False)
    if "date" not in daily.columns or "revenue" not in daily.columns:
        raise ValueError("daily_feature_store.csv must contain 'date' and 'revenue'.")

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date", "revenue"]).sort_values("date").reset_index(drop=True)
    if "cogs" not in daily.columns:
        daily["cogs"] = np.nan
    return daily


def _lightgbm_feature_columns(daily_features: pd.DataFrame) -> list[str]:
    preferred = [
        "day_of_week",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "promo_active_count",
        "promo_percentage_active_count",
        "promo_fixed_active_count",
        "promo_is_active",
        "revenue_lag_1",
        "revenue_lag_7",
        "revenue_lag_14",
        "revenue_lag_28",
        "revenue_roll_mean_7",
        "revenue_roll_mean_14",
        "revenue_roll_mean_28",
        "revenue_roll_std_7",
        "revenue_roll_std_14",
        "revenue_roll_std_28",
        "cogs_lag_1",
        "cogs_lag_7",
        "cogs_lag_14",
        "cogs_lag_28",
        "cogs_roll_mean_7",
        "cogs_roll_mean_14",
        "cogs_roll_mean_28",
    ]
    return [column for column in preferred if column in daily_features.columns]


def _prepare_lightgbm_training_frame(daily_features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = _lightgbm_feature_columns(daily_features)
    if not feature_cols:
        raise ValueError("No LightGBM feature columns found in daily_feature_store.csv.")

    train_frame = daily_features[["date", "revenue", "cogs"] + feature_cols].copy()
    train_frame = train_frame.dropna(subset=["revenue"] + feature_cols).reset_index(drop=True)
    if len(train_frame) < 60:
        raise ValueError("Not enough rows in feature store after lag-drop. Need at least 60 rows.")
    return train_frame, feature_cols


def _lightgbm_param_candidates(max_trials: int) -> list[dict[str, float | int]]:
    candidates = [
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63, "subsample": 0.9, "colsample_bytree": 0.8},
        {"n_estimators": 700, "learning_rate": 0.02, "num_leaves": 63, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "subsample": 1.0, "colsample_bytree": 0.8},
        {"n_estimators": 900, "learning_rate": 0.02, "num_leaves": 127, "subsample": 0.8, "colsample_bytree": 0.9},
        {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 95, "subsample": 1.0, "colsample_bytree": 1.0},
    ]
    if max_trials < 1:
        raise ValueError("lgbm_max_trials must be >= 1.")
    return candidates[:max_trials]


def _cross_validate_lightgbm(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_splits: int,
    valid_size: int,
    min_train_size: int,
    max_trials: int,
) -> dict[str, object]:
    fold_windows = _generate_expanding_window_splits(
        train_frame,
        n_splits=n_splits,
        valid_size=valid_size,
        min_train_size=min_train_size,
    )
    trial_results: list[dict[str, object]] = []
    best_trial: dict[str, object] | None = None
    for trial_idx, params in enumerate(_lightgbm_param_candidates(max_trials), start=1):
        fold_metrics: list[dict[str, float | int]] = []
        for fold_number, (train_end, valid_start, valid_end) in enumerate(fold_windows, start=1):
            fold_train = train_frame.iloc[:train_end].copy()
            fold_valid = train_frame.iloc[valid_start:valid_end].copy()

            model = LGBMRegressor(
                objective="regression",
                random_state=42,
                **params,
            )
            model.fit(fold_train[feature_cols], fold_train["revenue"])
            preds = model.predict(fold_valid[feature_cols])
            metrics = _evaluate(fold_valid["revenue"].to_numpy(), preds)
            fold_metrics.append(
                {
                    "fold": fold_number,
                    "train_rows": int(fold_train.shape[0]),
                    "valid_rows": int(fold_valid.shape[0]),
                    **metrics,
                }
            )

        avg_metrics = {
            "mae": float(np.mean([float(f["mae"]) for f in fold_metrics])),
            "rmse": float(np.mean([float(f["rmse"]) for f in fold_metrics])),
            "r2": float(np.mean([float(f["r2"]) for f in fold_metrics])),
        }
        trial_payload: dict[str, object] = {
            "trial": trial_idx,
            "params": params,
            "avg_metrics": avg_metrics,
            "fold_metrics": fold_metrics,
        }
        trial_results.append(trial_payload)
        if best_trial is None or float(avg_metrics["mae"]) < float(best_trial["avg_metrics"]["mae"]):  # type: ignore[index]
            best_trial = trial_payload

    if best_trial is None:
        raise ValueError("LightGBM tuning produced no valid trial.")
    return {
        "n_folds": n_splits,
        "valid_size": valid_size,
        "min_train_size": min_train_size,
        "best_trial": best_trial,
        "trial_results": trial_results,
    }


def _daily_promo_features_for_dates(dates: pd.Series, promotions: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    frame["promo_active_count"] = 0
    frame["promo_percentage_active_count"] = 0
    frame["promo_fixed_active_count"] = 0
    for promo in promotions.itertuples(index=False):
        mask = (frame["date"] >= promo.start_date) & (frame["date"] <= promo.end_date)
        frame.loc[mask, "promo_active_count"] += 1
        if promo.promo_type == "percentage":
            frame.loc[mask, "promo_percentage_active_count"] += 1
        elif promo.promo_type == "fixed":
            frame.loc[mask, "promo_fixed_active_count"] += 1
    frame["promo_is_active"] = (frame["promo_active_count"] > 0).astype("int8")
    return frame


def _compute_cogs_ratio(sales_df: pd.DataFrame, train_frame: pd.DataFrame) -> float:
    if "COGS" in sales_df.columns:
        ratio = float((sales_df["COGS"] / sales_df["Revenue"]).replace([np.inf, -np.inf], np.nan).dropna().mean())
        if not np.isnan(ratio):
            return ratio
    if "cogs" in train_frame.columns:
        ratio = float((train_frame["cogs"] / train_frame["revenue"]).replace([np.inf, -np.inf], np.nan).dropna().mean())
        if not np.isnan(ratio):
            return ratio
    return 0.1


def _build_lightgbm_feature_row(
    history: pd.DataFrame,
    *,
    current_date: pd.Timestamp,
    promo_features: pd.Series,
    feature_cols: list[str],
) -> dict[str, float]:
    history = history.sort_values("date")
    latest_revenue = float(history["revenue"].iloc[-1])
    latest_cogs = float(history["cogs"].iloc[-1])

    values: dict[str, float] = {
        "day_of_week": float(current_date.dayofweek),
        "week_of_year": float(current_date.isocalendar().week),
        "month": float(current_date.month),
        "quarter": float(((current_date.month - 1) // 3) + 1),
        "year": float(current_date.year),
        "is_weekend": float(1 if current_date.dayofweek in {5, 6} else 0),
        "is_month_start": float(1 if current_date.is_month_start else 0),
        "is_month_end": float(1 if current_date.is_month_end else 0),
        "promo_active_count": float(promo_features["promo_active_count"]),
        "promo_percentage_active_count": float(promo_features["promo_percentage_active_count"]),
        "promo_fixed_active_count": float(promo_features["promo_fixed_active_count"]),
        "promo_is_active": float(promo_features["promo_is_active"]),
    }

    for lag in (1, 7, 14, 28):
        lag_date = current_date - pd.Timedelta(days=lag)
        lag_rows = history.loc[history["date"] == lag_date]
        values[f"revenue_lag_{lag}"] = float(lag_rows["revenue"].iloc[-1]) if not lag_rows.empty else latest_revenue
        values[f"cogs_lag_{lag}"] = float(lag_rows["cogs"].iloc[-1]) if not lag_rows.empty else latest_cogs

    history_before = history.loc[history["date"] < current_date]
    for window in (7, 14, 28):
        revenue_window = history_before["revenue"].tail(window)
        cogs_window = history_before["cogs"].tail(window)
        values[f"revenue_roll_mean_{window}"] = float(revenue_window.mean()) if not revenue_window.empty else latest_revenue
        values[f"revenue_roll_std_{window}"] = (
            float(revenue_window.std(ddof=0)) if len(revenue_window) > 1 else 0.0
        )
        values[f"cogs_roll_mean_{window}"] = float(cogs_window.mean()) if not cogs_window.empty else latest_cogs

    return {column: float(values.get(column, 0.0)) for column in feature_cols}


def _predict_lightgbm_submission(
    *,
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict[str, float | int],
    template_dates: pd.Series,
    promotions: pd.DataFrame,
    cogs_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = LGBMRegressor(
        objective="regression",
        random_state=42,
        **best_params,
    )
    model.fit(train_frame[feature_cols], train_frame["revenue"])

    forecast_dates = pd.to_datetime(template_dates).sort_values().reset_index(drop=True)
    promo_daily = _daily_promo_features_for_dates(forecast_dates, promotions).set_index("date")

    history = train_frame[["date", "revenue", "cogs"]].copy().sort_values("date").reset_index(drop=True)
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date in forecast_dates:
        promo_features = promo_daily.loc[date]
        row_values = _build_lightgbm_feature_row(
            history,
            current_date=date,
            promo_features=promo_features,
            feature_cols=feature_cols,
        )
        feature_vector = pd.DataFrame([row_values], columns=feature_cols)
        revenue_pred = float(np.maximum(model.predict(feature_vector)[0], 0.0))
        cogs_pred = float(np.maximum(revenue_pred * cogs_ratio, 0.0))
        rows.append({"Date": date, "Revenue": revenue_pred, "COGS": cogs_pred})
        history = pd.concat(
            [
                history,
                pd.DataFrame({"date": [date], "revenue": [revenue_pred], "cogs": [cogs_pred]}),
            ],
            ignore_index=True,
        )

    prediction_frame = pd.DataFrame(rows)
    ordered = (
        pd.DataFrame({"Date": pd.to_datetime(template_dates)})
        .merge(prediction_frame, on="Date", how="left")
        .reset_index(drop=True)
    )
    return ordered["Revenue"].to_numpy(), ordered["COGS"].to_numpy()


def train_time_series_model(
    processed_dir: Path,
    *,
    raw_dir: Path,
    output_dir: Path,
    method: str = "auto",
    cv_folds: int = 5,
    cv_valid_size: int = 30,
    cv_min_train_size: int = 180,
    cv_only: bool = False,
    lgbm_max_trials: int = 6,
) -> dict[str, object]:
    """Train baseline model(s), evaluate, and generate submission."""
    sales_df = _load_sales(processed_dir, raw_dir)
    template_df = None if cv_only else _load_submission_template(processed_dir, raw_dir)
    methods = ["linear_regression", "seasonal_naive", "lightgbm"]
    cv_results = _cross_validate_methods(
        sales_df,
        methods=["linear_regression", "seasonal_naive"],
        n_splits=cv_folds,
        valid_size=cv_valid_size,
        min_train_size=cv_min_train_size,
    )
    daily_features = _load_daily_feature_store(processed_dir)
    train_frame, lgbm_feature_cols = _prepare_lightgbm_training_frame(daily_features)
    lgbm_cv = _cross_validate_lightgbm(
        train_frame,
        lgbm_feature_cols,
        n_splits=cv_folds,
        valid_size=cv_valid_size,
        min_train_size=cv_min_train_size,
        max_trials=lgbm_max_trials,
    )
    lgbm_metrics = lgbm_cv["best_trial"]["avg_metrics"]  # type: ignore[index]
    cv_results["lightgbm"] = {
        "n_folds": lgbm_cv["n_folds"],
        "valid_size": lgbm_cv["valid_size"],
        "min_train_size": lgbm_cv["min_train_size"],
        "avg_metrics": lgbm_metrics,
        "best_trial": lgbm_cv["best_trial"],
        "trial_results": lgbm_cv["trial_results"],
    }

    linear_metrics = cv_results["linear_regression"]["avg_metrics"]
    naive_metrics = cv_results["seasonal_naive"]["avg_metrics"]
    lgbm_mae = float(lgbm_metrics["mae"])

    if method == "auto":
        best_baseline_method = (
            "linear_regression" if float(linear_metrics["mae"]) <= float(naive_metrics["mae"]) else "seasonal_naive"
        )
        best_baseline_mae = float(cv_results[best_baseline_method]["avg_metrics"]["mae"])
        selected_method = "lightgbm" if lgbm_mae <= best_baseline_mae else best_baseline_method
    else:
        selected_method = method

    if selected_method not in methods:
        raise ValueError(f"Unsupported method: {selected_method}")

    selected_metrics = cv_results[selected_method]["avg_metrics"]
    submission_path: str | None = None

    if template_df is not None:
        if selected_method == "lightgbm":
            promotions = _load_promotions(processed_dir, raw_dir)
            cogs_ratio = _compute_cogs_ratio(sales_df, train_frame)
            revenue_pred, cogs_pred = _predict_lightgbm_submission(
                train_frame=train_frame,
                feature_cols=lgbm_feature_cols,
                best_params=cv_results["lightgbm"]["best_trial"]["params"],  # type: ignore[index]
                template_dates=template_df["Date"],
                promotions=promotions,
                cogs_ratio=cogs_ratio,
            )
        else:
            revenue_pred = _predict_with_method(sales_df, template_df["Date"], selected_method)
            cogs_pred = np.array([])
        submission = template_df.copy()
        submission["Revenue"] = np.maximum(revenue_pred, 0.0)
        if selected_method == "lightgbm":
            submission["COGS"] = cogs_pred
        elif "COGS" not in submission.columns:
            if "COGS" in sales_df.columns:
                ratio = float(
                    (sales_df["COGS"] / sales_df["Revenue"]).replace([np.inf, -np.inf], np.nan).dropna().mean()
                )
                if np.isnan(ratio):
                    ratio = 0.1
            else:
                ratio = 0.1
            submission["COGS"] = submission["Revenue"] * ratio

        output_dir.mkdir(parents=True, exist_ok=True)
        submission_file = output_dir / "submission.csv"
        submission.to_csv(submission_file, index=False)
        submission_path = str(submission_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "baseline_metrics.json"

    payload = {
        "selected_method": selected_method,
        "selected_metrics": selected_metrics,
        "linear_regression_metrics": linear_metrics,
        "seasonal_naive_metrics": naive_metrics,
        "lightgbm_metrics": lgbm_metrics,
        "cv": {
            "strategy": "expanding_window",
            "config": {
                "folds": cv_folds,
                "valid_size": cv_valid_size,
                "min_train_size": cv_min_train_size,
                "lgbm_max_trials": lgbm_max_trials,
            },
            "results": cv_results,
        },
        "submission_path": submission_path,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Datathon baseline forecast and generate submission.csv")
    parser.add_argument("--processed-dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--raw-dir", type=Path, default=Path("data") / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "processed")
    parser.add_argument(
        "--method",
        choices=["auto", "linear_regression", "seasonal_naive", "lightgbm"],
        default="auto",
        help="Forecasting baseline method.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of expanding-window CV folds.",
    )
    parser.add_argument(
        "--cv-valid-size",
        type=int,
        default=30,
        help="Validation horizon (rows) per CV fold.",
    )
    parser.add_argument(
        "--cv-min-train-size",
        type=int,
        default=180,
        help="Minimum training rows used for the first CV fold.",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Run CV evaluation only (skip submission.csv generation).",
    )
    parser.add_argument(
        "--lgbm-max-trials",
        type=int,
        default=6,
        help="Maximum LightGBM hyperparameter trials for CV tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = train_time_series_model(
        args.processed_dir,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        method=args.method,
        cv_folds=args.cv_folds,
        cv_valid_size=args.cv_valid_size,
        cv_min_train_size=args.cv_min_train_size,
        cv_only=args.cv_only,
        lgbm_max_trials=args.lgbm_max_trials,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
