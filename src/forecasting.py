"""Baseline forecasting script with leak-safe time-series baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    if "COGS" in sales_df.columns:
        sales_df["COGS"] = pd.to_numeric(sales_df["COGS"], errors="coerce")
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
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
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


def _fit_linear_regression(
    train_df: pd.DataFrame,
    predict_dates: pd.Series,
    *,
    target: str,
) -> np.ndarray:
    origin = train_df["Date"].min()
    x_train = _date_features(train_df["Date"], origin)
    y_train = train_df[target].to_numpy()

    model = LinearRegression()
    model.fit(x_train, y_train)

    x_predict = _date_features(predict_dates, origin)
    return model.predict(x_predict)


def _seasonal_naive_predict(
    history_df: pd.DataFrame,
    predict_dates: pd.Series,
    *,
    target: str,
) -> np.ndarray:
    known = {
        pd.Timestamp(row.Date): float(getattr(row, target))
        for row in history_df[["Date", target]].itertuples(index=False)
    }
    fallback = float(history_df[target].iloc[-1])
    predictions_by_date: dict[pd.Timestamp, float] = {}

    sorted_dates = pd.Series(pd.to_datetime(predict_dates)).sort_values()
    for date in sorted_dates:
        date = pd.Timestamp(date)
        lag_date = date - pd.Timedelta(days=7)
        if lag_date in known:
            pred = known[lag_date]
        else:
            past_dates = [d for d in known if d < date]
            pred = known[max(past_dates)] if past_dates else fallback
        predictions_by_date[date] = float(pred)
        known[date] = float(pred)

    return pd.Series(pd.to_datetime(predict_dates)).map(predictions_by_date).to_numpy(dtype=float)


def _annual_growth_factor(history_df: pd.DataFrame, target: str) -> tuple[int, float, float]:
    view = history_df[["Date", target]].copy()
    view["year"] = view["Date"].dt.year
    yearly = view.groupby("year", observed=True).agg(value=(target, "mean"), days=(target, "size"))
    complete_years = yearly[yearly["days"] >= 360]
    if complete_years.empty:
        complete_years = yearly

    last_year = int(complete_years.index.max())
    base_level = float(complete_years.loc[last_year, "value"])
    pct_change = complete_years["value"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if pct_change.empty:
        growth = 1.0
    else:
        growth = float(np.prod(1.0 + pct_change.clip(lower=-0.95)) ** (1.0 / len(pct_change)))
    return last_year, base_level, growth


def _seasonal_profile_predict(
    history_df: pd.DataFrame,
    predict_dates: pd.Series,
    *,
    target: str,
) -> np.ndarray:
    history = history_df[["Date", target]].dropna().copy()
    history["year"] = history["Date"].dt.year
    history["month"] = history["Date"].dt.month
    history["day"] = history["Date"].dt.day
    annual_mean = history.groupby("year", observed=True)[target].transform("mean")
    history["target_norm"] = history[target] / annual_mean.replace(0, np.nan)

    seasonal = (
        history.groupby(["month", "day"], observed=True)["target_norm"]
        .mean()
        .replace([np.inf, -np.inf], np.nan)
    )
    last_year, base_level, growth = _annual_growth_factor(history, target)

    predict = pd.DataFrame({"Date": pd.to_datetime(predict_dates)})
    predict["month"] = predict["Date"].dt.month
    predict["day"] = predict["Date"].dt.day
    predict["year"] = predict["Date"].dt.year
    joined = predict.join(seasonal.rename("seasonal_factor"), on=["month", "day"])
    joined["seasonal_factor"] = joined["seasonal_factor"].fillna(1.0)
    joined["years_ahead"] = joined["year"] - last_year
    return (base_level * np.power(growth, joined["years_ahead"]) * joined["seasonal_factor"]).to_numpy(dtype=float)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except ValueError:
        r2 = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _train_validation_split(sales_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = len(sales_df)
    if n_rows < 60:
        raise ValueError("Not enough rows in sales.csv. Need at least 60 rows for baseline split.")
    valid_size = max(30, int(n_rows * 0.2))
    split_idx = n_rows - valid_size
    return sales_df.iloc[:split_idx].copy(), sales_df.iloc[split_idx:].copy()


def train_time_series_model(
    processed_dir: Path,
    *,
    raw_dir: Path,
    output_dir: Path,
    method: str = "auto",
) -> dict[str, str | float]:
    """Train baseline model(s), evaluate, and generate submission."""
    sales_df = _load_sales(processed_dir, raw_dir)
    template_df = _load_submission_template(processed_dir, raw_dir)
    has_cogs = "COGS" in sales_df.columns

    train_df, valid_df = _train_validation_split(sales_df)
    y_valid = valid_df["Revenue"].to_numpy()

    linear_valid = _fit_linear_regression(train_df, valid_df["Date"], target="Revenue")
    naive_valid = _seasonal_naive_predict(train_df, valid_df["Date"], target="Revenue")
    seasonal_profile_valid = _seasonal_profile_predict(train_df, valid_df["Date"], target="Revenue")

    method_metrics = {
        "linear_regression": _evaluate(y_valid, linear_valid),
        "seasonal_naive": _evaluate(y_valid, naive_valid),
        "seasonal_profile": _evaluate(y_valid, seasonal_profile_valid),
    }

    if method == "auto":
        selected_method = min(method_metrics, key=lambda key: method_metrics[key]["mae"])
    else:
        selected_method = method

    if selected_method == "linear_regression":
        revenue_pred = _fit_linear_regression(sales_df, template_df["Date"], target="Revenue")
        cogs_pred = (
            _fit_linear_regression(sales_df, template_df["Date"], target="COGS")
            if has_cogs
            else None
        )
    elif selected_method == "seasonal_naive":
        revenue_pred = _seasonal_naive_predict(sales_df, template_df["Date"], target="Revenue")
        cogs_pred = (
            _seasonal_naive_predict(sales_df, template_df["Date"], target="COGS")
            if has_cogs
            else None
        )
    elif selected_method == "seasonal_profile":
        revenue_pred = _seasonal_profile_predict(sales_df, template_df["Date"], target="Revenue")
        cogs_pred = (
            _seasonal_profile_predict(sales_df, template_df["Date"], target="COGS")
            if has_cogs
            else None
        )
    else:
        raise ValueError(f"Unsupported method: {selected_method}")

    selected_metrics = method_metrics[selected_method]
    submission = template_df[["Date"]].copy()
    submission["Revenue"] = np.maximum(revenue_pred, 0.0)
    if cogs_pred is None:
        ratio = 0.1
        if has_cogs:
            ratio = float(
                (sales_df["COGS"] / sales_df["Revenue"])
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .mean()
            )
            if np.isnan(ratio):
                ratio = 0.1
        cogs_pred = submission["Revenue"].to_numpy() * ratio
    submission["COGS"] = np.maximum(cogs_pred, 0.0)
    submission["COGS"] = np.minimum(submission["COGS"], submission["Revenue"] * 0.99)
    submission["Date"] = pd.to_datetime(submission["Date"]).dt.strftime("%Y-%m-%d")
    submission = submission[["Date", "Revenue", "COGS"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    metrics_path = output_dir / "baseline_metrics.json"
    submission.to_csv(submission_path, index=False)

    payload = {
        "selected_method": selected_method,
        "selected_metrics": selected_metrics,
        "method_metrics": method_metrics,
        "submission_path": str(submission_path),
        "submission_rows": int(len(submission)),
        "submission_columns": list(submission.columns),
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
        choices=["auto", "linear_regression", "seasonal_naive", "seasonal_profile"],
        default="auto",
        help="Forecasting baseline method.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = train_time_series_model(
        args.processed_dir,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        method=args.method,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
