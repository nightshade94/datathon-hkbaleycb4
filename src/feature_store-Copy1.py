"""Feature-store construction for daily time-series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.csv_io import read_csv_directory, write_csv_file

DEFAULT_LAGS = (1, 7, 14, 28)
DEFAULT_WINDOWS = (7, 14, 28)


def _resolve_table(tables: dict[str, pd.DataFrame], *candidates: str) -> pd.DataFrame:
    """Resolve table by trying multiple stem candidates."""
    for candidate in candidates:
        if candidate in tables:
            return tables[candidate].copy()
    joined = ", ".join(candidates)
    raise KeyError(f"None of the requested tables exist: {joined}")


def _normalize_column_name(name: str) -> str:
    """Normalize heterogeneous column naming to a canonical snake_case field."""
    compact = name.strip().lower().replace("_", "")

    aliases = {
        "date": "date",
        "revenue": "revenue",
        "cogs": "cogs",
        "orderid": "order_id",
        "orderdate": "order_date",
        "customerid": "customer_id",
        "orderstatus": "order_status",
        "paymentvalue": "payment_value",
        "installments": "installments",
        "productid": "product_id",
        "quantity": "quantity",
        "unitprice": "unit_price",
        "discountamount": "discount_amount",
        "linegross": "line_gross",
        "linenet": "line_net",
        "haspromo1": "has_promo_1",
        "haspromo2": "has_promo_2",
        "promoid": "promo_id",
        "promoid2": "promo_id_2",
        "shipdate": "ship_date",
        "deliverydate": "delivery_date",
        "shippingfee": "shipping_fee",
        "returnid": "return_id",
        "returndate": "return_date",
        "refundamount": "refund_amount",
        "reviewid": "review_id",
        "reviewdate": "review_date",
        "rating": "rating",
        "sessions": "sessions",
        "uniquevisitors": "unique_visitors",
        "pageviews": "page_views",
        "bouncerate": "bounce_rate",
        "avgsessiondurationsec": "avg_session_duration_sec",
        "trafficsource": "traffic_source",
        "startdate": "start_date",
        "enddate": "end_date",
        "promotype": "promo_type",
        "missingshipmentflag": "missing_shipment_flag",
    }
    return aliases.get(compact, name.strip().lower())


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame columns to canonical names."""
    out = df.copy()
    out.columns = [_normalize_column_name(column) for column in out.columns]
    return out


def _add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add leak-safe calendar features."""
    out = daily.copy()
    out["day_of_week"] = out["date"].dt.dayofweek
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype("int16")
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["year"] = out["date"].dt.year
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype("int8")
    out["is_month_start"] = out["date"].dt.is_month_start.astype("int8")
    out["is_month_end"] = out["date"].dt.is_month_end.astype("int8")
    return out


def _add_lagged_features(
    daily: pd.DataFrame,
    columns: Iterable[str],
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Create lag and rolling features using shifted history to avoid leakage."""
    out = daily.copy()
    new_features: dict[str, pd.Series] = {}
    for column in columns:
        if column not in out.columns:
            continue

        for lag in lags:
            new_features[f"{column}_lag_{lag}"] = out[column].shift(lag)

        shifted = out[column].shift(1)
        for window in windows:
            new_features[f"{column}_roll_mean_{window}"] = shifted.rolling(window).mean()
            new_features[f"{column}_roll_std_{window}"] = shifted.rolling(window).std()

    if not new_features:
        return out

    lagged_frame = pd.DataFrame(new_features, index=out.index)
    return pd.concat([out, lagged_frame], axis=1)


def _build_promo_daily_features(
    daily_index: pd.Series,
    promotions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute daily counts of active promotions."""
    promo_daily = pd.DataFrame({"date": daily_index})
    promo_daily["promo_active_count"] = 0
    promo_daily["promo_percentage_active_count"] = 0
    promo_daily["promo_fixed_active_count"] = 0

    for row in promotions.itertuples(index=False):
        if pd.isna(row.start_date) or pd.isna(row.end_date):
            continue

        mask = (promo_daily["date"] >= row.start_date) & (promo_daily["date"] <= row.end_date)
        promo_daily.loc[mask, "promo_active_count"] += 1
        if row.promo_type == "percentage":
            promo_daily.loc[mask, "promo_percentage_active_count"] += 1
        elif row.promo_type == "fixed":
            promo_daily.loc[mask, "promo_fixed_active_count"] += 1

    return promo_daily


def create_daily_feature_store(
    processed_dir: Path,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Create a daily feature table for Revenue/COGS forecasting.

    The resulting table is leak-safe by construction:
    raw same-day explanatory signals are retained, but lag/rolling features
    are always computed from shifted history.
    """
    tables = read_csv_directory(processed_dir)

    sales = _normalize_columns(_resolve_table(tables, "sales"))
    orders = _normalize_columns(_resolve_table(tables, "orders"))
    payments = _normalize_columns(_resolve_table(tables, "payments"))
    shipments = _normalize_columns(_resolve_table(tables, "shipments"))
    order_items = _normalize_columns(_resolve_table(tables, "order_items", "orderitems"))
    returns = _normalize_columns(_resolve_table(tables, "returns"))
    reviews = _normalize_columns(_resolve_table(tables, "reviews"))
    promotions = _normalize_columns(_resolve_table(tables, "promotions"))
    web_traffic = _normalize_columns(_resolve_table(tables, "web_traffic", "webtraffic"))

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    shipments["ship_date"] = pd.to_datetime(shipments["ship_date"], errors="coerce")
    shipments["delivery_date"] = pd.to_datetime(shipments["delivery_date"], errors="coerce")
    returns["return_date"] = pd.to_datetime(returns["return_date"], errors="coerce")
    reviews["review_date"] = pd.to_datetime(reviews["review_date"], errors="coerce")
    promotions["start_date"] = pd.to_datetime(promotions["start_date"], errors="coerce")
    promotions["end_date"] = pd.to_datetime(promotions["end_date"], errors="coerce")
    web_traffic["date"] = pd.to_datetime(web_traffic["date"], errors="coerce")

    daily = sales[["date", "revenue", "cogs"]].sort_values("date").reset_index(drop=True)

    # --- Orders
    if "missing_shipment_flag" not in orders.columns:
        shipment_order_ids = set(shipments["order_id"].dropna().astype(int).tolist())
        orders["missing_shipment_flag"] = (~orders["order_id"].isin(shipment_order_ids)).astype("int8")

    order_daily = (
        orders.groupby("order_date", as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            unique_customers=("customer_id", "nunique"),
            delivered_orders=("order_status", lambda s: (s == "delivered").sum()),
            returned_orders=("order_status", lambda s: (s == "returned").sum()),
            cancelled_orders=("order_status", lambda s: (s == "cancelled").sum()),
            missing_shipment_orders=("missing_shipment_flag", "sum"),
        )
        .rename(columns={"order_date": "date"})
    )

    # --- Payments by order date
    payment_daily = (
        orders[["order_id", "order_date"]]
        .merge(payments[["order_id", "payment_value", "installments"]], on="order_id", how="left")
        .groupby("order_date", as_index=False)
        .agg(
            payment_value_total=("payment_value", "sum"),
            installments_mean=("installments", "mean"),
        )
        .rename(columns={"order_date": "date"})
    )

    # --- Order-items by order date
    if "line_gross" not in order_items.columns:
        order_items["line_gross"] = order_items["quantity"] * order_items["unit_price"]
    if "line_net" not in order_items.columns:
        order_items["line_net"] = order_items["line_gross"] - order_items["discount_amount"]
    if "has_promo_1" not in order_items.columns:
        order_items["has_promo_1"] = (
            order_items["promo_id"].fillna("NO_PROMO").astype(str).str.upper() != "NO_PROMO"
        ).astype("int8")
    if "has_promo_2" not in order_items.columns:
        order_items["has_promo_2"] = (
            order_items["promo_id_2"].fillna("NO_PROMO").astype(str).str.upper() != "NO_PROMO"
        ).astype("int8")

    line_daily = (
        orders[["order_id", "order_date"]]
        .merge(
            order_items[
                [
                    "order_id",
                    "quantity",
                    "line_gross",
                    "line_net",
                    "discount_amount",
                    "has_promo_1",
                    "has_promo_2",
                ]
            ],
            on="order_id",
            how="left",
        )
        .groupby("order_date", as_index=False)
        .agg(
            quantity_total=("quantity", "sum"),
            line_gross_total=("line_gross", "sum"),
            line_net_total=("line_net", "sum"),
            discount_total=("discount_amount", "sum"),
            promo_1_line_count=("has_promo_1", "sum"),
            promo_2_line_count=("has_promo_2", "sum"),
        )
        .rename(columns={"order_date": "date"})
    )

    # --- Shipping metrics by order date
    ship_daily = orders[["order_id", "order_date"]].merge(shipments, on="order_id", how="left")
    ship_daily["ship_lag_days"] = (ship_daily["ship_date"] - ship_daily["order_date"]).dt.days
    ship_daily["delivery_lag_days"] = (ship_daily["delivery_date"] - ship_daily["ship_date"]).dt.days

    shipping_daily = (
        ship_daily.groupby("order_date", as_index=False)
        .agg(
            shipped_order_count=("ship_date", lambda s: s.notna().sum()),
            shipping_fee_total=("shipping_fee", "sum"),
            ship_lag_days_mean=("ship_lag_days", "mean"),
            delivery_lag_days_mean=("delivery_lag_days", "mean"),
        )
        .rename(columns={"order_date": "date"})
    )

    # --- Returns by return date
    returns_daily = (
        returns.groupby("return_date", as_index=False)
        .agg(
            return_line_count=("return_id", "count"),
            refund_amount_total=("refund_amount", "sum"),
        )
        .rename(columns={"return_date": "date"})
    )

    # --- Reviews by review date
    reviews_daily = (
        reviews.groupby("review_date", as_index=False)
        .agg(
            review_count=("review_id", "count"),
            avg_rating=("rating", "mean"),
        )
        .rename(columns={"review_date": "date"})
    )

    # --- Promotion activity
    promo_daily = _build_promo_daily_features(daily["date"], promotions)

    # --- Merge all daily features
    daily = (
        daily.merge(order_daily, on="date", how="left")
        .merge(payment_daily, on="date", how="left")
        .merge(line_daily, on="date", how="left")
        .merge(shipping_daily, on="date", how="left")
        .merge(returns_daily, on="date", how="left")
        .merge(reviews_daily, on="date", how="left")
        .merge(web_traffic, on="date", how="left")
        .merge(promo_daily, on="date", how="left")
    )

    for column in daily.columns:
        if column == "date":
            continue
        if daily[column].dtype.kind in {"i", "u", "f"}:
            daily[column] = daily[column].fillna(0)

    daily = _add_calendar_features(daily)

    lag_candidates = [
        "revenue",
        "cogs",
        "order_count",
        "line_net_total",
        "payment_value_total",
        "sessions",
        "page_views",
        "discount_total",
        "shipping_fee_total",
        "refund_amount_total",
        "review_count",
        "avg_rating",
        "promo_active_count",
    ]
    daily = _add_lagged_features(daily, lag_candidates)

    # Keep only leak-safe same-day fields for forecasting.
    leak_prone_same_day = [
        "order_count",
        "unique_customers",
        "delivered_orders",
        "returned_orders",
        "cancelled_orders",
        "missing_shipment_orders",
        "payment_value_total",
        "installments_mean",
        "quantity_total",
        "line_gross_total",
        "line_net_total",
        "discount_total",
        "promo_1_line_count",
        "promo_2_line_count",
        "shipped_order_count",
        "shipping_fee_total",
        "ship_lag_days_mean",
        "delivery_lag_days_mean",
        "return_line_count",
        "refund_amount_total",
        "review_count",
        "avg_rating",
        "sessions",
        "unique_visitors",
        "page_views",
        "bounce_rate",
        "avg_session_duration_sec",
        "traffic_source",
    ]
    daily = daily.drop(columns=[col for col in leak_prone_same_day if col in daily.columns], errors="ignore")

    if output_path is None:
        output_path = processed_dir / "daily_feature_store.csv"
    write_csv_file(daily, output_path, index=False, date_format="%Y-%m-%d")
    return daily

