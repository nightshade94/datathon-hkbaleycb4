"""Data preparation utilities for cleaning and feature-readiness."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR
from src.csv_io import read_csv_directory, write_csv_directory

PROMO_MISSING_TOKEN = "NO_PROMO"
PROMO_ALL_CATEGORY_TOKEN = "ALL"

DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "customers": ("signup_date",),
    "inventory": ("snapshot_date",),
    "orders": ("order_date",),
    "promotions": ("start_date", "end_date"),
    "returns": ("return_date",),
    "reviews": ("review_date",),
    "sales": ("Date",),
    "sample_submission": ("Date",),
    "shipments": ("ship_date", "delivery_date"),
    "web_traffic": ("date",),
}

CATEGORICAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "customers": ("city", "gender", "age_group", "acquisition_channel"),
    "geography": ("city", "region", "district"),
    "orders": ("order_status", "payment_method", "device_type", "order_source"),
    "payments": ("payment_method",),
    "products": ("category", "segment", "size", "color"),
    "promotions": ("promo_type", "applicable_category", "promo_channel"),
    "returns": ("return_reason",),
    "reviews": ("review_title",),
    "web_traffic": ("traffic_source",),
}

REQUIRED_TABLES = {
    "customers",
    "geography",
    "inventory",
    "order_items",
    "orders",
    "payments",
    "products",
    "promotions",
    "returns",
    "reviews",
    "sales",
    "sample_submission",
    "shipments",
    "web_traffic",
}


def _clear_processed_csv_outputs(processed_dir: Path) -> None:
    """Remove stale CSV outputs before writing a fresh processed snapshot."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    for csv_file in processed_dir.glob("*.csv"):
        csv_file.unlink()


def _parse_date_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Parse known date columns with coercion."""
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def _cast_categorical_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Cast selected columns to categorical dtype when present."""
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].astype("category")
    return out


def _normalize_promo_code(value: object) -> str:
    """Map empty-like promo values to a single token."""
    if pd.isna(value):
        return PROMO_MISSING_TOKEN
    text = str(value).strip()
    if not text:
        return PROMO_MISSING_TOKEN
    return text


def _collapse_promo_codes(series: pd.Series) -> str:
    """Collapse multiple promo values from duplicate line-items."""
    values = sorted(
        {
            _normalize_promo_code(value)
            for value in series
            if _normalize_promo_code(value) != PROMO_MISSING_TOKEN
        }
    )
    if not values:
        return PROMO_MISSING_TOKEN
    return "|".join(values)


def _clean_order_items(order_items: pd.DataFrame) -> pd.DataFrame:
    """Clean and aggregate line-items at (order_id, product_id) level."""
    out = order_items.copy()

    out["promo_id"] = out["promo_id"].map(_normalize_promo_code)
    out["promo_id_2"] = out["promo_id_2"].map(_normalize_promo_code)
    out["line_gross"] = out["quantity"] * out["unit_price"]

    aggregated = (
        out.groupby(["order_id", "product_id"], as_index=False, sort=False)
        .agg(
            quantity=("quantity", "sum"),
            line_gross=("line_gross", "sum"),
            discount_amount=("discount_amount", "sum"),
            promo_id=("promo_id", _collapse_promo_codes),
            promo_id_2=("promo_id_2", _collapse_promo_codes),
        )
        .reset_index(drop=True)
    )

    aggregated["unit_price"] = aggregated["line_gross"] / aggregated["quantity"]
    aggregated["line_net"] = aggregated["line_gross"] - aggregated["discount_amount"]
    aggregated["has_promo_1"] = (aggregated["promo_id"] != PROMO_MISSING_TOKEN).astype("int8")
    aggregated["has_promo_2"] = (aggregated["promo_id_2"] != PROMO_MISSING_TOKEN).astype("int8")

    return aggregated[
        [
            "order_id",
            "product_id",
            "quantity",
            "unit_price",
            "discount_amount",
            "promo_id",
            "promo_id_2",
            "has_promo_1",
            "has_promo_2",
            "line_gross",
            "line_net",
        ]
    ]


def _clean_promotions(promotions: pd.DataFrame) -> pd.DataFrame:
    """Fill nullable business scope fields in promotions."""
    out = promotions.copy()
    out["applicable_category"] = (
        out["applicable_category"]
        .fillna(PROMO_ALL_CATEGORY_TOKEN)
        .astype(str)
        .str.strip()
        .replace("", PROMO_ALL_CATEGORY_TOKEN)
    )
    return out


def _impute_missing_shipments(
    orders: pd.DataFrame,
    shipments: pd.DataFrame,
) -> pd.DataFrame:
    """Impute shipment rows for shipped/delivered/returned orders that have none."""
    out = shipments.copy()
    order_view = orders[["order_id", "order_date", "order_status"]].copy()
    required_statuses = {"shipped", "delivered", "returned"}

    present_shipment_ids = set(out["order_id"].dropna().astype(int).tolist())
    candidates = order_view[
        order_view["order_status"].isin(required_statuses)
        & ~order_view["order_id"].isin(present_shipment_ids)
    ].copy()
    if candidates.empty:
        return out

    with_lag = order_view.merge(out, on="order_id", how="inner")
    with_lag["ship_lag_days"] = (with_lag["ship_date"] - with_lag["order_date"]).dt.days
    with_lag["delivery_lag_days"] = (with_lag["delivery_date"] - with_lag["ship_date"]).dt.days

    valid_ship_lag = with_lag.loc[with_lag["ship_lag_days"] >= 0, "ship_lag_days"]
    valid_delivery_lag = with_lag.loc[with_lag["delivery_lag_days"] >= 0, "delivery_lag_days"]

    ship_lag_days = int(valid_ship_lag.median()) if not valid_ship_lag.empty else 1
    delivery_lag_days = int(valid_delivery_lag.median()) if not valid_delivery_lag.empty else 4
    shipping_fee = float(out["shipping_fee"].median()) if not out.empty else 0.0

    candidates["ship_date"] = candidates["order_date"] + pd.to_timedelta(ship_lag_days, unit="D")
    candidates["delivery_date"] = candidates["ship_date"] + pd.to_timedelta(delivery_lag_days, unit="D")
    candidates["shipping_fee"] = round(shipping_fee, 2)

    additions = candidates[["order_id", "ship_date", "delivery_date", "shipping_fee"]]
    return (
        pd.concat([out, additions], ignore_index=True)
        .sort_values("order_id", kind="stable")
        .drop_duplicates(subset=["order_id"], keep="first")
        .reset_index(drop=True)
    )


def _impute_missing_return_details(
    orders: pd.DataFrame,
    returns: pd.DataFrame,
    order_items: pd.DataFrame,
) -> pd.DataFrame:
    """Impute minimal return records for orders marked as returned without details."""
    out = returns.copy()

    returned_orders = orders.loc[orders["order_status"] == "returned", ["order_id", "order_date"]].copy()
    existing_return_order_ids = set(out["order_id"].dropna().astype(int).tolist())
    missing = returned_orders[~returned_orders["order_id"].isin(existing_return_order_ids)].copy()
    if missing.empty:
        return out

    lag_view = out.merge(orders[["order_id", "order_date"]], on="order_id", how="inner")
    lag_days = (lag_view["return_date"] - lag_view["order_date"]).dt.days
    lag_days = lag_days[lag_days >= 0]
    median_return_lag = int(lag_days.median()) if not lag_days.empty else 14

    line_items = order_items.sort_values(["order_id", "product_id"], kind="stable")
    first_item = line_items.drop_duplicates(subset=["order_id"], keep="first").copy()
    missing = missing.merge(
        first_item[["order_id", "product_id", "quantity", "line_net"]],
        on="order_id",
        how="inner",
    )
    if missing.empty:
        return out

    numeric_suffix = (
        out["return_id"]
        .astype(str)
        .str.extract(r"(\d+)$")[0]
        .dropna()
        .astype(int)
    )
    next_id = int(numeric_suffix.max()) + 1 if not numeric_suffix.empty else 1

    unit_refund = (missing["line_net"] / missing["quantity"]).where(missing["quantity"] > 0, missing["line_net"])
    unit_refund = unit_refund.clip(lower=0)

    additions = pd.DataFrame(
        {
            "return_id": [f"RET-{next_id + idx:06d}" for idx in range(len(missing))],
            "order_id": missing["order_id"].to_numpy(),
            "product_id": missing["product_id"].to_numpy(),
            "return_date": missing["order_date"] + pd.to_timedelta(median_return_lag, unit="D"),
            "return_reason": "unknown",
            "return_quantity": 1,
            "refund_amount": unit_refund.round(2).to_numpy(),
        }
    )

    return (
        pd.concat([out, additions], ignore_index=True)
        .sort_values("return_id", kind="stable")
        .drop_duplicates(subset=["return_id"], keep="first")
        .reset_index(drop=True)
    )


def _extend_web_traffic_coverage(
    web_traffic: pd.DataFrame,
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """Backfill missing early web-traffic dates to align with sales range."""
    out = web_traffic.copy()
    out["imputed_flag"] = 0

    sales_start = sales["Date"].min()
    web_start = out["date"].min()
    if pd.isna(sales_start) or pd.isna(web_start) or web_start <= sales_start:
        return out

    missing_dates = pd.date_range(sales_start, web_start - pd.Timedelta(days=1), freq="D")
    if len(missing_dates) == 0:
        return out

    numeric_cols = [
        "sessions",
        "unique_visitors",
        "page_views",
        "bounce_rate",
        "avg_session_duration_sec",
    ]
    observed = out.copy()
    observed["dow"] = observed["date"].dt.dayofweek

    dow_medians = observed.groupby("dow")[numeric_cols].median()
    global_medians = observed[numeric_cols].median()

    source_mode_by_dow = observed.groupby("dow")["traffic_source"].agg(
        lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]
    )
    global_source_mode = (
        observed["traffic_source"].mode().iloc[0]
        if not observed["traffic_source"].mode().empty
        else "unknown"
    )

    imputed_rows: list[dict[str, object]] = []
    for date in missing_dates:
        dow = int(date.dayofweek)
        row: dict[str, object] = {"date": date, "imputed_flag": 1}
        for column in numeric_cols:
            if dow in dow_medians.index and not pd.isna(dow_medians.loc[dow, column]):
                value = dow_medians.loc[dow, column]
            else:
                value = global_medians[column]
            row[column] = float(value)
        row["traffic_source"] = source_mode_by_dow.get(dow, global_source_mode)
        imputed_rows.append(row)

    imputed = pd.DataFrame(imputed_rows)
    for column in ["sessions", "unique_visitors", "page_views"]:
        imputed[column] = imputed[column].round().astype(int)

    return (
        pd.concat([out, imputed], ignore_index=True)
        .sort_values("date", kind="stable")
        .drop_duplicates(subset=["date"], keep="first")
        .reset_index(drop=True)
    )


def _add_order_quality_flags(
    orders: pd.DataFrame,
    shipments: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Create order-level quality flags tied to shipment/return consistency."""
    out = orders.copy()

    shipment_order_ids = set(shipments["order_id"].dropna().astype(int).tolist())
    return_order_ids = set(returns["order_id"].dropna().astype(int).tolist())

    missing_shipment = ~out["order_id"].isin(shipment_order_ids)
    out["missing_shipment_flag"] = missing_shipment.astype("int8")
    out["delivered_without_shipment_flag"] = (
        (out["order_status"] == "delivered") & missing_shipment
    ).astype("int8")
    out["returned_without_return_detail_flag"] = (
        (out["order_status"] == "returned") & ~out["order_id"].isin(return_order_ids)
    ).astype("int8")
    out["has_return_detail_flag"] = out["order_id"].isin(return_order_ids).astype("int8")

    return out


def _validate_required_tables(raw_tables: dict[str, pd.DataFrame]) -> None:
    """Ensure all required source tables exist before processing."""
    missing = sorted(REQUIRED_TABLES - set(raw_tables.keys()))
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing required raw tables: {missing_str}")


def clean_all_data(raw_dir: Path, processed_dir: Path) -> None:
    """Load, clean, and persist all raw datasets.

    The cleaning pass applies:
    1. Date parsing and categorical casting by table schema.
    2. Missing-value imputation for promo/category business fields.
    3. Aggregation of duplicated line-items in ``order_items``.
    4. Order-level consistency flags for shipment/return anomalies.
    """
    raw_tables = read_csv_directory(raw_dir)
    _validate_required_tables(raw_tables)

    cleaned_tables: dict[str, pd.DataFrame] = {}
    for table_name, raw_table in raw_tables.items():
        table = raw_table.copy()
        table = _parse_date_columns(table, DATE_COLUMNS.get(table_name, ()))

        if table_name == "order_items":
            table = _clean_order_items(table)
        elif table_name == "promotions":
            table = _clean_promotions(table)

        table = _cast_categorical_columns(table, CATEGORICAL_COLUMNS.get(table_name, ()))
        cleaned_tables[table_name] = table.drop_duplicates(ignore_index=True)

    cleaned_tables["shipments"] = _impute_missing_shipments(
        orders=cleaned_tables["orders"],
        shipments=cleaned_tables["shipments"],
    )
    cleaned_tables["returns"] = _impute_missing_return_details(
        orders=cleaned_tables["orders"],
        returns=cleaned_tables["returns"],
        order_items=cleaned_tables["order_items"],
    )
    cleaned_tables["web_traffic"] = _extend_web_traffic_coverage(
        web_traffic=cleaned_tables["web_traffic"],
        sales=cleaned_tables["sales"],
    )
    cleaned_tables["orders"] = _add_order_quality_flags(
        orders=cleaned_tables["orders"],
        shipments=cleaned_tables["shipments"],
        returns=cleaned_tables["returns"],
    )

    _clear_processed_csv_outputs(processed_dir)
    write_csv_directory(
        cleaned_tables,
        processed_dir,
        index=False,
        date_format="%Y-%m-%d",
    )


def _run_default_pipeline() -> None:
    """Execute the default raw -> processed cleaning pipeline."""
    clean_all_data(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)

    # Imported lazily to avoid a hard dependency for callers that only clean.
    from src.feature_store import create_daily_feature_store

    create_daily_feature_store(processed_dir=PROCESSED_DIR)


if __name__ == "__main__":
    _run_default_pipeline()
