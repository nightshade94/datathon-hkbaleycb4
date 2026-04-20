"""Data preparation utilities."""

from pathlib import Path

import pandas as pd


FILE_ALIASES: dict[str, list[str]] = {
    "customers": ["customers.csv"],
    "geography": ["geography.csv"],
    "inventory": ["inventory.csv"],
    "orderitems": ["orderitems.csv", "order_items.csv"],
    "orders": ["orders.csv"],
    "payments": ["payments.csv"],
    "products": ["products.csv"],
    "promotions": ["promotions.csv"],
    "returns": ["returns.csv"],
    "reviews": ["reviews.csv"],
    "sales": ["sales.csv"],
    "samplesubmission": ["samplesubmission.csv", "sample_submission.csv"],
    "shipments": ["shipments.csv"],
    "webtraffic": ["webtraffic.csv", "web_traffic.csv"],
}

COLUMN_RENAME_MAPS: dict[str, dict[str, str]] = {
    "customers": {
        "customer_id": "customerid",
        "signup_date": "signupdate",
        "age_group": "agegroup",
        "acquisition_channel": "acquisitionchannel",
    },
    "orders": {
        "order_id": "orderid",
        "order_date": "orderdate",
        "customer_id": "customerid",
        "order_status": "orderstatus",
        "payment_method": "paymentmethod",
        "device_type": "devicetype",
        "order_source": "ordersource",
    },
    "orderitems": {
        "order_id": "orderid",
        "product_id": "productid",
        "unit_price": "unitprice",
        "discount_amount": "discountamount",
        "promo_id": "promoid",
        "promo_id_2": "promoid2",
    },
    "payments": {
        "order_id": "orderid",
        "payment_method": "paymentmethod",
        "payment_value": "paymentvalue",
    },
    "products": {
        "product_id": "productid",
        "product_name": "productname",
    },
    "promotions": {
        "promo_id": "promoid",
        "promo_name": "promoname",
        "promo_type": "promotype",
        "discount_value": "discountvalue",
        "start_date": "startdate",
        "end_date": "enddate",
        "applicable_category": "applicablecategory",
        "promo_channel": "promochannel",
        "stackable_flag": "stackableflag",
        "min_order_value": "minordervalue",
    },
    "returns": {
        "return_id": "returnid",
        "order_id": "orderid",
        "product_id": "productid",
        "return_date": "returndate",
        "return_reason": "returnreason",
        "return_quantity": "returnquantity",
        "refund_amount": "refundamount",
    },
    "reviews": {
        "review_id": "reviewid",
        "order_id": "orderid",
        "product_id": "productid",
        "customer_id": "customerid",
        "review_date": "reviewdate",
        "review_title": "reviewtitle",
    },
    "inventory": {
        "snapshot_date": "snapshotdate",
        "product_id": "productid",
        "stock_on_hand": "stockonhand",
        "units_received": "unitsreceived",
        "units_sold": "unitssold",
        "stockout_days": "stockoutdays",
        "days_of_supply": "daysofsupply",
        "fill_rate": "fillrate",
        "stockout_flag": "stockoutflag",
        "overstock_flag": "overstockflag",
        "reorder_flag": "reorderflag",
        "sell_through_rate": "sellthroughrate",
        "product_name": "productname",
    },
    "shipments": {
        "order_id": "orderid",
        "ship_date": "shipdate",
        "delivery_date": "deliverydate",
        "shipping_fee": "shippingfee",
    },
    "webtraffic": {
        "unique_visitors": "uniquevisitors",
        "page_views": "pageviews",
        "bounce_rate": "bouncerate",
        "avg_session_duration_sec": "avgsessiondurationsec",
        "traffic_source": "trafficsource",
    },
}

DATE_MAPPING: dict[str, list[str]] = {
    "customers": ["signupdate"],
    "promotions": ["startdate", "enddate"],
    "orders": ["orderdate"],
    "shipments": ["shipdate", "deliverydate"],
    "returns": ["returndate"],
    "reviews": ["reviewdate"],
    "sales": ["Date"],
    "inventory": ["snapshotdate"],
    "webtraffic": ["date"],
}

CATEGORICAL_COLS: dict[str, list[str]] = {
    "products": ["category", "segment", "size", "color"],
    "orders": ["orderstatus", "paymentmethod", "devicetype", "ordersource"],
}


def _load_all_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every expected CSV from *raw_dir* and normalize table/column names."""
    dataframes: dict[str, pd.DataFrame] = {}

    for table_name, candidates in FILE_ALIASES.items():
        file_path = next((raw_dir / name for name in candidates if (raw_dir / name).exists()), None)

        if file_path is None:
            print(f"Warning: File for table '{table_name}' not found. Checked: {candidates}")
            continue

        df = pd.read_csv(file_path, low_memory=False)
        df = df.rename(columns=COLUMN_RENAME_MAPS.get(table_name, {}))
        dataframes[table_name] = df
        print(f"Loaded {table_name} from {file_path.name}: {df.shape}")

    return dataframes


def _format_datetime(dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Parse known datetime columns."""
    for table, cols in DATE_MAPPING.items():
        if table not in dataframes:
            continue

        for col in cols:
            if col in dataframes[table].columns:
                dataframes[table][col] = pd.to_datetime(dataframes[table][col], errors="coerce")
            else:
                print(f"Warning: Column '{col}' not found in table '{table}'.")
    return dataframes


def _optimize_datatypes(dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Cast selected low-cardinality columns to pandas categorical dtype."""
    for table, cols in CATEGORICAL_COLS.items():
        if table not in dataframes:
            continue

        for col in cols:
            if col in dataframes[table].columns:
                dataframes[table][col] = dataframes[table][col].astype("category")
            else:
                print(f"Warning: Column '{col}' not found in table '{table}'.")
    return dataframes


def _handle_missing_values(dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply lightweight missing-value handling aligned with competition schema."""
    if "customers" in dataframes:
        cols_to_fill = ["gender", "agegroup", "acquisitionchannel"]
        existing_cols = [col for col in cols_to_fill if col in dataframes["customers"].columns]
        if existing_cols:
            dataframes["customers"][existing_cols] = dataframes["customers"][existing_cols].fillna(
                "Unknown"
            )

    if "orderitems" in dataframes:
        for col in ["promoid", "promoid2"]:
            if col in dataframes["orderitems"].columns:
                dataframes["orderitems"][col] = dataframes["orderitems"][col].fillna("No_Promo")

    if "promotions" in dataframes and "applicablecategory" in dataframes["promotions"].columns:
        dataframes["promotions"]["applicablecategory"] = dataframes["promotions"][
            "applicablecategory"
        ].fillna("All")

    return dataframes


def _save_processed_data(dataframes: dict[str, pd.DataFrame], processed_dir: Path) -> None:
    """Write all cleaned tables to *processed_dir* as CSV files."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    for table_name, df in dataframes.items():
        output_path = processed_dir / f"{table_name}.csv"
        df.to_csv(output_path, index=False)


def clean_all_data(raw_dir: Path, processed_dir: Path) -> None:
    """Load, clean, and persist all raw datasets.

    Reads every raw data file from *raw_dir*, applies cleaning steps
    (e.g. type casting, missing-value imputation, deduplication), and
    writes the resulting DataFrames to *processed_dir* in a format
    suitable for downstream modelling.

    Parameters
    ----------
    raw_dir:
        Directory that contains the original, unmodified data files.
    processed_dir:
        Directory where cleaned datasets will be saved.

    Returns
    -------
    None
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    df_dict = _load_all_data(raw_dir)
    df_dict = _format_datetime(df_dict)
    df_dict = _optimize_datatypes(df_dict)
    df_dict = _handle_missing_values(df_dict)
    _save_processed_data(df_dict, processed_dir)

    print("Data preprocessing completed!")
    print(f"Saved {len(df_dict)} tables to {processed_dir}")