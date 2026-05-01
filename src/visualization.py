"""Visualization utilities for EDA and model diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import EDA_IMAGES_DIR, PROCESSED_DIR, RAW_DIR


def set_plot_style() -> None:
    """Configure global Matplotlib / Seaborn aesthetics.

    Call this function once at the top of any script or notebook before
    generating plots.  Sets a consistent theme, font sizes, and figure
    DPI for all subsequent visualizations.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )


def _read_csv(data_dir: Path, stem: str) -> pd.DataFrame:
    path = data_dir / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset: {path}")
    return pd.read_csv(path, low_memory=False)


def _select_default_data_dir(raw_dir: Path, processed_dir: Path) -> Path:
    required_raw = ["sales.csv", "orders.csv", "order_items.csv", "products.csv"]
    if raw_dir.exists() and all((raw_dir / name).exists() for name in required_raw):
        return raw_dir
    return processed_dir


def _line_revenue(order_items: pd.DataFrame) -> pd.Series:
    quantity = pd.to_numeric(order_items["quantity"], errors="coerce").fillna(0.0)
    unit_price = pd.to_numeric(order_items["unit_price"], errors="coerce").fillna(0.0)
    discount = pd.to_numeric(order_items["discount_amount"], errors="coerce").fillna(0.0)
    return quantity * unit_price - discount


def _save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _plot_revenue_cogs_trend(sales: pd.DataFrame, output_dir: Path) -> Path:
    view = sales.copy()
    view["Date"] = pd.to_datetime(view["Date"], errors="coerce")
    view = view.dropna(subset=["Date"]).sort_values("Date")
    monthly = view.set_index("Date")[["Revenue", "COGS"]].resample("MS").sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly.index, monthly["Revenue"], label="Revenue", linewidth=1.8)
    ax.plot(monthly.index, monthly["COGS"], label="COGS", linewidth=1.8)
    ax.set_title("Monthly Revenue and COGS Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("VND")
    ax.legend()
    return _save_current_figure(output_dir / "01_trend_revenue_cogs.png")


def _plot_weekday_revenue(sales: pd.DataFrame, output_dir: Path) -> Path:
    view = sales.copy()
    view["Date"] = pd.to_datetime(view["Date"], errors="coerce")
    view = view.dropna(subset=["Date", "Revenue"])
    view["day_of_week"] = view["Date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = view.groupby("day_of_week", observed=True)["Revenue"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=weekday.index, y=weekday.values, ax=ax, color="#4C78A8")
    ax.set_title("Average Daily Revenue by Day of Week")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Average revenue")
    ax.tick_params(axis="x", rotation=25)
    return _save_current_figure(output_dir / "02_weekday_revenue.png")


def _plot_region_revenue(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    geography: pd.DataFrame,
    output_dir: Path,
) -> Path:
    items = order_items[["order_id"]].copy()
    items["line_revenue"] = _line_revenue(order_items)
    order_revenue = items.groupby("order_id", as_index=False, observed=True)["line_revenue"].sum()
    view = (
        orders[["order_id", "zip"]]
        .merge(order_revenue, on="order_id", how="inner")
        .merge(geography[["zip", "region"]], on="zip", how="left")
        .dropna(subset=["region"])
    )
    region = view.groupby("region", observed=True)["line_revenue"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=region.index, y=region.values, ax=ax, palette="Set2", hue=region.index, legend=False)
    ax.set_title("Total Transaction Revenue by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total revenue")
    return _save_current_figure(output_dir / "03_region_revenue.png")


def _plot_category_return_rate(
    order_items: pd.DataFrame,
    returns: pd.DataFrame,
    products: pd.DataFrame,
    output_dir: Path,
) -> Path:
    product_category = products[["product_id", "category"]].copy()
    item_counts = (
        order_items.merge(product_category, on="product_id", how="left")
        .dropna(subset=["category"])
        .groupby("category", observed=True)
        .size()
    )
    return_counts = (
        returns.merge(product_category, on="product_id", how="left")
        .dropna(subset=["category"])
        .groupby("category", observed=True)
        .size()
    )
    rates = (return_counts / item_counts).dropna().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=rates.index, y=rates.values * 100.0, ax=ax, color="#F58518")
    ax.set_title("Return Rate by Product Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Return records / order-item lines (%)")
    ax.tick_params(axis="x", rotation=20)
    return _save_current_figure(output_dir / "04_category_return_rate.png")


def _plot_traffic_quality(web_traffic: pd.DataFrame, output_dir: Path) -> Path:
    view = web_traffic.copy()
    view["bounce_rate"] = pd.to_numeric(view["bounce_rate"], errors="coerce")
    source = view.groupby("traffic_source", observed=True)["bounce_rate"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=source.index, y=source.values, ax=ax, color="#54A24B")
    ax.set_title("Average Bounce Rate by Traffic Source")
    ax.set_xlabel("Traffic source")
    ax.set_ylabel("Average bounce rate")
    ax.tick_params(axis="x", rotation=25)
    return _save_current_figure(output_dir / "05_traffic_bounce_rate.png")


def _plot_missing_values(tables: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    rows: list[dict[str, object]] = []
    for table_name, table in tables.items():
        missing_rate = table.isna().mean().sort_values(ascending=False)
        for column, rate in missing_rate.head(3).items():
            if rate > 0:
                rows.append({"field": f"{table_name}.{column}", "missing_rate": float(rate)})

    missing = pd.DataFrame(rows).sort_values("missing_rate", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values detected", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.barplot(data=missing, y="field", x="missing_rate", ax=ax, color="#B279A2")
        ax.set_title("Top Missing-Value Rates Across Source Tables")
        ax.set_xlabel("Missing rate")
        ax.set_ylabel("Table.column")
    return _save_current_figure(output_dir / "06_missing_values_top.png")


def generate_eda_plots(processed_dir: Path, output_dir: Path) -> list[Path]:
    """Create and save exploratory data analysis charts.

    Reads datasets from *processed_dir*, produces a
    standard suite of EDA plots (distributions, correlations, missing-
    value heatmaps, etc.), and writes each figure as a PNG file to
    *output_dir*.

    Parameters
    ----------
    processed_dir:
        Directory containing the cleaned datasets produced by
        :func:`src.data_prep.clean_all_data`.
    output_dir:
        Directory where generated plot images will be saved.

    Returns
    -------
    list[pathlib.Path]
        Paths to generated plot images.
    """
    set_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "sales": _read_csv(processed_dir, "sales"),
        "orders": _read_csv(processed_dir, "orders"),
        "order_items": _read_csv(processed_dir, "order_items"),
        "products": _read_csv(processed_dir, "products"),
        "returns": _read_csv(processed_dir, "returns"),
        "geography": _read_csv(processed_dir, "geography"),
        "web_traffic": _read_csv(processed_dir, "web_traffic"),
    }

    generated = [
        _plot_revenue_cogs_trend(tables["sales"], output_dir),
        _plot_weekday_revenue(tables["sales"], output_dir),
        _plot_region_revenue(tables["orders"], tables["order_items"], tables["geography"], output_dir),
        _plot_category_return_rate(tables["order_items"], tables["returns"], tables["products"], output_dir),
        _plot_traffic_quality(tables["web_traffic"], output_dir),
        _plot_missing_values(tables, output_dir),
    ]
    return generated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Datathon EDA figures.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing competition CSV files. Defaults to data/raw when available.",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=EDA_IMAGES_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir or _select_default_data_dir(args.raw_dir, args.processed_dir)
    generated = generate_eda_plots(data_dir, args.output_dir)
    print(f"Generated {len(generated)} EDA figure(s) from {data_dir}:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
