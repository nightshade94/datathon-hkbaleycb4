"""Solver for Part 1 multiple-choice questions.

The MCQ section is intentionally computed from the provided CSV files instead
of hard-coding answer keys. By default the script uses ``data/raw`` because the
questions refer to the original tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR


ChoiceMap = dict[str, str]
Details = dict[str, Any]


Q1_CHOICES = {"A": 30.0, "B": 90.0, "C": 144.0, "D": 365.0}
Q5_CHOICES = {"A": 12.0, "B": 25.0, "C": 39.0, "D": 54.0}
Q2_CHOICES: ChoiceMap = {
    "A": "Premium",
    "B": "Performance",
    "C": "Activewear",
    "D": "Standard",
}
Q3_CHOICES: ChoiceMap = {
    "A": "defective",
    "B": "wrong_size",
    "C": "changed_mind",
    "D": "not_as_described",
}
Q4_CHOICES: ChoiceMap = {
    "A": "organic_search",
    "B": "paid_search",
    "C": "email_campaign",
    "D": "social_media",
}
Q6_CHOICES: ChoiceMap = {"A": "55+", "B": "25-34", "C": "35-44", "D": "45-54"}
Q7_CHOICES: ChoiceMap = {"A": "West", "B": "Central", "C": "East", "D": "approximately_equal"}
Q8_CHOICES: ChoiceMap = {
    "A": "credit_card",
    "B": "cod",
    "C": "paypal",
    "D": "bank_transfer",
}
Q9_CHOICES: ChoiceMap = {"A": "S", "B": "M", "C": "L", "D": "XL"}
Q10_CHOICES = {"A": 1, "B": 3, "C": 6, "D": 12}


def _read_csv(data_dir: Path, stem: str, **kwargs: Any) -> pd.DataFrame:
    path = data_dir / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset: {path}")
    read_kwargs = {"low_memory": False}
    read_kwargs.update(kwargs)
    return pd.read_csv(path, **read_kwargs)


def _select_default_data_dir(raw_dir: Path, processed_dir: Path) -> Path:
    required_raw = ["orders.csv", "order_items.csv", "products.csv", "sales.csv"]
    if raw_dir.exists() and all((raw_dir / name).exists() for name in required_raw):
        return raw_dir
    return processed_dir


def _closest_numeric_answer(value: float, choices: dict[str, float]) -> str:
    return min(choices, key=lambda key: abs(value - choices[key]))


def _max_choice_by_series(values: pd.Series, choices: ChoiceMap) -> str:
    available = {key: label for key, label in choices.items() if label in values.index}
    if not available:
        raise ValueError(f"None of the expected choices appear in the computed series: {choices}")
    return max(available, key=lambda key: float(values.loc[available[key]]))


def _min_choice_by_series(values: pd.Series, choices: ChoiceMap) -> str:
    available = {key: label for key, label in choices.items() if label in values.index}
    if not available:
        raise ValueError(f"None of the expected choices appear in the computed series: {choices}")
    return min(available, key=lambda key: float(values.loc[available[key]]))


def _valid_promo_mask(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    missing_tokens = {"", "nan", "none", "no_promo", "null"}
    return ~text.isin(missing_tokens)


def _line_revenue(order_items: pd.DataFrame) -> pd.Series:
    quantity = pd.to_numeric(order_items["quantity"], errors="coerce").fillna(0.0)
    unit_price = pd.to_numeric(order_items["unit_price"], errors="coerce").fillna(0.0)
    discount = pd.to_numeric(order_items["discount_amount"], errors="coerce").fillna(0.0)
    return quantity * unit_price - discount


def _solve_q1(orders: pd.DataFrame) -> tuple[str, Details]:
    view = orders[["customer_id", "order_date"]].copy()
    view["order_date"] = pd.to_datetime(view["order_date"], errors="coerce")
    view = view.dropna(subset=["customer_id", "order_date"]).sort_values(["customer_id", "order_date"])
    gaps = view.groupby("customer_id", observed=True)["order_date"].diff().dt.days.dropna()
    median_gap = float(gaps.median())
    answer = _closest_numeric_answer(median_gap, Q1_CHOICES)
    return answer, {"median_inter_order_gap_days": median_gap}


def _solve_q2(products: pd.DataFrame) -> tuple[str, Details]:
    view = products[["segment", "price", "cogs"]].copy()
    view["price"] = pd.to_numeric(view["price"], errors="coerce")
    view["cogs"] = pd.to_numeric(view["cogs"], errors="coerce")
    view = view.dropna(subset=["segment", "price", "cogs"])
    view = view[view["price"] > 0].copy()
    view["gross_margin_rate"] = (view["price"] - view["cogs"]) / view["price"]
    margins = view.groupby("segment", observed=True)["gross_margin_rate"].mean().sort_values(ascending=False)
    answer = _max_choice_by_series(margins, Q2_CHOICES)
    return answer, {
        "selected_segment": Q2_CHOICES[answer],
        "selected_margin": float(margins.loc[Q2_CHOICES[answer]]),
        "top_segment_overall": str(margins.index[0]),
        "top_margin_overall": float(margins.iloc[0]),
    }


def _solve_q3(returns: pd.DataFrame, products: pd.DataFrame) -> tuple[str, Details]:
    view = returns.merge(products[["product_id", "category"]], on="product_id", how="left")
    counts = view.loc[view["category"] == "Streetwear", "return_reason"].value_counts()
    answer = _max_choice_by_series(counts, Q3_CHOICES)
    return answer, {"selected_reason": Q3_CHOICES[answer], "count": int(counts.loc[Q3_CHOICES[answer]])}


def _solve_q4(web_traffic: pd.DataFrame) -> tuple[str, Details]:
    view = web_traffic[["traffic_source", "bounce_rate"]].copy()
    view["bounce_rate"] = pd.to_numeric(view["bounce_rate"], errors="coerce")
    rates = view.dropna().groupby("traffic_source", observed=True)["bounce_rate"].mean().sort_values()
    answer = _min_choice_by_series(rates, Q4_CHOICES)
    return answer, {
        "selected_source": Q4_CHOICES[answer],
        "selected_bounce_rate": float(rates.loc[Q4_CHOICES[answer]]),
        "lowest_source_overall": str(rates.index[0]),
        "lowest_bounce_rate_overall": float(rates.iloc[0]),
    }


def _solve_q5(order_items: pd.DataFrame) -> tuple[str, Details]:
    promo_rate = float(_valid_promo_mask(order_items["promo_id"]).mean() * 100.0)
    answer = _closest_numeric_answer(promo_rate, Q5_CHOICES)
    return answer, {"promo_line_rate_percent": promo_rate}


def _solve_q6(orders: pd.DataFrame, customers: pd.DataFrame) -> tuple[str, Details]:
    customer_view = customers[["customer_id", "age_group"]].dropna(subset=["age_group"]).copy()
    order_counts = orders.groupby("customer_id", observed=True).size().rename("order_count")
    view = customer_view.merge(order_counts, on="customer_id", how="left")
    view["order_count"] = view["order_count"].fillna(0)
    avg_orders = view.groupby("age_group", observed=True)["order_count"].mean().sort_values(ascending=False)
    answer = _max_choice_by_series(avg_orders, Q6_CHOICES)
    return answer, {
        "selected_age_group": Q6_CHOICES[answer],
        "avg_orders_per_customer": float(avg_orders.loc[Q6_CHOICES[answer]]),
    }


def _solve_q7(orders: pd.DataFrame, order_items: pd.DataFrame, geography: pd.DataFrame) -> tuple[str, Details]:
    items = order_items[["order_id"]].copy()
    items["line_revenue"] = _line_revenue(order_items)
    order_revenue = items.groupby("order_id", as_index=False, observed=True)["line_revenue"].sum()
    view = (
        orders[["order_id", "zip"]]
        .merge(order_revenue, on="order_id", how="inner")
        .merge(geography[["zip", "region"]], on="zip", how="left")
        .dropna(subset=["region"])
    )
    region_revenue = view.groupby("region", observed=True)["line_revenue"].sum().sort_values(ascending=False)
    if region_revenue.empty:
        raise ValueError("Could not compute region-level revenue for Q7.")

    max_value = float(region_revenue.iloc[0])
    min_value = float(region_revenue.iloc[-1])
    spread = (max_value - min_value) / max_value if max_value else 0.0
    if spread <= 0.02:
        answer = "D"
    else:
        answer = _max_choice_by_series(region_revenue, {k: v for k, v in Q7_CHOICES.items() if k != "D"})

    return answer, {
        "selected_region": Q7_CHOICES[answer],
        "region_revenue": {str(k): float(v) for k, v in region_revenue.items()},
        "relative_spread": spread,
    }


def _solve_q8(orders: pd.DataFrame) -> tuple[str, Details]:
    counts = orders.loc[orders["order_status"] == "cancelled", "payment_method"].value_counts()
    answer = _max_choice_by_series(counts, Q8_CHOICES)
    return answer, {"selected_payment_method": Q8_CHOICES[answer], "count": int(counts.loc[Q8_CHOICES[answer]])}


def _solve_q9(order_items: pd.DataFrame, returns: pd.DataFrame, products: pd.DataFrame) -> tuple[str, Details]:
    product_size = products[["product_id", "size"]].copy()
    return_counts = (
        returns.merge(product_size, on="product_id", how="left")
        .dropna(subset=["size"])
        .groupby("size", observed=True)
        .size()
    )
    item_counts = (
        order_items.merge(product_size, on="product_id", how="left")
        .dropna(subset=["size"])
        .groupby("size", observed=True)
        .size()
    )
    rates = (return_counts / item_counts).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    answer = _max_choice_by_series(rates, Q9_CHOICES)
    return answer, {"selected_size": Q9_CHOICES[answer], "return_rate": float(rates.loc[Q9_CHOICES[answer]])}


def _solve_q10(payments: pd.DataFrame) -> tuple[str, Details]:
    view = payments[["installments", "payment_value"]].copy()
    view["installments"] = pd.to_numeric(view["installments"], errors="coerce")
    view["payment_value"] = pd.to_numeric(view["payment_value"], errors="coerce")
    averages = view.dropna().groupby("installments", observed=True)["payment_value"].mean().sort_values(ascending=False)
    choices = {key: str(value) for key, value in Q10_CHOICES.items()}
    averages.index = averages.index.astype(int).astype(str)
    answer = _max_choice_by_series(averages, choices)
    return answer, {
        "selected_installments": Q10_CHOICES[answer],
        "avg_payment_value": float(averages.loc[str(Q10_CHOICES[answer])]),
    }


def solve_mcq_details(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Compute MCQ answers and supporting evidence from local CSV files."""
    orders = _read_csv(data_dir, "orders", parse_dates=["order_date"])
    order_items = _read_csv(data_dir, "order_items")
    products = _read_csv(data_dir, "products")
    customers = _read_csv(data_dir, "customers")
    returns = _read_csv(data_dir, "returns")
    web_traffic = _read_csv(data_dir, "web_traffic")
    geography = _read_csv(data_dir, "geography")
    payments = _read_csv(data_dir, "payments")

    solvers = {
        "Q1": lambda: _solve_q1(orders),
        "Q2": lambda: _solve_q2(products),
        "Q3": lambda: _solve_q3(returns, products),
        "Q4": lambda: _solve_q4(web_traffic),
        "Q5": lambda: _solve_q5(order_items),
        "Q6": lambda: _solve_q6(orders, customers),
        "Q7": lambda: _solve_q7(orders, order_items, geography),
        "Q8": lambda: _solve_q8(orders),
        "Q9": lambda: _solve_q9(order_items, returns, products),
        "Q10": lambda: _solve_q10(payments),
    }

    results: dict[str, dict[str, Any]] = {}
    for question, solver in solvers.items():
        answer, details = solver()
        results[question] = {"answer": answer, **details}
    return results


def solve_mcq(processed_dir: Path) -> dict[str, str]:
    """Return only the selected answer choice for each MCQ question."""
    return {question: result["answer"] for question, result in solve_mcq_details(processed_dir).items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Datathon Round 1 Part 1 MCQ answers.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing competition CSV files. Defaults to data/raw when available.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full supporting evidence as JSON instead of a compact answer list.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir or _select_default_data_dir(RAW_DIR, PROCESSED_DIR)
    details = solve_mcq_details(data_dir)

    if args.json:
        payload = {"data_dir": str(data_dir), "answers": details}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"Data directory: {data_dir}")
    for question in sorted(details, key=lambda value: int(value[1:])):
        print(f"{question}: {details[question]['answer']}")


if __name__ == "__main__":
    main()
