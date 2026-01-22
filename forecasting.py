from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


def _month_range(start: str = "2025-01-01", end: str = "2026-12-01") -> List[pd.Timestamp]:
    months = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq="MS")
    return list(months)


@st.cache_data(show_spinner=False)
def build_baseline(long_df: pd.DataFrame) -> pd.DataFrame:
    # Use last 12 months average sales per active shop by country, then project forward assuming constant shops
    last_month = long_df["Month"].max()
    window_start = last_month - pd.DateOffset(months=11)
    last12 = long_df[(long_df["Month"] >= window_start) & (long_df["Month"] <= last_month)].copy()

    active = (
        last12.assign(is_active=last12["Sales"] > 0)
        .groupby(["Country", "Month"], as_index=False)["is_active"].sum()
        .rename(columns={"is_active": "active_shops"})
    )
    per_shop = (
        last12.groupby(["Country", "Month"], as_index=False)["Sales"].sum().merge(active, on=["Country", "Month"], how="left")
    )
    per_shop["active_shops"] = per_shop["active_shops"].replace(0, np.nan)
    per_shop["avg_per_shop"] = per_shop["Sales"] / per_shop["active_shops"]

    # Average across last 12 months
    country_stats = per_shop.groupby("Country", as_index=False).agg(
        avg_per_shop=("avg_per_shop", "mean"), current_active_shops=("active_shops", "mean")
    )
    
    # Handle NaN values before converting to int - replace with 1 as fallback
    country_stats["current_active_shops"] = country_stats["current_active_shops"].fillna(1.0)
    country_stats["avg_per_shop"] = country_stats["avg_per_shop"].fillna(0.0)
    # Ensure no infinite values
    country_stats["current_active_shops"] = country_stats["current_active_shops"].replace([np.inf, -np.inf], 1.0)
    country_stats["avg_per_shop"] = country_stats["avg_per_shop"].replace([np.inf, -np.inf], 0.0)
    # Round and convert to int safely
    country_stats["current_active_shops"] = (
        country_stats["current_active_shops"].round().fillna(1).replace([np.inf, -np.inf], 1).astype(int)
    )

    months = _month_range()
    rows = []
    for _, r in country_stats.iterrows():
        for m in months:
            rows.append(
                {
                    "Country": r["Country"],
                    "Month": m,
                    "Sales": float(r["avg_per_shop"]) * int(r["current_active_shops"]),
                    "Scenario": "Baseline",
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_adjusted_scenario(
    baseline: pd.DataFrame,
    growth_by_country: Dict[str, float],  # e.g. {"CI": 0.10}
    new_shop_rate_by_country: Dict[str, float],  # annual rate, e.g. 0.10
    target_per_shop_by_country: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    df = baseline.copy()
    df.sort_values(["Country", "Month"], inplace=True)
    results = []
    for country, g in df.groupby("Country"):
        growth = growth_by_country.get(country, 0.0)
        open_rate = new_shop_rate_by_country.get(country, 0.0)
        target_per_shop = (target_per_shop_by_country or {}).get(country, np.nan)

        # Infer baseline per-shop and number of shops
        monthly_baseline = g["Sales"].iloc[0]
        # Use 12-month decompose: per_shop * shops_count
        # We cannot recover the split directly; assume per-shop is stable across months in baseline period.
        # Estimate per_shop from first year average if provided; else divide by an inferred shop count (>=1)
        # Here we infer shop_count from a heuristic: keep baseline value as total; simulate shops evolving.
        shops = 1
        per_shop = monthly_baseline
        
        # Handle NaN values safely
        if (monthly_baseline is None) or np.isnan(monthly_baseline) or monthly_baseline <= 0:
            monthly_baseline = 1000.0  # Default fallback value
            per_shop = monthly_baseline
        
        if not np.isnan(target_per_shop) and target_per_shop > 0:
            per_shop = target_per_shop
            # Ensure safe division
            shops = 1
            if per_shop is not None and per_shop > 0 and monthly_baseline is not None:
                if (not np.isnan(monthly_baseline)) and (not np.isnan(per_shop)):
                    ratio = monthly_baseline / per_shop
                    if not np.isnan(ratio) and np.isfinite(ratio):
                        try:
                            shops = max(int(round(ratio)), 1)
                        except ValueError:
                            shops = 1

        for idx, row in g.reset_index(drop=True).iterrows():
            # Apply annualized growth to per_shop compounded monthly
            monthly_growth = (1.0 + growth) ** (1.0 / 12.0) - 1.0
            per_shop = per_shop * (1.0 + monthly_growth)

            # Add shops gradually based on annual rate, distributed monthly
            monthly_open_rate = open_rate / 12.0
            shops = max(int(round(shops * (1.0 + monthly_open_rate))), 1)

            results.append(
                {
                    "Country": country,
                    "Month": row["Month"],
                    "Sales": float(per_shop * shops),
                    "Scenario": "Ajusté",
                    "per_shop": float(per_shop),
                    "shops": int(shops),
                }
            )
    return pd.DataFrame(results)


