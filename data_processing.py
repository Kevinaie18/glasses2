from __future__ import annotations

import io
import os
from datetime import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "sample_data.xlsx"
)


@st.cache_data(show_spinner=False)
def load_excel(override_path: Optional[str] = None, uploaded_bytes: Optional[bytes] = None) -> pd.DataFrame:
    """Load Excel as a raw wide table. Prefer uploaded file; else local default.

    Returns the dataframe without transformations.
    """
    if uploaded_bytes is not None:
        excel_bytes = io.BytesIO(uploaded_bytes)
        df = pd.read_excel(excel_bytes, engine="openpyxl")
        return df

    path = override_path or DEFAULT_DATA_PATH
    if not os.path.exists(path):
        # Fallback 1: look for expected filename at project root
        root_candidate = os.path.join(os.path.dirname(__file__), "Copie de Lapaire_Vente mensuelles usd shop.xlsx")
        if os.path.exists(root_candidate):
            return pd.read_excel(root_candidate, engine="openpyxl")
        # Fallback 2: first .xlsx file in project root
        root_dir = os.path.dirname(__file__)
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(".xlsx"):
                return pd.read_excel(os.path.join(root_dir, fname), engine="openpyxl")
        raise FileNotFoundError(
            f"Excel not found at {path}. Upload a file in the sidebar or place the Excel at this path."
        )
    return pd.read_excel(path, engine="openpyxl")


def _detect_month_columns(df: pd.DataFrame) -> List[object]:
    """Identify monthly columns including datetime-typed headers and date-like strings.

    Returns the columns in their original type to preserve melt correctness.
    """
    candidate_cols: List[object] = []
    for col in df.columns:
        # Accept datetime-like header types directly
        if isinstance(col, (pd.Timestamp, dt, np.datetime64)):
            candidate_cols.append(col)
            continue
        # Accept strings that look like dates
        if isinstance(col, str):
            if any(col.startswith(prefix) for prefix in ("202", "201")):
                # e.g., '2023-01', '2023-01-01'
                candidate_cols.append(col)
                continue
            if len(col) >= 8 and ("/" in col or "-" in col):
                parts = col.replace("-", "/").split("/")
                if len(parts) == 3 and all(part.isdigit() for part in parts):
                    candidate_cols.append(col)

    # Ensure parseable to datetime if string
    parsed_set = set()
    for c in candidate_cols:
        if isinstance(c, str):
            s = pd.to_datetime(pd.Series([c]), dayfirst=True, errors="coerce")
            if not s.isna().all():
                parsed_set.add(c)
        else:
            parsed_set.add(c)

    # Keep original order
    ordered = [c for c in df.columns if c in parsed_set]
    return ordered


@st.cache_data(show_spinner=False)
def to_long_format(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Transform the raw wide table to a normalized long format.

    Output columns: Outlet, Country, Month (datetime64[ns]), Sales (float), Year (int)
    """
    df = df_wide.copy()
    if "Outlet" not in df.columns or "Country" not in df.columns:
        raise ValueError("Input Excel must contain 'Outlet' and 'Country' columns.")

    month_columns = _detect_month_columns(df)
    if not month_columns:
        raise ValueError("Could not detect monthly columns in the Excel sheet.")

    id_cols = ["Outlet", "Country"]
    value_cols = month_columns
    long_df = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="MonthStr", value_name="Sales")

    long_df["Month"] = pd.to_datetime(long_df["MonthStr"], dayfirst=True, errors="coerce")
    long_df = long_df.dropna(subset=["Month"]).copy()
    long_df["Sales"] = pd.to_numeric(long_df["Sales"], errors="coerce").fillna(0.0)
    long_df["Year"] = long_df["Month"].dt.year
    long_df.sort_values(["Country", "Outlet", "Month"], inplace=True)
    long_df.reset_index(drop=True, inplace=True)
    return long_df[["Outlet", "Country", "Month", "Year", "Sales"]]


@st.cache_data(show_spinner=False)
def get_available_filters(long_df: pd.DataFrame) -> Tuple[List[str], List[str], List[pd.Timestamp]]:
    """Get available filter options from the data."""
    countries = clean_and_sort_unique(long_df["Country"])
    outlets = clean_and_sort_unique(long_df["Outlet"])
    months = clean_and_sort_months(long_df["Month"])
    return countries, outlets, months


@st.cache_data(show_spinner=False)
def filter_data(
    long_df: pd.DataFrame,
    countries: Optional[List[str]] = None,
    outlets: Optional[List[str]] = None,
    start_month: Optional[pd.Timestamp] = None,
    end_month: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    df = long_df
    if countries:
        df = df[df["Country"].isin(countries)]
    if outlets:
        df = df[df["Outlet"].isin(outlets)]
    if start_month is not None:
        df = df[df["Month"] >= pd.to_datetime(start_month)]
    if end_month is not None:
        df = df[df["Month"] <= pd.to_datetime(end_month)]
    return df.copy()


@st.cache_data(show_spinner=False)
def compute_active_shops(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["is_active"] = df["Sales"] > 0
    active = (
        df.groupby(["Country", "Month"], as_index=False)["is_active"].sum().rename(columns={"is_active": "active_shops"})
    )
    return active


@st.cache_data(show_spinner=False)
def compute_country_monthly_average(long_df: pd.DataFrame) -> pd.DataFrame:
    sales = long_df.groupby(["Country", "Month"], as_index=False)["Sales"].sum()
    active = compute_active_shops(long_df)
    merged = sales.merge(active, on=["Country", "Month"], how="left")
    merged["active_shops"] = merged["active_shops"].replace(0, np.nan)
    merged["avg_sales_per_active_shop"] = merged["Sales"] / merged["active_shops"]
    return merged


def last_n_months(long_df: pd.DataFrame, n: int) -> List[pd.Timestamp]:
    months = sorted(long_df["Month"].dropna().unique().tolist())
    return months[-n:] if len(months) >= n else months


def clean_and_sort_unique(series: pd.Series) -> List[str]:
    """
    Clean a pandas series and return sorted unique values as strings.
    Handles mixed data types and removes problematic values.
    """
    cleaned = series.dropna()
    if cleaned.empty:
        return []
    
    # Convert to string and remove problematic values
    cleaned = cleaned.astype(str)
    cleaned = cleaned[cleaned != 'nan']
    cleaned = cleaned[cleaned != '']
    cleaned = cleaned[cleaned != 'None']
    
    return sorted(cleaned.unique())


def clean_and_sort_months(series: pd.Series) -> List[pd.Timestamp]:
    """
    Clean a pandas series of dates and return sorted unique values.
    """
    cleaned = series.dropna()
    if cleaned.empty:
        return []
    
    return sorted(cleaned.unique())



