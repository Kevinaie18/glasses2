from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ------------------------------
# Schema normalization utilities
# ------------------------------

def _first_of_month(dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt, errors="coerce")
    return pd.to_datetime(dict(year=dt.dt.year, month=dt.dt.month, day=1))


@st.cache_data(show_spinner=False)
def normalize_shops_schema(
    df: pd.DataFrame,
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize input DataFrame to the minimal schema required by the shop views:
      - country: str
      - shop_id: str|int
      - shop_open_date: datetime (if missing, inferred as first month with sales_usd>0)
      - period: datetime (first day of month)
      - sales_usd: float

    If project-specific names differ, pass a mapping dict: {standard_name: project_column}.
    """
    df_local = df.copy()

    # Default mapping based on the current project (long format)
    default_mapping = {
        "country": "Country",
        "shop_id": "Outlet",
        "period": "Month",
        "sales_usd": "Sales",
        # Optional: "shop_open_date": "OpenDate"
    }
    if mapping:
        default_mapping.update(mapping)

    # Rename columns if present; else create
    rename_pairs = {}
    for std_col, project_col in default_mapping.items():
        if project_col in df_local.columns:
            rename_pairs[project_col] = std_col
    df_local = df_local.rename(columns=rename_pairs)

    # Ensure required columns exist
    required = ["country", "shop_id", "period", "sales_usd"]
    for col in required:
        if col not in df_local.columns:
            raise KeyError(f"Missing required column '{col}' after normalization")

    # Types & normalization
    df_local["period"] = _first_of_month(df_local["period"])
    df_local["sales_usd"] = pd.to_numeric(df_local["sales_usd"], errors="coerce").fillna(0.0)

    # Open date
    if "shop_open_date" not in df_local.columns:
        # Infer first month with strictly positive sales
        first_sales = (
            df_local[df_local["sales_usd"] > 0]
            .groupby("shop_id", as_index=False)["period"].min()
            .rename(columns={"period": "shop_open_date"})
        )
        df_local = df_local.merge(first_sales, on="shop_id", how="left")

    return df_local


# ------------------------------
# Common calculations
# ------------------------------

def months_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_datetime(a, errors="coerce")
    b = pd.to_datetime(b, errors="coerce")
    return (a.dt.year * 12 + a.dt.month) - (b.dt.year * 12 + b.dt.month)


def bucket_ca(sales_usd: float) -> str:
    if sales_usd >= 10_000:
        return "10k+"
    if sales_usd >= 7_000:
        return "7-10k"
    if sales_usd >= 5_000:
        return "5-7k"
    if sales_usd >= 2_000:
        return "2-5k"
    return "0-2k"


def bucket_maturity(age_months: int) -> str:
    if age_months <= 6:
        return "0-6m"
    if 7 <= age_months <= 10:
        return "6-10m"
    return "11m+"


# ------------------------------
# View 1 — Répartition boutiques par tranches de CA
# ------------------------------

@st.cache_data(show_spinner=False)
def compute_view1_ca_buckets_table(
    df: pd.DataFrame,
    countries: Optional[List[str]] = None,
    start_period: Optional[pd.Timestamp] = None,
    end_period: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    d = normalize_shops_schema(df)
    if countries:
        d = d[d["country"].isin(countries)]
    if start_period is not None:
        d = d[d["period"] >= pd.to_datetime(start_period)]
    if end_period is not None:
        d = d[d["period"] <= pd.to_datetime(end_period)]

    if d.empty:
        return pd.DataFrame()

    d = d.assign(ca_bucket=d["sales_usd"].map(bucket_ca))
    counts = d.groupby(["period", "ca_bucket"], as_index=False)["shop_id"].count().rename(columns={"shop_id": "count"})
    pivot = counts.pivot(index="period", columns="ca_bucket", values="count").fillna(0).astype(int)

    # Ensure all buckets/ordering are present
    bucket_order = ["10k+", "7-10k", "5-7k", "2-5k", "0-2k"]
    for b in bucket_order:
        if b not in pivot.columns:
            pivot[b] = 0
    pivot = pivot[bucket_order]
    pivot["Total"] = pivot.sum(axis=1)

    # Percentages
    pct = pivot[bucket_order].div(pivot["Total"].replace(0, np.nan), axis=0) * 100
    pct = pct.add_suffix("_pct")
    out = pd.concat([pivot, pct], axis=1).reset_index()
    out = out.sort_values("period")
    out["period_str"] = out["period"].dt.strftime("%m-%y")
    cols = ["period", "period_str"] + bucket_order + ["Total"] + [f"{b}_pct" for b in bucket_order]
    return out[cols]


def plot_view1_ca_buckets_table(table_df: pd.DataFrame):
    if table_df.empty:
        st.info("Aucune donnée pour la période/pays sélectionnés.")
        return
    display = table_df.copy()
    for col in ["10k+", "7-10k", "5-7k", "2-5k", "0-2k", "Total"]:
        display[col] = display[col].astype(int)
    for col in ["10k+_pct", "7-10k_pct", "5-7k_pct", "2-5k_pct", "0-2k_pct"]:
        display[col] = display[col].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    st.dataframe(
        display.drop(columns=["period"]).rename(columns={"period_str": "Période"}),
        use_container_width=True,
        hide_index=True,
    )


def build_view1_ca_buckets_matrix(table_df: pd.DataFrame) -> pd.DataFrame:
    """Rotate table so that CA buckets are rows and periods are columns (MM-YY), with Total row.
    Matches the reading direction shown by the user's screenshot."""
    if table_df.empty:
        return pd.DataFrame()
    bucket_order = ["10k+", "7-10k", "5-7k", "2-5k", "0-2k"]
    t = table_df.copy()
    t["period_str"] = t["period"].dt.strftime("%d/%m/%Y")
    # Build values matrix for counts only (no percentages)
    value_cols = bucket_order + ["Total"]
    melted = t.melt(id_vars=["period_str"], value_vars=value_cols, var_name="bucket", value_name="count")
    # Pivot to columns=periods
    matrix = (
        melted.pivot(index="bucket", columns="period_str", values="count")
        .reindex(bucket_order + ["Total"])
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"bucket": "Tranche CA (USD/mois)"})
    )
    # Friendly French labels
    label_map = {
        "10k+": "+10 kUSD par mois",
        "7-10k": "Entre 7 et 10k USD par mois",
        "5-7k": "Entre 5 et 7k USD par mois",
        "2-5k": "Entre 2 et 5k USD par mois",
        "0-2k": "≤ 2k USD par mois",
        "Total": "Total",
    }
    matrix["Tranche CA (USD/mois)"] = matrix["Tranche CA (USD/mois)"].map(lambda x: label_map.get(x, x))
    return matrix


def plot_view1_ca_buckets_matrix(matrix_df: pd.DataFrame):
    if matrix_df.empty:
        st.info("Aucune donnée pour la période/pays sélectionnés.")
        return
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)


# ------------------------------
# View 2 — Poids du CA par maturité
# ------------------------------

@st.cache_data(show_spinner=False)
def compute_view2_maturity_share(
    df: pd.DataFrame,
    countries: Optional[List[str]] = None,
    start_period: Optional[pd.Timestamp] = None,
    end_period: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, bool]:
    d = normalize_shops_schema(df)
    if countries:
        d = d[d["country"].isin(countries)]
    if start_period is not None:
        d = d[d["period"] >= pd.to_datetime(start_period)]
    if end_period is not None:
        d = d[d["period"] <= pd.to_datetime(end_period)]

    if d.empty:
        return pd.DataFrame(), False

    # Require open dates for this view
    d = d[d["shop_open_date"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), False

    d["age_months"] = months_diff(d["period"], d["shop_open_date"]).clip(lower=0)
    # Guard against NaN before int casting
    d["age_months"] = d["age_months"].fillna(0).astype(int)
    d["maturity_bucket"] = d["age_months"].map(bucket_maturity)
    agg = d.groupby(["period", "maturity_bucket"], as_index=False)["sales_usd"].sum()
    total = agg.groupby("period", as_index=False)["sales_usd"].sum().rename(columns={"sales_usd": "total_sales"})
    out = agg.merge(total, on="period", how="left")
    out["share_pct"] = np.where(out["total_sales"] > 0, out["sales_usd"] / out["total_sales"] * 100.0, 0.0)
    out["CA_KUSD"] = out["sales_usd"] / 1000.0
    out = out.sort_values(["period", "maturity_bucket"]).reset_index(drop=True)
    return out, True


def plot_view2_maturity_share(maturity_df: pd.DataFrame, mode: str = "percent"):
    if maturity_df.empty:
        st.info("Aucune donnée de maturité disponible (dates d'ouverture manquantes).")
        return
    display = maturity_df.copy()
    display["period_str"] = display["period"].dt.strftime("%m-%y")
    if mode == "percent":
        fig = px.area(
            display,
            x="period_str",
            y="share_pct",
            color="maturity_bucket",
            category_orders={"maturity_bucket": ["0-6m", "6-10m", "11m+"]},
            labels={"period_str": "Période", "share_pct": "Part (%)", "maturity_bucket": "Maturité"},
        )
        fig.update_layout(yaxis_tickformat=".0f")
    else:
        fig = px.bar(
            display,
            x="period_str",
            y="CA_KUSD",
            color="maturity_bucket",
            barmode="stack",
            category_orders={"maturity_bucket": ["0-6m", "6-10m", "11m+"]},
            labels={"period_str": "Période", "CA_KUSD": "CA (k USD)", "maturity_bucket": "Maturité"},
        )
        fig.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)
    # Export equivalent table
    table = display.pivot_table(index="period_str", columns="maturity_bucket", values="share_pct" if mode=="percent" else "CA_KUSD", aggfunc="sum").fillna(0)
    st.dataframe(table.reset_index(), use_container_width=True, hide_index=True)


# ------------------------------
# View 3 — Combo chart CA & réseau boutiques
# ------------------------------

@st.cache_data(show_spinner=False)
def compute_view3_combo(
    df: pd.DataFrame,
    countries: Optional[List[str]] = None,
    start_period: Optional[pd.Timestamp] = None,
    end_period: Optional[pd.Timestamp] = None,
    active_threshold_usd: float = 0.0,
) -> pd.DataFrame:
    d = normalize_shops_schema(df)
    if countries:
        d = d[d["country"].isin(countries)]
    if start_period is not None:
        d = d[d["period"] >= pd.to_datetime(start_period)]
    if end_period is not None:
        d = d[d["period"] <= pd.to_datetime(end_period)]

    if d.empty:
        return pd.DataFrame()

    monthly_sales = d.groupby("period", as_index=False)["sales_usd"].sum().rename(columns={"sales_usd": "total_sales_usd"})
    active = (
        d[d["sales_usd"] > float(active_threshold_usd)]
        .groupby("period", as_index=False)["shop_id"].nunique()
        .rename(columns={"shop_id": "active_shops"})
    )
    out = monthly_sales.merge(active, on="period", how="left").fillna({"active_shops": 0})
    out["CA_KUSD"] = out["total_sales_usd"] / 1000.0
    return out.sort_values("period")


def plot_view3_combo(combo_df: pd.DataFrame, y1_max: Optional[float] = None, y2_max: Optional[float] = None):
    if combo_df.empty:
        st.info("Aucune donnée pour le combo CA & boutiques actives.")
        return
    combo_df = combo_df.copy()
    combo_df["period_str"] = combo_df["period"].dt.strftime("%m-%y")

    fig = make_combo_figure(combo_df, y1_max=y1_max, y2_max=y2_max)
    st.plotly_chart(fig, use_container_width=True)


def make_combo_figure(combo_df: pd.DataFrame, y1_max: Optional[float] = None, y2_max: Optional[float] = None) -> go.Figure:
    fig = go.Figure()
    # Use USD for display to resemble example; show formatted labels on bars
    bar_values_usd = combo_df["total_sales_usd"] if "total_sales_usd" in combo_df.columns else combo_df["CA_KUSD"] * 1000.0
    bar_text = bar_values_usd.map(lambda v: f"{float(v):,.0f}".replace(",", " "))
    fig.add_bar(
        x=combo_df["period_str"],
        y=bar_values_usd,
        name="CA (USD)",
        marker_color="#3B82F6",
        text=bar_text,
        textposition="outside",
        textfont=dict(color="#374151"),
        yaxis="y1",
    )
    fig.add_trace(
        go.Scatter(
            x=combo_df["period_str"],
            y=combo_df["active_shops"],
            name="Boutiques actives",
            mode="lines+markers+text",
            marker=dict(color="#E53935"),
            line=dict(color="#E53935", width=3, shape="spline"),
            text=combo_df["active_shops"].astype(int).astype(str),
            textposition="bottom center",
            yaxis="y2",
        )
    )
    # Build axis ranges if provided
    yaxis_layout = dict(title="CA (USD)", rangemode="tozero", tickformat=",.0f")
    if y1_max is not None and np.isfinite(y1_max):
        yaxis_layout.update(dict(range=[0, float(y1_max)]))

    yaxis2_layout = dict(title="Boutiques actives", overlaying="y", side="right", rangemode="tozero")
    if y2_max is not None and np.isfinite(y2_max):
        yaxis2_layout.update(dict(range=[0, float(y2_max)]))

    fig.update_layout(
        xaxis=dict(title="", tickangle=45),
        yaxis=yaxis_layout,
        yaxis2=yaxis2_layout,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=20, r=20, b=40, l=40),
    )
    fig.update_yaxes(tickformat=",.0f")
    return fig


# ------------------------------
# Exports helpers
# ------------------------------

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.read()


