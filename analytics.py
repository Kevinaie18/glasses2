from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def compute_growth_metrics(long_df: pd.DataFrame, entity: str = "Country", by_country: bool = False) -> pd.DataFrame:
    if entity not in {"Country", "Outlet"}:
        raise ValueError("entity must be 'Country' or 'Outlet'")
    df = long_df.copy()
    # Grouping keys: for Outlet, optionally include Country to compute per-country outlet growth
    if entity == "Country":
        group_keys = ["Country", "Month"]
        change_keys = ["Country"]
    else:
        group_keys = (["Country", "Outlet", "Month"] if by_country else ["Outlet", "Month"])  # type: ignore
        change_keys = (["Country", "Outlet"] if by_country else ["Outlet"])  # type: ignore
    df = df.groupby(group_keys, as_index=False)["Sales"].sum()
    df.sort_values(group_keys, inplace=True)
    df["MoM"] = df.groupby(change_keys)["Sales"].pct_change(periods=1)
    df["YoY"] = df.groupby(change_keys)["Sales"].pct_change(periods=12)
    return df


@st.cache_data(show_spinner=False)
def compute_top_bottom_deciles_by_year(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = long_df.copy()
    annual = df.groupby(["Year", "Outlet"], as_index=False)["Sales"].sum().rename(columns={"Sales": "AnnualSales"})
    result_top: List[pd.DataFrame] = []
    result_bottom: List[pd.DataFrame] = []
    for year, g in annual.groupby("Year"):
        threshold_top = np.quantile(g["AnnualSales"], 0.9) if len(g) > 0 else np.nan
        threshold_bottom = np.quantile(g["AnnualSales"], 0.1) if len(g) > 0 else np.nan
        result_top.append(g[g["AnnualSales"] >= threshold_top].assign(Decile="Top 10%"))
        result_bottom.append(g[g["AnnualSales"] <= threshold_bottom].assign(Decile="Bottom 10%"))
    top_df = pd.concat(result_top, ignore_index=True) if result_top else pd.DataFrame()
    bottom_df = pd.concat(result_bottom, ignore_index=True) if result_bottom else pd.DataFrame()
    return top_df, bottom_df


@st.cache_data(show_spinner=False)
def compute_top_bottom_deciles_for_period(long_df: pd.DataFrame, by_country: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Top/Bottom 10% based on total Sales over the provided filtered period.

    If by_country is True, compute deciles within each country; else global.
    """
    df = long_df.copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if by_country:
        grouped = df.groupby(["Country", "Outlet"], as_index=False)["Sales"].sum().rename(columns={"Sales": "PeriodSales"})
        top_parts: List[pd.DataFrame] = []
        bot_parts: List[pd.DataFrame] = []
        for country, g in grouped.groupby("Country"):
            if g.empty:
                continue
            t = np.quantile(g["PeriodSales"], 0.9) if len(g) > 0 else np.nan
            b = np.quantile(g["PeriodSales"], 0.1) if len(g) > 0 else np.nan
            top_parts.append(g[g["PeriodSales"] >= t].assign(Decile="Top 10%"))
            bot_parts.append(g[g["PeriodSales"] <= b].assign(Decile="Bottom 10%"))
        return (
            pd.concat(top_parts, ignore_index=True) if top_parts else pd.DataFrame(),
            pd.concat(bot_parts, ignore_index=True) if bot_parts else pd.DataFrame(),
        )
    else:
        grouped = df.groupby(["Outlet"], as_index=False)["Sales"].sum().rename(columns={"Sales": "PeriodSales"})
        if grouped.empty:
            return pd.DataFrame(), pd.DataFrame()
        t = np.quantile(grouped["PeriodSales"], 0.9)
        b = np.quantile(grouped["PeriodSales"], 0.1)
        return (
            grouped[grouped["PeriodSales"] >= t].assign(Decile="Top 10%"),
            grouped[grouped["PeriodSales"] <= b].assign(Decile="Bottom 10%"),
        )


@st.cache_data(show_spinner=False)
def detect_probable_closures(long_df: pd.DataFrame, months_without_sales: int = 3) -> pd.DataFrame:
    df = long_df.copy()
    df.sort_values(["Outlet", "Month"], inplace=True)
    
    results = []
    for outlet, g in df.groupby("Outlet"):
        # Check if this outlet ever had sales > 0 (was active before)
        had_sales_before = (g["Sales"] > 0).any()
        
        if not had_sales_before:
            # Boutique qui n'a jamais eu de CA - pas considérée comme fermée
            continue
        
        # Boutique qui avait du CA avant - analyser après le premier mois actif
        first_active_idx = g.index[g["Sales"] > 0][0]
        post_open = g.loc[first_active_idx + 1 :].copy()
        post_open["no_sales"] = post_open["Sales"] <= 0

        # Compute consecutive zero-sales streaks after opening
        def max_consecutive_zero(s: pd.Series) -> int:
            max_streak = 0
            current = 0
            for v in s.tolist():
                if v:  # no_sales = True
                    current += 1
                    max_streak = max(max_streak, current)
                else:
                    current = 0
            return max_streak

        max_zero_streak = max_consecutive_zero(post_open["no_sales"]) if not post_open.empty else 0

        # Only include boutiques at risk
        if max_zero_streak >= months_without_sales:
            # Get country information
            country = g["Country"].iloc[0] if not g.empty else "N/A"
            
            # Get last active month and last sales amount
            last_active = g[g["Sales"] > 0].iloc[-1] if len(g[g["Sales"] > 0]) > 0 else None
            last_active_month = last_active["Month"] if last_active is not None else "N/A"
            last_sales = last_active["Sales"] if last_active is not None else 0
            
            # Determine closure reason and risk level
            if max_zero_streak >= 12:
                risk_level = "🔴 CRITIQUE"
                closure_reason = "Fermeture probable (≥12 mois sans CA après ouverture)"
            elif max_zero_streak >= 6:
                risk_level = "🟠 ÉLEVÉ"
                closure_reason = "Risque de fermeture (≥6 mois sans CA après ouverture)"
            else:
                risk_level = "🟡 MODÉRÉ"
                closure_reason = f"Inactivité (≥{months_without_sales} mois sans CA après ouverture)"

            results.append({
                "Pays": country,
                "Boutique": outlet,
                "Niveau de risque": risk_level,
                "Mois sans CA": max_zero_streak,
                "Dernier CA actif": f"{last_sales:,.0f}".replace(",", " ") if last_sales > 0 else "N/A",
                "Dernier mois actif": last_active_month.strftime("%m-%Y") if last_active_month != "N/A" else "N/A",
                "Raison": closure_reason
            })
    
    # Sort by risk level (critical first, then high, then moderate)
    if results:
        risk_order = {"🔴 CRITIQUE": 1, "🟠 ÉLEVÉ": 2, "🟡 MODÉRÉ": 3}
        results.sort(key=lambda x: risk_order.get(x["Niveau de risque"], 4))
    
    return pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def detect_cruise_speed(
    long_df: pd.DataFrame, stability_threshold: float = 0.10, consecutive_months: int = 3
) -> pd.DataFrame:
    df = long_df.copy()
    df.sort_values(["Outlet", "Month"], inplace=True)
    results: List[Dict] = []
    for outlet, g in df.groupby("Outlet"):
        g = g.set_index("Month").asfreq("MS").fillna(0.0).reset_index()
        g["ma3"] = g["Sales"].rolling(window=3, min_periods=3).mean()
        g["ma3_change"] = g["ma3"].pct_change()
        # Find first index where abs change <= threshold for N consecutive months
        achieved_idx: Optional[int] = None
        consecutive = 0
        for idx in range(len(g)):
            change = g.loc[idx, "ma3_change"]
            if not np.isnan(change) and abs(change) <= stability_threshold:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= consecutive_months:
                achieved_idx = idx
                break
        if achieved_idx is not None:
            cruise_start = g.loc[achieved_idx, "Month"]
            cruise_avg = g.loc[achieved_idx:, "ma3"].mean()
            results.append(
                {"Outlet": outlet, "cruise_start": cruise_start, "cruise_avg": cruise_avg, "achieved": True}
            )
        else:
            results.append({"Outlet": outlet, "cruise_start": pd.NaT, "cruise_avg": np.nan, "achieved": False})
    return pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def compute_kpis(long_df: pd.DataFrame) -> Dict[str, float]:
    """Compute executive KPIs aligned with UI expectations.

    Returns keys:
    - total_revenue: LTM revenue (sum over last 12 months window)
    - ltm_growth: Growth of last 12 months vs prior 12 months
    - mom_growth: Latest MoM percentage
    - yoy_growth: Latest YoY percentage
    - avg_revenue_per_shop: Latest month total divided by active shops
    - top_20_percent_share: Share of revenue from top 20% outlets over last 12 months
    """
    df = long_df.copy()
    if df.empty:
        return {
            "total_revenue": 0.0,
            "ltm_growth": np.nan,
            "mom_growth": np.nan,
            "yoy_growth": np.nan,
            "avg_revenue_per_shop": np.nan,
            "top_20_percent_share": np.nan,
        }

    # Aggregate monthly
    monthly = df.groupby("Month", as_index=False)["Sales"].sum().sort_values("Month")
    max_month = monthly["Month"].max()
    window_start = max_month - pd.DateOffset(months=11)
    last12 = monthly[monthly["Month"] >= window_start] if not pd.isna(max_month) else monthly
    prev12_start = window_start - pd.DateOffset(months=12)
    prev12 = monthly[(monthly["Month"] >= prev12_start) & (monthly["Month"] < window_start)] if not pd.isna(max_month) else pd.DataFrame(columns=monthly.columns)

    total_revenue = float(last12["Sales"].sum()) if not last12.empty else float(monthly["Sales"].sum())
    ltm_growth = np.nan
    if not last12.empty and not prev12.empty:
        prev_sum = float(prev12["Sales"].sum())
        ltm_growth = ((total_revenue - prev_sum) / prev_sum) * 100.0 if prev_sum > 0 else np.nan

    # MoM and YoY latest
    monthly["MoM"] = monthly["Sales"].pct_change(1) * 100.0
    monthly["YoY"] = monthly["Sales"].pct_change(12) * 100.0
    mom_growth = float(monthly["MoM"].dropna().iloc[-1]) if monthly["MoM"].dropna().size else np.nan
    yoy_growth = float(monthly["YoY"].dropna().iloc[-1]) if monthly["YoY"].dropna().size else np.nan

    # Avg revenue per active shop at latest month
    latest_month = max_month
    latest_df = df[df["Month"] == latest_month]
    if latest_df.empty:
        avg_rev_per_shop = np.nan
    else:
        # Active shop if Sales > 0 in latest month
        active_shops = latest_df.groupby("Outlet", as_index=False)["Sales"].sum()
        num_active = int((active_shops["Sales"] > 0).sum())
        total_latest = float(active_shops["Sales"].sum())
        avg_rev_per_shop = (total_latest / num_active) if num_active > 0 else np.nan

    # Top 20% share over last 12 months
    window_df = df[df["Month"] >= window_start] if not pd.isna(max_month) else df
    by_outlet = window_df.groupby("Outlet", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    total_window = float(by_outlet["Sales"].sum()) if not by_outlet.empty else 0.0
    top_20_percent_share = np.nan
    if not by_outlet.empty and total_window > 0:
        cutoff = max(int(np.ceil(0.2 * len(by_outlet))), 1)
        top_sum = float(by_outlet.head(cutoff)["Sales"].sum())
        top_20_percent_share = (top_sum / total_window) * 100.0

    return {
        "total_revenue": total_revenue,
        "ltm_growth": ltm_growth,
        "mom_growth": mom_growth,
        "yoy_growth": yoy_growth,
        "avg_revenue_per_shop": float(avg_rev_per_shop) if not isinstance(avg_rev_per_shop, float) else avg_rev_per_shop,
        "top_20_percent_share": top_20_percent_share,
    }


def _label_recommendation(
    relative_to_cruise: float,
    yoy_growth: float,
    volatility_cv: float,
    months_under_cruise: int,
) -> str:
    # Simplified rule-set aligned with PRD wording
    low_volatility = volatility_cv <= 0.25 if not np.isnan(volatility_cv) else False
    negative_trend = yoy_growth < -0.05 if not np.isnan(yoy_growth) else False
    strong_trend = yoy_growth > 0.15 if not np.isnan(yoy_growth) else False

    if relative_to_cruise >= 0.10 and strong_trend and low_volatility:
        return "Renforcer"
    if abs(relative_to_cruise) <= 0.10 and not negative_trend:
        return "Maintenir"
    if relative_to_cruise <= -0.10 and months_under_cruise >= 2 and strong_trend:
        return "Corriger"
    if relative_to_cruise <= -0.10 and months_under_cruise >= 6:
        return "Relocaliser"
    if relative_to_cruise <= -0.10 and months_under_cruise >= 9 and negative_trend:
        return "Fermer"
    return "Maintenir"


@st.cache_data(show_spinner=False)
def build_recommendations(long_df: pd.DataFrame) -> pd.DataFrame:
    # Pre-computations
    cruise = detect_cruise_speed(long_df)
    growth_outlet = compute_growth_metrics(long_df, entity="Outlet")
    # Volatility per outlet over last 12 months
    vol = (
        long_df.groupby(["Outlet", "Month"], as_index=False)["Sales"].sum().groupby("Outlet")["Sales"]
        .rolling(12, min_periods=6)
        .apply(lambda s: s.std() / s.mean() if s.mean() else np.nan, raw=False)
        .reset_index()
        .rename(columns={"level_1": "row_idx", "Sales": "volatility_cv"})
    )
    # Current snapshot per outlet: last available month metrics
    latest_sales = long_df.sort_values("Month").groupby("Outlet").tail(1)[["Outlet", "Month", "Sales"]]

    # Merge auxiliary data
    latest = latest_sales.merge(cruise, on="Outlet", how="left")
    # Get YoY for outlet at last month
    yoy_latest = (
        growth_outlet.dropna(subset=["YoY"])  # ensure meaningful
        .sort_values(["Outlet", "Month"]).groupby("Outlet").tail(1)[["Outlet", "YoY"]]
    )
    latest = latest.merge(yoy_latest, on="Outlet", how="left")
    # Volatility latest value
    vol_latest = vol.sort_values("row_idx").groupby("Outlet").tail(1)[["Outlet", "volatility_cv"]]
    latest = latest.merge(vol_latest, on="Outlet", how="left")

    # Compute relative_to_cruise and months_under_cruise
    def compute_months_under(row: pd.Series) -> int:
        # Approximate months under cruise if cruise achieved; else 0
        # Count last consecutive months where Sales < cruise_avg
        outlet = row["Outlet"]
        subset = long_df[long_df["Outlet"] == outlet].sort_values("Month")
        under = 0
        cruise_avg = row.get("cruise_avg", np.nan)
        if np.isnan(cruise_avg):
            return 0
        for _, r in subset.tail(24).iterrows():  # last 24 months
            if r["Sales"] < cruise_avg * 0.9:
                under += 1
            else:
                under = 0
        return under

    latest["relative_to_cruise"] = (latest["Sales"] - latest["cruise_avg"]) / latest["cruise_avg"]
    latest["relative_to_cruise"] = latest["relative_to_cruise"].replace([np.inf, -np.inf], np.nan)
    latest["months_under_cruise"] = latest.apply(compute_months_under, axis=1)
    latest["Recommendation"] = latest.apply(
        lambda r: _label_recommendation(
            relative_to_cruise=float(r.get("relative_to_cruise", np.nan)),
            yoy_growth=float(r.get("YoY", np.nan)),
            volatility_cv=float(r.get("volatility_cv", np.nan)),
            months_under_cruise=int(r.get("months_under_cruise", 0)),
        ),
        axis=1,
    )
    return latest[[
        "Outlet",
        "Month",
        "Sales",
        "cruise_start",
        "cruise_avg",
        "relative_to_cruise",
        "months_under_cruise",
        "YoY",
        "volatility_cv",
        "Recommendation",
    ]]


@st.cache_data(show_spinner=False)
def compute_ytd_by_country(long_df: pd.DataFrame, months_span: List[int] = list(range(1, 8))) -> pd.DataFrame:
    df = long_df.copy()
    df["month_num"] = df["Month"].dt.month
    ytd = df[df["month_num"].isin(months_span)].groupby(["Country", "Year"], as_index=False)["Sales"].sum()
    pivot = ytd.pivot_table(index=["Country"], columns="Year", values="Sales", aggfunc="sum", fill_value=0.0)
    pivot = pivot.reset_index()
    return pivot


@st.cache_data(show_spinner=False)
def compute_seasonality_factors_country(long_df: pd.DataFrame, lookback_months: int = 24) -> pd.DataFrame:
    df = long_df.sort_values("Month").copy()
    if df.empty:
        return pd.DataFrame(columns=["Country", "month_of_year", "seasonality_factor"])
    max_month = df["Month"].max()
    window_start = max_month - pd.DateOffset(months=lookback_months - 1)
    recent = df[df["Month"] >= window_start].copy()
    recent["moy"] = recent["Month"].dt.month
    by_country = recent.groupby(["Country"], as_index=False)["Sales"].mean().rename(columns={"Sales": "mean_overall"})
    by_moy = recent.groupby(["Country", "moy"], as_index=False)["Sales"].mean().rename(columns={"Sales": "mean_moy"})
    merged = by_moy.merge(by_country, on="Country", how="left")
    merged["seasonality_factor"] = merged.apply(
        lambda r: (r["mean_moy"] / r["mean_overall"]) if r["mean_overall"] and not np.isnan(r["mean_overall"]) else np.nan,
        axis=1,
    )
    merged = merged.rename(columns={"moy": "month_of_year"})
    return merged[["Country", "month_of_year", "seasonality_factor"]]


@st.cache_data(show_spinner=False)
def detect_anomalies_country(long_df: pd.DataFrame, lookback_months: int = 24, z_thresh: float = 2.0) -> pd.DataFrame:
    df = long_df.copy()
    df = df.groupby(["Country", "Month"], as_index=False)["Sales"].sum().sort_values(["Country", "Month"]) 
    max_month = df["Month"].max()
    if pd.isna(max_month):
        return pd.DataFrame(columns=["Country", "Month", "Sales", "z" , "is_anomaly"]) 
    window_start = max_month - pd.DateOffset(months=lookback_months - 1)
    recent = df[df["Month"] >= window_start].copy()
    anomalies_list = []
    for country, g in recent.groupby("Country"):
        mu = g["Sales"].mean()
        sigma = g["Sales"].std()
        if sigma and sigma > 0:
            z = (g["Sales"] - mu) / sigma
        else:
            z = pd.Series([0]*len(g), index=g.index)
        temp = g.copy()
        temp["z"] = z
        temp["is_anomaly"] = temp["z"].abs() >= z_thresh
        anomalies_list.append(temp)
    return pd.concat(anomalies_list, ignore_index=True) if anomalies_list else pd.DataFrame()


@st.cache_data(show_spinner=False)
def compute_cagr_country_and_global(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    df = long_df.copy()
    country_month = df.groupby(["Country", "Month"], as_index=False)["Sales"].sum().sort_values(["Country", "Month"]) 
    cagr_rows = []
    for country, g in country_month.groupby("Country"):
        first = g.iloc[0]["Sales"]
        last = g.iloc[-1]["Sales"]
        n_months = max(len(g), 1)
        n_years = n_months / 12.0
        if first > 0 and n_years > 0:
            cagr = (last / first) ** (1.0 / n_years) - 1.0
        else:
            cagr = np.nan
        cagr_rows.append({"Country": country, "CAGR": cagr})
    cagr_df = pd.DataFrame(cagr_rows)
    # Global CAGR
    global_month = df.groupby("Month", as_index=False)["Sales"].sum().sort_values("Month")
    if not global_month.empty:
        g_first = global_month.iloc[0]["Sales"]
        g_last = global_month.iloc[-1]["Sales"]
        g_years = max(len(global_month) / 12.0, 1e-6)
        global_cagr = (g_last / g_first) ** (1.0 / g_years) - 1.0 if g_first > 0 else np.nan
    else:
        global_cagr = np.nan
    return cagr_df, float(global_cagr) if not np.isnan(global_cagr) else np.nan


@st.cache_data(show_spinner=False)
def cruise_aggregates_by_country(long_df: pd.DataFrame) -> pd.DataFrame:
    # For each outlet compute months until cruise_start from first non-zero sales
    cruise = detect_cruise_speed(long_df)
    df = long_df.copy()
    df = df.sort_values(["Outlet", "Month"]).copy()
    first_active_month = (
        df[df["Sales"] > 0].groupby("Outlet", as_index=False)["Month"].min().rename(columns={"Month": "first_active"})
    )
    aux = cruise.merge(first_active_month, on="Outlet", how="left")
    aux["months_to_cruise"] = (aux["cruise_start"] - aux["first_active"]).dt.days.div(30.44)
    aux["months_to_cruise"] = aux["months_to_cruise"].where(aux["achieved"], np.nan)
    outlet_country = df.groupby("Outlet", as_index=False)["Country"].first()
    aux = aux.merge(outlet_country, on="Outlet", how="left")
    agg = aux.groupby("Country", as_index=False).agg(
        pct_achieved=("achieved", lambda s: float(s.sum()) / float(len(s)) if len(s) else np.nan),
        mean_months_to_cruise=("months_to_cruise", "mean"),
        median_months_to_cruise=("months_to_cruise", "median"),
    )
    return agg


@st.cache_data(show_spinner=False)
def country_alerts(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    # Decline >= 3 months (MoM negative three consecutive)
    growth = compute_growth_metrics(df, entity="Country").sort_values(["Country", "Month"]) 
    alerts = []
    for country, g in growth.groupby("Country"):
        mom = g["MoM"].fillna(0).values.tolist()
        decline3 = False
        streak = 0
        for v in mom[-12:]:
            if v < 0:
                streak += 1
                if streak >= 3:
                    decline3 = True
            else:
                streak = 0
        # Volatility abnormal over last 12 months
        last12 = df[df["Country"] == country].groupby("Month", as_index=False)["Sales"].sum().sort_values("Month").tail(12)
        vol = (last12["Sales"].std() / last12["Sales"].mean()) if not last12.empty and last12["Sales"].mean() else np.nan
        # Inactivity
        latest = last12.iloc[-1]["Sales"] if not last12.empty else 0.0
        inactivity = latest <= 0 or (last12["Sales"] <= 0).rolling(3, min_periods=3).sum().fillna(0).ge(3).any()
        alerts.append({
            "Country": country,
            "decline_3m": decline3,
            "volatility_cv": vol,
            "volatility_flag": bool(vol > 0.20) if not np.isnan(vol) else False,
            "inactivity_flag": bool(inactivity),
        })
    return pd.DataFrame(alerts)


@st.cache_data(show_spinner=False)
def cruise_status_by_country(long_df: pd.DataFrame) -> pd.DataFrame:
    """For each country, compute shops at/below cruise and percentages."""
    cruise = detect_cruise_speed(long_df)
    if cruise.empty:
        return pd.DataFrame(columns=["Country", "total_shops", "achieved_count", "below_count", "pct_below"])
    df = long_df.sort_values(["Outlet", "Month"])  # preserve outlet-country mapping
    outlet_country = df.groupby("Outlet", as_index=False)["Country"].first()
    joined = cruise.merge(outlet_country, on="Outlet", how="left")
    stats = joined.groupby("Country", as_index=False).agg(
        total_shops=("Outlet", "count"),
        achieved_count=("achieved", "sum"),
    )
    stats["below_count"] = stats["total_shops"] - stats["achieved_count"]
    stats["pct_below"] = stats.apply(
        lambda r: float(r["below_count"]) / float(r["total_shops"]) if r["total_shops"] else np.nan, axis=1
    )
    return stats


@st.cache_data(show_spinner=False)
def build_ca_opportunity_risk_dataset(
    long_df: pd.DataFrame,
    lookback_months: int = 24,
    seasonality_df: Optional[pd.DataFrame] = None,
    anomalies_df: Optional[pd.DataFrame] = None,
    countries_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fuse seasonality factors with anomalies to compute expected sales, percent gap, and CA signal.

    Returns columns: Country, Month, Sales, seasonality_factor, interpretation, z, severity,
    Ecart_%_vs_attendu, Signal_CA
    """
    if long_df is None or long_df.empty:
        return pd.DataFrame(
            columns=[
                "Country",
                "Month",
                "Sales",
                "seasonality_factor",
                "interpretation",
                "z",
                "severity",
                "Ecart_%_vs_attendu",
                "Signal_CA",
            ]
        )

    df = long_df.copy()
    df = df.groupby(["Country", "Month"], as_index=False)["Sales"].sum().sort_values(["Country", "Month"])

    # Restrict to lookback window
    max_month = df["Month"].max()
    if pd.isna(max_month):
        return pd.DataFrame()
    window_start = max_month - pd.DateOffset(months=lookback_months - 1)
    recent = df[df["Month"] >= window_start].copy()

    # Seasonality
    if seasonality_df is None or seasonality_df.empty:
        seasonality_df = compute_seasonality_factors_country(long_df, lookback_months=lookback_months)
    seas = seasonality_df.copy()
    seas = seas.rename(columns={"month_of_year": "moy"})

    # Country monthly overall mean over the same window (expected baseline)
    country_mean = recent.groupby("Country", as_index=False)["Sales"].mean().rename(columns={"Sales": "mean_overall"})

    # Anomalies
    if anomalies_df is None or anomalies_df.empty:
        anomalies_df = detect_anomalies_country(long_df, lookback_months=lookback_months)
    anom = anomalies_df.copy()

    # Harmonize date linkage
    anom["moy"] = anom["Month"].dt.month

    # Merge seasonality factor onto anomalies using Country + month-of-year
    fused = anom.merge(seas, on=["Country", "moy"], how="left")
    fused = fused.merge(country_mean, on="Country", how="left")

    # Interpretation from seasonality_factor if not provided
    def _interpret(f: float) -> str:
        if pd.isna(f):
            return "N/A"
        if f > 1.05:
            return "Pic"
        if f < 0.95:
            return "Creux"
        return "Normal"

    fused["interpretation"] = fused.get("interpretation", pd.Series(index=fused.index, dtype="object"))
    fused.loc[fused["interpretation"].isna() | (fused["interpretation"] == ""), "interpretation"] = fused["seasonality_factor"].apply(_interpret)

    # Expected sales and percent gap
    fused["Sales_attendu"] = fused.apply(
        lambda r: (r["seasonality_factor"] * r["mean_overall"]) if pd.notna(r.get("seasonality_factor", np.nan)) and pd.notna(r.get("mean_overall", np.nan)) else np.nan,
        axis=1,
    )
    fused["Ecart_%_vs_attendu"] = fused.apply(
        lambda r: ((r["Sales"] / r["Sales_attendu"]) - 1.0) * 100.0 if pd.notna(r.get("Sales_attendu", np.nan)) and r.get("Sales_attendu", 0) > 0 else np.nan,
        axis=1,
    )

    # Severity friendly labels if missing
    if "severity" not in fused.columns:
        fused["severity"] = fused["z"].apply(lambda x: "Critique" if abs(x) > 3 else "Modéré" if abs(x) > 2.5 else "Léger")

    # Signal logic
    def _signal(row: pd.Series) -> str:
        ecart = row.get("Ecart_%_vs_attendu", np.nan)
        f = row.get("seasonality_factor", np.nan)
        if pd.isna(ecart) or pd.isna(f):
            return "Observation"
        if ecart > 10.0 and f < 0.95:
            return "Opportunité"
        if ecart < -10.0 and f > 1.05:
            return "Risque"
        return "Observation"

    fused["Signal_CA"] = fused.apply(_signal, axis=1)

    # Filter by countries if provided
    if countries_filter:
        fused = fused[fused["Country"].isin(countries_filter)]

    # Keep only requested columns and sort by absolute gap desc
    keep_cols = [
        "Country",
        "Month",
        "Sales",
        "seasonality_factor",
        "interpretation",
        "z",
        "severity",
        "Ecart_%_vs_attendu",
        "Signal_CA",
    ]
    fused = fused[keep_cols].copy()
    fused = fused.sort_values("Ecart_%_vs_attendu", key=lambda x: x.abs(), ascending=False)
    return fused


@st.cache_data(show_spinner=False)
def analyze_cruise_speed_by_country(long_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze cruise speed by country with detailed metrics.
    
    Returns DataFrame with columns:
    - Country: country name
    - total_shops: total number of shops in country
    - shops_at_cruise: number of shops at cruise speed
    - pct_at_cruise: percentage of shops at cruise speed
    - avg_cruise_ca_per_shop: average CA per shop when at cruise speed
    - shops_below_avg_3m: number of shops below 3-month average
    - pct_below_avg_3m: percentage of shops below 3-month average
    - avg_months_to_cruise: average months to reach cruise speed
    """
    if long_df.empty:
        return pd.DataFrame()
    
    df = long_df.copy()
    
    # Get cruise speed data
    cruise_data = detect_cruise_speed(df)
    
    # Merge with country information
    cruise_with_country = cruise_data.merge(
        df[["Outlet", "Country"]].drop_duplicates(), 
        on="Outlet", 
        how="left"
    )
    
    # Calculate time to cruise speed for each shop
    df_sorted = df.sort_values(["Outlet", "Month"])
    first_active_month = (
        df_sorted[df_sorted["Sales"] > 0].groupby("Outlet", as_index=False)["Month"].min().rename(columns={"Month": "first_active"})
    )
    cruise_with_first = cruise_data.merge(first_active_month, on="Outlet", how="left")
    cruise_with_first["months_to_cruise"] = (
        cruise_with_first["cruise_start"] - cruise_with_first["first_active"]
    ).dt.days.div(30.44)
    cruise_with_first["months_to_cruise"] = cruise_with_first["months_to_cruise"].where(
        cruise_with_first["achieved"], np.nan
    )
    
    # Merge with country information for cruise data
    cruise_with_country = cruise_with_first.merge(
        df[["Outlet", "Country"]].drop_duplicates(), 
        on="Outlet", 
        how="left"
    )
    
    # Calculate 3-month average for each shop
    df_sorted["ma3"] = df_sorted.groupby("Outlet")["Sales"].rolling(
        window=3, min_periods=3
    ).mean().reset_index(0, drop=True)
    
    # Get latest month sales for comparison
    latest_month = df["Month"].max()
    latest_sales = df[df["Month"] == latest_month][["Outlet", "Sales"]].rename(columns={"Sales": "Sales_latest"})
    latest_sales = latest_sales.merge(
        df[["Outlet", "Country"]].drop_duplicates(), 
        on="Outlet", 
        how="left"
    )
    
    # Get latest 3-month average for each shop
    latest_ma3 = df_sorted.groupby("Outlet")["ma3"].last().reset_index().rename(columns={"ma3": "ma3_latest"})
    latest_ma3 = latest_ma3.merge(
        df[["Outlet", "Country"]].drop_duplicates(), 
        on="Outlet", 
        how="left"
    )
    
    # Merge all data
    comparison = latest_sales.merge(latest_ma3, on=["Outlet", "Country"])
    
    # Calculate shops below average
    comparison["below_avg"] = comparison["Sales_latest"] < comparison["ma3_latest"]
    
    # Aggregate by country
    country_stats = []
    
    for country in df["Country"].unique():
        country_cruise = cruise_with_country[cruise_with_country["Country"] == country]
        country_comparison = comparison[comparison["Country"] == country]
        
        total_shops = len(country_comparison)
        if total_shops == 0:
            continue
            
        # Cruise speed stats
        shops_at_cruise = int(country_cruise["achieved"].sum())
        pct_at_cruise = (shops_at_cruise / total_shops * 100) if total_shops > 0 else 0
        
        # Average CA per shop at cruise speed
        cruise_shops = country_cruise[country_cruise["achieved"] == True]
        avg_cruise_ca = cruise_shops["cruise_avg"].mean() if not cruise_shops.empty else 0
        
        # Average months to reach cruise speed
        avg_months_to_cruise = country_cruise["months_to_cruise"].mean() if not country_cruise.empty else np.nan
        
        # Shops below 3-month average
        shops_below_avg = int(country_comparison["below_avg"].sum())
        pct_below_avg = (shops_below_avg / total_shops * 100) if total_shops > 0 else 0
        
        country_stats.append({
            "Country": country,
            "total_shops": total_shops,
            "shops_at_cruise": shops_at_cruise,
            "pct_at_cruise": round(pct_at_cruise, 1),
            "avg_cruise_ca_per_shop": round(avg_cruise_ca, 0) if not pd.isna(avg_cruise_ca) else 0,
            "avg_months_to_cruise": round(avg_months_to_cruise, 1) if not pd.isna(avg_months_to_cruise) else np.nan,
            "shops_below_avg_3m": shops_below_avg,
            "pct_below_avg_3m": round(pct_below_avg, 1)
        })
    
    return pd.DataFrame(country_stats).sort_values("pct_at_cruise", ascending=False)


