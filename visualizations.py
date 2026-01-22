from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px


def line_global_sales(long_df: pd.DataFrame):
    monthly = long_df.groupby("Month", as_index=False)["Sales"].sum()
    fig = px.line(monthly, x="Month", y="Sales", title="CA Global Mensuel")
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat="%m-%y")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def line_country_avg_per_active_shop(avg_df: pd.DataFrame):
    fig = px.line(
        avg_df,
        x="Month",
        y="avg_per_shop",
        color="Country",
        title="CA mensuel moyen par boutique active (par pays)",
    )
    fig.update_layout(hovermode="x unified", yaxis_title="CA moyen par boutique (USD)")
    fig.update_xaxes(tickformat="%m-%y")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def heatmap_top5_shops(long_df: pd.DataFrame, country: str, last_n: int = 12, top_n: int = 5):
    dfc = long_df[long_df["Country"] == country].copy()
    if dfc.empty:
        return px.imshow(np.zeros((1, 1)), title="Aucune donnée")
    months = sorted(dfc["Month"].dropna().unique().tolist())
    months = months[-last_n:] if len(months) >= last_n else months
    last_n_months = dfc[dfc["Month"].isin(months)]
    totals = last_n_months.groupby("Outlet", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(top_n)
    top_boutiques = last_n_months[last_n_months["Outlet"].isin(totals["Outlet"])].copy()
    pivot = top_boutiques.pivot_table(index="Outlet", columns="Month", values="Sales", aggfunc="sum").fillna(0.0)
    fig = px.imshow(
        pivot.values,
        labels=dict(x="Mois", y="Boutique", color="Ventes (USD)"),
        x=[m.strftime("%m-%y") for m in pivot.columns],
        y=pivot.index.tolist(),
        aspect="auto",
        title=f"Heatmap Top {top_n} boutiques – {country} ({last_n} derniers mois)",
        color_continuous_scale="Viridis",
    )
    return fig


def decile_bars(decile_df: pd.DataFrame, decile_label: str):
    if decile_df.empty:
        return px.bar(pd.DataFrame({"Outlet": [], "AnnualSales": []}), x="Outlet", y="AnnualSales", title=f"{decile_label} – Aucune donnée")
    fig = px.bar(decile_df, x="Outlet", y="AnnualSales", color="Year", barmode="group", title=f"{decile_label} par année")
    # Format y tick labels with thousands separator and no decimals
    fig.update_yaxes(tickformat=",.0f")
    return fig


def forecast_chart(forecast_df: pd.DataFrame, country: Optional[str] = None):
    df = forecast_df.copy()
    if country:
        df = df[df["Country"] == country]
    fig = px.line(df, x="Month", y="Sales", color="Scenario", facet_row="Country", title="Prévisions 2025–2026")
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat="%m-%y")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def country_share_bar(long_df: pd.DataFrame):
    if long_df.empty:
        return px.bar(pd.DataFrame({"Country": [], "Share": []}), x="Country", y="Share", title="Poids pays dans le CA global (%)")
    # Use the currently filtered period to compute dynamic share
    period = long_df.groupby("Country", as_index=False)["Sales"].sum()
    total = period["Sales"].sum()
    # Use ratio 0..1 so that percent tickformat renders correctly
    period["Share"] = np.where(total > 0, period["Sales"] / total, 0.0)
    fig = px.bar(
        period.sort_values("Share", ascending=False),
        x="Country",
        y="Share",
        title="Poids pays dans le CA global (%) – Période sélectionnée",
    )
    fig.update_yaxes(tickformat=",.0%")
    return fig


def bar_latest_growth(growth_df: pd.DataFrame, entity: str):
    if growth_df.empty:
        return px.bar(pd.DataFrame({entity: [], "YoY": []}), x=entity, y="YoY", title="Variation YoY – Aucune donnée")
    latest = growth_df.sort_values("Month").groupby(entity).tail(1)
    latest = latest.replace([np.inf, -np.inf], np.nan)
    latest = latest.dropna(subset=["YoY"]) if latest["YoY"].notna().any() else latest
    fig = px.bar(
        latest.sort_values("YoY", ascending=False),
        x=entity,
        y="YoY",
        title=f"Variation YoY la plus récente par {entity.lower()}",
    )
    fig.update_yaxes(tickformat=",.0%")
    return fig


def heatmap_ecart_vs_attendu(
    fused_df: pd.DataFrame,
    countries: Optional[List[str]] = None,
    zmin: float = -40.0,
    zmax: float = 40.0,
    color_scale: str = "RdYlGn",
):
    df = fused_df.copy()
    if countries:
        df = df[df["Country"].isin(countries)]
    if df.empty:
        return px.imshow(np.zeros((1, 1)), title="Aucune donnée")
    
    # Format month for better readability
    df["MonthStr"] = df["Month"].dt.strftime("%m-%Y")
    
    # Create pivot table
    pivot = df.pivot_table(index="Country", columns="MonthStr", values="Ecart_%_vs_attendu", aggfunc="mean")
    
    # Ensure columns sorted by date order
    try:
        sorted_cols = sorted(pivot.columns, key=lambda s: pd.to_datetime(s, format="%m-%Y"))
        pivot = pivot[sorted_cols]
    except Exception:
        pass
    
    # Format values for better readability
    pivot_formatted = pivot.round(1)
    
    fig = px.imshow(
        pivot_formatted.values,
        labels=dict(x="Mois", y="Pays", color="Écart % vs attendu"),
        x=pivot_formatted.columns.tolist(),
        y=pivot_formatted.index.tolist(),
        aspect="auto",
        color_continuous_scale=color_scale,
        zmin=zmin,
        zmax=zmax,
        title="🎯 Heatmap - Sur/sous-performance vs attendu (CA)",
    )
    
    # Add text annotations to show values on cells
    for i in range(len(pivot_formatted.index)):
        for j in range(len(pivot_formatted.columns)):
            value = pivot_formatted.iloc[i, j]
            if not pd.isna(value):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{value:.1f}%",
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1
                )
    
    # Improve layout and readability
    fig.update_layout(
        xaxis_title="📅 Mois (MM-YYYY)",
        yaxis_title="🌍 Pays",
        coloraxis_colorbar=dict(
            title="Écart % vs attendu",
            titleside="right",
            tickformat=".1f",
            ticksuffix="%"
        ),
        height=500,
        margin=dict(l=100, r=100, t=100, b=100)
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=-45)
    
    # Add hover template for better information
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>" +
                     "Mois: %{x}<br>" +
                     "Écart: %{z:.1f}%<br>" +
                     "<extra></extra>"
    )
    
    return fig


def bubble_ca_opportunity_risk(
    fused_df: pd.DataFrame,
    min_abs_z: float = 0.0,
    min_sales: float = 0.0,
    only_signals: Optional[List[str]] = None,
):
    df = fused_df.copy()
    if min_abs_z > 0:
        df = df[df["z"].abs() >= min_abs_z]
    if min_sales > 0:
        df = df[df["Sales"] >= min_sales]
    if only_signals:
        df = df[df["Signal_CA"].isin(only_signals)]
    if df.empty:
        return px.scatter(pd.DataFrame({"z": [], "seasonality_factor": [], "Sales": [], "Signal_CA": []}), x="z", y="seasonality_factor", title="Aucune donnée")
    
    # Create a more interpretable color scheme
    color_map = {
        "Opportunité": "#00FF00",  # Bright green for opportunities
        "Risque": "#FF0000",       # Bright red for risks
        "Observation": "#FFA500"   # Orange for neutral
    }
    
    # Map colors to signals
    df["color"] = df["Signal_CA"].map(color_map)
    
    fig = px.scatter(
        df,
        x="z",
        y="seasonality_factor",
        size="Sales",
        color="Signal_CA",
        color_discrete_map=color_map,
        hover_data={
            "Country": True,
            "Month_Str": df["Month"].dt.strftime("%m-%Y"),
            "Sales": ":,",
            "Ecart_%_vs_attendu": ".1f",
            "z": ".2f",
        },
        title="🎯 Opportunités et Risques (CA vs Attendu) - Focus PE",
        size_max=50,  # Control maximum bubble size
    )
    
    # Improve layout and readability
    fig.update_layout(
        xaxis_title="📊 Z-Score (Écart vs moyenne historique)",
        yaxis_title="📈 Facteur de saisonnalité",
        height=500,
        margin=dict(l=100, r=100, t=100, b=100),
        legend=dict(
            title="Signal CA",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add reference lines and zones for better interpretation
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Saisonnalité normale (1.0)")
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Performance moyenne (z=0)")
    
    # Add zones for better interpretation
    fig.add_hrect(y0=0.95, y1=1.05, fillcolor="lightgray", opacity=0.2, 
                  annotation_text="Zone normale", annotation_position="top left")
    
    # Improve hover template
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                     "Mois: %{customdata[1]}<br>" +
                     "CA: %{customdata[2]} USD<br>" +
                     "Écart: %{customdata[3]}%<br>" +
                     "Z-Score: %{customdata[4]}<br>" +
                     "Saisonnalité: %{y:.3f}<br>" +
                     "<extra></extra>"
    )
    
    # Update y-axis range for better visibility
    y_min = max(0.5, df["seasonality_factor"].min() * 0.9)
    y_max = min(2.0, df["seasonality_factor"].max() * 1.1)
    fig.update_yaxes(range=[y_min, y_max])
    
    return fig


