from __future__ import annotations

import os
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import (
    build_recommendations,
    compute_growth_metrics,
    compute_kpis,
    compute_top_bottom_deciles_by_year,
    detect_probable_closures,
    detect_cruise_speed,
    compute_ytd_by_country,
    compute_seasonality_factors_country,
    detect_anomalies_country,
    compute_cagr_country_and_global,
    cruise_aggregates_by_country,
    country_alerts,
    build_ca_opportunity_risk_dataset,
    analyze_cruise_speed_by_country,
)
from data_processing import (
    load_excel,
    to_long_format,
    get_available_filters,
    clean_and_sort_unique,
    clean_and_sort_months,
    DEFAULT_DATA_PATH,
    filter_data,
)
from forecasting import build_adjusted_scenario, build_baseline
from utils.export_utils import default_output_filename, export_excel, export_pdf_simple
from visualizations import (
    decile_bars,
    forecast_chart,
    heatmap_top5_shops,
    line_country_avg_per_active_shop,
    line_global_sales,
    country_share_bar,
    bar_latest_growth,
    heatmap_ecart_vs_attendu,
    bubble_ca_opportunity_risk,
)
from shops_views import (
    compute_view1_ca_buckets_table,
    plot_view1_ca_buckets_table,
    build_view1_ca_buckets_matrix,
    plot_view1_ca_buckets_matrix,
    compute_view2_maturity_share,
    plot_view2_maturity_share,
    compute_view3_combo,
    plot_view3_combo,
    df_to_excel_bytes,
)

# --- V1.5: Imports IA ---
from components.ai_widgets import render_ai_tab
# --- Fin imports IA ---


def load_data():
    """Loads data from DEFAULT_DATA_PATH or user uploads."""

# --- V1.5: Fonctions helper pour l'IA ---

def get_top_performers(df: pd.DataFrame, n: int = 5) -> list:
    """Retourne les n meilleures boutiques du dernier mois."""
    if df.empty or "Month" not in df.columns or "Sales" not in df.columns:
        return []
    
    latest_month = df["Month"].max()
    latest = df[df["Month"] == latest_month]
    
    if latest.empty:
        return []
    
    top = latest.nlargest(n, "Sales")
    return [
        f"{row['Outlet']} ({row['Country']}): {row['Sales']:,.0f} USD"
        for _, row in top.iterrows()
    ]


def get_underperformers(df: pd.DataFrame, n: int = 5) -> list:
    """Retourne les n boutiques les moins performantes (actives) du dernier mois."""
    if df.empty or "Month" not in df.columns or "Sales" not in df.columns:
        return []
    
    latest_month = df["Month"].max()
    latest = df[df["Month"] == latest_month]
    
    # Exclure les boutiques à 0 (potentiellement fermées)
    active = latest[latest["Sales"] > 0]
    
    if active.empty:
        return []
    
    bottom = active.nsmallest(n, "Sales")
    return [
        f"{row['Outlet']} ({row['Country']}): {row['Sales']:,.0f} USD"
        for _, row in bottom.iterrows()
    ]


def get_country_breakdown(df: pd.DataFrame) -> dict:
    """Retourne la performance par pays pour la période."""
    if df.empty or "Country" not in df.columns or "Sales" not in df.columns:
        return {}
    
    breakdown = {}
    for country in df["Country"].unique():
        country_df = df[df["Country"] == country]
        breakdown[country] = {
            "ca": country_df["Sales"].sum(),
            "avg": country_df["Sales"].mean(),
            "shops": country_df["Outlet"].nunique() if "Outlet" in country_df.columns else 0,
        }
    
    return breakdown

# --- Fin fonctions helper IA ---


def load_data():
    """Loads data from DEFAULT_DATA_PATH or user uploads."""
    if os.path.exists(DEFAULT_DATA_PATH):
        df = load_excel()
        long_df = to_long_format(df)
        if "Month" in long_df.columns:
            long_df["Month"] = pd.to_datetime(long_df["Month"], errors="coerce")
        return long_df
    
    st.sidebar.info("Importez un fichier Excel via la barre latérale pour commencer.")
    uploaded_file = st.sidebar.file_uploader("Excel (xlsx)", type=["xlsx"], key="main_uploader")
    if uploaded_file is None:
        return None
    df = load_excel(uploaded_bytes=uploaded_file.read())
    long_df = to_long_format(df)
    if "Month" in long_df.columns:
        long_df["Month"] = pd.to_datetime(long_df["Month"], errors="coerce")
    return long_df


def sidebar_controls(long_df: pd.DataFrame) -> Dict[str, Any]:
    """Sidebar controls for filtering and configuration."""
    st.sidebar.header("🎛️ Contrôles")
    
    # Quick period filters
    st.sidebar.subheader("📅 Période d'analyse")
    period_filter = st.sidebar.radio(
        "Période rapide",
        ["LTM", "Last 6M", "Last 3M", "Personnalisée"],
        index=0,
        key="period_radio"
    )
    
    # Date range based on period filter
    if period_filter == "LTM":
        end_date = long_df["Month"].max()
        start_date = end_date - pd.DateOffset(months=12)
    elif period_filter == "Last 6M":
        end_date = long_df["Month"].max()
        start_date = end_date - pd.DateOffset(months=6)
    elif period_filter == "Last 3M":
        end_date = long_df["Month"].max()
        start_date = end_date - pd.DateOffset(months=3)
    else:  # Custom
        date_range = st.sidebar.date_input(
            "Période personnalisée",
            value=(long_df["Month"].min(), long_df["Month"].max()),
            min_value=long_df["Month"].min(),
            max_value=long_df["Month"].max(),
            key="custom_date_range"
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
        else:
            end_date = long_df["Month"].max()
            start_date = end_date - pd.DateOffset(months=12)
    
    # Quick country filters
    st.sidebar.subheader("🌍 Filtres pays")
    # Clean country data to avoid sorting errors with mixed types
    all_countries = clean_and_sort_unique(long_df["Country"])
    
    # Select all option
    if st.sidebar.button("✅ Tous les pays", key="select_all_countries"):
        st.session_state.selected_countries = all_countries
    if st.sidebar.button("❌ Aucun pays", key="clear_countries"):
        st.session_state.selected_countries = []
    
    # Initialize session state for countries if not exists
    if "selected_countries" not in st.session_state:
        st.session_state.selected_countries = all_countries
    
    # Country multiselect
    selected_countries = st.sidebar.multiselect(
        "Sélectionner les pays",
        options=all_countries,
        default=st.session_state.selected_countries,
        key="country_multiselect"
    )
    
    # Update session state
    st.session_state.selected_countries = selected_countries
    
    # If no countries selected, select all
    if not selected_countries:
        selected_countries = all_countries
        st.session_state.selected_countries = all_countries
    
    # File uploader
    st.sidebar.subheader("📁 Données")
    uploaded = st.sidebar.file_uploader("Excel (xlsx)", type=["xlsx"], key="sidebar_uploader")
    
    # Configuration sliders
    st.sidebar.subheader("⚙️ Configuration")
    closure_threshold = st.sidebar.slider(
        "Seuil fermeture (mois sans CA)", min_value=1, max_value=12, value=2, key="closure_threshold"
    )
    heatmap_months = st.sidebar.slider(
        "Derniers mois pour heatmap", min_value=3, max_value=12, value=12, key="heatmap_months"
    )
    heatmap_top_n = st.sidebar.slider(
        "Top N boutiques", min_value=3, max_value=10, value=5, key="heatmap_top_n"
    )
    
    # Forecasting parameters
    st.sidebar.subheader("🔮 Prévisions")
    growth_by_country = st.sidebar.slider(
        "Croissance par pays (%)", min_value=-50, max_value=100, value=10, key="growth_by_country"
    )
    new_shop_rate_by_country = st.sidebar.slider(
        "Taux ouverture boutiques (%)", min_value=0, max_value=50, value=5, key="new_shop_rate_by_country"
    )
    target_per_shop_by_country = st.sidebar.slider(
        "CA cible par boutique (k USD)", min_value=1, max_value=100, value=20, key="target_per_shop_by_country"
    )
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "selected_countries": selected_countries,
        "uploaded_file": uploaded,
        "closure_threshold": closure_threshold,
        "heatmap_months": heatmap_months,
        "heatmap_top_n": heatmap_top_n,
        "growth_by_country": growth_by_country,
        "new_shop_rate_by_country": new_shop_rate_by_country,
        "target_per_shop_by_country": target_per_shop_by_country,
    }


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Lapaire Dashboard",
        page_icon="👓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("👓 Lapaire Dashboard - Analyse Réseau")
    st.caption("Dashboard d'analyse des performances du réseau de boutiques")
    
    # Load data
    long_df = load_data()
    if long_df is None:
        st.error("❌ Impossible de charger les données. Vérifiez le fichier Excel.")
        return
    
    # Build sidebar controls after we have data
    controls = sidebar_controls(long_df)
    
    # Handle new file upload from sidebar
    if controls["uploaded_file"] is not None:
        working_df = load_excel(uploaded_bytes=controls["uploaded_file"].read())
        long_df = to_long_format(working_df)
        if "Month" in long_df.columns:
            long_df["Month"] = pd.to_datetime(long_df["Month"], errors="coerce")
        # Rebuild controls with new data
        controls = sidebar_controls(long_df)
    
    # Apply filters
    filtered = filter_data(
        long_df,
        countries=controls["selected_countries"],
        start_month=controls["start_date"],
        end_month=controls["end_date"],
    )
    
    # Compute KPIs
    kpis = compute_kpis(filtered)
    cruise_df = detect_cruise_speed(filtered)
    pct_at_cruise = float(cruise_df["achieved"].mean()) if not cruise_df.empty else np.nan
    closures = detect_probable_closures(filtered, months_without_sales=controls["closure_threshold"])
    probable_closures = len(closures) if not closures.empty else 0
    
    # Executive KPI ribbon
    st.markdown("---")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "💰 CA Total LTM",
            f"{kpis['total_revenue']:,.0f}".replace(",", " "),
            f"{kpis['ltm_growth']:+.1f}%",
            help="Chiffre d'affaires total sur les 12 derniers mois"
        )
    
    with col2:
        st.metric(
            "📈 Croissance MoM",
            f"{kpis['mom_growth']:+.1f}%",
            help="Croissance mois sur mois"
        )
    
    with col3:
        st.metric(
            "📊 Croissance YoY",
            f"{kpis['yoy_growth']:+.1f}%",
            help="Croissance année sur année"
        )
    
    with col4:
        st.metric(
            "🏪 CA Moyen/Boutique",
            f"{kpis['avg_revenue_per_shop']:,.0f}".replace(",", " "),
            help="Chiffre d'affaires moyen par boutique active"
        )
    
    with col5:
        st.metric(
            "🎯 Top 20% Part",
            f"{kpis['top_20_percent_share']:.1f}%",
            help="Part du chiffre d'affaires des 20% meilleures boutiques"
        )
    
    with col6:
        st.metric(
            "⚡ % à Rythme",
            f"{pct_at_cruise:.1f}%" if not np.isnan(pct_at_cruise) else "N/A",
            help="Pourcentage de boutiques au rythme de croisière"
        )
    
    st.markdown("---")
    
    # Investment memo
    st.markdown("### 📋 Mémo d'Investissement")
    st.info(f"""
    **Résumé Exécutif**: Le réseau Lapaire affiche un CA total de {kpis['total_revenue']:,.0f} USD sur les 12 derniers mois, 
    avec une croissance {kpis['yoy_growth']:+.1f}% YoY et {kpis['mom_growth']:+.1f}% MoM. 
    {probable_closures} boutiques sont identifiées comme fermetures probables. 
    Le réseau maintient {pct_at_cruise:.1f}% de ses boutiques au rythme de croisière optimal.
    """)
    
    # Tabbed interface
    tab_overview, tab_countries, tab_shops, tab_forecasts, tab_alerts, tab_ai, tab_memo = st.tabs([
        "📊 Vue d'ensemble", "🌍 Pays", "🏪 Boutiques", "🔮 Prévisions", "⚠️ Alertes", "🤖 IA", "📝 Mémo"
    ])
    
    with tab_overview:
        st.subheader("📊 Vue d'ensemble du réseau")
        
        # Remplacement par le combo CA mensuel (kUSD) & Boutiques actives
        combo_overview = compute_view3_combo(
            filtered,
            countries=controls["selected_countries"],
            start_period=controls["start_date"],
            end_period=controls["end_date"],
            active_threshold_usd=0.0,
        )
        # Axis scaling filters
        col_scale1, col_scale2, _ = st.columns([1,1,2])
        with col_scale1:
            y1_max = st.number_input("Max axe CA (USD)", min_value=0.0, value=float(combo_overview["total_sales_usd"].max()) if not combo_overview.empty else 0.0, step=1000.0, key="ov_y1")
        with col_scale2:
            y2_max = st.number_input("Max axe Boutiques", min_value=0.0, value=float(combo_overview["active_shops"].max()) if not combo_overview.empty else 0.0, step=1.0, key="ov_y2")
        plot_view3_combo(combo_overview, y1_max=y1_max, y2_max=y2_max)
        
        # View 1: Répartition par tranches de CA (déplacé depuis l'onglet Boutiques)
        st.markdown("---")
        st.subheader("📊 Répartition boutiques par tranches de CA")
        st.markdown("*Distribution des boutiques par segments de chiffre d'affaires mensuel*")
        
        # Controls for CA buckets view
        col_ca1, col_ca2 = st.columns(2)
        with col_ca1:
            ca_countries = st.multiselect(
                "Pays (vue CA)",
                options=clean_and_sort_unique(filtered["Country"]),
                default=clean_and_sort_unique(filtered["Country"]),
                key="overview_ca_countries"
            )
        
        # Compute and display CA buckets table
        ca_buckets_table = compute_view1_ca_buckets_table(
            filtered,
            countries=ca_countries,
            start_period=controls["start_date"],
            end_period=controls["end_date"],
        )
        # Matrix rotated view to match requested orientation
        ca_buckets_matrix = build_view1_ca_buckets_matrix(ca_buckets_table)
        plot_view1_ca_buckets_matrix(ca_buckets_matrix)
        
        # Export buttons
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            st.download_button(
                "📥 Exporter (CSV)",
                data=ca_buckets_matrix.to_csv(index=False).encode("utf-8"),
                file_name="repartition_tranches_ca_overview.csv",
                mime="text/csv",
            )
        with col_export2:
            st.download_button(
                "📥 Exporter (XLSX)",
                data=df_to_excel_bytes(ca_buckets_matrix, sheet_name="Repartition_CA"),
                file_name="repartition_tranches_ca_overview.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        
        st.markdown("---")
        
        # YTD comparisons with proper formatting
        ytd_by_country = compute_ytd_by_country(filtered)
        if not ytd_by_country.empty:
            st.subheader("📅 Comparaisons YTD par pays")
            # Format numbers with space separator and 2 decimals
            ytd_display = ytd_by_country.copy()
            for col in ytd_by_country.columns:
                if col != "Country" and ytd_by_country[col].dtype in ["float64", "int64"]:
                    ytd_display[col] = ytd_by_country[col].map(lambda x: f"{float(x):,.2f}".replace(",", " ") if pd.notna(x) else "")
            st.dataframe(ytd_display, use_container_width=True, hide_index=True)
        
        # Seasonality and anomalies
        seasonality = compute_seasonality_factors_country(filtered)
        anomalies = detect_anomalies_country(filtered)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Saisonnalité par pays")
            if not seasonality.empty:
                # Pivot table: countries as rows, months as columns
                seasonality_pivot = seasonality.pivot_table(
                    index="Country", 
                    columns="month_of_year", 
                    values="seasonality_factor", 
                    aggfunc="first"
                ).round(3)
                
                # Rename columns to month names
                month_names = {
                    1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Juin",
                    7: "Juil", 8: "Août", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc"
                }
                seasonality_pivot.columns = [month_names.get(col, str(col)) for col in seasonality_pivot.columns]
                
                # Format the values
                seasonality_display = seasonality_pivot.copy()
                for col in seasonality_display.columns:
                    seasonality_display[col] = seasonality_display[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
                
                st.dataframe(seasonality_display, use_container_width=True)
                
                # Add explanation
                with st.expander("ℹ️ Explication des facteurs de saisonnalité", expanded=False):
                    st.markdown("""
                    **Facteur > 1.0** : Mois de pic (CA supérieur à la moyenne)
                    **Facteur = 1.0** : Mois normal (CA égal à la moyenne)  
                    **Facteur < 1.0** : Mois creux (CA inférieur à la moyenne)
                    """)
            else:
                st.info("Aucune donnée de saisonnalité disponible")
        
        with col2:
            st.subheader("🚨 Détection d'anomalies")
            if not anomalies.empty:
                # Filter only anomalies and format for better readability
                anomalies_filtered = anomalies[anomalies["is_anomaly"]].copy()
                if not anomalies_filtered.empty:
                    anomalies_filtered["Sales"] = anomalies_filtered["Sales"].map(lambda x: f"{x:,.0f}".replace(",", " "))
                    anomalies_filtered["z"] = anomalies_filtered["z"].round(2)
                    anomalies_filtered["Month"] = anomalies_filtered["Month"].dt.strftime("%m-%Y")
                    anomalies_filtered["severity"] = anomalies_filtered["z"].apply(
                        lambda x: "🔴 Critique" if abs(x) > 3 else "🟡 Modéré" if abs(x) > 2.5 else "🟠 Léger"
                    )
                    
                    # Add explanation column
                    anomalies_filtered["explication"] = anomalies_filtered["z"].apply(
                        lambda x: "Écart très important vs moyenne historique" if abs(x) > 3 
                        else "Écart significatif vs moyenne historique" if abs(x) > 2.5 
                        else "Écart notable vs moyenne historique"
                    )
                    
                    anomalies_display = anomalies_filtered[["Country", "Month", "Sales", "z", "severity", "explication"]]
                    st.dataframe(anomalies_display, use_container_width=True, hide_index=True)
                    
                    # Add explanation
                    with st.expander("ℹ️ Explication des anomalies détectées", expanded=False):
                        st.markdown("""
                        **Z-score** : Mesure l'écart en nombre d'écarts-types par rapport à la moyenne historique.
                        
                        **Seuils** :
                        - **🔴 Critique** : |z| > 3 (écart très important)
                        - **🟡 Modéré** : 2.5 < |z| ≤ 3 (écart significatif)  
                        - **🟠 Léger** : 2 < |z| ≤ 2.5 (écart notable)
                        
                        **Interprétation** : Une anomalie indique un CA très différent de la tendance historique du pays.
                        """)
                else:
                    st.success("✅ Aucune anomalie détectée - données cohérentes")
            else:
                st.info("Aucune donnée d'anomalie disponible")
        
        # PE-oriented CA opportunity/risk fusion visuals
        st.subheader("📌 Sur/sous-performance vs attendu (focus PE)")
        st.markdown("""
        **🎯 Objectif de cette analyse :** Identifier les opportunités d'investissement et les risques cachés en analysant 
        la performance des boutiques par rapport aux attentes saisonnières et aux anomalies de marché.
        
        **📊 Méthodologie :** 
        - **Saisonnalité** : Calcul du CA attendu basé sur les facteurs saisonniers historiques par pays
        - **Écart vs attendu** : Mesure de la sur/sous-performance en % par rapport aux attentes
        - **Signal CA** : Classification automatique en "Opportunité" (surperformance en période creuse), 
          "Risque" (sous-performance en période de pic) ou "Observation" (performance normale)
        
        **💡 Utilisation PE :** 
        - **Heatmap** : Visualise les écarts de performance par pays et par mois (vert = surperformance, rouge = sous-performance)
        - **Bubble Chart** : Combine z-score d'anomalie, facteur saisonnier et volume de CA pour identifier les marchés prioritaires
        """)
        
        pe_countries = st.multiselect(
            "Pays (PE)", 
            options=clean_and_sort_unique(filtered["Country"]), 
            default=clean_and_sort_unique(filtered["Country"]), 
            key="pe_countries"
        )
        zmin, zmax = st.slider("Échelle Heatmap (Écart %)", min_value=-100, max_value=100, value=(-40, 40), key="pe_heatmap_scale")
        min_abs_z = st.slider("Filtre z-score min (bubbles)", min_value=0.0, max_value=5.0, value=2.0, step=0.1, key="pe_min_z")
        min_sales = st.slider("Filtre CA min (bubbles)", min_value=0, max_value=100000, value=0, step=1000, key="pe_min_sales")
        only_signals = st.multiselect("Signaux", options=["Opportunité", "Risque", "Observation"], default=["Opportunité", "Risque"], key="pe_signals")
        
        fused = build_ca_opportunity_risk_dataset(filtered, countries_filter=pe_countries)
        colh, colb = st.columns(2)
        with colh:
            st.plotly_chart(heatmap_ecart_vs_attendu(fused, countries=pe_countries, zmin=float(zmin), zmax=float(zmax), color_scale="RdYlGn"), use_container_width=True)
        with colb:
            st.plotly_chart(bubble_ca_opportunity_risk(fused, min_abs_z=float(min_abs_z), min_sales=float(min_sales), only_signals=only_signals), use_container_width=True)
    
    with tab_countries:
        st.subheader("🌍 Analyse par pays")
        
        # Country average revenue per active shop
        avg_per_shop = filtered.groupby(["Country", "Month"], as_index=False).agg({
            "Sales": "sum",
            "Outlet": "nunique"
        }).assign(avg_per_shop=lambda x: x["Sales"] / x["Outlet"])
        
        st.plotly_chart(line_country_avg_per_active_shop(avg_per_shop), use_container_width=True)
        
        # Cruise speed analysis by country (detailed)
        cruise_by_country = cruise_aggregates_by_country(filtered)
        st.subheader("⚡ Analyse rythme de croisière par pays")
        
        # Add detailed explanation of cruise speed calculation
        with st.expander("ℹ️ Calcul du rythme de croisière", expanded=True):
            st.markdown("""
            **🎯 Définition :** Une boutique est "à rythme de croisière" quand elle atteint une **stabilité financière durable**.
            
            **📊 Méthode de calcul :**
            1. **Moyenne mobile 3 mois** : Calculée sur les 3 derniers mois consécutifs
            2. **Seuil de stabilité** : Variation ≤ 10% entre les moyennes mobiles successives
            3. **Période de validation** : 3 mois consécutifs de stabilité requise
            
            **🔍 Formule appliquée :**
            ```
            MA3_mois_N = (CA_mois_N + CA_mois_N-1 + CA_mois_N-2) / 3
            Variation = |(MA3_mois_N - MA3_mois_N-1) / MA3_mois_N-1|
            Rythme de croisière = Variation ≤ 10% sur 3 mois consécutifs
            ```
            
            **💡 Interprétation business :**
            - **À rythme** : Boutique mature, prévisible, optimale pour la planification
            - **En dessous** : Boutique en développement ou en difficulté, nécessite attention
            - **Temps moyen** : Indique la maturité typique du marché local
            """)
        
        try:
            detailed_cruise = analyze_cruise_speed_by_country(filtered)
        except Exception:
            detailed_cruise = pd.DataFrame()
        if not detailed_cruise.empty:
            disp = detailed_cruise.copy()
            # Format the average CA per shop
            disp["avg_cruise_ca_per_shop"] = disp["avg_cruise_ca_per_shop"].map(lambda x: f"{x:,.0f}".replace(",", " ") if x > 0 else "N/A")
            # Format the average months to cruise
            disp["avg_months_to_cruise"] = disp["avg_months_to_cruise"].map(lambda x: f"{x:.1f} mois" if pd.notna(x) and x > 0 else "N/A")
            
            st.dataframe(
                disp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Country": "Pays",
                    "total_shops": "Total boutiques",
                    "shops_at_cruise": "À rythme",
                    "pct_at_cruise": "% à rythme",
                    "avg_cruise_ca_per_shop": "CA moyen à rythme (USD)",
                    "avg_months_to_cruise": "Temps moyen (mois)",
                    "shops_below_avg_3m": "En dessous 3M",
                    "pct_below_avg_3m": "% en dessous 3M"
                }
            )
        elif not cruise_by_country.empty:
            st.dataframe(cruise_by_country, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune donnée disponible pour le rythme de croisière par pays.")
        
        # Heatmaps
        with st.expander(f"Heatmaps Top {controls['heatmap_top_n']} boutiques ({controls['heatmap_months']} derniers mois)", expanded=False):
            cols = st.columns(min(3, len(controls["selected_countries"])) or 1)
            for idx, country in enumerate(controls["selected_countries"]):
                with cols[idx % len(cols)]:
                    st.plotly_chart(heatmap_top5_shops(filtered, country, controls['heatmap_months'], controls['heatmap_top_n']), use_container_width=True)
    
    with tab_shops:
        st.subheader("🏪 Analyse par boutique")
        
        # Top/Bottom deciles
        deciles = compute_top_bottom_deciles_by_year(filtered)
        if isinstance(deciles, tuple) and len(deciles) == 2:
            top_deciles, bottom_deciles = deciles
            if not top_deciles.empty or not bottom_deciles.empty:
                st.subheader("🏆 Top 10% et Bottom 10% par année")
                st.markdown("*Performance relative des boutiques par rapport à la moyenne du réseau*")
                
                # Display top and bottom separately for better clarity
                col1, col2 = st.columns(2)
                with col1:
                    if not top_deciles.empty:
                        st.markdown("**🥇 Top 10% - Meilleures performances**")
                        st.dataframe(top_deciles, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucune donnée Top 10% disponible")
                
                with col2:
                    if not bottom_deciles.empty:
                        st.markdown("**🥉 Bottom 10% - Performances à améliorer**")
                        st.dataframe(bottom_deciles, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucune donnée Bottom 10% disponible")
                
                # Combine both for visualization
                if not top_deciles.empty or not bottom_deciles.empty:
                    combined_deciles = pd.concat([top_deciles, bottom_deciles], ignore_index=True)
                    st.plotly_chart(decile_bars(combined_deciles, "Decile"), use_container_width=True)
        else:
            if not deciles.empty:
                st.subheader("🏆 Top 10% et Bottom 10% par année")
                st.markdown("*Performance relative des boutiques par rapport à la moyenne du réseau*")
                st.dataframe(deciles, use_container_width=True, hide_index=True)
                st.plotly_chart(decile_bars(deciles, "Decile"), use_container_width=True)
            else:
                st.info("Aucune donnée de déciles disponible")
        
        # Growth metrics
        growth_country = compute_growth_metrics(filtered, entity="Country")
        growth_outlet = compute_growth_metrics(filtered, entity="Outlet", by_country=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(bar_latest_growth(growth_outlet, "Outlet"), use_container_width=True)
        with col2:
            st.info("💡 Croissance YoY par boutique - Sélectionnez un pays pour filtrer les résultats")
        
        # Probable closures - only show boutiques at risk
        if not closures.empty:
            st.subheader("🚨 Détection des fermetures probables")
            st.markdown("*Boutiques présentant un risque de fermeture basé sur l'inactivité*")
            
            # Display risk summary
            col1, col2, col3 = st.columns(3)
            with col1:
                critical_count = len(closures[closures["Niveau de risque"] == "🔴 CRITIQUE"])
                st.metric("🔴 Critique", critical_count)
            with col2:
                high_count = len(closures[closures["Niveau de risque"] == "🟠 ÉLEVÉ"])
                st.metric("🟠 Élevé", high_count)
            with col3:
                moderate_count = len(closures[closures["Niveau de risque"] == "🟡 MODÉRÉ"])
                st.metric("🟡 Modéré", moderate_count)
            
            # Display detailed table
            st.dataframe(closures, use_container_width=True, hide_index=True)
            
            # Add explanation
            with st.expander("ℹ️ Explication des niveaux de risque", expanded=False):
                st.markdown("""
                **🔴 CRITIQUE** : ≥12 mois sans CA après ouverture - Fermeture très probable
                **🟠 ÉLEVÉ** : 6-11 mois sans CA après ouverture - Risque de fermeture élevé  
                **🟡 MODÉRÉ** : 3-5 mois sans CA après ouverture - Inactivité à surveiller
                
                *Note : Seules les boutiques qui ont eu du CA avant sont analysées*
                """)
        else:
            st.success("✅ Aucune boutique à risque de fermeture détectée")
        
        # Recommendations
        recommendations = build_recommendations(filtered)
        if not recommendations.empty:
            st.subheader("🎯 Plan d'Action par Boutique")
            st.markdown("*Actions prioritaires pour optimiser la performance du réseau*")
            
            # Add action-oriented explanation
            with st.expander("ℹ️ Guide d'utilisation des recommandations", expanded=True):
                st.markdown("""
                **🚀 RENFORCER** : Boutiques performantes à développer (augmenter les ressources, étendre l'offre)
                **✅ MAINTENIR** : Boutiques stables à conserver (maintenir le niveau actuel)
                **⚠️ CORRIGER** : Boutiques en difficulté à redresser (formation, support, ajustement stratégique)
                **🔄 RELOCALISER** : Boutiques mal positionnées à déplacer (nouveau site, fusion)
                **❌ FERMER** : Boutiques non viables à fermer (libération de ressources)
                
                *Note : Les recommandations sont basées sur la performance relative au rythme de croisière, la croissance YoY et la volatilité*
                """)
            
            # Display recommendations with better formatting
            st.dataframe(recommendations, use_container_width=True, hide_index=True)
            
            # Add summary metrics
            if not recommendations.empty:
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    renforcer_count = len(recommendations[recommendations["Recommendation"] == "RENFORCER"])
                    st.metric("🚀 À Renforcer", renforcer_count)
                with col2:
                    maintenir_count = len(recommendations[recommendations["Recommendation"] == "MAINTENIR"])
                    st.metric("✅ À Maintenir", maintenir_count)
                with col3:
                    corriger_count = len(recommendations[recommendations["Recommendation"] == "CORRIGER"])
                    st.metric("⚠️ À Corriger", corriger_count)
                with col4:
                    relocaliser_count = len(recommendations[recommendations["Recommendation"] == "RELOCALISER"])
                    st.metric("🔄 À Relocaliser", relocaliser_count)
                with col5:
                    fermer_count = len(recommendations[recommendations["Recommendation"] == "FERMER"])
                    st.metric("❌ À Fermer", fermer_count)

        st.markdown("---")
    
    with tab_forecasts:
        st.subheader("🔮 Prévisions 2025-2026")
        st.markdown("""
        **Projection de CA par pays** basée sur :
        - Performance du réseau (CA historique, croissance)
        - Saisonnalité observée
        - Potentiel d'ouverture de nouvelles boutiques
        - Temps d'arrivée à croisière
        """)
        
        baseline = build_baseline(filtered)
        
        # Per-country forecasting parameters (sliders) in sidebar
        st.sidebar.subheader("⚙️ Paramètres par pays")
        countries_list = clean_and_sort_unique(filtered["Country"])
        
        # Create sliders for each country
        growth_dict = {}
        new_shop_dict = {}
        target_dict = {}
        
        for country in countries_list:
            st.sidebar.markdown(f"**{country}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                growth_dict[country] = st.slider(
                    f"Croissance {country}", 
                    min_value=-50.0, 
                    max_value=100.0, 
                    value=float(controls["growth_by_country"]), 
                    step=5.0,
                    key=f"growth_{country}"
                ) / 100.0
            with col2:
                new_shop_dict[country] = st.slider(
                    f"Ouverture {country}", 
                    min_value=0.0, 
                    max_value=50.0, 
                    value=float(controls["new_shop_rate_by_country"]), 
                    step=1.0,
                    key=f"new_shop_{country}"
                ) / 100.0
            with col3:
                target_dict[country] = st.slider(
                    f"Cible {country}", 
                    min_value=1.0, 
                    max_value=50.0, 
                    value=float(controls["target_per_shop_by_country"]) / 1000.0, 
                    step=0.5,
                    key=f"target_{country}"
                ) * 1000.0
            st.sidebar.markdown("---")
        
        adjusted = build_adjusted_scenario(
            baseline,
            growth_by_country=growth_dict,
            new_shop_rate_by_country=new_shop_dict,
            target_per_shop_by_country=target_dict,
        )
        forecast_df = pd.concat([baseline, adjusted], ignore_index=True)
        
        # Build per-country landing 2025 and projection 2026 summary table
        f = forecast_df.copy()
        f["Year"] = f["Month"].dt.year
        
        # Calculate 2025 landing (sum of all months in 2025)
        landing_2025 = f[f["Year"] == 2025].groupby(["Country", "Scenario"], as_index=False)["Sales"].sum().rename(columns={"Sales": "CA_2025"})
        
        # Calculate 2026 projection (sum of all months in 2026)
        proj_2026 = f[f["Year"] == 2026].groupby(["Country", "Scenario"], as_index=False)["Sales"].sum().rename(columns={"Sales": "CA_2026"})
        
        # Merge and create summary table
        summary = landing_2025.merge(proj_2026, on=["Country", "Scenario"], how="outer").fillna(0.0)
        
        # Add growth rate
        summary["Croissance_2025_2026"] = np.where(
            summary["CA_2025"] > 0,
            ((summary["CA_2026"] - summary["CA_2025"]) / summary["CA_2025"] * 100).round(1),
            0.0
        )
        
        # Split numeric summaries by scenario and add Total row
        base_num = summary[summary["Scenario"] == "Baseline"][["Country", "CA_2025", "CA_2026", "Croissance_2025_2026"]].copy()
        adj_num = summary[summary["Scenario"] == "Ajusté"][["Country", "CA_2025", "CA_2026", "Croissance_2025_2026"]].copy()
        
        # Sort countries to ensure UG comes before Total
        def sort_countries_for_display(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            # Create a custom sorting key: put UG first, then others alphabetically, then Total last
            def sort_key(country):
                if country == "UG":
                    return "0"  # First
                elif country == "Total":
                    return "Z"  # Last
                else:
                    return country  # Alphabetical order
            return df.sort_values("Country", key=lambda x: x.map(sort_key))
        
        base_num = sort_countries_for_display(base_num)
        adj_num = sort_countries_for_display(adj_num)
        
        def add_total_row(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            ca25 = float(df["CA_2025"].sum())
            ca26 = float(df["CA_2026"].sum())
            growth = ((ca26 - ca25) / ca25 * 100.0) if ca25 > 0 else 0.0
            total_row = pd.DataFrame({
                "Country": ["Total"],
                "CA_2025": [ca25],
                "CA_2026": [ca26],
                "Croissance_2025_2026": [round(growth, 1)]
            })
            return pd.concat([df, total_row], ignore_index=True)
        
        base_num = add_total_row(base_num)
        adj_num = add_total_row(adj_num)
        
        # Format for display
        def format_display(df: pd.DataFrame) -> pd.DataFrame:
            d = df.copy()
            d["CA_2025"] = d["CA_2025"].map(lambda x: f"{x:,.0f}".replace(",", " "))
            d["CA_2026"] = d["CA_2026"].map(lambda x: f"{x:,.0f}".replace(",", " "))
            d["Croissance_2025_2026"] = d["Croissance_2025_2026"].map(lambda x: f"{x:+.1f}%")
            return d
        
        baseline_summary = format_display(base_num)
        adjusted_summary = format_display(adj_num)
        
        # Display scenario tables with KPIs in rows, years in columns (no numeric index columns)
        st.markdown("### 📊 Récapitulatif Prévisions par Pays")

        def pivot_kpis_rows(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            # Preserve the original order of countries from the sorted DataFrame
            country_order = df["Country"].tolist()
            
            # Country as row index, KPI as row labels, Years as columns
            tidy = df.melt(id_vars=["Country"], value_vars=["CA_2025", "CA_2026", "Croissance_2025_2026"],
                           var_name="KPI", value_name="Valeur")
            # Split KPI into label + year where applicable
            def split_kpi(k: str) -> tuple[str, str]:
                if k.startswith("CA_"):
                    return ("CA (USD)", k.split("_")[1])
                if k == "Croissance_2025_2026":
                    return ("Croissance 2025→2026", "2025→2026")
                return (k, "")
            parts = tidy["KPI"].apply(split_kpi)
            tidy["Label"] = parts.apply(lambda t: t[0])
            tidy["Col"] = parts.apply(lambda t: t[1])
            # For growth, place in a dedicated column
            ca = tidy[tidy["Label"] == "CA (USD)"].pivot(index="Country", columns="Col", values="Valeur").reset_index()
            growth = tidy[tidy["Label"] == "Croissance 2025→2026"][["Country", "Valeur"]].rename(columns={"Valeur": "Croissance 2025→2026"})
            pivot = ca.merge(growth, on="Country", how="left")
            
            # Reorder countries to match the original sorted order
            pivot["Country"] = pd.Categorical(pivot["Country"], categories=country_order, ordered=True)
            pivot = pivot.sort_values("Country")
            
            return pivot

        if not baseline_summary.empty:
            st.markdown("#### 🎯 Scénario Baseline (sans ajustements)")
            base_rows = pivot_kpis_rows(base_num)
            # Format numbers and ensure correct column order
            def fmt_cols(df_src: pd.DataFrame) -> pd.DataFrame:
                d = df_src.copy()
                if "2025" in d.columns:
                    d["2025"] = d["2025"].map(lambda x: f"{float(x):,.0f}".replace(",", " ") if pd.notna(x) and not isinstance(x, str) else str(x))
                if "2026" in d.columns:
                    d["2026"] = d["2026"].map(lambda x: f"{float(x):,.0f}".replace(",", " ") if pd.notna(x) and not isinstance(x, str) else str(x))
                if "Croissance 2025→2026" in d.columns:
                    d["Croissance 2025→2026"] = d["Croissance 2025→2026"].map(lambda x: f"{float(x):+.1f}%" if pd.notna(x) and not isinstance(x, str) else str(x))
                return d
            st.dataframe(fmt_cols(base_rows).rename(columns={"Country": "Pays"}), use_container_width=True, hide_index=True)

        if not adjusted_summary.empty:
            st.markdown("#### 🔮 Scénario Ajusté (avec hypothèses)")
            adj_rows = pivot_kpis_rows(adj_num)
            st.dataframe(fmt_cols(adj_rows).rename(columns={"Country": "Pays"}), use_container_width=True, hide_index=True)
        
        # Show assumptions used
        with st.expander("ℹ️ Hypothèses utilisées", expanded=False):
            # Calculate totals for summary
            total_countries = len(adjusted_summary) if not adjusted_summary.empty else 0
            total_ca_2025 = adjusted_summary["CA_2025"].str.replace(" ", "").astype(float).sum() if not adjusted_summary.empty else 0
            total_ca_2026 = adjusted_summary["CA_2026"].str.replace(" ", "").astype(float).sum() if not adjusted_summary.empty else 0
            total_growth = ((total_ca_2026 - total_ca_2025) / total_ca_2025 * 100) if total_ca_2025 > 0 else 0
            
            st.markdown(f"""
            **Paramètres d'ajustement :**
            - **Croissance par pays**: {controls['growth_by_country']}%
            - **Taux d'ouverture de nouvelles boutiques**: {controls['new_shop_rate_by_country']}%
            - **CA cible par boutique**: {controls['target_per_shop_by_country']:,} USD
            - **Période d'analyse**: {controls['start_date'].strftime('%m-%Y')} à {controls['end_date'].strftime('%m-%Y')}
            
            **Impact global :**
            - **Nombre de pays analysés**: {total_countries}
            - **CA total 2025**: {total_ca_2025:,.0f} USD
            - **CA total 2026**: {total_ca_2026:,.0f} USD
            - **Croissance globale 2025→2026**: {total_growth:+.1f}%
            """)
    
    with tab_alerts:
        st.subheader("⚠️ Alertes et indicateurs")
        
        # Country alerts
        alerts = country_alerts(filtered)
        if not alerts.empty:
            st.subheader("🚨 Alertes par pays")
            st.dataframe(alerts, use_container_width=True, hide_index=True)
        
        # Cruise speed distribution
        if not cruise_df.empty:
            st.subheader("⚡ Analyse du Rythme de Croisière par Pays")
            st.markdown("*Performance et stabilité des boutiques dans chaque marché*")
            
            # Add detailed explanation of cruise speed calculation
            with st.expander("ℹ️ Méthodologie du Rythme de Croisière", expanded=True):
                st.markdown("""
                **🎯 Définition du Rythme de Croisière :**
                Une boutique est considérée "à rythme de croisière" quand elle atteint une **stabilité financière durable**.
                
                **📊 Calcul technique :**
                1. **Moyenne mobile 3 mois** : Calculée sur les 3 derniers mois consécutifs
                2. **Seuil de stabilité** : Variation ≤ 10% entre les moyennes mobiles successives
                3. **Période de validation** : 3 mois consécutifs de stabilité requise
                
                **🔍 Formule appliquée :**
                ```
                Variation = |(MA3_mois_N - MA3_mois_N-1) / MA3_mois_N-1|
                Rythme de croisière = Variation ≤ 10% sur 3 mois consécutifs
                ```
                
                **💡 Interprétation business :**
                - **À rythme** : Boutique mature, prévisible, optimale pour la planification
                - **En dessous** : Boutique en développement ou en difficulté, nécessite attention
                - **Temps moyen** : Indique la maturité typique du marché local
                """)
            
            # Get detailed country analysis
            cruise_country_analysis = analyze_cruise_speed_by_country(filtered)
            
            if not cruise_country_analysis.empty:
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_shops = cruise_country_analysis["total_shops"].sum()
                    at_cruise = cruise_country_analysis["shops_at_cruise"].sum()
                    pct_at_cruise = (at_cruise / total_shops * 100) if total_shops > 0 else 0
                    st.metric("Boutiques à rythme", f"{at_cruise}/{total_shops}")
                with col2:
                    st.metric("% à rythme", f"{pct_at_cruise:.1f}%")
                with col3:
                    avg_cruise_ca = cruise_country_analysis["avg_cruise_ca_per_shop"].mean()
                    st.metric("CA moyen à rythme", f"{avg_cruise_ca:,.0f}".replace(",", " "))
                
                # Display detailed country table
                st.markdown("#### 📊 Analyse par pays")
                st.markdown("*Détail du rythme de croisière et performance des 3 derniers mois*")
                
                display_cruise = cruise_country_analysis.copy()
                display_cruise["avg_cruise_ca_per_shop"] = display_cruise["avg_cruise_ca_per_shop"].map(
                    lambda x: f"{x:,.0f}".replace(",", " ") if x > 0 else "N/A"
                )
                
                st.dataframe(
                    display_cruise,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Country": "Pays",
                        "total_shops": "Total boutiques",
                        "shops_at_cruise": "À rythme",
                        "pct_at_cruise": "% à rythme",
                        "avg_cruise_ca_per_shop": "CA moyen à rythme (USD)",
                        "shops_below_avg_3m": "En dessous 3M",
                        "pct_below_avg_3m": "% en dessous 3M"
                    }
                )
                
                # Add explanation
                with st.expander("ℹ️ Explication des métriques", expanded=False):
                    st.markdown("""
                    **Rythme de croisière** : Une boutique est considérée à rythme de croisière quand sa moyenne mobile 3 mois 
                    varie de moins de 10% sur 3 mois consécutifs.
                    
                    **CA moyen à rythme** : CA mensuel moyen des boutiques quand elles sont à rythme de croisière.
                    
                    **En dessous 3M** : Nombre de boutiques dont le CA du dernier mois est inférieur à leur moyenne mobile 3 mois.
                    """)
            else:
                st.info("Aucune donnée disponible pour l'analyse du rythme de croisière par pays.")
    
    # --- V1.5: Onglet Assistant IA ---
    with tab_ai:
        st.subheader("🤖 Assistant IA Lapaire")
        
        # Préparer les données pour l'IA
        top_performers = get_top_performers(filtered, n=5)
        underperformers = get_underperformers(filtered, n=5)
        period_str = f"{controls['start_date']:%m/%Y} - {controls['end_date']:%m/%Y}"
        country_breakdown = get_country_breakdown(filtered)
        
        # Ajouter pct_at_cruise aux KPIs si manquant
        kpis_for_ai = kpis.copy()
        if "pct_at_cruise" not in kpis_for_ai:
            kpis_for_ai["pct_at_cruise"] = pct_at_cruise if not np.isnan(pct_at_cruise) else 0.0
        
        # Convertir closures DataFrame en liste de dicts
        alerts_list = closures.to_dict("records") if not closures.empty else []
        
        # Rendre l'onglet IA complet
        render_ai_tab(
            df=filtered,
            kpis=kpis_for_ai,
            alerts=alerts_list,
            top_performers=top_performers,
            underperformers=underperformers,
            period=period_str,
            country_breakdown=country_breakdown,
        )
    # --- Fin onglet IA ---
    
    with tab_memo:
        st.subheader("📝 Export et documentation")
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📊 Télécharger Excel",
                data=export_excel(filtered, kpis, cruise_df, closures),
                file_name="lapaire_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        
        with col2:
            st.download_button(
                label="📄 Télécharger PDF",
                data=export_pdf_simple(filtered, kpis, cruise_df, closures),
                file_name="lapaire_analysis.pdf",
                mime="application/pdf",
            )
        
        # Data freshness
        st.info(f"📅 Données mises à jour le: {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M')}")
        
        # Footnotes and assumptions
        with st.expander("📋 Notes et hypothèses", expanded=False):
            st.markdown("""
            **Méthodologie**:
            - Les fermetures probables sont détectées après 2 mois consécutifs sans CA (configurable)
            - Le rythme de croisière est défini comme une variation ≤10% sur 3 mois consécutifs
            - Les prévisions intègrent la saisonnalité historique et les tendances de croissance
            - Les recommandations sont basées sur les performances relatives et la position dans le réseau
            """)

if __name__ == "__main__":
    main()


