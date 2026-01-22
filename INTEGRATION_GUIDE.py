"""
INSTRUCTIONS DE MODIFICATION - app.py
=====================================

Ce fichier décrit les modifications à apporter à app.py pour intégrer l'IA.
Les modifications sont minimales et non-breaking.

"""

# =============================================================================
# MODIFICATION 1: Ajouter les imports en haut du fichier (après les autres imports)
# =============================================================================

IMPORTS_TO_ADD = '''
# --- V1.5: Imports IA ---
from components.ai_widgets import (
    render_ai_tab,
    render_ai_status_badge,
)
# --- Fin imports IA ---
'''

# =============================================================================
# MODIFICATION 2: Ajouter des fonctions helper (avant la fonction main())
# =============================================================================

HELPER_FUNCTIONS = '''
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
'''

# =============================================================================
# MODIFICATION 3: Ajouter l'onglet IA dans les tabs (ligne ~298)
# =============================================================================

# AVANT (ligne ~298):
TABS_BEFORE = '''
    tab_overview, tab_countries, tab_shops, tab_forecasts, tab_alerts, tab_memo = st.tabs([
        "📊 Vue d'ensemble", "🌍 Pays", "🏪 Boutiques", "🔮 Prévisions", "⚠️ Alertes", "📝 Mémo"
    ])
'''

# APRÈS:
TABS_AFTER = '''
    tab_overview, tab_countries, tab_shops, tab_forecasts, tab_alerts, tab_ai, tab_memo = st.tabs([
        "📊 Vue d'ensemble", "🌍 Pays", "🏪 Boutiques", "🔮 Prévisions", "⚠️ Alertes", "🤖 IA", "📝 Mémo"
    ])
'''

# =============================================================================
# MODIFICATION 4: Ajouter le contenu de l'onglet IA (après tab_alerts, avant tab_memo)
# =============================================================================

# À insérer juste avant "with tab_memo:" (vers ligne ~987)
AI_TAB_CONTENT = '''
    # --- V1.5: Onglet Assistant IA ---
    with tab_ai:
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
'''

# =============================================================================
# RÉSUMÉ DES MODIFICATIONS
# =============================================================================

SUMMARY = """
RÉSUMÉ DES MODIFICATIONS À APPORTER À app.py:
=============================================

1. IMPORTS (en haut du fichier, après les autres imports):
   - Ajouter: from components.ai_widgets import render_ai_tab, render_ai_status_badge

2. FONCTIONS HELPER (avant main()):
   - Ajouter: get_top_performers()
   - Ajouter: get_underperformers()
   - Ajouter: get_country_breakdown()

3. TABS (ligne ~298):
   - Ajouter "tab_ai" dans la liste des tabs
   - Ajouter "🤖 IA" dans les labels

4. CONTENU ONGLET IA (avant tab_memo, vers ligne ~987):
   - Ajouter le bloc "with tab_ai:" avec l'appel à render_ai_tab()

FICHIERS À COPIER DANS LE REPO:
==============================
- ai/__init__.py
- ai/client.py
- ai/narrative.py
- ai/chat.py
- ai/alerts.py
- components/__init__.py
- components/ai_widgets.py
- .streamlit/secrets.toml.template -> .streamlit/secrets.toml (avec votre clé)

MISE À JOUR requirements.txt:
============================
- Ajouter: anthropic>=0.40.0,<1.0.0

CONFIGURATION:
=============
1. Créer un compte sur console.anthropic.com
2. Générer une clé API
3. Copier secrets.toml.template vers secrets.toml
4. Remplacer "sk-ant-api03-VOTRE-CLE-ICI" par votre vraie clé
5. Ajouter secrets.toml à .gitignore!

Pour Streamlit Cloud:
- Aller dans Settings > Secrets
- Ajouter: ANTHROPIC_API_KEY = "votre-clé"
"""

if __name__ == "__main__":
    print(SUMMARY)
