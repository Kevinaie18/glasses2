"""
Chat Q&A sur les données Lapaire.
Permet de poser des questions en langage naturel sur les performances.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from .client import call_claude


# Prompt système pour le Q&A
SYSTEM_QA = """Tu es un assistant data analyst pour Lapaire, un réseau de boutiques d'optique en Afrique.

Tu réponds aux questions sur les performances des boutiques en utilisant UNIQUEMENT les données fournies.

Règles:
- Sois précis et cite les chiffres exacts
- Si tu ne peux pas répondre avec les données disponibles, dis-le clairement
- Utilise le format markdown pour structurer tes réponses
- Reste concis (max 150 mots sauf si on demande plus de détails)
- Pour les montants, utilise le format "XX XXX USD" avec espace comme séparateur de milliers

Tu réponds toujours en français."""


def get_suggested_questions(df: pd.DataFrame) -> List[str]:
    """
    Retourne une liste de questions suggérées basées sur les données.
    
    Args:
        df: DataFrame des données de ventes
    
    Returns:
        Liste de questions pertinentes
    """
    suggestions = [
        "Quelles boutiques sous-performent ce mois?",
        "Quel pays a la meilleure croissance?",
        "Combien de boutiques sont au rythme de croisière?",
    ]
    
    # Ajouter des suggestions contextuelles
    if "Country" in df.columns:
        countries = df["Country"].unique()
        if len(countries) >= 2:
            c1, c2 = countries[0], countries[1]
            suggestions.append(f"Compare {c1} vs {c2}")
    
    if "Month" in df.columns:
        suggestions.append("Quelle est la tendance des 3 derniers mois?")
    
    return suggestions[:5]  # Max 5 suggestions


def _prepare_data_context(df: pd.DataFrame, kpis: Dict) -> str:
    """Prépare le contexte des données pour Claude."""
    
    # Infos générales
    countries = df["Country"].unique().tolist() if "Country" in df.columns else []
    outlets_count = df["Outlet"].nunique() if "Outlet" in df.columns else 0
    
    # Période
    if "Month" in df.columns:
        min_month = df["Month"].min()
        max_month = df["Month"].max()
        date_range = f"{min_month:%m/%Y} - {max_month:%m/%Y}"
    else:
        date_range = "N/A"
    
    # Stats par pays
    country_stats = ""
    if "Country" in df.columns and "Sales" in df.columns:
        stats = df.groupby("Country").agg({
            "Sales": ["sum", "mean", "count"],
            "Outlet": "nunique"
        }).round(0)
        stats.columns = ["CA_Total", "CA_Moyen", "Nb_Lignes", "Nb_Boutiques"]
        country_stats = stats.to_string()
    
    # Top/Bottom boutiques du dernier mois
    top5_text = ""
    bottom5_text = ""
    if "Month" in df.columns and "Sales" in df.columns and "Outlet" in df.columns:
        latest_month = df["Month"].max()
        recent = df[df["Month"] == latest_month].copy()
        
        if not recent.empty:
            top5 = recent.nlargest(5, "Sales")[["Outlet", "Country", "Sales"]]
            top5_text = top5.to_string(index=False)
            
            active = recent[recent["Sales"] > 0]
            if not active.empty:
                bottom5 = active.nsmallest(5, "Sales")[["Outlet", "Country", "Sales"]]
                bottom5_text = bottom5.to_string(index=False)
    
    # Tendance mensuelle
    monthly_trend = ""
    if "Month" in df.columns and "Sales" in df.columns:
        monthly = df.groupby("Month")["Sales"].sum().tail(6)
        if len(monthly) > 1:
            monthly_trend = "CA mensuel (6 derniers mois):\n"
            for month, sales in monthly.items():
                monthly_trend += f"  {month:%m/%Y}: {sales:,.0f} USD\n"
    
    context = f"""DONNÉES LAPAIRE DISPONIBLES:
=====================================
Période: {date_range}
Pays: {', '.join(countries)}
Nombre total de boutiques: {outlets_count}

KPIs CALCULÉS:
- CA Total LTM: {kpis.get('total_revenue', 0):,.0f} USD
- Croissance YoY: {kpis.get('yoy_growth', 0):+.1f}%
- Croissance MoM: {kpis.get('mom_growth', 0):+.1f}%
- CA moyen/boutique: {kpis.get('avg_revenue_per_shop', 0):,.0f} USD
- % au rythme de croisière: {kpis.get('pct_at_cruise', 0):.1f}%

STATISTIQUES PAR PAYS:
{country_stats}

{monthly_trend}

TOP 5 BOUTIQUES (dernier mois):
{top5_text}

BOTTOM 5 BOUTIQUES ACTIVES (dernier mois):
{bottom5_text}
"""
    
    return context


def answer_data_question(
    question: str,
    df: pd.DataFrame,
    kpis: Dict,
    additional_context: str = "",
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Répond à une question sur les données en langage naturel.
    
    Args:
        question: Question de l'utilisateur
        df: DataFrame des données de ventes
        kpis: Dictionnaire des KPIs calculés
        additional_context: Contexte supplémentaire optionnel
    
    Returns:
        Tuple (réponse texte, DataFrame résultat si applicable)
    """
    
    # Préparer le contexte
    data_context = _prepare_data_context(df, kpis)
    
    # Ajouter contexte supplémentaire si fourni
    if additional_context:
        data_context += f"\n\nCONTEXTE ADDITIONNEL:\n{additional_context}"
    
    prompt = f"""{data_context}

=====================================
QUESTION DE L'UTILISATEUR: {question}
=====================================

Réponds à cette question en utilisant UNIQUEMENT les données ci-dessus.

Instructions:
1. Si la question est claire, réponds directement avec les chiffres
2. Si tu as besoin de faire un calcul simple, fais-le et montre le résultat
3. Si la question n'est pas claire, demande des précisions
4. Si tu ne peux pas répondre avec les données disponibles, explique pourquoi

Format ta réponse en markdown."""

    response = call_claude(
        prompt=prompt,
        system=SYSTEM_QA,
        max_tokens=500,
        temperature=0.5,  # Plus déterministe pour les questions de données
    )
    
    return response, None


def answer_comparison_question(
    entity1: str,
    entity2: str,
    df: pd.DataFrame,
    comparison_type: str = "country",
) -> str:
    """
    Répond à une question de comparaison entre deux entités.
    
    Args:
        entity1: Première entité (pays ou boutique)
        entity2: Deuxième entité
        df: DataFrame des données
        comparison_type: "country" ou "outlet"
    
    Returns:
        Analyse comparative formatée
    """
    
    col = "Country" if comparison_type == "country" else "Outlet"
    
    if col not in df.columns:
        return f"❌ Colonne '{col}' non trouvée dans les données."
    
    # Filtrer les données pour chaque entité
    df1 = df[df[col] == entity1]
    df2 = df[df[col] == entity2]
    
    if df1.empty:
        return f"❌ Aucune donnée trouvée pour '{entity1}'"
    if df2.empty:
        return f"❌ Aucune donnée trouvée pour '{entity2}'"
    
    # Calculer les stats
    stats1 = {
        "ca_total": df1["Sales"].sum(),
        "ca_moyen": df1["Sales"].mean(),
        "nb_mois": df1["Month"].nunique() if "Month" in df1.columns else 0,
    }
    stats2 = {
        "ca_total": df2["Sales"].sum(),
        "ca_moyen": df2["Sales"].mean(),
        "nb_mois": df2["Month"].nunique() if "Month" in df2.columns else 0,
    }
    
    if comparison_type == "country":
        stats1["nb_boutiques"] = df1["Outlet"].nunique()
        stats2["nb_boutiques"] = df2["Outlet"].nunique()
    
    prompt = f"""Compare {entity1} vs {entity2}:

{entity1.upper()}:
- CA Total: {stats1['ca_total']:,.0f} USD
- CA Moyen/mois: {stats1['ca_moyen']:,.0f} USD
- Nombre de mois de données: {stats1['nb_mois']}
{f"- Nombre de boutiques: {stats1.get('nb_boutiques', 'N/A')}" if comparison_type == "country" else ""}

{entity2.upper()}:
- CA Total: {stats2['ca_total']:,.0f} USD
- CA Moyen/mois: {stats2['ca_moyen']:,.0f} USD
- Nombre de mois de données: {stats2['nb_mois']}
{f"- Nombre de boutiques: {stats2.get('nb_boutiques', 'N/A')}" if comparison_type == "country" else ""}

Fournis une analyse comparative en 4-5 phrases:
1. Qui performe le mieux globalement?
2. Quelle est la différence en %?
3. Points forts de chaque entité
4. Recommandation

Format markdown avec emojis."""

    return call_claude(
        prompt=prompt,
        system=SYSTEM_QA,
        max_tokens=400,
        temperature=0.6,
    )


def explain_kpi(kpi_name: str, kpi_value: float, context: Dict) -> str:
    """
    Explique un KPI en langage simple.
    
    Args:
        kpi_name: Nom du KPI
        kpi_value: Valeur du KPI
        context: Contexte additionnel (benchmarks, historique)
    
    Returns:
        Explication du KPI
    """
    
    prompt = f"""Explique ce KPI à un investisseur non-technique:

KPI: {kpi_name}
Valeur: {kpi_value}
Contexte: {context}

En 2-3 phrases:
1. Que signifie ce KPI?
2. Cette valeur est-elle bonne ou mauvaise?
3. Que devrait faire le management?

Style: simple, direct, actionnable."""

    return call_claude(
        prompt=prompt,
        system=SYSTEM_QA,
        max_tokens=200,
        temperature=0.7,
    )
