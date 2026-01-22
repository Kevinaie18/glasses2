"""
Génération de synthèses narratives pour le board.
Transforme les KPIs et données en mémo exécutif actionnable.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .client import call_claude


# Prompt système pour les synthèses
SYSTEM_NARRATIVE = """Tu es un analyste financier senior spécialisé dans le retail en Afrique.
Tu prépares des mémos pour des investisseurs Private Equity.

Ton style:
- Factuel et chiffré (toujours citer les données)
- Concis (phrases courtes, pas de jargon)
- Orienté décision (qu'est-ce que ça implique?)
- Utilise des emojis pour les tendances: 📈 hausse, 📉 baisse, ⚠️ attention, ✅ positif

Tu écris en français."""


def generate_executive_summary(
    kpis: Dict,
    alerts: List[Dict],
    top_performers: List[str],
    underperformers: List[str],
    period: str,
    country_breakdown: Optional[Dict] = None,
) -> str:
    """
    Génère un résumé exécutif automatique pour le board.
    
    Args:
        kpis: Dictionnaire des KPIs calculés
        alerts: Liste des alertes actives
        top_performers: Liste des meilleures boutiques
        underperformers: Liste des boutiques en difficulté
        period: Période d'analyse (ex: "01/2024 - 12/2024")
        country_breakdown: Optionnel, performance par pays
    
    Returns:
        Texte du résumé exécutif formaté en markdown
    """
    
    # Formater les alertes
    if alerts:
        alerts_text = "\n".join([
            f"- [{a.get('Niveau de risque', a.get('niveau', 'N/A'))}] "
            f"{a.get('Boutique', a.get('boutique', 'N/A'))}: "
            f"{a.get('Raison', a.get('message', 'N/A'))}"
            for a in alerts[:5]
        ])
    else:
        alerts_text = "Aucune alerte critique ce mois"
    
    # Formater les top/bottom performers
    top_text = ", ".join(top_performers[:5]) if top_performers else "N/A"
    under_text = ", ".join(underperformers[:5]) if underperformers else "N/A"
    
    # Construire la section pays si disponible
    country_section = ""
    if country_breakdown:
        country_lines = []
        for country, data in country_breakdown.items():
            if isinstance(data, dict):
                ca = data.get('ca', data.get('sales', 0))
                growth = data.get('growth', data.get('yoy', 0))
                country_lines.append(f"- {country}: {ca:,.0f} USD ({growth:+.1f}% YoY)")
            else:
                country_lines.append(f"- {country}: {data:,.0f} USD")
        country_section = f"""
PERFORMANCE PAR PAYS:
{chr(10).join(country_lines)}
"""
    
    prompt = f"""PÉRIODE D'ANALYSE: {period}

KPIs DU RÉSEAU LAPAIRE:
- CA Total LTM: {kpis.get('total_revenue', 0):,.0f} USD
- Croissance YoY: {kpis.get('yoy_growth', 0):+.1f}%
- Croissance MoM: {kpis.get('mom_growth', 0):+.1f}%
- CA moyen par boutique: {kpis.get('avg_revenue_per_shop', 0):,.0f} USD
- % boutiques au rythme de croisière: {kpis.get('pct_at_cruise', 0):.1f}%
- Part du Top 20%: {kpis.get('top_20_percent_share', 0):.1f}%
{country_section}
TOP 5 BOUTIQUES: {top_text}
BOUTIQUES EN DIFFICULTÉ: {under_text}

ALERTES ACTIVES ({len(alerts)} total):
{alerts_text}

---

Rédige un EXECUTIVE SUMMARY de 150-200 mots avec cette structure:

## 📊 Headline
Une phrase d'accroche chiffrée résumant la tendance principale.

## Performance
2-3 phrases sur les résultats clés (CA, croissance, points forts).

## ⚠️ Points de vigilance
1-2 phrases sur les risques ou problèmes identifiés.

## 🎯 Outlook
1 phrase sur les perspectives et actions recommandées.

---
Utilise le format markdown. Sois factuel et direct."""

    return call_claude(
        prompt=prompt,
        system=SYSTEM_NARRATIVE,
        max_tokens=600,
        temperature=0.7,
    )


def generate_country_analysis(
    country: str,
    country_kpis: Dict,
    shops_data: List[Dict],
    period: str,
) -> str:
    """
    Génère une analyse détaillée pour un pays spécifique.
    
    Args:
        country: Code pays (ex: "UG", "KE")
        country_kpis: KPIs du pays
        shops_data: Données des boutiques du pays
        period: Période d'analyse
    
    Returns:
        Analyse formatée en markdown
    """
    
    # Statistiques boutiques
    num_shops = len(shops_data)
    if shops_data:
        avg_sales = sum(s.get('sales', 0) for s in shops_data) / num_shops
        top_shop = max(shops_data, key=lambda x: x.get('sales', 0))
        bottom_shop = min(shops_data, key=lambda x: x.get('sales', 0))
    else:
        avg_sales = 0
        top_shop = {}
        bottom_shop = {}
    
    prompt = f"""Analyse le pays {country} du réseau Lapaire pour la période {period}.

DONNÉES:
- Nombre de boutiques: {num_shops}
- CA total: {country_kpis.get('total', 0):,.0f} USD
- Croissance YoY: {country_kpis.get('yoy_growth', 0):+.1f}%
- CA moyen/boutique: {avg_sales:,.0f} USD
- Meilleure boutique: {top_shop.get('name', 'N/A')} ({top_shop.get('sales', 0):,.0f} USD)
- Plus faible: {bottom_shop.get('name', 'N/A')} ({bottom_shop.get('sales', 0):,.0f} USD)

Fournis une analyse en 3-4 phrases couvrant:
1. Performance globale du pays
2. Disparités entre boutiques
3. Recommandation principale

Format markdown, style concis et factuel."""

    return call_claude(
        prompt=prompt,
        system=SYSTEM_NARRATIVE,
        max_tokens=300,
        temperature=0.7,
    )


def generate_monthly_highlights(
    current_month_data: Dict,
    previous_month_data: Dict,
    period: str,
) -> str:
    """
    Génère les faits marquants du mois vs mois précédent.
    
    Args:
        current_month_data: Données du mois en cours
        previous_month_data: Données du mois précédent
        period: Période (ex: "Décembre 2024")
    
    Returns:
        Liste des highlights en markdown
    """
    
    prompt = f"""Compare les performances de {period} vs le mois précédent.

MOIS EN COURS:
- CA Total: {current_month_data.get('total_sales', 0):,.0f} USD
- Boutiques actives: {current_month_data.get('active_shops', 0)}
- Top pays: {current_month_data.get('top_country', 'N/A')}

MOIS PRÉCÉDENT:
- CA Total: {previous_month_data.get('total_sales', 0):,.0f} USD
- Boutiques actives: {previous_month_data.get('active_shops', 0)}
- Top pays: {previous_month_data.get('top_country', 'N/A')}

Génère 3-5 bullet points des faits marquants:
- Évolutions significatives (>10%)
- Anomalies ou surprises
- Tendances à surveiller

Format: liste markdown avec emojis."""

    return call_claude(
        prompt=prompt,
        system=SYSTEM_NARRATIVE,
        max_tokens=300,
        temperature=0.7,
    )
