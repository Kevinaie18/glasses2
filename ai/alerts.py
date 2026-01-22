"""
Alertes intelligentes enrichies par IA.
Transforme les alertes brutes en insights actionnables avec contexte et recommandations.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from .client import call_claude


# Prompt système pour les alertes
SYSTEM_ALERTS = """Tu es un expert en operations retail spécialisé dans les réseaux de boutiques en Afrique.

Tu analyses des alertes de performance pour identifier les vraies urgences et filtrer le bruit.

Ton rôle:
- Contextualiser l'alerte (est-ce normal? attendu? inquiétant?)
- Proposer des actions concrètes et réalistes
- Ajuster le niveau d'urgence si nécessaire
- Éviter les faux positifs (boutique en travaux, saisonnalité, etc.)

Tu réponds TOUJOURS en JSON valide."""


def enrich_alert(
    alert: Dict,
    context: Optional[Dict] = None,
) -> Dict:
    """
    Enrichit une alerte avec diagnostic et recommandations IA.
    
    Args:
        alert: Dictionnaire de l'alerte brute avec clés possibles:
            - boutique/Boutique: nom de la boutique
            - pays/Pays: code pays
            - niveau/Niveau de risque: sévérité brute
            - message/Raison: description du problème
            - mois_sans_ca/Mois sans CA: nombre de mois sans ventes
        context: Contexte optionnel avec:
            - age_months: ancienneté de la boutique
            - last_3m_trend: tendance 3 derniers mois
            - country_avg: moyenne du pays
            - current_sales: CA actuel
            - historical_pattern: pattern historique
    
    Returns:
        Alerte enrichie avec diagnostic, actions et urgence ajustée
    """
    
    # Normaliser les clés de l'alerte
    boutique = alert.get("Boutique", alert.get("boutique", "N/A"))
    pays = alert.get("Pays", alert.get("pays", alert.get("Country", "N/A")))
    niveau = alert.get("Niveau de risque", alert.get("niveau", "N/A"))
    message = alert.get("Raison", alert.get("message", "N/A"))
    mois_sans_ca = alert.get("Mois sans CA", alert.get("mois_sans_ca", 0))
    
    # Préparer le contexte
    ctx = context or {}
    age_months = ctx.get("age_months", "N/A")
    last_3m = ctx.get("last_3m_trend", "N/A")
    country_avg = ctx.get("country_avg", 0)
    current_sales = ctx.get("current_sales", 0)
    
    prompt = f"""ALERTE À ANALYSER:
==================
Boutique: {boutique}
Pays: {pays}
Niveau brut: {niveau}
Problème: {message}
Mois sans CA: {mois_sans_ca}

CONTEXTE DISPONIBLE:
==================
- Ancienneté boutique: {age_months} mois
- Tendance 3 derniers mois: {last_3m}
- Moyenne CA pays: {country_avg:,.0f} USD
- CA actuel boutique: {current_sales:,.0f} USD

ANALYSE DEMANDÉE:
==================
1. Diagnostic: Quelle est la cause probable? (2 phrases max)
2. Actions: Que faire concrètement? (2-3 actions prioritaires)
3. Urgence ajustée: Le niveau brut est-il justifié?
   - "Critique" = action immédiate requise (risque fermeture, perte client majeure)
   - "Élevé" = action cette semaine
   - "Modéré" = à monitorer, action ce mois
   - "Faible" = information, pas d'action urgente

IMPORTANT: Réponds UNIQUEMENT avec un JSON valide, sans texte avant ni après:

{{
    "diagnostic": "Explication courte de la situation",
    "actions": [
        "Action 1 concrète",
        "Action 2 concrète"
    ],
    "urgence_ajustee": "Critique|Élevé|Modéré|Faible",
    "justification_urgence": "Pourquoi ce niveau d'urgence"
}}"""

    response = call_claude(
        prompt=prompt,
        system=SYSTEM_ALERTS,
        max_tokens=400,
        temperature=0.5,
    )
    
    # Parser la réponse JSON
    enriched_alert = alert.copy()
    
    try:
        # Chercher le JSON dans la réponse
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            enriched_alert["diagnostic"] = parsed.get("diagnostic", "")
            enriched_alert["actions"] = parsed.get("actions", [])
            enriched_alert["urgence_ajustee"] = parsed.get("urgence_ajustee", niveau)
            enriched_alert["justification_urgence"] = parsed.get("justification_urgence", "")
            enriched_alert["enriched"] = True
        else:
            # Si pas de JSON valide, utiliser la réponse brute
            enriched_alert["diagnostic"] = response
            enriched_alert["enriched"] = False
            
    except json.JSONDecodeError:
        enriched_alert["diagnostic"] = response
        enriched_alert["enriched"] = False
    
    return enriched_alert


def enrich_alerts_batch(
    alerts: List[Dict],
    contexts: Optional[Dict[str, Dict]] = None,
    max_alerts: int = 10,
) -> List[Dict]:
    """
    Enrichit plusieurs alertes en batch.
    Limite le nombre pour éviter trop d'appels API.
    
    Args:
        alerts: Liste des alertes brutes
        contexts: Dict {boutique_name: context_dict} pour chaque boutique
        max_alerts: Nombre max d'alertes à enrichir
    
    Returns:
        Liste des alertes enrichies
    """
    
    if not alerts:
        return []
    
    contexts = contexts or {}
    enriched = []
    
    # Trier par sévérité pour prioriser les plus critiques
    severity_order = {
        "🔴 CRITIQUE": 0,
        "CRITIQUE": 0,
        "🟠 ÉLEVÉ": 1,
        "ÉLEVÉ": 1,
        "🟡 MODÉRÉ": 2,
        "MODÉRÉ": 2,
        "🟢 FAIBLE": 3,
        "FAIBLE": 3,
    }
    
    sorted_alerts = sorted(
        alerts,
        key=lambda a: severity_order.get(
            a.get("Niveau de risque", a.get("niveau", "MODÉRÉ")),
            2
        )
    )
    
    for alert in sorted_alerts[:max_alerts]:
        boutique = alert.get("Boutique", alert.get("boutique", ""))
        ctx = contexts.get(boutique, {})
        enriched.append(enrich_alert(alert, ctx))
    
    # Ajouter les alertes non enrichies (au-delà de la limite)
    for alert in sorted_alerts[max_alerts:]:
        alert_copy = alert.copy()
        alert_copy["enriched"] = False
        alert_copy["diagnostic"] = "Non analysé (limite atteinte)"
        enriched.append(alert_copy)
    
    return enriched


def generate_alerts_summary(alerts: List[Dict]) -> str:
    """
    Génère un résumé des alertes pour le dashboard.
    
    Args:
        alerts: Liste des alertes (enrichies ou non)
    
    Returns:
        Résumé textuel des alertes
    """
    
    if not alerts:
        return "✅ **Aucune alerte active** - Le réseau fonctionne normalement."
    
    # Compter par niveau
    counts = {"Critique": 0, "Élevé": 0, "Modéré": 0, "Faible": 0}
    
    for alert in alerts:
        niveau = alert.get("urgence_ajustee", alert.get("Niveau de risque", alert.get("niveau", "Modéré")))
        # Nettoyer le niveau des emojis
        niveau_clean = niveau.replace("🔴 ", "").replace("🟠 ", "").replace("🟡 ", "").replace("🟢 ", "")
        niveau_clean = niveau_clean.upper()
        
        if "CRITIQUE" in niveau_clean:
            counts["Critique"] += 1
        elif "ÉLEV" in niveau_clean or "ELEV" in niveau_clean:
            counts["Élevé"] += 1
        elif "MODÉR" in niveau_clean or "MODER" in niveau_clean:
            counts["Modéré"] += 1
        else:
            counts["Faible"] += 1
    
    # Construire le résumé
    summary_parts = []
    
    if counts["Critique"] > 0:
        summary_parts.append(f"🔴 **{counts['Critique']} critique(s)**")
    if counts["Élevé"] > 0:
        summary_parts.append(f"🟠 **{counts['Élevé']} élevé(s)**")
    if counts["Modéré"] > 0:
        summary_parts.append(f"🟡 {counts['Modéré']} modéré(s)")
    if counts["Faible"] > 0:
        summary_parts.append(f"🟢 {counts['Faible']} faible(s)")
    
    total = len(alerts)
    summary = f"**{total} alerte(s) active(s)**: " + " | ".join(summary_parts)
    
    # Ajouter les alertes critiques en détail
    critiques = [a for a in alerts if "CRITIQUE" in str(a.get("urgence_ajustee", a.get("Niveau de risque", ""))).upper()]
    
    if critiques:
        summary += "\n\n**⚠️ Actions immédiates requises:**\n"
        for alert in critiques[:3]:
            boutique = alert.get("Boutique", alert.get("boutique", "N/A"))
            diagnostic = alert.get("diagnostic", alert.get("Raison", alert.get("message", "")))
            summary += f"- **{boutique}**: {diagnostic[:100]}...\n"
    
    return summary


def prioritize_alerts(alerts: List[Dict]) -> List[Dict]:
    """
    Trie les alertes par priorité business.
    
    Args:
        alerts: Liste des alertes
    
    Returns:
        Alertes triées par priorité décroissante
    """
    
    def priority_score(alert: Dict) -> int:
        """Calcule un score de priorité (plus bas = plus urgent)."""
        score = 100
        
        # Niveau d'urgence
        niveau = str(alert.get("urgence_ajustee", alert.get("Niveau de risque", ""))).upper()
        if "CRITIQUE" in niveau:
            score -= 50
        elif "ÉLEV" in niveau or "ELEV" in niveau:
            score -= 30
        elif "MODÉR" in niveau or "MODER" in niveau:
            score -= 10
        
        # Durée du problème
        mois_sans_ca = alert.get("Mois sans CA", alert.get("mois_sans_ca", 0))
        if isinstance(mois_sans_ca, (int, float)):
            score -= min(mois_sans_ca * 5, 25)  # Max -25 points
        
        # Si enrichi par IA
        if alert.get("enriched"):
            # Ajuster selon l'analyse IA
            actions = alert.get("actions", [])
            if len(actions) > 2:
                score -= 5  # Plus d'actions = plus complexe
        
        return score
    
    return sorted(alerts, key=priority_score)
