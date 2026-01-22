"""
Composants Streamlit pour l'intégration IA.
Widgets réutilisables pour le dashboard Lapaire V1.5.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from ai.client import is_ai_available
from ai.narrative import generate_executive_summary
from ai.chat import answer_data_question, get_suggested_questions
from ai.alerts import enrich_alerts_batch, generate_alerts_summary


def render_ai_status_badge():
    """Affiche un badge indiquant si l'IA est disponible."""
    if is_ai_available():
        st.success("🤖 Assistant IA actif", icon="✅")
    else:
        st.warning("🤖 Assistant IA non configuré", icon="⚠️")
        with st.expander("Comment activer l'IA?"):
            st.markdown("""
            1. Créez un compte sur [console.anthropic.com](https://console.anthropic.com)
            2. Générez une clé API
            3. Ajoutez-la dans `.streamlit/secrets.toml`:
            ```toml
            ANTHROPIC_API_KEY = "sk-ant-votre-clé"
            ```
            4. Redémarrez l'application
            """)


def render_ai_summary_widget(
    kpis: Dict,
    alerts: List[Dict],
    top_performers: List[str],
    underperformers: List[str],
    period: str,
    country_breakdown: Optional[Dict] = None,
):
    """
    Widget de synthèse IA automatique.
    
    Args:
        kpis: Dictionnaire des KPIs
        alerts: Liste des alertes actives
        top_performers: Liste des top boutiques
        underperformers: Liste des boutiques en difficulté
        period: Période d'analyse
        country_breakdown: Performance par pays (optionnel)
    """
    
    with st.container(border=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            st.markdown("### 🤖 Synthèse IA")
        
        with col2:
            regenerate = st.button("🔄", help="Régénérer la synthèse", key="regen_summary")
        
        if not is_ai_available():
            st.info("💡 Activez l'IA pour générer des synthèses automatiques.")
            return
        
        # Clé de cache basée sur la période et les KPIs
        cache_key = f"ai_summary_{period}_{kpis.get('total_revenue', 0):.0f}"
        
        # Générer ou récupérer du cache
        if regenerate or cache_key not in st.session_state:
            with st.spinner("Génération de la synthèse..."):
                # Ajouter pct_at_cruise si manquant
                if "pct_at_cruise" not in kpis:
                    kpis["pct_at_cruise"] = 0.0
                
                summary = generate_executive_summary(
                    kpis=kpis,
                    alerts=alerts,
                    top_performers=top_performers,
                    underperformers=underperformers,
                    period=period,
                    country_breakdown=country_breakdown,
                )
                st.session_state[cache_key] = summary
        
        # Afficher la synthèse
        st.markdown(st.session_state[cache_key])
        
        # Métadonnées
        st.caption(f"Généré le {datetime.now():%d/%m/%Y à %H:%M}")


def render_ai_chat_widget(
    df: pd.DataFrame,
    kpis: Dict,
    key_prefix: str = "chat",
):
    """
    Widget de chat Q&A sur les données.
    
    Args:
        df: DataFrame des données de ventes
        kpis: Dictionnaire des KPIs
        key_prefix: Préfixe pour les clés Streamlit (évite les conflits)
    """
    
    st.markdown("### 💬 Posez vos questions")
    
    if not is_ai_available():
        st.info("💡 Activez l'IA pour poser des questions sur vos données.")
        return
    
    # Suggestions de questions
    suggestions = get_suggested_questions(df)
    
    st.markdown("**Questions suggérées:**")
    cols = st.columns(len(suggestions))
    
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(
            suggestion,
            key=f"{key_prefix}_sug_{i}",
            use_container_width=True,
        ):
            st.session_state[f"{key_prefix}_question"] = suggestion
    
    # Champ de saisie
    question = st.text_input(
        "Votre question:",
        value=st.session_state.get(f"{key_prefix}_question", ""),
        placeholder="Ex: Quel pays a la meilleure croissance?",
        key=f"{key_prefix}_input",
    )
    
    col_btn, col_clear = st.columns([1, 1])
    
    with col_btn:
        analyze = st.button(
            "🔍 Analyser",
            type="primary",
            key=f"{key_prefix}_analyze",
            use_container_width=True,
        )
    
    with col_clear:
        if st.button(
            "🗑️ Effacer",
            key=f"{key_prefix}_clear",
            use_container_width=True,
        ):
            st.session_state[f"{key_prefix}_question"] = ""
            st.session_state[f"{key_prefix}_history"] = []
            st.rerun()
    
    # Historique des conversations
    if f"{key_prefix}_history" not in st.session_state:
        st.session_state[f"{key_prefix}_history"] = []
    
    # Traiter la question
    if analyze and question:
        with st.spinner("Analyse en cours..."):
            answer, _ = answer_data_question(question, df, kpis)
            
            # Ajouter à l'historique
            st.session_state[f"{key_prefix}_history"].append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().strftime("%H:%M"),
            })
    
    # Afficher l'historique (du plus récent au plus ancien)
    if st.session_state[f"{key_prefix}_history"]:
        st.markdown("---")
        st.markdown("**Historique:**")
        
        for i, entry in enumerate(reversed(st.session_state[f"{key_prefix}_history"][-5:])):
            with st.container(border=True):
                st.markdown(f"**Q ({entry['timestamp']}):** {entry['question']}")
                st.markdown(entry['answer'])


def render_ai_alerts_widget(
    alerts: List[Dict],
    contexts: Optional[Dict[str, Dict]] = None,
    max_display: int = 10,
):
    """
    Widget d'alertes intelligentes enrichies.
    
    Args:
        alerts: Liste des alertes brutes
        contexts: Contextes par boutique pour l'enrichissement
        max_display: Nombre max d'alertes à afficher
    """
    
    st.markdown("### 🚨 Alertes Intelligentes")
    
    if not alerts:
        st.success("✅ Aucune alerte active - Tout va bien!")
        return
    
    # Résumé des alertes
    summary = generate_alerts_summary(alerts)
    st.markdown(summary)
    
    st.markdown("---")
    
    # Option d'enrichissement IA
    if is_ai_available():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**Détail des alertes:**")
        
        with col2:
            enrich = st.button(
                "🤖 Enrichir avec IA",
                key="enrich_alerts",
                help="Ajouter diagnostic et recommandations IA",
            )
        
        # Enrichir si demandé
        cache_key = f"enriched_alerts_{len(alerts)}"
        
        if enrich or cache_key in st.session_state:
            if enrich:
                with st.spinner("Analyse des alertes..."):
                    enriched = enrich_alerts_batch(
                        alerts,
                        contexts=contexts,
                        max_alerts=min(max_display, 10),
                    )
                    st.session_state[cache_key] = enriched
            
            alerts_to_display = st.session_state.get(cache_key, alerts)
        else:
            alerts_to_display = alerts
    else:
        alerts_to_display = alerts
        st.info("💡 Activez l'IA pour enrichir les alertes avec des recommandations.")
    
    # Afficher les alertes
    for i, alert in enumerate(alerts_to_display[:max_display]):
        boutique = alert.get("Boutique", alert.get("boutique", "N/A"))
        niveau = alert.get("urgence_ajustee", alert.get("Niveau de risque", alert.get("niveau", "N/A")))
        
        # Couleur selon le niveau
        if "CRITIQUE" in str(niveau).upper():
            border_color = "red"
            icon = "🔴"
        elif "ÉLEV" in str(niveau).upper() or "ELEV" in str(niveau).upper():
            border_color = "orange"
            icon = "🟠"
        elif "MODÉR" in str(niveau).upper() or "MODER" in str(niveau).upper():
            border_color = "yellow"
            icon = "🟡"
        else:
            border_color = "green"
            icon = "🟢"
        
        with st.expander(f"{icon} **{boutique}** - {niveau}", expanded=(i < 3)):
            # Infos de base
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Pays:** {alert.get('Pays', alert.get('pays', 'N/A'))}")
            with col2:
                st.markdown(f"**Mois sans CA:** {alert.get('Mois sans CA', alert.get('mois_sans_ca', 'N/A'))}")
            
            # Raison
            raison = alert.get("Raison", alert.get("message", "N/A"))
            st.markdown(f"**Problème:** {raison}")
            
            # Si enrichi par IA
            if alert.get("enriched"):
                st.markdown("---")
                st.markdown("**🤖 Analyse IA:**")
                st.markdown(alert.get("diagnostic", ""))
                
                actions = alert.get("actions", [])
                if actions:
                    st.markdown("**Actions recommandées:**")
                    for action in actions:
                        st.markdown(f"- {action}")
                
                justif = alert.get("justification_urgence", "")
                if justif:
                    st.caption(f"💡 {justif}")


def render_ai_tab(
    df: pd.DataFrame,
    kpis: Dict,
    alerts: List[Dict],
    top_performers: List[str],
    underperformers: List[str],
    period: str,
    country_breakdown: Optional[Dict] = None,
):
    """
    Rendu complet de l'onglet IA.
    Combine tous les widgets IA dans une mise en page cohérente.
    
    Args:
        df: DataFrame des données
        kpis: Dictionnaire des KPIs
        alerts: Liste des alertes
        top_performers: Top boutiques
        underperformers: Boutiques en difficulté
        period: Période d'analyse
        country_breakdown: Performance par pays
    """
    
    st.subheader("🤖 Assistant IA Lapaire")
    
    # Badge de statut
    render_ai_status_badge()
    
    st.markdown("---")
    
    # Layout en 2 colonnes
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Synthèse narrative
        render_ai_summary_widget(
            kpis=kpis,
            alerts=alerts,
            top_performers=top_performers,
            underperformers=underperformers,
            period=period,
            country_breakdown=country_breakdown,
        )
    
    with col_right:
        # Chat Q&A
        render_ai_chat_widget(df, kpis)
    
    st.markdown("---")
    
    # Alertes intelligentes (pleine largeur)
    render_ai_alerts_widget(alerts)
