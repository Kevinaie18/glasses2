"""
Module IA pour Lapaire Dashboard V1.5
Intégration Claude API pour synthèses, Q&A et alertes intelligentes.
"""

from .client import get_client, call_claude
from .narrative import generate_executive_summary
from .chat import answer_data_question, get_suggested_questions
from .alerts import enrich_alert, enrich_alerts_batch

__all__ = [
    "get_client",
    "call_claude",
    "generate_executive_summary",
    "answer_data_question",
    "get_suggested_questions",
    "enrich_alert",
    "enrich_alerts_batch",
]
