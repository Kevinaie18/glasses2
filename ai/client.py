"""
Client Claude API pour Lapaire Dashboard.
Gestion centralisée des appels API avec retry, cache et error handling.
"""

from __future__ import annotations

import time
from typing import Optional

import streamlit as st

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Configuration
DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # secondes


@st.cache_resource
def get_client() -> Optional[anthropic.Anthropic]:
    """
    Retourne un client Anthropic singleton.
    La clé API est lue depuis st.secrets.
    
    Returns:
        Client Anthropic ou None si non disponible
    """
    if not ANTHROPIC_AVAILABLE:
        st.warning("⚠️ Package 'anthropic' non installé. Installez-le avec: pip install anthropic")
        return None
    
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        st.warning(
            "🔑 Clé API Anthropic manquante. "
            "Ajoutez ANTHROPIC_API_KEY dans .streamlit/secrets.toml"
        )
        return None
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Erreur initialisation client Anthropic: {e}")
        return None


def call_claude(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 1000,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> str:
    """
    Appelle l'API Claude avec gestion d'erreurs et retry.
    
    Args:
        prompt: Le prompt utilisateur
        system: Instructions système (optionnel)
        max_tokens: Limite de tokens en sortie
        model: Modèle à utiliser
        temperature: Créativité (0.0-1.0)
    
    Returns:
        Réponse de Claude ou message d'erreur
    """
    client = get_client()
    
    if client is None:
        return "❌ Assistant IA non disponible. Vérifiez la configuration de la clé API."
    
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temperature,
            }
            
            if system:
                kwargs["system"] = system
            
            response = client.messages.create(**kwargs)
            
            # Extraire le texte de la réponse
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                return "❌ Réponse vide de l'API"
                
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return "⏳ API temporairement surchargée. Réessayez dans quelques instants."
            
        except anthropic.APIConnectionError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return "🌐 Erreur de connexion à l'API. Vérifiez votre connexion internet."
            
        except anthropic.AuthenticationError:
            return "🔑 Clé API invalide. Vérifiez votre configuration."
            
        except anthropic.APIError as e:
            return f"❌ Erreur API: {str(e)}"
            
        except Exception as e:
            return f"❌ Erreur inattendue: {str(e)}"
    
    return "❌ Échec après plusieurs tentatives"


def is_ai_available() -> bool:
    """Vérifie si l'IA est disponible et configurée."""
    return get_client() is not None
