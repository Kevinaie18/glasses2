# Lapaire Dashboard V1.5 - Intégration IA

## 🎯 Vue d'ensemble

La V1.5 ajoute une couche d'intelligence artificielle au dashboard existant **sans modifier l'architecture**. Le fichier Excel reste la source de données unique.

### Nouvelles fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| 🤖 **Synthèse IA** | Génération automatique d'un mémo exécutif pour le board |
| 💬 **Chat Q&A** | Posez des questions en français sur vos données |
| 🚨 **Alertes intelligentes** | Diagnostic et recommandations contextualisées |

## 📁 Structure des fichiers

```
glasses/
├── app.py                    # ✏️ Modifié (ajout onglet IA)
├── ai/                       # 🆕 Nouveau module
│   ├── __init__.py
│   ├── client.py             # Client Claude API
│   ├── narrative.py          # Synthèses narratives
│   ├── chat.py               # Q&A sur les données
│   └── alerts.py             # Alertes intelligentes
├── components/               # 🆕 Nouveau module
│   ├── __init__.py
│   └── ai_widgets.py         # Widgets Streamlit
├── requirements.txt          # ✏️ Modifié (+anthropic)
└── .streamlit/
    └── secrets.toml          # 🆕 Configuration API (ne pas commit!)
```

## 🚀 Installation

### 1. Copier les fichiers

```bash
# Copier les nouveaux modules
cp -r ai/ votre-repo/glasses/
cp -r components/ votre-repo/glasses/
```

### 2. Mettre à jour requirements.txt

Ajouter cette ligne :
```
anthropic>=0.40.0,<1.0.0
```

### 3. Configurer la clé API

```bash
# Copier le template
cp .streamlit/secrets.toml.template .streamlit/secrets.toml

# Éditer et ajouter votre clé
nano .streamlit/secrets.toml
```

Contenu de `secrets.toml` :
```toml
ANTHROPIC_API_KEY = "sk-ant-api03-VOTRE-VRAIE-CLE"
```

⚠️ **Important** : Ajoutez `secrets.toml` à `.gitignore` !

### 4. Modifier app.py

Voir le fichier `app_v15.py` pour la version complète, ou appliquer manuellement les modifications décrites dans `INTEGRATION_GUIDE.py`.

**Modifications requises :**

1. **Imports** (en haut du fichier) :
```python
from components.ai_widgets import render_ai_tab
```

2. **Fonctions helper** (avant `main()`) :
```python
def get_top_performers(df, n=5): ...
def get_underperformers(df, n=5): ...
def get_country_breakdown(df): ...
```

3. **Ajouter l'onglet IA** dans les tabs :
```python
tab_overview, tab_countries, tab_shops, tab_forecasts, tab_alerts, tab_ai, tab_memo = st.tabs([
    "📊 Vue d'ensemble", "🌍 Pays", "🏪 Boutiques", "🔮 Prévisions", "⚠️ Alertes", "🤖 IA", "📝 Mémo"
])
```

4. **Contenu de l'onglet** (avant `with tab_memo:`) :
```python
with tab_ai:
    render_ai_tab(
        df=filtered,
        kpis=kpis_for_ai,
        alerts=alerts_list,
        top_performers=top_performers,
        underperformers=underperformers,
        period=period_str,
        country_breakdown=country_breakdown,
    )
```

### 5. Installer et lancer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Déploiement Streamlit Cloud

1. Push le code sur GitHub (sans `secrets.toml`)
2. Sur [share.streamlit.io](https://share.streamlit.io) :
   - Aller dans **Settings > Secrets**
   - Ajouter :
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-api03-VOTRE-CLE"
   ```
3. Redéployer l'app

## 🔑 Obtenir une clé API Anthropic

1. Créer un compte sur [console.anthropic.com](https://console.anthropic.com)
2. Aller dans **Settings > API Keys**
3. Cliquer **Create Key**
4. Ajouter un moyen de paiement (facturation à l'usage)

**Coût estimé** : ~$10-15/mois pour un usage normal

## 💡 Utilisation

### Synthèse IA

Dans l'onglet **🤖 IA**, la synthèse est générée automatiquement à partir des KPIs. Cliquez sur 🔄 pour régénérer.

### Chat Q&A

Posez des questions en français :
- "Quelles boutiques sous-performent ?"
- "Compare Uganda vs Kenya"
- "Quelle est la tendance ce trimestre ?"

### Alertes intelligentes

Cliquez sur **🤖 Enrichir avec IA** pour ajouter :
- Diagnostic de la situation
- Actions recommandées
- Urgence ajustée

## ⚠️ Limitations

- L'IA nécessite une connexion internet
- Coût API proportionnel à l'usage
- Les réponses sont générées, pas calculées (possibilité d'approximations)

## 📞 Support

En cas de problème :
1. Vérifier que la clé API est valide
2. Vérifier la connexion internet
3. Consulter les logs Streamlit

---

**Version** : 1.5.0  
**Date** : Janvier 2026  
**Compatibilité** : Streamlit 1.36+, Python 3.10+
