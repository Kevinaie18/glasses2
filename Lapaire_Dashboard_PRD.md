
# Product Requirements Document (PRD) – Lapaire Sales Performance Dashboard

## 1. Objectif du projet
Développer un dashboard interactif de suivi et d’analyse des performances du réseau de boutiques Lapaire, afin de :
- Suivre le chiffre d’affaires (CA) par pays, par boutique et dans le temps
- Identifier les tendances, pics saisonniers et anomalies
- Suivre la montée en puissance des nouvelles boutiques (rythme de croisière)
- Générer des prévisions ajustables pour 2025 et 2026
- Faciliter la prise de décision stratégique pour le conseil d’administration

## 2. Données en entrée
**Fichier Excel source :**
- Colonnes :  
  - `Outlet` : Nom de la boutique  
  - `Country` : Code pays (`CI`, `ML`, `TG`, `BJ`, `BF`, `UG`, `TZ`, `SN`, `MA`)  
  - `01/01/2023` → `01/07/2025` : CA mensuel en USD  
  - `Total général` : CA cumulé  
  - `Nbr de mois d'activité`  
  - `Monthly average (USD)`  

## 3. Fonctionnalités principales du dashboard
### 3.1. Vue d’ensemble
- Graphique d’évolution du CA global
- Graphique d’évolution du CA par pays (moyenne mensuelle par boutique)
- Comparaison YoY et MoM avec pourcentage de variation

### 3.2. Analyse par pays
- Répartition du CA total par pays et par boutique
- Heatmap mensuelle pour le **Top 5 boutiques** par pays sur les 12 derniers mois
- Analyse saisonnalité (pics / creux) et anomalies
- Poids relatif de chaque pays dans le CA global

### 3.3. Analyse par boutique
- Délai d’atteinte du rythme de croisière (moyenne mobile 3 mois, variation ≤10 %)
- CA moyen après rythme atteint
- Boutiques n’ayant pas encore atteint leur rythme
- Détection de fermetures probables (inactivité ≥3 mois)

### 3.4. Prévisions
- Baseline forecast 2025 et 2026 :
  - Ajustement dynamique via sliders :  
    - Croissance CA (%) par pays  
    - Taux d’ouverture de nouvelles boutiques  
    - Niveau de CA cible par boutique  
  - Projection mensuelle et cumulée
- Simulation impact stratégique (ouverture / fermeture / relocalisation)

### 3.5. Recommandations
- Classification automatique :  
  - **Renforcer** : >+10 % au-dessus du rythme ≥3 mois, YoY >+15 %, volatilité faible  
  - **Maintenir** : ±10 % autour du rythme, tendance stable  
  - **Corriger** : <−10 % du rythme ≥2 mois mais tendance positive récente  
  - **Relocaliser** : sous-rythme ≥6 mois + cannibalisation probable  
  - **Fermer** : sous-rythme ≥9–12 mois, tendance négative, faible contribution  
- Justification textuelle pour chaque recommandation
- Score pays pour décision surpondération / désinvestissement

## 4. Interactivité attendue
- Sélection de pays / boutiques via menu déroulant
- Filtre par période
- Sliders pour ajuster hypothèses de prévisions
- Bouton d’export PDF/Excel des visualisations et analyses

## 5. Visualisations
1. **Courbe CA global et par pays**
2. **Heatmaps** mensuelles par boutique
3. **Diagramme délai moyen** pour atteindre le rythme de croisière par pays
4. **Carte géographique** avec code couleur selon recommandation
5. **Graphique prévisionnel** 2025–2026 (baseline vs scénario ajusté)

## 6. Contraintes techniques
- **Backend** : Python (Pandas, NumPy, Prophet/Scikit-learn pour prévisions)
- **Frontend** : Streamlit
- **Fichiers de données** : Upload Excel dynamique
- **Compatibilité** : Doit tourner localement et être facilement déployable sur Streamlit Cloud
- **Code** : Éviter doublons pays dans baseline forecast

## 7. Livrables attendus
1. Script Python Streamlit complet (`dashboard.py`)
2. Modules séparés pour :
   - Chargement et nettoyage des données (`data_loader.py`)
   - Calculs et indicateurs (`analytics.py`)
   - Visualisations (`visuals.py`)
   - Prévisions (`forecast.py`)
3. Fichiers de configuration (`requirements.txt`, `README.md`)
4. Version prête à être poussée sur GitHub
