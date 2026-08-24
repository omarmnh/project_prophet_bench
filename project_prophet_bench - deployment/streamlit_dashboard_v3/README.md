# Dashboard Bench & Opportunités — Guide rapide

## Installation

```bash
pip install -r requirements.txt
```

## Assistant IA (Gemini) — configuration de la clé API

L'assistant IA de la sidebar nécessite une clé API Gemini, jamais écrite en dur dans le code :

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# puis éditez .streamlit/secrets.toml et collez votre clé :
# GEMINI_API_KEY = "votre-cle"
```

`.streamlit/secrets.toml` est exclu par `.gitignore` — ne le committez jamais. En déploiement sur
Streamlit Community Cloud, collez le même contenu dans Settings > Secrets de l'application (pas de
fichier à créer). Sans clé configurée, le dashboard fonctionne normalement ; seule la section
Assistant IA affiche un message d'avertissement au lieu du chat.

Si une clé a déjà été partagée en clair quelque part (chat, code, dépôt Git), considérez-la comme
compromise et régénérez-en une nouvelle sur https://aistudio.google.com/apikey.

## Lancement (mode démo, données simulées)

```bash
streamlit run app.py
```

L'application démarre directement avec un jeu de données simulé (mêmes colonnes,
mêmes ordres de grandeur que `rapport_par_semaine_domaine_v2.xlsx`), donc vous
pouvez voir le rendu immédiatement sans rien configurer.

## Tester avec vos VRAIES données (sans toucher au code)

Ouvrez la sidebar → section **« Utiliser mes propres données »** → déposez
`rapport_par_semaine_domaine_v2.xlsx`. Les pages Vue d'ensemble et EDA
afficheront alors votre historique réel (la page Prévisions restera en mode
statistique simulé tant que l'étape suivante n'est pas faite).

## Basculer sur votre VRAI pipeline Prophet

1. `pip install prophet`
2. Vérifiez que `prophet_pipeline.py` est bien à la racine du projet (déjà en place —
   c'est votre code de cascade Opportunités → Bench/TACE, nettoyé pour être importable).
3. Dans `config.py` : `USE_REAL_MODEL = True`
4. Relancez `streamlit run app.py`.

La page Prévisions appellera alors automatiquement `predict_opportunities()` puis
`predict_bench_tace()` pour le domaine et l'horizon choisis dans l'interface —
aucune autre modification n'est nécessaire.

## Structure du projet

```
app.py                  routage + filtres globaux (sidebar)
config.py               constantes (domaines, couleurs, horizons par défaut)
data_loader.py           chargement réel / génération de données simulées
forecasting.py           bascule démo <-> pipeline Prophet réel
prophet_pipeline.py       votre pipeline Prophet en cascade
ai_assistant.py           Assistant IA (Gemini) dans la sidebar
styling.py               thème visuel (CSS + template Plotly)
utils.py                 KPI, métriques d'erreur (MAE/MAPE/WAPE), export CSV
views/
    overview.py          Page 1 : Vue d'ensemble
    eda.py               Page 2 : Analyse Exploratoire (EDA)
    predictions.py       Page 3 : Prévisions
.streamlit/config.toml  thème dark corporate
.streamlit/secrets.toml.example  template de la clé API Gemini (à copier)
.gitignore               exclut secrets.toml du versioning
```
