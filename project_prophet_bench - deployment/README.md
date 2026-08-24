# Pilotage Prédictif du Bench & du TACE — Prophet + Streamlit

Projet de prévision du nombre de consultants en intercontrat (Bench) et du Taux d'Activité
Congés Exclus (TACE) par domaine (Data/Cloud, DevOps/Infra, Agile & Delivery, Cybersécurité,
Dev Fullstack), via un modèle Prophet en cascade (Opportunités → Bench/TACE), restitué dans un
dashboard interactif Streamlit incluant un assistant IA (Gemini).

## 📁 Structure du projet

```
project_prophet_bench/
├── cleaning & agregation & transformation.ipynb   # Nettoyage, agrégation et mise en forme des données brutes
├── training models and evaluation.ipynb           # Entraînement Prophet + grid-search hyperparamètres + évaluation
├── rapport_par_semaine_domaine_training.xlsx      # ⭐ Dataset à utiliser pour l'entraînement (voir ci-dessous)
├── rapport_par_semaine_domaine_apres forecasting.xlsx  # Export de contrôle après génération des prévisions
├── rapport_nbr_consultant_par_semaine.xlsx        # Export intermédiaire (étape d'agrégation)
├── sujet_bench_triee.xlsx                         # Données brutes triées (étape de nettoyage)
├── sujet_bench_domaine.xlsx                       # Données brutes par domaine (étape de nettoyage)
├── sujet_bench_agrandie.xlsx                      # Données brutes enrichies (étape de nettoyage)
├── rapport et ppt/                                # Livrables de présentation (rapport, support PPT)
└── streamlit_dashboard_v3/                        # Application Streamlit (le dashboard à exécuter)
    ├── app.py
    ├── config.py
    ├── data_loader.py
    ├── forecasting.py
    ├── prophet_pipeline.py
    ├── ai_assistant.py
    ├── styling.py
    ├── utils.py
    ├── views/
    ├── .streamlit/
    │   ├── config.toml
    │   └── secrets.toml.example
    ├── requirements.txt
    └── README.md
```

Les notebooks (`cleaning & agregation & transformation.ipynb`, `training models and evaluation.ipynb`)
documentent la démarche data science complète (nettoyage, feature engineering, grid-search des
hyperparamètres Prophet, évaluation MAE/MAPE/WAPE). Le dossier `streamlit_dashboard_v3/` contient
l'application finale, autonome, qui restitue les résultats sous forme de dashboard décisionnel.

## ⭐ Quel dataset utiliser pour l'entraînement ?

**`rapport_par_semaine_domaine_training.xlsx`** est le fichier de référence à utiliser, aussi bien
pour ré-exécuter les notebooks que pour alimenter le dashboard Streamlit avec de vraies données.

Il contient une ligne par semaine et par domaine, avec les colonnes suivantes :

| Colonne | Description |
|---|---|
| `ds` | Date du lundi de la semaine (format date) |
| `y` | Nombre de consultants en bench cette semaine-là |
| `domaine` | Domaine technologique (Agile & Delivery, Cybersécurité, Data/Cloud, Dev Fullstack, DevOps / Infra) |
| `effectif_total` | Effectif total du domaine |
| `tace` | Taux d'Activité Congés Exclus (%) = (effectif_total − y) / effectif_total × 100 |
| `volume_opportunites` | Nombre d'opportunités commerciales sur la semaine |

Les autres fichiers Excel du dossier racine (`sujet_bench_triee`, `sujet_bench_domaine`,
`sujet_bench_agrandie`, `rapport_nbr_consultant_par_semaine`) sont des **exports intermédiaires**
générés au fil des étapes de nettoyage dans le notebook `cleaning & agregation & transformation.ipynb`
— ils ne doivent pas être utilisés directement pour l'entraînement ou pour le dashboard.
`rapport_par_semaine_domaine_apres forecasting.xlsx` est quant à lui un export de **contrôle**,
généré après coup pour vérifier visuellement la qualité des prévisions — ce n'est pas un fichier
d'entrée.

## 🚀 Lancer le dashboard Streamlit

```bash
cd streamlit_dashboard_v3
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur (`http://localhost:8501`). Par défaut,
elle démarre avec des données **simulées** pour permettre une prise en main immédiate sans rien
configurer. Pour utiliser les vraies données :

- **Option rapide (démo/présentation)** : dans la sidebar, section « 📁 Utiliser mes propres
  données », déposez `rapport_par_semaine_domaine_training.xlsx` (`.xlsx` ou `.csv` acceptés).
- **Option permanente** : copiez `rapport_par_semaine_domaine_training.xlsx` directement dans le
  dossier `streamlit_dashboard_v3/` et renommez-le selon la valeur de `REAL_DATA_PATH` définie dans
  `config.py` (par défaut `rapport_par_semaine_domaine_v2.xlsx` — à adapter si besoin).

## 🔮 Basculer du modèle de démonstration vers le vrai modèle Prophet

Par défaut, le dashboard utilise un modèle statistique simplifié pour permettre une démonstration
rapide sans dépendance lourde. Pour activer le véritable pipeline Prophet en cascade (celui
documenté et entraîné dans `training models and evaluation.ipynb`) :

1. `pip install prophet`
2. S'assurer que `rapport_par_semaine_domaine_training.xlsx` est bien accessible (voir section
   précédente)
3. Dans `streamlit_dashboard_v3/config.py`, passer `USE_REAL_MODEL = True`
4. Relancer `streamlit run app.py`

## 🤖 Assistant IA (Gemini) — configuration de la clé API

Le dashboard intègre un assistant conversationnel (Google Gemini) dans la sidebar, permettant
d'interroger les prévisions en langage naturel. La clé API n'est **jamais** écrite en dur dans le
code : elle est lue exclusivement via `st.secrets`.

**En local :**
```bash
cd streamlit_dashboard_v3
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Puis ouvrir `.streamlit/secrets.toml` et remplacer la valeur par votre propre clé :
```toml
GEMINI_API_KEY = "votre-cle-api-gemini"
```
Clé disponible/à régénérer sur : https://aistudio.google.com/apikey

**En déploiement (Streamlit Community Cloud ou équivalent) :** ne pas créer de fichier
`secrets.toml` sur le serveur — coller le même contenu directement dans *Settings > Secrets* de
l'application, via l'interface web.

`.streamlit/secrets.toml` est exclu du suivi Git via `.gitignore` : il ne doit **jamais** être
commité ni poussé sur GitHub. Seul `.streamlit/secrets.toml.example` (sans clé réelle) est
versionné, à titre de modèle pour les prochaines personnes qui cloneront le dépôt. Sans clé
configurée, le reste du dashboard fonctionne normalement ; seule la section Assistant IA affiche
un message d'avertissement invitant à configurer la clé.

## 📓 Rôle des notebooks

- **`cleaning & agregation & transformation.ipynb`** : nettoyage des exports bruts
  (`sujet_bench_*.xlsx`), agrégation hebdomadaire par domaine, calcul du TACE, production du
  dataset final `rapport_par_semaine_domaine_training.xlsx`.
- **`training models and evaluation.ipynb`** : recherche des hyperparamètres Prophet optimaux par
  domaine (grid-search sur `changepoint_prior_scale`, `seasonality_prior_scale`,
  `holidays_prior_scale`), entraînement du modèle en cascade (Opportunités → Bench/TACE), et
  évaluation via MAE/MAPE/WAPE. Les hyperparamètres retenus sont ceux repris dans
  `streamlit_dashboard_v3/prophet_pipeline.py`.



