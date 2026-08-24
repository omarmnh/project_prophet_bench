"""
config.py
----------
Point d'entrée UNIQUE pour tous les paramètres "métier" et visuels de l'application.
Modifier ce fichier suffit à adapter le dashboard (nouveaux domaines, nouvelles
couleurs, nouveaux horizons par défaut...) sans toucher au reste du code.
"""

from datetime import date

# ------------------------------------------------------------------
# 1. IDENTITÉ DE L'APPLICATION
# ------------------------------------------------------------------
APP_TITLE = "Pilotage Bench & Opportunités"
APP_ICON = "📊"
APP_SUBTITLE = "Dashboard prédictif — Data / DevOps / Tech"

# ------------------------------------------------------------------
# 2. DOMAINES MÉTIER
#    /!\ Doit correspondre EXACTEMENT aux valeurs de la colonne "domaine"
#    de rapport_par_semaine_domaine_v2.xlsx
# ------------------------------------------------------------------
DOMAINES = [
    "Agile & Delivery",
    "Cybersécurité",
    "Data/Cloud",
    "Dev Fullstack",
    "DevOps / Infra",
]

# ------------------------------------------------------------------
# 3. HORIZON DE PRÉVISION
#    Le cahier des charges demande un horizon ADAPTATIF selon le domaine
#    (ex : 1 mois par défaut pour la Data, 2 mois pour le DevOps).
# ------------------------------------------------------------------
HORIZON_OPTIONS = {
    "1 mois (~4 semaines)": 4,
    "2 mois (~8 semaines)": 8,
    "3 mois (~12 semaines)": 12,
}

DEFAULT_HORIZON_WEEKS = {
    "Agile & Delivery": 4,
    "Cybersécurité": 4,
    "Data/Cloud": 4,      # 1 mois : domaine au turnover rapide, on veut du court terme
    "Dev Fullstack": 4,
    "DevOps / Infra": 8,  # 2 mois : cycles de staffing plus longs
}

# Seuil d'alerte métier sur le TACE (taux d'occupation).
# En-dessous de ce seuil, on considère le niveau de bench comme préoccupant.
ALERT_TACE_THRESHOLD = 80.0

# ------------------------------------------------------------------
# 3bis. HORIZON DE CALCUL PROPHET (interne, différent de l'horizon affiché)
#    La prévision affichée doit toujours partir d'AUJOURD'HUI (temps réel),
#    pas de la dernière date du fichier Excel (qui peut dater de plusieurs
#    mois). On demande donc à Prophet un horizon de calcul volontairement
#    large -> le filtrage sur "aujourd'hui -> aujourd'hui + horizon choisi"
#    se fait ensuite à l'affichage (voir forecasting.py / views/predictions.py).
# ------------------------------------------------------------------
MIN_FORECAST_PERIODS_WEEKS = 120  # plancher (reprend la valeur de votre ancien dashboard)
FORECAST_BUFFER_WEEKS = 8         # marge de sécurité ajoutée au-delà de la fenêtre demandée

# ------------------------------------------------------------------
# 4. CHARTE GRAPHIQUE
#    Historique -> bleu / gris | Prédiction -> violet / vert | Alerte -> rouge / orange
# ------------------------------------------------------------------
COLOR_HISTORIQUE = "#6C8EBF"          # bleu corporate (données réelles)
COLOR_HISTORIQUE_GRIS = "#8A94A6"     # gris neutre (séries secondaires)
COLOR_PREDICTION = "#8A7FFF"          # violet (projection IA) — couleur principale
COLOR_PREDICTION_VERT = "#3DDC97"     # vert (variante de projection, ex. scénario optimiste)
COLOR_ALERTE_ROUGE = "#E85C4A"        # rouge (alerte / baisse forte)
COLOR_ALERTE_ORANGE = "#F5A623"       # orange (avertissement / vigilance)
COLOR_CI_FILL = "rgba(138, 127, 255, 0.18)"  # remplissage de l'intervalle de confiance

# Palette qualitative pour comparer plusieurs domaines sur un même graphique.
# On évite volontairement le rouge/orange, réservés au code couleur "alerte".
DOMAIN_COLORS = {
    "Agile & Delivery": "#6C8EBF",
    "Cybersécurité": "#4A4E69",
    "Data/Cloud": "#8A7FFF",
    "Dev Fullstack": "#3DDC97",
    "DevOps / Infra": "#5AC8FA",
}

# ------------------------------------------------------------------
# 5. BASCULE DÉMO <-> PRODUCTION
#    Passez USE_REAL_MODEL à True dès que prophet_pipeline.py + prophet
#    sont installés et que vos données réelles sont disponibles.
# ------------------------------------------------------------------
USE_REAL_MODEL = True
REAL_DATA_PATH = "rapport_par_semaine_domaine_training.xlsx"

TODAY = date.today()
