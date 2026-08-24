"""
app.py
------
Point d'entrée de l'application Streamlit.

Structure du projet
    app.py                  -> routage entre les 3 pages + filtres globaux (sidebar)
    config.py                -> constantes (domaines, couleurs, horizons par défaut)
    data_loader.py            -> chargement des données réelles / génération de données simulées
    forecasting.py            -> bascule démo <-> pipeline Prophet réel
    prophet_pipeline.py        -> VOTRE code de prévision en cascade (Prophet)
    ai_assistant.py            -> Assistant IA (Gemini) dans la sidebar
    styling.py                 -> thème visuel (CSS + template Plotly)
    utils.py                   -> KPI, métriques d'erreur, export CSV
    views/
        overview.py             -> Page 1 : Vue d'ensemble
        eda.py                  -> Page 2 : Analyse Exploratoire (EDA)
        predictions.py          -> Page 3 : Prévisions (le cœur du projet)

Lancement :
    pip install -r requirements.txt
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # puis y coller votre clé Gemini
    streamlit run app.py
"""
from __future__ import annotations
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import streamlit as st

from ai_assistant import render_sidebar_assistant
from config import APP_ICON, APP_SUBTITLE, APP_TITLE, DOMAINES
from data_loader import load_data
from styling import inject_custom_css, register_plotly_template
from views import eda, overview, predictions

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
inject_custom_css()
register_plotly_template()


def render_sidebar() -> tuple[str, list[str], tuple, pd.DataFrame]:
    """
    Affiche la navigation + les filtres globaux, charge les données et renvoie
    les choix de l'utilisateur ainsi que le DataFrame chargé.

    /!\\ IMPORTANT (bug corrigé) : on NE stocke PAS le fichier uploadé dans
    st.session_state suivi d'un st.rerun(). `st.file_uploader` conserve sa
    valeur d'un rerun à l'autre (le fichier reste "attaché" dans le navigateur
    tant qu'on ne le retire pas), donc un `if uploaded is not None: st.rerun()`
    se redéclenche à l'infini -> boucle de reruns qui fait planter l'app.
    Streamlit relance déjà le script automatiquement à chaque interaction :
    on lit simplement la valeur du widget et on l'utilise dans le MÊME run.
    """
    st.sidebar.title(f"{APP_ICON} {APP_TITLE}")
    st.sidebar.caption(APP_SUBTITLE)
    st.sidebar.divider()

    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio(
        "Page", ["Vue d'ensemble", "Analyse Exploratoire (EDA)", "Prévisions"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Filtres globaux")
    domaines_selected = st.sidebar.multiselect(
        "Domaine(s)", options=DOMAINES, default=DOMAINES,
        help="Filtre appliqué aux pages Vue d'ensemble et EDA. "
             "La page Prévisions dispose de son propre sélecteur (1 domaine à la fois, "
             "car chaque modèle s'entraîne séparément par domaine).",
    )

    with st.sidebar.expander("📁 Utiliser mes propres données"):
        st.caption("Format attendu : ds, y, domaine, effectif_total, tace, volume_opportunites")
        uploaded_file = st.file_uploader(
            "Fichier Excel ou CSV", type=["xlsx", "csv"], label_visibility="collapsed",
        )

    df = load_data(uploaded_file)

    min_date, max_date = df["ds"].min().date(), df["ds"].max().date()
    date_range = st.sidebar.slider(
        "Période d'analyse", min_value=min_date, max_value=max_date,
        value=(min_date, max_date), format="DD/MM/YYYY",
        help="N'affecte que les pages Vue d'ensemble et EDA : la page Prévisions "
             "s'entraîne toujours sur l'historique complet disponible.",
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Données à jour au {max_date.strftime('%d/%m/%Y')} • {len(df)} lignes chargées")

    return page, domaines_selected, date_range, df


def main() -> None:
    page, domaines_selected, date_range, df = render_sidebar()
    render_sidebar_assistant(df)

    if page == "Prévisions":
        # Important : la page Prévisions reçoit TOUJOURS l'historique complet (non filtré
        # par période), pour que le modèle s'entraîne sur toutes les données disponibles
        # et que l'horizon parte bien de la dernière semaine réellement connue.
        predictions.render(df, domaines_selected)
        return

    df_period = df[(df["ds"].dt.date >= date_range[0]) & (df["ds"].dt.date <= date_range[1])]

    if page == "Vue d'ensemble":
        overview.render(df_period, domaines_selected)
    else:
        eda.render(df_period, domaines_selected)


if __name__ == "__main__":
    main()
