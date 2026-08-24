"""
views/predictions.py
---------------------
Page 3 : Module de Prévisions — le cœur du projet.

Sélection d'un domaine + horizon adaptatif -> prévision (historique continue +
projection en pointillés + intervalle de confiance) -> tableau détaillé ->
métriques de fiabilité -> export CSV.

⏱️ TEMPS RÉEL (comme votre ancien dashboard) : la partie "prévision à venir"
(tableau + export) est toujours filtrée sur [aujourd'hui ; aujourd'hui + horizon
choisi], et non sur [dernière date du fichier ; +horizon]. forecasting.py se
charge de demander à Prophet un horizon de calcul assez large en coulisses
pour que cette fenêtre soit toujours couverte, même avec un fichier ancien.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    ALERT_TACE_THRESHOLD,
    COLOR_ALERTE_ROUGE,
    COLOR_CI_FILL,
    COLOR_HISTORIQUE,
    COLOR_PREDICTION,
    DEFAULT_HORIZON_WEEKS,
    HORIZON_OPTIONS,
)
from forecasting import get_forecast_for_domain
from utils import compute_error_metrics, to_csv_bytes


def render(df: pd.DataFrame, domaines_selected: list[str]) -> None:
    st.title("🔮 Prévisions")
    st.caption("Projection du bench et du taux d'occupation (TACE) par domaine.")

    if not domaines_selected:
        st.info("Sélectionnez au moins un domaine dans la barre latérale.")
        return

    domaine = st.selectbox(
        "Domaine à prévoir", sorted(domaines_selected),
        help="La prévision est calculée domaine par domaine : le pipeline s'entraîne "
             "séparément pour chacun (comme dans votre pipeline Prophet en cascade).",
    )

    horizon_weeks = _render_horizon_selector(domaine)

    with st.spinner(f"Calcul de la prévision pour « {domaine} »..."):
        df_forecast = get_forecast_for_domain(df, domaine, horizon_weeks)

    # Fenêtre "temps réel" : toujours ancrée sur AUJOURD'HUI, pas sur la
    # dernière date du fichier de données.
    aujourdhui = pd.Timestamp.now().normalize()
    date_limite = aujourdhui + pd.Timedelta(weeks=horizon_weeks)

    st.info(
        f"📍 Prévisions affichées du **{aujourdhui.strftime('%d/%m/%Y')}** "
        f"(aujourd'hui) au **{date_limite.strftime('%d/%m/%Y')}**."
    )

    _render_chart(df_forecast, domaine, date_limite)
    st.divider()
    _render_table(df_forecast, aujourdhui, date_limite)
    st.divider()
    _render_model_performance(df_forecast)
    _render_download(df_forecast, domaine, aujourdhui, date_limite)


def _render_horizon_selector(domaine: str) -> int:
    default_weeks = DEFAULT_HORIZON_WEEKS.get(domaine, 4)
    default_label = next(
        (label for label, wk in HORIZON_OPTIONS.items() if wk == default_weeks),
        list(HORIZON_OPTIONS.keys())[0],
    )
    label = st.select_slider(
        "Horizon de prévision", options=list(HORIZON_OPTIONS.keys()), value=default_label,
        help="Horizon par défaut adapté au domaine (ex : 1 mois pour la Data, "
             "2 mois pour le DevOps) — librement modifiable. Compté à partir "
             "d'aujourd'hui, pas de la dernière date du fichier.",
    )
    return HORIZON_OPTIONS[label]


def _render_chart(df_forecast: pd.DataFrame, domaine: str, date_limite: pd.Timestamp) -> None:
    metric_choice = st.radio("Métrique affichée", ["TACE (%)", "Bench (nb personnes)"], horizontal=True)

    last_hist_date = df_forecast.loc[df_forecast["y"].notna(), "ds"].max()
    hist = df_forecast[df_forecast["ds"] <= last_hist_date]
    # >= last_hist_date (et non >) pour que la ligne pointillée reparte du dernier point réel,
    # et <= date_limite pour ne pas afficher tout l'horizon de calcul interne (qui va bien
    # au-delà, volontairement, pour garantir que la fenêtre temps réel soit toujours couverte).
    fut = df_forecast[(df_forecast["ds"] >= last_hist_date) & (df_forecast["ds"] <= date_limite)]

    if metric_choice == "TACE (%)":
        y_hist_col, y_pred_col = "tace", "tace_pred"
        effectif = df_forecast["effectif_total"].iloc[0]
        # Moins de bench => plus de TACE : les bornes s'inversent en passant de bench à tace
        fut_upper = (effectif - fut["bench_min"]) / effectif * 100
        fut_lower = (effectif - fut["bench_max"]) / effectif * 100
        y_axis_title = "TACE (%)"
    else:
        y_hist_col, y_pred_col = "y", "bench_pred"
        fut_upper = fut["bench_max"]
        fut_lower = fut["bench_min"]
        y_axis_title = "Bench (nb personnes)"

    fig = go.Figure()

    # Historique : ligne continue bleu/gris
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist[y_hist_col], mode="lines", name="Historique",
        line=dict(color=COLOR_HISTORIQUE, width=2.5),
    ))

    # Bande d'incertitude (remplissage violet clair)
    fig.add_trace(go.Scatter(
        x=list(fut["ds"]) + list(fut["ds"][::-1]),
        y=list(fut_upper) + list(fut_lower[::-1]),
        fill="toself", fillcolor=COLOR_CI_FILL,
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name="Intervalle de confiance",
    ))

    # Prédiction : ligne pointillée violette
    fig.add_trace(go.Scatter(
        x=fut["ds"], y=fut[y_pred_col], mode="lines", name="Prévision",
        line=dict(color=COLOR_PREDICTION, width=2.5, dash="dash"),
    ))

    # Seuil d'alerte métier (pertinent uniquement pour la vue TACE)
    if metric_choice == "TACE (%)":
        fig.add_hline(
            y=ALERT_TACE_THRESHOLD, line_dash="dot", line_color=COLOR_ALERTE_ROUGE,
            annotation_text=f"Seuil d'alerte ({ALERT_TACE_THRESHOLD:.0f}%)",
            annotation_position="bottom right",
        )
        fig.add_hrect(y0=0, y1=ALERT_TACE_THRESHOLD, fillcolor=COLOR_ALERTE_ROUGE, opacity=0.06, line_width=0)

    fig.update_layout(
        height=460, yaxis_title=y_axis_title, xaxis_title="Semaine",
        hovermode="x unified", title=f"{domaine} — historique & prévision",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Comment lire ce graphique :** la ligne pleine correspond aux données réelles, "
        "la ligne pointillée à la prévision du modèle (y compris la période entre la dernière "
        "donnée connue et aujourd'hui, que le modèle doit combler), et la zone ombrée "
        "matérialise l'incertitude. La zone rouge signale une situation de TACE sous le "
        "seuil d'alerte."
    )


def _render_table(df_forecast: pd.DataFrame, aujourdhui: pd.Timestamp, date_limite: pd.Timestamp) -> None:
    st.subheader("⏱️ Prévisions en temps réel")

    future = df_forecast[(df_forecast["ds"] >= aujourdhui) & (df_forecast["ds"] <= date_limite)].copy()

    if future.empty:
        st.warning(
            "⚠️ Aucune donnée prédite pour cette période. Vérifiez que l'horizon de calcul "
            "interne (config.MIN_FORECAST_PERIODS_WEEKS) est suffisant, ou augmentez-le."
        )
        return

    future_display = future[["ds", "bench_pred", "bench_min", "bench_max", "tace_pred"]].rename(columns={
        "ds": "Semaine", "bench_pred": "Bench prévu", "bench_min": "Bench min",
        "bench_max": "Bench max", "tace_pred": "TACE prévu (%)",
    })

    st.dataframe(
        future_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Semaine": st.column_config.DateColumn("Semaine", format="DD/MM/YYYY"),
            "TACE prévu (%)": st.column_config.ProgressColumn(
                "TACE prévu (%)", min_value=0, max_value=100, format="%.1f%%",
            ),
        },
    )


def _render_model_performance(df_forecast: pd.DataFrame) -> None:
    with st.expander("📐 Performances du modèle", expanded=False):
        
        metrics = compute_error_metrics(df_forecast)
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", metrics["MAE"], help="Erreur absolue moyenne, en nombre de personnes.")
        c2.metric(
            "MAPE", f"{metrics['MAPE']}%",
            help="Erreur moyenne en %. Sensible aux semaines où le bench est proche de 0.",
        )
        c3.metric(
            "WAPE", f"{metrics['WAPE']}%",
            help="Erreur pondérée en % — plus robuste que le MAPE, à privilégier pour ce cas d'usage.",
        )
        st.caption(f"Calculé sur {metrics['n_obs']} semaines d'historique.")


def _render_download(
    df_forecast: pd.DataFrame, domaine: str, aujourdhui: pd.Timestamp, date_limite: pd.Timestamp
) -> None:
    future = df_forecast[(df_forecast["ds"] >= aujourdhui) & (df_forecast["ds"] <= date_limite)]
    if future.empty:
        return
    csv_bytes = to_csv_bytes(future)
    st.download_button(
        "⬇️ Télécharger les prévisions (CSV)",
        data=csv_bytes,
        file_name=f"previsions_{domaine.replace(' ', '_').replace('/', '-')}.csv",
        mime="text/csv",
    )
