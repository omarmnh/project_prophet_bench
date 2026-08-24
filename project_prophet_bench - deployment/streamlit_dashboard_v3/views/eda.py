"""
views/eda.py
------------
Page 2 : Analyse Exploratoire (EDA) — évolution historique détaillée,
répartition par domaine et top compétences demandées.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DOMAIN_COLORS
from data_loader import generate_dummy_skills


def render(df: pd.DataFrame, domaines_selected: list[str]) -> None:
    st.title("🔎 Analyse Exploratoire (EDA)")
    st.caption("Explorez l'historique en détail avant de passer aux prévisions.")

    df_f = df[df["domaine"].isin(domaines_selected)]
    if df_f.empty:
        st.info("Sélectionnez au moins un domaine dans la barre latérale.")
        return

    tab_evol, tab_repartition, tab_skills = st.tabs(
        ["📈 Évolution historique", "🥧 Répartition", "🏷️ Top compétences"]
    )

    with tab_evol:
        _render_evolution(df_f)
    with tab_repartition:
        _render_repartition(df_f)
    with tab_skills:
        _render_skills(domaines_selected)


def _render_evolution(df_f: pd.DataFrame) -> None:
    metric_label = st.selectbox(
        "Indicateur", ["Bench (nb personnes)", "TACE (%)", "Volume d'opportunités"], index=0,
    )
    metric_col = {
        "Bench (nb personnes)": "y", "TACE (%)": "tace", "Volume d'opportunités": "volume_opportunites",
    }[metric_label]

    fig = px.line(
        df_f.sort_values("ds"), x="ds", y=metric_col, color="domaine",
        color_discrete_map=DOMAIN_COLORS,
    )
    fig.update_layout(height=430, yaxis_title=metric_label, xaxis_title="Semaine", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Lecture :** chaque courbe représente un domaine. Survolez le graphique pour "
        "comparer les valeurs semaine par semaine entre domaines."
    )


def _render_repartition(df_f: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    with col1:
        effectifs = df_f.drop_duplicates("domaine")[["domaine", "effectif_total"]]
        fig_pie = px.pie(
            effectifs, names="domaine", values="effectif_total",
            color="domaine", color_discrete_map=DOMAIN_COLORS, hole=0.45,
        )
        fig_pie.update_layout(height=380, title="Répartition de l'effectif total par domaine")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        tace_moyen = df_f.groupby("domaine", as_index=False)["tace"].mean().sort_values("tace")
        fig_bar = px.bar(
            tace_moyen, x="tace", y="domaine", orientation="h",
            color="domaine", color_discrete_map=DOMAIN_COLORS, text_auto=".1f",
        )
        fig_bar.update_layout(
            height=380, title="TACE moyen par domaine (%)", showlegend=False,
            xaxis_title="TACE moyen (%)", yaxis_title="",
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def _render_skills(domaines_selected: list[str]) -> None:
    st.caption(
        "🔗 Données simulées — à remplacer par une extraction NLP réelle des intitulés "
        "de poste / descriptions d'opportunités (TF-IDF, spaCy, etc.)."
    )
    df_skills = generate_dummy_skills()
    df_skills = df_skills[df_skills["domaine"].isin(domaines_selected)]

    domaine_choice = st.selectbox("Domaine", sorted(domaines_selected))
    top = df_skills[df_skills["domaine"] == domaine_choice].sort_values("frequence", ascending=True).tail(10)

    fig = go.Figure(go.Bar(
        x=top["frequence"], y=top["competence"], orientation="h",
        marker_color=DOMAIN_COLORS.get(domaine_choice),
    ))
    fig.update_layout(
        height=380, title=f"Top compétences demandées — {domaine_choice}",
        xaxis_title="Nb de mentions (simulé)",
    )
    st.plotly_chart(fig, use_container_width=True)
