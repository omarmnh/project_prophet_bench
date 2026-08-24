"""
views/overview.py
------------------
Page 1 : Vue d'ensemble (macro) — KPIs exécutifs + comparaison globale des domaines.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DOMAIN_COLORS
from utils import growth_by_domain, pct_growth


def render(df: pd.DataFrame, domaines_selected: list[str]) -> None:
    st.title("📊 Vue d'ensemble")
    st.caption("Résumé exécutif de l'activité Bench & Opportunités sur les domaines sélectionnés.")

    df_f = df[df["domaine"].isin(domaines_selected)]
    if df_f.empty:
        st.info("Sélectionnez au moins un domaine dans la barre latérale pour afficher la vue d'ensemble.")
        return

    _render_kpis(df_f)
    st.divider()
    _render_comparison_chart(df_f)


def _render_kpis(df_f: pd.DataFrame) -> None:
    last_date = df_f["ds"].max()

    # KPI 1 : volume total d'opportunités sur les 4 dernières semaines vs les 4 précédentes
    last_4w = df_f[df_f["ds"] > last_date - pd.Timedelta(weeks=4)]
    prev_4w = df_f[(df_f["ds"] <= last_date - pd.Timedelta(weeks=4)) &
                   (df_f["ds"] > last_date - pd.Timedelta(weeks=8))]
    vol_total = int(last_4w["volume_opportunites"].sum())
    vol_prev = prev_4w["volume_opportunites"].sum()

    # KPI 2 : domaine en plus forte croissance (sur le volume d'opportunités)
    growth_df = growth_by_domain(df_f, metric="volume_opportunites", window=4)
    top_domain_row = growth_df.iloc[0] if not growth_df.empty else None

    # KPI 3 : TACE moyen (période récente vs précédente)
    tace_recent = last_4w["tace"].mean()
    tace_prev = prev_4w["tace"].mean()

    # KPI 4 : effectif en bench sur la dernière semaine connue vs la semaine précédente
    last_week = df_f[df_f["ds"] == last_date]
    prev_week = df_f[df_f["ds"] == last_date - pd.Timedelta(weeks=1)]
    bench_last = int(last_week["y"].sum())
    bench_prev = int(prev_week["y"].sum()) if not prev_week.empty else None

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Volume d'opportunités (4 dern. sem.)", f"{vol_total}",
        delta=f"{pct_growth(vol_total, vol_prev)}%" if vol_prev else None,
        help="Somme des opportunités commerciales sur les 4 dernières semaines, "
             "comparée aux 4 semaines précédentes.",
    )

    if top_domain_row is not None:
        c2.metric(
            "Domaine en plus forte croissance", top_domain_row["domaine"],
            delta=f"{top_domain_row['growth_pct']}%",
            help="Domaine dont le volume d'opportunités a le plus progressé récemment.",
        )
    else:
        c2.metric("Domaine en plus forte croissance", "—")

    c3.metric(
        "TACE moyen", f"{tace_recent:.1f}%",
        delta=f"{round(tace_recent - tace_prev, 1)} pts" if pd.notna(tace_prev) else None,
        help="Taux d'occupation moyen (part du staff en mission). "
             "Plus il est élevé, moins il y a de bench.",
    )

    c4.metric(
        "Effectif en bench (dern. semaine)", f"{bench_last}",
        delta=f"{bench_last - bench_prev:+d}" if bench_prev is not None else None,
        delta_color="inverse",
        help="Nombre total de personnes sans mission sur la dernière semaine connue "
             "(une hausse est signalée en rouge : situation à surveiller).",
    )


def _render_comparison_chart(df_f: pd.DataFrame) -> None:
    st.subheader("Évolution comparée des domaines")
    metric_label = st.radio(
        "Indicateur à comparer", ["TACE (%)", "Bench (nb personnes)", "Opportunités"],
        horizontal=True, label_visibility="collapsed",
    )
    metric_col = {
        "TACE (%)": "tace", "Bench (nb personnes)": "y", "Opportunités": "volume_opportunites",
    }[metric_label]

    fig = go.Figure()
    for dom, g in df_f.groupby("domaine"):
        g = g.sort_values("ds")
        fig.add_trace(go.Scatter(
            x=g["ds"], y=g[metric_col], mode="lines", name=dom,
            line=dict(color=DOMAIN_COLORS.get(dom), width=2),
        ))
    fig.update_layout(height=420, yaxis_title=metric_label, xaxis_title="Semaine", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
