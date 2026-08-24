"""
styling.py
----------
Thème visuel de l'application : CSS custom (cartes KPI, sidebar) + template
Plotly commun à tous les graphiques. Le template utilise des fonds transparents
pour bien s'adapter aussi bien au thème dark corporate (.streamlit/config.toml)
qu'à un thème clair, sans dupliquer la logique de couleur dans chaque page.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from config import COLOR_HISTORIQUE, COLOR_PREDICTION


def inject_custom_css() -> None:
    """Injecte un peu de CSS pour un rendu plus 'produit' que le Streamlit par défaut."""
    st.markdown(
        """
        <style>
        /* Cartes KPI : légère teinte + bordure + coins arrondis */
        div[data-testid="stMetric"] {
            background-color: rgba(138, 127, 255, 0.06);
            border: 1px solid rgba(138, 127, 255, 0.25);
            border-radius: 10px;
            padding: 14px 16px 10px 16px;
        }
        div[data-testid="stMetricLabel"] {
            font-weight: 500;
            opacity: 0.85;
        }
        /* Titres un peu plus aérés */
        h1, h2, h3 {
            letter-spacing: 0.2px;
        }
        /* Sidebar : un peu plus de respiration en haut */
        section[data-testid="stSidebar"] {
            padding-top: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def register_plotly_template() -> None:
    """
    Déclare un template Plotly "corporate_dark" réutilisé par toutes les pages
    (fond transparent, grille discrète, légende horizontale, tooltip uniforme).
    Appelé une seule fois au démarrage de l'app (voir app.py).
    """
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=13, color="#B8BCC8"),
        colorway=[COLOR_HISTORIQUE, COLOR_PREDICTION, "#3DDC97", "#5AC8FA", "#4A4E69"],
        xaxis=dict(showgrid=True, gridcolor="rgba(138,148,166,0.15)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(138,148,166,0.15)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1E2230", font_size=12, font_family="Inter"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    pio.templates["corporate_dark"] = template
    pio.templates.default = "corporate_dark"
