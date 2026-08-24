"""
forecasting.py
--------------
Point d'entrée UNIQUE utilisé par la page Prévisions pour obtenir une
prévision (historique + futur) pour un domaine donné.

Ce module fait la bascule entre :
  - le pipeline Prophet réel (prophet_pipeline.py), si config.USE_REAL_MODEL = True
  - un mode "démo" statistique léger (par défaut), qui renvoie EXACTEMENT le
    même format de colonnes que la vraie fonction predict_bench_tace().

⏱️ PRÉVISION "TEMPS RÉEL" (comme votre ancien dashboard) :
   La fenêtre affichée à l'écran doit toujours démarrer à AUJOURD'HUI, pas à
   la dernière date du fichier Excel (qui peut dater de plusieurs mois). On
   ne peut pas demander directement "periods=aujourd'hui+3 mois" à Prophet
   (son paramètre `periods` compte des semaines à partir de la DERNIÈRE DATE
   DU FICHIER, pas depuis aujourd'hui) : on calcule donc ici un horizon de
   calcul plus large que nécessaire (cf. _compute_required_periods), et c'est
   views/predictions.py qui filtre ensuite l'affichage sur
   [aujourd'hui ; aujourd'hui + horizon choisi].

👉 POUR BRANCHER VOS VRAIS MODÈLES, RIEN À CHANGER ICI :
   1. pip install prophet
   2. Vérifiez que prophet_pipeline.py est à la racine du projet (déjà fait)
   3. Dans config.py : USE_REAL_MODEL = True
   4. C'est tout : get_forecast_for_domain() appellera automatiquement
      predict_opportunities() puis predict_bench_tace() pour vous.

Colonnes garanties en sortie de get_forecast_for_domain() :
    ds, domaine, effectif_total, bench_pred, bench_min, bench_max, tace_pred, y, tace
    (y / tace = valeurs réelles historiques -> NaN sur la partie future)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import FORECAST_BUFFER_WEEKS, MIN_FORECAST_PERIODS_WEEKS, USE_REAL_MODEL


def get_forecast_for_domain(df_all: pd.DataFrame, domaine: str, horizon_weeks: int) -> pd.DataFrame:
    """
    Fonction publique appelée par views/predictions.py.

    `horizon_weeks` est la fenêtre que l'UTILISATEUR veut voir AFFICHÉE à partir
    d'aujourd'hui (4/8/12 semaines). En interne, on demande à Prophet un horizon
    de calcul plus large (`periods_weeks`) pour être sûr de couvrir jusqu'à
    aujourd'hui + horizon_weeks, même si le fichier de données est ancien.
    """
    df_dom = df_all[df_all["domaine"] == domaine]
    last_hist_date = df_dom["ds"].max()
    periods_weeks = _compute_required_periods(last_hist_date, horizon_weeks)

    if USE_REAL_MODEL:
        result = _real_forecast(df_all, domaine, periods_weeks)
        if result is not None:
            return result
        st.warning(
            "⚠️ Le pipeline Prophet réel n'a pas pu être exécuté "
            "(module manquant, `prophet` non installé, ou erreur d'entraînement) — "
            "bascule automatique sur la prévision simulée."
        )
    return _dummy_forecast(df_all, domaine, periods_weeks)


def _compute_required_periods(
    last_hist_date: pd.Timestamp,
    horizon_weeks: int,
    min_weeks: int = MIN_FORECAST_PERIODS_WEEKS,
    buffer_weeks: int = FORECAST_BUFFER_WEEKS,
) -> int:
    """
    Détermine combien de semaines demander à Prophet pour être certain que la
    fenêtre "temps réel" [aujourd'hui ; aujourd'hui + horizon_weeks] soit bien
    couverte par la prévision, même si `last_hist_date` est ancienne.

    Reprend l'astuce de votre ancien dashboard (periods=120 codé en dur), en la
    rendant automatique : si vos données vieillissent encore, l'horizon de
    calcul s'élargit tout seul au lieu de tomber court.
    """
    today = pd.Timestamp.now().normalize()
    weeks_since_last_hist = max(0, (today - last_hist_date).days // 7)
    return max(min_weeks, weeks_since_last_hist + horizon_weeks + buffer_weeks)


def _real_forecast(df_all: pd.DataFrame, domaine: str, periods_weeks: int):
    """Appelle votre vrai pipeline Prophet en cascade (cf. prophet_pipeline.py)."""
    try:
        from prophet_pipeline import holidays_df, predict_bench_tace, predict_opportunities

        df_dom = df_all[df_all["domaine"] == domaine].sort_values("ds")
        df_opp = predict_opportunities(df_dom, domaine, holidays_df, periods=periods_weeks)
        if df_opp is None:
            return None

        df_final = predict_bench_tace(df_dom, df_opp, domaine, holidays_df, periods=periods_weeks)
        if df_final is None:
            return None

        df_final = df_final.merge(df_dom[["ds", "y", "tace"]], on="ds", how="left")
        return df_final
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _dummy_forecast(df_all: pd.DataFrame, domaine: str, periods_weeks: int) -> pd.DataFrame:
    """
    Prévision simulée (tendance récente + saisonnalité + incertitude croissante
    avec l'horizon), utilisée en mode démo. Reproduit la FORME d'un résultat
    Prophet (bande de confiance qui s'élargit avec l'horizon) sans dépendre de
    la librairie `prophet`.

    🔗 Cette fonction n'est JAMAIS appelée quand USE_REAL_MODEL = True.
    """
    df_dom = df_all[df_all["domaine"] == domaine].sort_values("ds").reset_index(drop=True)
    effectif = df_dom["effectif_total"].iloc[-1]
    y_hist = df_dom["y"].to_numpy(dtype=float)
    n = len(df_dom)

    # --- 1. "Fitted" historique : moyenne mobile lissée -> sert de référence pour le backtest
    window = 5
    fitted = pd.Series(y_hist).rolling(window, min_periods=1, center=True).mean().to_numpy()

    # --- 2. Projection future : tendance récente + légère saisonnalité + incertitude croissante
    recent = y_hist[-13:] if n >= 13 else y_hist
    trend_slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) > 1 else 0.0
    last_value = fitted[-1]
    residual_std = float(np.std(y_hist[-13:] - fitted[-13:])) if n >= 13 else (float(np.std(y_hist)) or 1.0)
    residual_std = residual_std or 1.0

    future_idx = np.arange(1, periods_weeks + 1)
    seasonal = 2.0 * np.sin(2 * np.pi * future_idx / 13)
    future_bench = last_value + trend_slope * future_idx + seasonal
    future_bench = np.clip(future_bench, 0, effectif)

    # L'incertitude grandit avec l'horizon, comme le ferait un vrai intervalle Prophet
    uncertainty = residual_std * (1 + 0.18 * future_idx)
    future_lower = np.clip(future_bench - uncertainty, 0, effectif)
    future_upper = np.clip(future_bench + uncertainty, 0, effectif)

    future_dates = pd.date_range(
        start=df_dom["ds"].iloc[-1] + pd.Timedelta(weeks=1), periods=periods_weeks, freq="W-MON"
    )

    df_hist_out = pd.DataFrame({
        "ds": df_dom["ds"],
        "bench_pred": fitted.round(),
        "bench_min": np.clip(fitted - residual_std, 0, effectif).round(),
        "bench_max": np.clip(fitted + residual_std, 0, effectif).round(),
    })
    df_future_out = pd.DataFrame({
        "ds": future_dates,
        "bench_pred": future_bench.round(),
        "bench_min": future_lower.round(),
        "bench_max": future_upper.round(),
    })

    df_final = pd.concat([df_hist_out, df_future_out], ignore_index=True)
    df_final["effectif_total"] = effectif
    df_final["tace_pred"] = (effectif - df_final["bench_pred"]) / effectif * 100
    df_final["domaine"] = domaine
    df_final = df_final.merge(df_dom[["ds", "y", "tace"]], on="ds", how="left")
    return df_final
