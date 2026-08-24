"""
utils.py
--------
Fonctions transverses : calcul de KPI, métriques d'erreur, export CSV.
Volontairement indépendant de Streamlit -> facilement testable unitairement
(ex: pytest) indépendamment de l'interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pct_growth(current: float, previous: float) -> float:
    """Croissance en % entre deux valeurs, robuste à previous == 0 / NaN."""
    if previous in (0, None) or pd.isna(previous):
        return 0.0
    return round((current - previous) / previous * 100, 1)


def growth_by_domain(df: pd.DataFrame, metric: str = "volume_opportunites", window: int = 4) -> pd.DataFrame:
    """
    Calcule, pour chaque domaine, la croissance (%) entre les `window` dernières
    semaines et les `window` semaines précédentes. Sert au KPI "domaine en plus
    forte croissance" de la page Vue d'ensemble.
    """
    rows = []
    for dom, g in df.groupby("domaine"):
        g = g.sort_values("ds")
        recent = g[metric].tail(window).mean()
        previous = g[metric].tail(window * 2).head(window).mean()
        rows.append({
            "domaine": dom,
            "recent": recent,
            "previous": previous,
            "growth_pct": pct_growth(recent, previous),
        })
    return pd.DataFrame(rows).sort_values("growth_pct", ascending=False).reset_index(drop=True)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Ignore les semaines où y_true == 0 (division par zéro) — dans ce cas
    préférer le WAPE, plus robuste.
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error : rapporte l'erreur absolue totale au
    volume total réel. Plus robuste que le MAPE quand certaines semaines ont
    un bench proche de 0 (cas fréquent sur des petits domaines).
    """
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def compute_error_metrics(df_forecast: pd.DataFrame) -> dict:
    """
    Calcule MAE / MAPE / WAPE sur la portion HISTORIQUE du dataframe de
    prévision (là où la vraie valeur `y` est connue), comparée à `bench_pred`.

    NB — En production : remplacez ce calcul "in-sample" par un vrai backtest
    hors-échantillon, par ex. via `Prophet.cross_validation` +
    `performance_metrics` sur une fenêtre glissante, pour une estimation de
    fiabilité non biaisée.
    """
    hist = df_forecast.dropna(subset=["y", "bench_pred"])
    if hist.empty:
        return {"MAE": float("nan"), "MAPE": float("nan"), "WAPE": float("nan"), "n_obs": 0}

    y_true = hist["y"].to_numpy(dtype=float)
    y_pred = hist["bench_pred"].to_numpy(dtype=float)
    return {
        "MAE": round(mae(y_true, y_pred), 2),
        "MAPE": round(mape(y_true, y_pred), 1),
        "WAPE": round(wape(y_true, y_pred), 1),
        "n_obs": int(len(hist)),
    }


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Encode un DataFrame en CSV, séparateur ';' et encodage utf-8-sig
    (rendu correct des accents et compatibilité Excel FR par défaut).
    """
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")
