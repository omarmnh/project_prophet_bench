"""
prophet_pipeline.py
--------------------
Pipeline de prévision en cascade (Prophet) : Opportunités -> Bench -> TACE.

⚠️ Ce fichier reprend VOTRE code tel quel (mêmes hyperparamètres, même logique
de cascade avec le régresseur `opp_lag4`). Il a seulement été réorganisé pour
être proprement importable depuis le dashboard (cf. forecasting.py) :
  - le chargement du fichier Excel n'est plus exécuté au niveau module
    (sinon le simple `import` du fichier plantait si le chemin Windows
    n'existait pas) ;
  - le bloc Streamlit (bouton, selectbox, line_chart) a été retiré, car
    l'interface est maintenant gérée par app.py / views/predictions.py.

Pour activer ce pipeline réel dans le dashboard :
    1. pip install prophet
    2. Placez rapport_par_semaine_domaine_v2.xlsx à la racine du projet
       (ou utilisez le bouton "Utiliser mes propres données" dans la sidebar)
    3. Dans config.py : USE_REAL_MODEL = True
"""

import logging

import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# ==========================================
# 1. HYPERPARAMÈTRES OPTIMAUX (issus de votre grid-search)
# ==========================================

BEST_PARAMS_OPP = {
    "Agile & Delivery": {"cps": 0.30, "sps": 0.1, "hps": 1.0},
    "Cybersécurité": {"cps": 0.01, "sps": 0.1, "hps": 0.1},
    "Data/Cloud": {"cps": 0.05, "sps": 10.0, "hps": 1.0},
    "Dev Fullstack": {"cps": 0.01, "sps": 0.1, "hps": 1.0},
    "DevOps / Infra": {"cps": 0.30, "sps": 0.1, "hps": 1.0},
}

BEST_PARAMS_BENCH = {
    "Agile & Delivery": {"cps": 0.05, "sps": 10.0, "hps": 10.0},
    "Cybersécurité": {"cps": 0.05, "sps": 10.0, "hps": 10.0},
    "Data/Cloud": {"cps": 0.30, "sps": 10.0, "hps": 10.0},
    "Dev Fullstack": {"cps": 0.05, "sps": 0.1, "hps": 1.0},
    "DevOps / Infra": {"cps": 0.30, "sps": 10.0, "hps": 0.1},
}


# ==========================================
# 2. JOURS FÉRIÉS / PÉRIODES SPÉCIALES (régresseur "holidays" Prophet)
# ==========================================

holidays_df = pd.DataFrame({
    "holiday": [
        # --- 2023 ---
        "fete_du_travail", "victoire_1945", "ascension", "fete_nationale",
        "vacances_ete", "assomption", "noel",
        # --- 2024 ---
        "fete_du_travail", "victoire_1945", "ascension", "fete_nationale",
        "vacances_ete", "assomption", "noel",
        # --- 2025 ---
        "fete_du_travail", "victoire_1945", "ascension", "fete_nationale",
        "vacances_ete", "assomption", "noel",
    ],
    "ds": pd.to_datetime([
        "2023-05-01", "2023-05-08", "2023-05-18", "2023-07-14",
        "2023-08-01", "2023-08-15", "2023-12-25",
        "2024-05-01", "2024-05-08", "2024-05-09", "2024-07-14",
        "2024-08-01", "2024-08-15", "2024-12-25",
        "2025-05-01", "2025-05-08", "2025-05-29", "2025-07-14",
        "2025-08-01", "2025-08-15", "2025-12-25",
    ]),
    "lower_window": [
        0, 0, 0, 0, 0, -1, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
    ],
    "upper_window": [
        0, 0, 3, 2, 14, 0, 6,
        0, 0, 3, 0, 14, 1, 6,
        1, 1, 3, 2, 14, 2, 7,
    ],
})


# ==========================================
# 3. FONCTIONS DE PRÉDICTION EN CASCADE
# ==========================================

def predict_opportunities(df_dom, dom_name, holidays_df, periods=12):
    """Étape 1 : prédit le volume d'opportunités sur les prochaines semaines."""
    try:
        df_opp = df_dom[["ds", "volume_opportunites"]].rename(columns={"volume_opportunites": "y"})
        params = BEST_PARAMS_OPP.get(dom_name, {"cps": 0.05, "sps": 1.0, "hps": 1.0})

        model_opp = Prophet(
            holidays=holidays_df,
            changepoint_prior_scale=params["cps"],
            seasonality_prior_scale=params["sps"],
            holidays_prior_scale=params["hps"],
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        model_opp.fit(df_opp)

        future_opp = model_opp.make_future_dataframe(periods=periods, freq="W-MON")
        forecast_opp = model_opp.predict(future_opp)

        df_opp_complete = forecast_opp[["ds", "yhat"]].rename(columns={"yhat": "volume_opportunites"})
        df_opp_complete["volume_opportunites"] = df_opp_complete["volume_opportunites"].round().astype(int)
        df_opp_complete["volume_opportunites"] = df_opp_complete["volume_opportunites"].clip(lower=0)

        return df_opp_complete

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction des opportunités pour {dom_name}: {e}")
        return None


def predict_bench_tace(df_dom, df_opp_complete, dom_name, holidays_df, periods=12, regressor_source="predicted"):
    """
    Étape 2 : utilise les opportunités décalées de 4 semaines pour prédire le bench et le TACE.

    regressor_source :
        "predicted" (défaut, comportement cascade original — inchangé) : le régresseur
            `opp_lag4` est construit à partir des opportunités PRÉDITES par le modèle
            Prophet de l'étape 1 (`yhat`), y compris sur l'historique. C'est ce que fait
            réellement run_forecasting_pipeline() en production (le vrai volume futur
            n'existe pas, donc on est obligé de partir d'une prédiction).
        "actual" : sur l'historique, le régresseur `opp_lag4` est construit à partir des
            VRAIES opportunités observées (colonne `volume_opportunites` de df_dom).
            Sur les semaines futures (au-delà de la dernière date connue), on retombe
            automatiquement sur les opportunités prédites, faute de vraie donnée future.
            -> Permet de vérifier si l'écart de WAPE observé entre votre évaluation
            "offline" et le dashboard vient de la propagation d'erreur en cascade.
    """
    try:
        df_opp_complete = df_opp_complete.copy()

        if regressor_source == "actual":
            df_real_opp = df_dom[["ds", "volume_opportunites"]].copy()
            df_opp_complete = df_opp_complete.merge(
                df_real_opp, on="ds", how="left", suffixes=("_pred", "_real")
            )
            # Vraie valeur si disponible (historique), sinon valeur prédite (futur)
            df_opp_complete["volume_opportunites"] = (
                df_opp_complete["volume_opportunites_real"]
                .combine_first(df_opp_complete["volume_opportunites_pred"])
            )
            df_opp_complete = df_opp_complete[["ds", "volume_opportunites"]]
        elif regressor_source != "predicted":
            raise ValueError(f"regressor_source doit valoir 'predicted' ou 'actual', reçu: {regressor_source!r}")

        df_opp_complete["opp_lag4"] = df_opp_complete["volume_opportunites"].shift(4)

        df_train = pd.merge(df_dom[["ds", "y"]], df_opp_complete[["ds", "opp_lag4"]], on="ds", how="right")
        df_train_clean = df_train.dropna(subset=["y", "opp_lag4"]).copy()

        params = BEST_PARAMS_BENCH.get(dom_name, {"cps": 0.05, "sps": 1.0, "hps": 1.0})
        model_bench = Prophet(
            holidays=holidays_df,
            changepoint_prior_scale=params["cps"],
            seasonality_prior_scale=params["sps"],
            holidays_prior_scale=params["hps"],
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        model_bench.add_regressor("opp_lag4")
        model_bench.fit(df_train_clean)

        future_bench = df_train[["ds", "opp_lag4"]].copy()
        future_bench["opp_lag4"] = future_bench["opp_lag4"].bfill().ffill()

        forecast_bench = model_bench.predict(future_bench)

        df_final = forecast_bench[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        df_final.rename(
            columns={"yhat": "bench_pred", "yhat_lower": "bench_min", "yhat_upper": "bench_max"},
            inplace=True,
        )

        df_final["bench_pred"] = df_final["bench_pred"].clip(lower=0).round()
        df_final["bench_min"] = df_final["bench_min"].clip(lower=0).round()

        last_effectif = df_dom["effectif_total"].iloc[-1]
        df_final["effectif_total"] = last_effectif
        df_final["tace_pred"] = (
            (df_final["effectif_total"] - df_final["bench_pred"]) / df_final["effectif_total"] * 100
        )
        df_final["domaine"] = dom_name

        return df_final

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction du bench pour {dom_name}: {e}")
        return None


def run_forecasting_pipeline(df_all, holidays_df, periods=12):
    """Orchestre les deux étapes pour tous les domaines et consolide les résultats."""
    domaines = df_all["domaine"].unique()
    all_predictions = []

    for dom in domaines:
        df_dom = df_all[df_all["domaine"] == dom].sort_values("ds")
        df_opp_complete = predict_opportunities(df_dom, dom, holidays_df, periods)

        if df_opp_complete is not None:
            df_final = predict_bench_tace(df_dom, df_opp_complete, dom, holidays_df, periods)
            if df_final is not None:
                df_final = pd.merge(df_final, df_dom[["ds", "y", "tace"]], on="ds", how="left")
                all_predictions.append(df_final)

    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    return pd.DataFrame()


def load_real_data(path: str = "rapport_par_semaine_domaine_v2.xlsx") -> pd.DataFrame:
    """Charge votre fichier réel (utilisé uniquement en exécution autonome, voir plus bas)."""
    return pd.read_excel(path)


# ==========================================
# Exécution autonome (facultative) : tester le pipeline hors Streamlit
# via `python prophet_pipeline.py`
# ==========================================
if __name__ == "__main__":
    df_all = load_real_data()
    df_results = run_forecasting_pipeline(df_all, holidays_df, periods=12)
    print(df_results.head(20))
