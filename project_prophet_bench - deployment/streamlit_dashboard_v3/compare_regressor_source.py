"""
compare_regressor_source.py
----------------------------
Script de diagnostic à exécuter EN LOCAL (là où `prophet` est installé), pour
objectiver l'écart de WAPE constaté entre votre évaluation "offline" et le
dashboard Streamlit.

Compare, pour chaque domaine, DEUX variantes du modèle Bench (même hyperparamètres,
même données — seul le régresseur change) :

    "predicted" -> opp_lag4 = opportunités PRÉDITES par le modèle Prophet #1 (yhat),
                   y compris sur l'historique. C'est le comportement réel de la
                   cascade en production (run_forecasting_pipeline / dashboard).

    "actual"    -> opp_lag4 = VRAIES opportunités observées sur l'historique
                   (fallback sur les valeurs prédites uniquement pour les
                   semaines futures, où la vraie donnée n'existe pas encore).
                   C'est probablement ce qu'utilisait votre évaluation offline.

Si la colonne "actual" se rapproche de votre screenshot (~9-15 % de WAPE) et que
la colonne "predicted" se rapproche de ce que montre Streamlit (~14-20 %), la
propagation d'erreur en cascade est confirmée comme cause principale de l'écart.

Lancement :
    pip install prophet   (si pas déjà fait)
    python compare_regressor_source.py
"""
import numpy as np
import pandas as pd

from prophet_pipeline import holidays_df, load_real_data, predict_bench_tace, predict_opportunities


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom else float("nan")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_domain(df_all: pd.DataFrame, dom_name: str, periods: int = 12) -> dict | None:
    df_dom = df_all[df_all["domaine"] == dom_name].sort_values("ds")
    df_opp = predict_opportunities(df_dom, dom_name, holidays_df, periods=periods)
    if df_opp is None:
        return None

    results = {}
    for source in ("predicted", "actual"):
        df_final = predict_bench_tace(
            df_dom, df_opp, dom_name, holidays_df, periods=periods, regressor_source=source
        )
        if df_final is None:
            results[source] = {"MAE": float("nan"), "MAPE": float("nan"), "WAPE": float("nan"), "n_obs": 0}
            continue

        merged = df_final.merge(df_dom[["ds", "y"]], on="ds", how="inner")
        merged = merged.dropna(subset=["y", "bench_pred"])
        y_true = merged["y"].to_numpy(dtype=float)
        y_pred = merged["bench_pred"].to_numpy(dtype=float)

        results[source] = {
            "MAE": round(mae(y_true, y_pred), 2),
            "MAPE": round(mape(y_true, y_pred), 1),
            "WAPE": round(wape(y_true, y_pred), 1),
            "n_obs": int(len(merged)),
        }
    return results


def main() -> None:
    df_all = load_real_data()  # rapport_par_semaine_domaine_v2.xlsx, à la racine du projet
    domaines = df_all["domaine"].unique()

    rows = []
    for dom in domaines:
        print(f"Traitement : {dom}...")
        res = evaluate_domain(df_all, dom)
        if res is None:
            continue
        rows.append({
            "domaine": dom,
            "WAPE prédit (cascade réelle)": res["predicted"]["WAPE"],
            "WAPE réel (backtest historique)": res["actual"]["WAPE"],
            "écart (pts)": round(res["predicted"]["WAPE"] - res["actual"]["WAPE"], 1),
        })

    df_compare = pd.DataFrame(rows)
    print()
    print(df_compare.to_string(index=False))
    print()
    print(
        "Lecture : si 'WAPE réel' est proche de votre screenshot (~9-15 %) et que 'WAPE prédit'\n"
        "est proche de ce que montre Streamlit (~14-20 %), la propagation d'erreur en cascade\n"
        "(le modèle Bench hérite de l'imprécision du modèle Opportunités, même sur l'historique)\n"
        "est bien la cause principale de l'écart observé."
    )


if __name__ == "__main__":
    main()
