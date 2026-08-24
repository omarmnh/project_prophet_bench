"""
data_loader.py
--------------
Chargement des données réelles (xlsx) OU génération de données simulées ayant
EXACTEMENT le même schéma que rapport_par_semaine_domaine_v2.xlsx, pour pouvoir
lancer le dashboard immédiatement sans dépendance externe.

Colonnes attendues (identiques au fichier réel, vérifiées sur votre export) :
    ds                     -> date du lundi de la semaine (datetime)
    y                      -> nombre de personnes en bench cette semaine-là
    domaine                -> domaine technologique
    effectif_total         -> effectif total du domaine (constant dans le temps)
    tace                   -> taux d'occupation (%) = (effectif_total - y) / effectif_total * 100
    volume_opportunites    -> nombre d'opportunités commerciales sur la semaine
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from config import DOMAINES, REAL_DATA_PATH

# Effectifs "réalistes" par domaine (ordres de grandeur observés sur l'historique réel)
_EFFECTIFS = {
    "Agile & Delivery": 64,
    "Cybersécurité": 55,
    "Data/Cloud": 192,
    "Dev Fullstack": 60,
    "DevOps / Infra": 79,
}

# Volume moyen d'opportunités hebdomadaires par domaine (calibré sur l'historique réel)
_OPP_BASE = {
    "Agile & Delivery": 4,
    "Cybersécurité": 4,
    "Data/Cloud": 19,
    "Dev Fullstack": 6,
    "DevOps / Infra": 9,
}


def load_data(_uploaded_file=None) -> pd.DataFrame:
    """
    Point d'entrée UNIQUE pour récupérer df_all dans le reste de l'application.

    Ordre de priorité :
      1. Fichier uploadé via la sidebar (st.file_uploader) si présent — .xlsx OU .csv.
      2. REAL_DATA_PATH si le fichier existe à côté de app.py.
      3. Sinon (mode démo) -> génération d'un dataset simulé au format identique.

    🔗 EN PRODUCTION : vous pouvez aussi simplement remplacer le corps de cette
    fonction par : `return pd.read_excel(REAL_DATA_PATH)`.

    ⚠️ Volontairement PAS de @st.cache_data ici : cette fonction dépend d'un
    fichier uploadé (objet non hashable de façon fiable). Le préfixer par "_"
    pour le cache aurait exclu ce paramètre du calcul de la clé -> Streamlit
    aurait alors renvoyé le même résultat mis en cache (les données simulées)
    même après un nouvel upload, puisque rien d'autre ne change dans la
    signature de l'appel. La génération des données démo (ci-dessous) reste,
    elle, mise en cache normalement — c'est la seule partie coûteuse.
    """
    if _uploaded_file is not None:
        return _postprocess(_read_any_format(_uploaded_file))

    if Path(REAL_DATA_PATH).exists():
        return _postprocess(pd.read_excel(REAL_DATA_PATH))

    return _postprocess(generate_dummy_data())


def _read_any_format(file) -> pd.DataFrame:
    """Lit un fichier .xlsx OU .csv selon son extension (upload sidebar)."""
    name = getattr(file, "name", "")
    if name.lower().endswith(".csv"):
        # sep=None + engine="python" -> détecte automatiquement ',' ou ';'
        return pd.read_csv(file, sep=None, engine="python")
    return pd.read_excel(file)


def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values(["domaine", "ds"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def generate_dummy_data(start: str = "2023-01-16", n_weeks: int = 130, seed: int = 42) -> pd.DataFrame:
    """
    Génère un jeu de données hebdomadaire simulé (tendance + saisonnalité + bruit),
    statistiquement cohérent avec le fichier réel, pour chacun des domaines.

    🔗 Purement pour la démo : à remplacer par load_data() une fois vos vraies
    données disponibles (aucun autre fichier n'a besoin d'être modifié).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_weeks, freq="W-MON")

    frames = []
    for dom in DOMAINES:
        effectif = _EFFECTIFS.get(dom, 80)
        opp_base = _OPP_BASE.get(dom, 6)

        t = np.arange(n_weeks)
        trend = 1 + 0.15 * (t / n_weeks)                                    # légère croissance de la demande
        seasonality = 1 + 0.25 * np.sin(2 * np.pi * t / 52) + 0.1 * np.sin(2 * np.pi * t / 13)
        noise = rng.normal(1, 0.18, n_weeks)
        volume_opportunites = np.clip(opp_base * trend * seasonality * noise, 0, None).round().astype(int)

        # Le bench réagit avec un décalage : les opportunités d'il y a ~4 semaines
        # absorbent une partie du bench (même logique de lag que le vrai pipeline)
        opp_lag = pd.Series(volume_opportunites).shift(4).fillna(opp_base).to_numpy()
        base_bench = effectif * 0.18  # ~18% de bench "structurel"
        bench = base_bench - 0.35 * (opp_lag - opp_base) + rng.normal(0, effectif * 0.03, n_weeks)
        bench = np.clip(bench, 0, effectif * 0.5).round().astype(int)

        tace = (effectif - bench) / effectif * 100

        frames.append(pd.DataFrame({
            "ds": dates,
            "y": bench,
            "domaine": dom,
            "effectif_total": effectif,
            "tace": tace,
            "volume_opportunites": volume_opportunites,
        }))

    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def generate_dummy_skills(seed: int = 7) -> pd.DataFrame:
    """
    Jeu de données simulé "top compétences demandées" par domaine, utilisé
    sur la page EDA.

    🔗 En production, remplacez par une extraction NLP réelle (ex: TF-IDF /
    spaCy sur les intitulés ou descriptions des opportunités commerciales).
    """
    rng = np.random.default_rng(seed)
    skills_by_domain = {
        "Agile & Delivery": ["Scrum", "SAFe", "Kanban", "Jira", "Product Owner", "Coaching Agile"],
        "Cybersécurité": ["SOC", "Pentest", "ISO 27001", "SIEM", "IAM", "GRC"],
        "Data/Cloud": ["Python", "Spark", "Snowflake", "Airflow", "AWS", "Databricks"],
        "Dev Fullstack": ["React", "Node.js", "TypeScript", "Java", "Spring Boot", "API REST"],
        "DevOps / Infra": ["Kubernetes", "Terraform", "Docker", "CI/CD", "AWS", "Ansible"],
    }
    rows = []
    for dom, skills in skills_by_domain.items():
        counts = rng.integers(8, 45, size=len(skills))
        for skill, count in zip(skills, counts):
            rows.append({"domaine": dom, "competence": skill, "frequence": int(count)})
    return pd.DataFrame(rows)
