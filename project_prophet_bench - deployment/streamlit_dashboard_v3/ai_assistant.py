"""
ai_assistant.py
----------------
Assistant IA (Google Gemini) intégré dans la sidebar, pour interroger en
langage naturel les prévisions Bench/TACE calculées par le pipeline.

🔐 SÉCURITÉ DE LA CLÉ API — LIRE AVANT UTILISATION :
   La clé API n'est JAMAIS écrite en dur dans ce fichier ni ailleurs dans le
   code. Elle est lue EXCLUSIVEMENT via st.secrets, qui va la chercher :
     - en local : dans .streamlit/secrets.toml (fichier à créer vous-même à
       partir de .streamlit/secrets.toml.example — JAMAIS commité dans Git,
       voir .gitignore)
     - en déploiement (Streamlit Community Cloud) : dans Settings > Secrets
       de l'application, saisie directement dans l'interface web, jamais
       dans le code source ni dans un fichier versionné

   Si une clé API a déjà été partagée en clair quelque part (chat, code,
   dépôt Git, capture d'écran...), elle doit être considérée comme
   compromise : régénérez-en une nouvelle immédiatement sur
   https://aistudio.google.com/apikey avant toute mise en production, et
   révoquez l'ancienne.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import DEFAULT_HORIZON_WEEKS, DOMAINES
from forecasting import get_forecast_for_domain

GEMINI_MODEL_NAME = "gemini-3.5-flash"


def _get_api_key() -> str | None:
    """Lit la clé API UNIQUEMENT depuis st.secrets. Aucun fallback en dur, jamais."""
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        # st.secrets peut lever une exception si aucun fichier secrets.toml n'existe du tout
        return None


def _get_model(system_instruction: str):
    """Configure et retourne le modèle Gemini, ou None si la clé est absente/invalide/librairie manquante."""
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # import différé : ne casse pas l'app si non installé

        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL_NAME, system_instruction=system_instruction)
    except Exception as e:
        st.session_state["ai_config_error"] = str(e)
        return None


@st.cache_data(show_spinner="Génération des prévisions pour l'assistant IA...")
def _build_context_dataframe(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule, pour CHAQUE domaine, la fenêtre de prévision "temps réel" (aujourd'hui
    -> aujourd'hui + horizon par défaut du domaine) et les concatène. C'est ce
    dataframe qui sert de contexte factuel à l'assistant IA (colonnes bench_pred /
    tace_pred / domaine) — équivalent de votre st.session_state['df_results'],
    mais construit ici à partir de forecasting.get_forecast_for_domain() pour
    rester cohérent avec le reste du dashboard (bascule démo/réel, fenêtrage
    temps réel déjà en place).
    """
    aujourdhui = pd.Timestamp.now().normalize()
    frames = []
    for dom in DOMAINES:
        horizon = DEFAULT_HORIZON_WEEKS.get(dom, 4)
        df_forecast = get_forecast_for_domain(df_all, dom, horizon)
        date_limite = aujourdhui + pd.Timedelta(weeks=horizon)
        future = df_forecast[(df_forecast["ds"] >= aujourdhui) & (df_forecast["ds"] <= date_limite)]
        frames.append(future)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_system_instruction(df_results: pd.DataFrame) -> str:
    summary_data = (
        df_results.groupby("domaine")
        .agg({"bench_pred": ["mean", "max"], "tace_pred": ["mean", "min"]})
        .round(1)
        .to_string()
    )
    return f"""Tu es un expert RH et Business Analyst spécialisé en ESN.
Analyse les données de prévision TACE (%) et Bench (nombre de consultants) ci-dessous :
{summary_data}

Instructions :
- Sois concis, clair et direct.
- Utilise des puces si nécessaire.
- Propose des recommandations concrètes d'optimisation de staffing."""


def _ask_gemini(question: str) -> str:
    """Envoie la question à Gemini avec l'historique de conversation, retourne la réponse (ou un message d'erreur lisible)."""
    df_results = st.session_state.get("df_results")
    if df_results is None or df_results.empty:
        return (
            "Aucune donnée de prévision disponible pour construire le contexte. "
            "Cliquez sur 🔄 Actualiser pour générer le contexte de l'assistant."
        )

    system_instruction = _build_system_instruction(df_results)
    model = _get_model(system_instruction)
    if model is None:
        return (
            "⚠️ Assistant IA indisponible : clé GEMINI_API_KEY absente ou invalide, ou "
            "librairie `google-generativeai` non installée. Consultez .streamlit/secrets.toml.example."
        )

    # Historique converti au format attendu par l'API Gemini (rôle "model", pas "assistant")
    gemini_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in st.session_state.messages[:-1]  # on exclut la question qu'on vient d'ajouter
    ]

    try:
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(question)
        return response.text
    except Exception as e:
        return f"⚠️ Erreur lors de l'appel à Gemini : {e}"


def _handle_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("L'assistant réfléchit..."):
        answer = _ask_gemini(question)
    st.session_state.messages.append({"role": "assistant", "content": answer})


def render_sidebar_assistant(df_all: pd.DataFrame) -> None:
    """Point d'entrée : à appeler depuis app.py, après le chargement des données."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.sidebar.divider()
    st.sidebar.subheader("🤖 Assistant IA - Analyse TACE & Bench")

    if _get_api_key() is None:
        st.sidebar.warning(
            "Clé API Gemini non configurée. Ajoutez `GEMINI_API_KEY` dans "
            "`.streamlit/secrets.toml` (voir `.streamlit/secrets.toml.example`) "
            "pour activer l'assistant."
        )
        return

    if st.session_state.get("df_results") is None:
        st.session_state["df_results"] = _build_context_dataframe(df_all)

    col_refresh, col_clear = st.sidebar.columns(2)
    if col_refresh.button("🔄 Actualiser", use_container_width=True,
                           help="Recalcule les prévisions utilisées comme contexte par l'IA"):
        st.session_state["df_results"] = _build_context_dataframe(df_all)
    if col_clear.button("🗑️ Effacer chat", use_container_width=True):
        st.session_state.messages = []

    with st.sidebar.expander("💡 Questions rapides", expanded=False):
        if st.button("📊 Résumé global des prévisions", use_container_width=True, key="q_resume"):
            _handle_question("Fais-moi un résumé global des prévisions pour les prochains mois.")

        if st.button("⚠️ Domaine le plus critique", use_container_width=True, key="q_critique"):
            _handle_question("Quel est le domaine le plus critique qui risque d'avoir le plus de bench ?")

        domaine_choisi = st.selectbox("Analyser un pôle spécifique", options=DOMAINES, key="q_domaine_select")
        if st.button("💡 Stratégie RH pour ce pôle", use_container_width=True, key="q_domaine_btn"):
            _handle_question(f"Quelle stratégie RH recommanderais-tu pour le pôle {domaine_choisi} ?")

    # Historique de la conversation. st.chat_message fonctionne dans la sidebar
    # (contrairement à st.chat_input, explicitement interdit par Streamlit hors
    # de la zone principale — voir docs.streamlit.io/develop/api-reference/chat/st.chat_input).
    chat_container = st.sidebar.container(height=320)
    with chat_container:
        if not st.session_state.messages:
            st.caption("Posez une question ou choisissez une suggestion ci-dessus.")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Saisie libre : st.chat_input est INCOMPATIBLE avec la sidebar (limitation
    # documentée de Streamlit, StreamlitAPIException si on essaie). On utilise
    # donc un st.form + st.text_input, qui offre une UX quasi identique
    # (validation à l'Entrée, champ vidé après envoi grâce à clear_on_submit).
    with st.sidebar.form("ai_chat_form", clear_on_submit=True):
        question = st.text_input(
            "Votre question", placeholder="Posez votre question à l'IA...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Envoyer ➤", use_container_width=True)

    if submitted and question.strip():
        _handle_question(question.strip())
