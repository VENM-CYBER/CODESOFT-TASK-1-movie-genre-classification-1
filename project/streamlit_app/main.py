"""
streamlit_app/main.py
----------------------
Multi-page Streamlit application for the Movie Genre Classification System.

Run: streamlit run streamlit_app/main.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="CineGenre AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import page modules ───────────────────────────────────────────────────
from streamlit_app.pages import home, dataset_insights, genre_predictor, model_performance, about

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,600;1,9..40,300&display=swap');

/* ── Root vars ── */
:root {
    --bg:       #0d0f1a;
    --surface:  #161928;
    --border:   #252840;
    --accent:   #6C63FF;
    --accent2:  #FF6584;
    --accent3:  #43B89C;
    --text:     #e8eaf6;
    --muted:    #7b7fa6;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── Hide default hamburger + footer ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 0.08em;
    color: var(--accent);
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: var(--accent); font-size: 1.6rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #9c94ff);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.6rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Text area ── */
.stTextArea textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── DataFrames ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: var(--accent); }

/* ── Genre badge ── */
.genre-badge {
    display: inline-block;
    padding: 0.35rem 1.1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.06em;
    background: linear-gradient(135deg, var(--accent), #9c94ff);
    color: white;
}

/* ── Card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🎬 CineGenre AI")
    st.markdown("<p style='color:var(--muted);margin-top:-0.5rem;font-size:0.85rem;'>Movie Genre Classifier</p>", unsafe_allow_html=True)
    st.markdown("---")

    PAGES = {
        "🏠  Home":               "home",
        "📊  Dataset Insights":   "dataset",
        "🎯  Genre Predictor":    "predictor",
        "📈  Model Performance":  "performance",
        "ℹ️   About Project":     "about",
    }

    if "page" not in st.session_state:
        st.session_state.page = "home"

    for label, key in PAGES.items():
        active = st.session_state.page == key
        style  = "color:var(--accent);font-weight:700;" if active else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='color:var(--muted);font-size:0.75rem;text-align:center;'>"
        "Built with ❤️ using Streamlit<br>& Scikit-Learn</p>",
        unsafe_allow_html=True,
    )


# ── Page router ────────────────────────────────────────────────────────────
page = st.session_state.get("page", "home")

if page == "home":
    home.render()
elif page == "dataset":
    dataset_insights.render()
elif page == "predictor":
    genre_predictor.render()
elif page == "performance":
    model_performance.render()
elif page == "about":
    about.render()
