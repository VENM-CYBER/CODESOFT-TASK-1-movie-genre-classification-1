"""
streamlit_app/pages/home.py
----------------------------
Home page for the CineGenre AI Streamlit app.
"""

import streamlit as st
import os


def render():
    # ── Hero ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:3rem 0 2rem;'>
        <div style='font-size:4rem;'>🎬</div>
        <h1 style='font-family:"Bebas Neue",sans-serif;font-size:3.8rem;
                   letter-spacing:0.08em;color:#6C63FF;margin:0;line-height:1;'>
            CINEGENRE AI
        </h1>
        <p style='color:#7b7fa6;font-size:1.15rem;margin-top:0.6rem;'>
            NLP-Powered Movie Genre Classification System
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature cards row ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    cards = [
        ("🧠", "3 ML Models", "Logistic Regression, Naive Bayes & Linear SVM"),
        ("⚡", "Instant Prediction", "Get genre predictions in milliseconds"),
        ("📊", "10 Genres", "Action, Drama, Horror, Sci-Fi & more"),
        ("📈", "Full Analytics", "EDA, confusion matrices & model comparison"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center;'>
                <div style='font-size:2rem;'>{icon}</div>
                <strong style='font-size:1rem;'>{title}</strong>
                <p style='color:#7b7fa6;font-size:0.82rem;margin-top:0.4rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Quick-start columns ────────────────────────────────────────────────
    left, right = st.columns([1.8, 1.2], gap="large")

    with left:
        st.markdown("### 🚀 How it works")
        steps = [
            ("1", "Input", "Paste any movie plot description into the text area."),
            ("2", "Process", "The text is cleaned, lemmatized & vectorized with TF-IDF."),
            ("3", "Predict", "Our best ML model predicts the genre + confidence scores."),
            ("4", "Explore", "View top-3 genres, probability bars & save a report."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex;gap:1rem;align-items:flex-start;margin-bottom:0.9rem;'>
                <div style='min-width:2rem;height:2rem;border-radius:50%;
                            background:linear-gradient(135deg,#6C63FF,#9c94ff);
                            display:flex;align-items:center;justify-content:center;
                            color:white;font-weight:700;font-size:0.9rem;'>{num}</div>
                <div>
                    <strong>{title}</strong>
                    <p style='color:#7b7fa6;font-size:0.85rem;margin:0;'>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("### 🎭 Supported Genres")
        genres = [
            ("🔫", "Action"),    ("😄", "Comedy"),   ("😢", "Drama"),
            ("👻", "Horror"),    ("💕", "Romance"),   ("🚀", "Sci-Fi"),
            ("😰", "Thriller"),  ("✨", "Animation"), ("🎥", "Documentary"),
            ("🧙", "Fantasy"),
        ]
        g_cols = st.columns(2)
        for i, (emoji, genre) in enumerate(genres):
            with g_cols[i % 2]:
                st.markdown(
                    f"<div style='padding:0.4rem 0;color:#e8eaf6;'>"
                    f"{emoji} {genre}</div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ── Model status ───────────────────────────────────────────────────────
    model_path = os.path.join("artifacts", "models", "best_model.pkl")
    meta_path  = os.path.join("artifacts", "models", "model_meta.json")

    if os.path.exists(model_path):
        import json
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        st.success(
            f"✅  **Model loaded and ready!**  "
            f"Best model: **{meta.get('best_model', 'N/A')}**  |  "
            f"F1 Score: **{meta.get('best_f1', 0):.4f}**"
        )
    else:
        st.warning(
            "⚠️  No trained model found. "
            "Run `python train.py` to train the model before using the predictor."
        )

    # ── CTA ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c_l, c_c, c_r = st.columns([1, 1, 1])
    with c_c:
        if st.button("🎯  Go to Genre Predictor", use_container_width=True):
            st.session_state.page = "predictor"
            st.rerun()
