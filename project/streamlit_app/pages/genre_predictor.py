"""
streamlit_app/pages/genre_predictor.py
----------------------------------------
Genre Predictor page for the CineGenre AI Streamlit app.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.utils.helpers import (
    SAMPLE_PLOTS, append_to_history,
    generate_prediction_report_text,
)


GENRE_COLORS = {
    "Action":      "#FF6584", "Comedy":      "#F5A623",
    "Drama":       "#4A90E2", "Horror":      "#CE93D8",
    "Romance":     "#FF9A9E", "Sci-Fi":      "#43B89C",
    "Thriller":    "#6C63FF", "Animation":   "#81C784",
    "Documentary": "#64B5F6", "Fantasy":     "#FFD54F",
}


@st.cache_resource(show_spinner="Loading model …")
def _load_predictor():
    from src.models.predictor import GenrePredictor
    return GenrePredictor()


def _confidence_bar(genre: str, confidence: float, rank: int) -> str:
    color  = GENRE_COLORS.get(genre, "#6C63FF")
    pct    = confidence * 100
    medals = ["🥇", "🥈", "🥉"]
    medal  = medals[rank] if rank < 3 else "  "
    return f"""
    <div style='margin:0.6rem 0;'>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.25rem;'>
            <span>{medal} <strong>{genre}</strong></span>
            <span style='color:{color};font-weight:700;'>{pct:.1f}%</span>
        </div>
        <div style='background:#252840;border-radius:999px;height:10px;overflow:hidden;'>
            <div style='width:{pct:.1f}%;height:100%;border-radius:999px;
                        background:linear-gradient(90deg,{color},white10);
                        transition:width 0.5s ease;'></div>
        </div>
    </div>
    """


def render():
    st.markdown("""
    <h1 style='font-family:"Bebas Neue",sans-serif;font-size:2.6rem;
               letter-spacing:0.06em;color:#6C63FF;margin-bottom:0;'>
        🎯 Genre Predictor
    </h1>
    <p style='color:#7b7fa6;margin-top:0.3rem;'>
        Enter a movie plot description and let the AI classify its genre.
    </p>
    """, unsafe_allow_html=True)

    # ── Model check ───────────────────────────────────────────────────────
    if not os.path.exists(os.path.join("artifacts", "models", "best_model.pkl")):
        st.error("❌  No trained model found. Run `python train.py` first.")
        return

    predictor = _load_predictor()

    # ── Sample plots selector ─────────────────────────────────────────────
    with st.expander("💡  Load a sample plot", expanded=False):
        selected_genre = st.selectbox("Pick a genre:", ["— choose —"] + sorted(SAMPLE_PLOTS.keys()))
        if selected_genre != "— choose —":
            if st.button("📋  Load this sample"):
                st.session_state["plot_input"] = SAMPLE_PLOTS[selected_genre]
                st.rerun()

    # ── Main input ────────────────────────────────────────────────────────
    default_text = st.session_state.get("plot_input", "")
    plot_text = st.text_area(
        "📝  Movie Plot Description",
        value=default_text,
        height=180,
        placeholder="e.g. A retired spy is pulled back into the field when a dangerous cartel kidnaps his daughter…",
        key="plot_textarea",
    )

    col_pred, col_clear = st.columns([2, 1])
    with col_pred:
        predict_clicked = st.button("🔍  Predict Genre", use_container_width=True)
    with col_clear:
        if st.button("🗑️  Clear", use_container_width=True):
            st.session_state["plot_input"] = ""
            st.rerun()

    # ── Prediction results ────────────────────────────────────────────────
    if predict_clicked and plot_text.strip():
        with st.spinner("Analysing …"):
            predicted    = predictor.predict_genre(plot_text)
            top3         = predictor.predict_top3_genres(plot_text)
            all_probs    = predictor.predict_probability(plot_text)

        # Append to history
        append_to_history(plot_text, predicted, top3)

        st.markdown("---")
        r1, r2 = st.columns([1.2, 1.8], gap="large")

        # ── Left: primary prediction ───────────────────────────────────
        with r1:
            color = GENRE_COLORS.get(predicted, "#6C63FF")
            conf  = top3[0]["confidence"] * 100
            st.markdown(f"""
            <div class='card' style='text-align:center;border-top:3px solid {color};'>
                <p style='color:#7b7fa6;font-size:0.85rem;margin-bottom:0.4rem;'>PRIMARY PREDICTION</p>
                <div style='font-size:3rem;margin:0.5rem 0;'>
                    {_genre_emoji(predicted)}
                </div>
                <div class='genre-badge' style='background:linear-gradient(135deg,{color},{color}aa);'>
                    {predicted.upper()}
                </div>
                <p style='margin-top:1rem;font-size:0.9rem;color:#7b7fa6;'>
                    Confidence: <strong style='color:{color};'>{conf:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Right: top-3 + all probabilities ──────────────────────────
        with r2:
            st.markdown("#### 🏆 Top 3 Predictions")
            bars_html = "".join(_confidence_bar(item["genre"], item["confidence"], i)
                                for i, item in enumerate(top3))
            st.markdown(f"<div class='card'>{bars_html}</div>", unsafe_allow_html=True)

        # ── Full probability chart ─────────────────────────────────────
        st.markdown("#### 📊 All Genre Probabilities")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        labels = [g for g, _ in sorted_probs]
        values = [p * 100 for _, p in sorted_probs]
        bar_colors = [GENRE_COLORS.get(g, "#6C63FF") for g in labels]

        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=bar_colors,
            marker_line_width=0,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eaf6", height=320,
            yaxis=dict(title="Confidence (%)", gridcolor="#252840", range=[0, max(values)*1.18]),
            xaxis=dict(gridcolor="#252840"),
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Download report ────────────────────────────────────────────
        report = generate_prediction_report_text(plot_text, predicted, top3, all_probs)
        st.download_button(
            "📥  Download Prediction Report",
            data=report,
            file_name=f"genre_report_{predicted.lower()}.txt",
            mime="text/plain",
        )

    elif predict_clicked:
        st.warning("Please enter a plot description before predicting.")

    # ── Word count ────────────────────────────────────────────────────────
    if plot_text.strip():
        wc = len(plot_text.split())
        st.caption(f"Word count: {wc}")


def _genre_emoji(genre: str) -> str:
    mapping = {
        "Action": "🔫", "Comedy": "😄", "Drama": "😢", "Horror": "👻",
        "Romance": "💕", "Sci-Fi": "🚀", "Thriller": "😰",
        "Animation": "✨", "Documentary": "🎥", "Fantasy": "🧙",
    }
    return mapping.get(genre, "🎬")
