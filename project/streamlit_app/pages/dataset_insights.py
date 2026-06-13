"""
streamlit_app/pages/dataset_insights.py
-----------------------------------------
Dataset Insights page for the CineGenre AI Streamlit app.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PROCESSED_PATH = os.path.join("data", "processed", "movies_processed.csv")
RAW_PATH       = os.path.join("data", "raw", "movies.csv")

PALETTE = [
    "#6C63FF", "#FF6584", "#43B89C", "#F5A623", "#4A90E2",
    "#E57373", "#64B5F6", "#81C784", "#FFD54F", "#CE93D8",
]


@st.cache_data(show_spinner="Loading dataset …")
def _load():
    path = PROCESSED_PATH if os.path.exists(PROCESSED_PATH) else RAW_PATH
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def render():
    st.markdown("""
    <h1 style='font-family:"Bebas Neue",sans-serif;font-size:2.6rem;
               letter-spacing:0.06em;color:#6C63FF;margin-bottom:0;'>
        📊 Dataset Insights
    </h1>
    <p style='color:#7b7fa6;margin-top:0.3rem;'>
        Explore the movie dataset used to train the genre classifier.
    </p>
    """, unsafe_allow_html=True)

    df = _load()
    if df is None:
        st.warning("Dataset not found. Run `python train.py` to generate and process data.")
        return

    df["word_count"] = df["plot"].str.split().apply(len)

    # ── KPI row ───────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Movies",    f"{len(df):,}")
    k2.metric("Unique Genres",   df["genre"].nunique())
    k3.metric("Avg Plot Length", f"{df['word_count'].mean():.0f} words")
    k4.metric("Max Plot Length", f"{df['word_count'].max()} words")

    st.markdown("---")

    # ── Genre distribution ────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Genre Distribution", "📏 Plot Length", "🔍 Data Explorer"])

    with tab1:
        counts = df["genre"].value_counts().reset_index()
        counts.columns = ["genre", "count"]

        c_l, c_r = st.columns(2, gap="large")

        with c_l:
            fig_bar = px.bar(
                counts, x="count", y="genre", orientation="h",
                color="genre", color_discrete_sequence=PALETTE,
                title="Movies per Genre",
                text="count",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8eaf6", showlegend=False,
                yaxis=dict(title=""), xaxis=dict(title="Count", gridcolor="#252840"),
                title_font_size=14, height=380,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_r:
            fig_pie = px.pie(
                counts, values="count", names="genre",
                color_discrete_sequence=PALETTE,
                title="Genre Proportions",
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e8eaf6",
                title_font_size=14, height=380,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        c_l, c_r = st.columns(2, gap="large")

        with c_l:
            fig_hist = px.histogram(
                df, x="word_count", nbins=40,
                color_discrete_sequence=["#6C63FF"],
                title="Overall Word-Count Distribution",
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8eaf6", bargap=0.05,
                xaxis=dict(title="Words", gridcolor="#252840"),
                yaxis=dict(title="Frequency", gridcolor="#252840"),
                title_font_size=14, height=360,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with c_r:
            fig_box = px.box(
                df, x="genre", y="word_count",
                color="genre", color_discrete_sequence=PALETTE,
                title="Word Count by Genre",
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8eaf6", showlegend=False,
                xaxis=dict(title="Genre", tickangle=35, gridcolor="#252840"),
                yaxis=dict(title="Word Count", gridcolor="#252840"),
                title_font_size=14, height=360,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        # Filters
        f1, f2, f3 = st.columns([1.5, 1, 1])
        with f1:
            search = st.text_input("🔍 Search plot text", "")
        with f2:
            genre_filter = st.multiselect(
                "Filter by genre",
                sorted(df["genre"].unique()),
                default=[],
            )
        with f3:
            wc_range = st.slider(
                "Word count range",
                int(df["word_count"].min()), int(df["word_count"].max()),
                (int(df["word_count"].min()), int(df["word_count"].max())),
            )

        mask = (
            (df["word_count"] >= wc_range[0]) &
            (df["word_count"] <= wc_range[1])
        )
        if genre_filter:
            mask &= df["genre"].isin(genre_filter)
        if search.strip():
            mask &= df["plot"].str.lower().str.contains(search.lower(), na=False)

        filtered = df[mask][["movie_title", "genre", "word_count", "plot"]].reset_index(drop=True)
        st.markdown(f"**{len(filtered):,}** movies match your filters")
        st.dataframe(filtered, use_container_width=True, height=400)

    # ── Saved charts ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🖼️  Saved EDA Charts")
    charts = {
        "Genre Distribution":       "genre_distribution.png",
        "Plot Length Distribution": "plot_length_distribution.png",
        "Top Words per Genre":      "top_words_per_genre.png",
        "Word Cloud":               "word_cloud.png",
    }
    chart_cols = st.columns(2)
    for i, (title, fname) in enumerate(charts.items()):
        path = os.path.join("artifacts", "reports", fname)
        with chart_cols[i % 2]:
            if os.path.exists(path):
                st.image(path, caption=title, use_column_width=True)
            else:
                st.info(f"'{fname}' not found. Run train.py to generate.")
