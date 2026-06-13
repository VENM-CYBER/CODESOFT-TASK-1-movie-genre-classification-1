"""
streamlit_app/pages/model_performance.py
------------------------------------------
Model Performance page for the CineGenre AI Streamlit app.
"""

import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from src.utils.helpers import load_model_metrics, load_model_meta, load_history


PALETTE = ["#6C63FF", "#FF6584", "#43B89C"]


def render():
    st.markdown("""
    <h1 style='font-family:"Bebas Neue",sans-serif;font-size:2.6rem;
               letter-spacing:0.06em;color:#6C63FF;margin-bottom:0;'>
        📈 Model Performance
    </h1>
    <p style='color:#7b7fa6;margin-top:0.3rem;'>
        Compare all trained models and explore evaluation metrics.
    </p>
    """, unsafe_allow_html=True)

    metrics = load_model_metrics()
    meta    = load_model_meta()

    if not metrics:
        st.warning("No model metrics found. Run `python train.py` to train the models.")
        return

    model_names  = list(metrics.keys())
    best_name    = meta.get("best_model", model_names[0])

    # ── KPI row ────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    best = metrics[best_name]
    k1.metric("Best Model",    best_name.split()[0] + " " + best_name.split()[1] if len(best_name.split()) > 1 else best_name)
    k2.metric("Accuracy",      f"{best['accuracy']*100:.1f}%")
    k3.metric("F1 (Weighted)", f"{best['f1_weighted']*100:.1f}%")
    k4.metric("CV F1 (Mean)",  f"{best['cv_f1_mean']*100:.1f}%")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Comparison", "📋 Classification Report", "🔲 Confusion Matrix", "📜 Prediction History"
    ])

    # ── Tab 1: Comparison ─────────────────────────────────────────────────
    with tab1:
        metric_keys = ["accuracy", "precision", "recall", "f1_weighted", "cv_f1_mean"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1 Weighted", "CV F1 Mean"]

        # Bar chart
        fig = go.Figure()
        for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            values = [metrics[n][key] for n in model_names]
            fig.add_trace(go.Bar(
                name=label, x=model_names, y=values,
                marker_color=PALETTE[i % len(PALETTE)],
                text=[f"{v:.3f}" for v in values], textposition="outside",
            ))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eaf6", height=420,
            yaxis=dict(title="Score", range=[0, 1.15], gridcolor="#252840"),
            xaxis=dict(gridcolor="#252840"),
            title="Model Comparison Dashboard",
            title_font_size=15,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        rows = []
        for name in model_names:
            m = metrics[name]
            rows.append({
                "Model":            name,
                "Accuracy":         f"{m['accuracy']:.4f}",
                "Precision":        f"{m['precision']:.4f}",
                "Recall":           f"{m['recall']:.4f}",
                "F1 Weighted":      f"{m['f1_weighted']:.4f}",
                "CV F1 Mean":       f"{m['cv_f1_mean']:.4f}",
                "CV F1 ±Std":       f"{m['cv_f1_std']:.4f}",
                "Best": "✅" if name == best_name else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Tab 2: Classification report ─────────────────────────────────────
    with tab2:
        selected_model = st.selectbox("Select model:", model_names,
                                       index=model_names.index(best_name))
        report = metrics[selected_model].get("classification_report", {})
        if report:
            rows = []
            for label, vals in report.items():
                if isinstance(vals, dict):
                    rows.append({
                        "Class":     label,
                        "Precision": f"{vals.get('precision', 0):.3f}",
                        "Recall":    f"{vals.get('recall', 0):.3f}",
                        "F1-Score":  f"{vals.get('f1-score', 0):.3f}",
                        "Support":   int(vals.get("support", 0)),
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Classification report not available.")

    # ── Tab 3: Confusion matrix ────────────────────────────────────────────
    with tab3:
        cm_model = st.selectbox("Select model  ", model_names,
                                 index=model_names.index(best_name), key="cm_model")
        cm_data  = metrics[cm_model].get("confusion_matrix", [])
        if cm_data:
            cm       = np.array(cm_data)
            # Get class labels from classification report
            cr_keys  = [k for k in metrics[cm_model]["classification_report"].keys()
                        if k not in ("accuracy", "macro avg", "weighted avg")]
            if len(cr_keys) != cm.shape[0]:
                cr_keys = [str(i) for i in range(cm.shape[0])]

            cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            view = st.radio("View", ["Counts", "Normalised"], horizontal=True)
            data = cm if view == "Counts" else cm_norm
            fmt  = ".0f" if view == "Counts" else ".2f"

            fig_cm = px.imshow(
                data,
                x=cr_keys, y=cr_keys,
                color_continuous_scale="Blues",
                text_auto=fmt,
                labels=dict(x="Predicted", y="True"),
                title=f"Confusion Matrix — {cm_model} ({view})",
                aspect="auto",
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e8eaf6", height=550,
                margin=dict(l=60, r=20, t=50, b=60),
                coloraxis_showscale=False,
            )
            fig_cm.update_xaxes(tickangle=40)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Confusion matrix not available.")

        # Show saved image if present
        img_path = os.path.join("artifacts", "reports", "confusion_matrix.png")
        if os.path.exists(img_path):
            with st.expander("View full-resolution saved chart"):
                st.image(img_path, use_column_width=True)

    # ── Tab 4: Prediction history ─────────────────────────────────────────
    with tab4:
        history = load_history()
        if not history:
            st.info("No predictions made yet. Go to the Genre Predictor page and make some predictions!")
        else:
            st.markdown(f"**{len(history)}** predictions recorded")
            rows = []
            for entry in reversed(history):
                top1 = entry["top3"][0] if entry.get("top3") else {}
                rows.append({
                    "Timestamp":   entry.get("timestamp", ""),
                    "Genre":       entry.get("predicted_genre", ""),
                    "Confidence":  f"{top1.get('confidence', 0)*100:.1f}%",
                    "Plot Snippet": entry.get("plot_snippet", "")[:80],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Genre distribution of predictions
            pred_counts = pd.Series(
                [r["Genre"] for r in rows]
            ).value_counts().reset_index()
            pred_counts.columns = ["Genre", "Count"]
            fig_hist = px.bar(
                pred_counts, x="Genre", y="Count",
                color="Genre", color_discrete_sequence=PALETTE * 4,
                title="Prediction History — Genre Distribution",
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8eaf6", showlegend=False,
                xaxis=dict(gridcolor="#252840"),
                yaxis=dict(gridcolor="#252840"),
                height=300, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            if st.button("🗑️  Clear History"):
                from src.utils.helpers import clear_history
                clear_history()
                st.success("History cleared.")
                st.rerun()
