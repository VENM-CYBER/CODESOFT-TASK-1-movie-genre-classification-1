"""
streamlit_app/pages/about.py
------------------------------
About Project page for the CineGenre AI Streamlit app.
"""

import streamlit as st


def render():
    st.markdown("""
    <h1 style='font-family:"Bebas Neue",sans-serif;font-size:2.6rem;
               letter-spacing:0.06em;color:#6C63FF;margin-bottom:0;'>
        ℹ️ About This Project
    </h1>
    <p style='color:#7b7fa6;margin-top:0.3rem;'>
        Architecture, tech stack, and project information.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Overview ────────────────────────────────────────────────────────
    st.markdown("""
    ### 🎬 CineGenre AI — Movie Genre Classification System

    CineGenre AI is a production-ready Natural Language Processing (NLP) system that
    automatically predicts the genre of a movie based solely on its plot description.
    It compares three classical ML algorithms (Logistic Regression, Multinomial Naive Bayes,
    and Linear SVM) and selects the best-performing model automatically.

    The project is portfolio-grade and deployment-ready on **Streamlit Cloud**.
    """)

    # ── Architecture ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏗️ System Architecture")

    st.markdown("""
    <div class='card'>
    <pre style='color:#e8eaf6;font-size:0.82rem;line-height:1.7;'>
    ┌─────────────────────────────────────────────────────────┐
    │                  CineGenre AI Pipeline                  │
    └─────────────────────────────────────────────────────────┘

    ┌──────────┐    ┌──────────────┐    ┌──────────────────┐
    │  Raw CSV │───▶│ Data Loader  │───▶│  Text Cleaner    │
    │ (plots)  │    │ & Validator  │    │  (NLTK pipeline) │
    └──────────┘    └──────────────┘    └────────┬─────────┘
                                                 │
                                                 ▼
                                        ┌──────────────────┐
                                        │  TF-IDF          │
                                        │  Vectorizer      │
                                        │  (n-gram 1–2)    │
                                        └────────┬─────────┘
                                                 │
                         ┌───────────────────────┼────────────────────────┐
                         ▼                       ▼                        ▼
               ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
               │ Logistic         │   │ Multinomial      │   │ Linear SVM       │
               │ Regression       │   │ Naive Bayes      │   │ (LinearSVC)      │
               └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
                        └────────────────┬──────┘                      │
                                         └─────────────────────────────┘
                                                      │
                                                      ▼
                                           ┌──────────────────┐
                                           │  Auto-select     │
                                           │  Best Model      │
                                           │  (by F1 Score)   │
                                           └────────┬─────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │  best_model.pkl      │
                                         │  tfidf_vectorizer.pkl│
                                         └──────────────────────┘
    </pre>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech stack ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")

    c1, c2, c3 = st.columns(3)
    categories = [
        ("🐍 Backend", [
            ("Python 3.11+", "Core language"),
            ("Scikit-Learn", "ML algorithms & pipelines"),
            ("Pandas / NumPy", "Data manipulation"),
            ("NLTK", "Text preprocessing & lemmatization"),
            ("Joblib", "Model serialisation"),
        ]),
        ("🎨 Frontend", [
            ("Streamlit", "Web application framework"),
            ("Plotly", "Interactive charts"),
            ("Matplotlib / Seaborn", "Static EDA charts"),
            ("WordCloud", "Visualisation"),
            ("Custom CSS", "Dark-mode premium UI"),
        ]),
        ("🚀 Deployment", [
            ("Streamlit Cloud", "Primary hosting"),
            ("FastAPI", "REST API (optional)"),
            ("Docker", "Containerisation"),
            ("GitHub Actions", "CI/CD"),
            ("pytest", "Test suite"),
        ]),
    ]
    for col, (cat_title, items) in zip([c1, c2, c3], categories):
        with col:
            st.markdown(f"**{cat_title}**")
            for name, desc in items:
                st.markdown(
                    f"<div style='padding:0.3rem 0;'>"
                    f"<span style='color:#6C63FF;'>●</span> "
                    f"<strong>{name}</strong>"
                    f"<span style='color:#7b7fa6;font-size:0.82rem;'> — {desc}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── NLP pipeline ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔤 NLP Preprocessing Pipeline")
    steps = [
        ("1", "Lowercase", "Convert all text to lowercase for uniformity"),
        ("2", "Remove URLs", "Strip http/https links"),
        ("3", "Remove HTML", "Clean any embedded HTML tags"),
        ("4", "Remove Punctuation", "Remove all punctuation marks"),
        ("5", "Remove Numbers", "Strip standalone numeric tokens"),
        ("6", "Remove Stopwords", "Filter common English stopwords via NLTK"),
        ("7", "Lemmatize", "Reduce words to base form (e.g. 'running' → 'run')"),
        ("8", "TF-IDF", "Convert clean text to numerical feature matrix (1–2 ngrams)"),
    ]
    cols = st.columns(4)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(f"""
            <div class='card' style='text-align:center;min-height:110px;'>
                <div style='font-size:1.4rem;font-weight:800;color:#6C63FF;'>{num}</div>
                <strong style='font-size:0.9rem;'>{title}</strong>
                <p style='color:#7b7fa6;font-size:0.77rem;margin-top:0.3rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Folder structure ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📁 Project Structure")
    st.code("""
movie-genre-classification/
│
├── data/
│   ├── raw/                    # Raw CSV dataset
│   └── processed/              # Preprocessed dataset
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Data loading & cleaning
│   │   └── generate_dataset.py # Synthetic dataset generator
│   ├── preprocessing/
│   │   └── text_cleaner.py     # Full NLP cleaning pipeline
│   ├── models/
│   │   ├── trainer.py          # Training, evaluation & saving
│   │   └── predictor.py        # Prediction API
│   ├── utils/
│   │   └── helpers.py          # History, reports, sample plots
│   └── visualization/
│       └── eda.py              # All EDA charts
│
├── artifacts/
│   ├── models/                 # best_model.pkl, tfidf_vectorizer.pkl
│   └── reports/                # PNG charts + metrics JSON
│
├── streamlit_app/
│   ├── main.py                 # Multi-page app entry point
│   └── pages/                  # One module per page
│
├── tests/                      # pytest test suite
│
├── train.py                    # Full training pipeline script
├── predict.py                  # CLI prediction tool
├── app.py                      # Streamlit launcher
├── requirements.txt
└── README.md
    """, language="")

    # ── Author ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class='card' style='text-align:center;'>
        <h3 style='margin-bottom:0.5rem;'>👨‍💻 Author</h3>
        <p style='color:#7b7fa6;font-size:0.9rem;'>
            Built as a portfolio-quality ML project demonstrating NLP, Scikit-Learn, 
            and full-stack deployment skills.
        </p>
        <p style='margin-top:0.8rem;'>
            <a href='https://github.com' style='color:#6C63FF;text-decoration:none;'>
                🔗 GitHub
            </a>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <a href='https://linkedin.com' style='color:#6C63FF;text-decoration:none;'>
                💼 LinkedIn
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)
