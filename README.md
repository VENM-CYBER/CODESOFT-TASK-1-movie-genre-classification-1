# 🎬 CineGenre AI — Movie Genre Classification System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B?logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

**An NLP-powered system that classifies movie genres from plot descriptions using classical ML algorithms.**

[Live Demo](#deployment) · [API Docs](#rest-api) · [Installation](#installation) · [Usage](#usage)

</div>

---

## 📌 Project Overview

CineGenre AI is a **production-ready NLP pipeline** that automatically predicts the genre of a movie (Action, Comedy, Drama, Horror, Romance, Sci-Fi, Thriller, Animation, Documentary, Fantasy) from a plain-text plot summary.

The system trains and compares three classical machine learning algorithms, automatically selects the best-performing model, and serves predictions through a premium dark-mode Streamlit web application and an optional FastAPI REST endpoint.

This project is designed to be:
- ✅ **Portfolio-grade** — clean code, full documentation, professional UI
- ✅ **Production-ready** — model serialisation, CI/CD, Docker, REST API
- ✅ **Internship-level** — demonstrates end-to-end ML engineering skills

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **3 ML Models** | Logistic Regression, Multinomial Naive Bayes, Linear SVM |
| 🔤 **Full NLP Pipeline** | Lowercase → clean → stopword removal → lemmatisation → TF-IDF |
| 🏆 **Auto Model Selection** | Best model chosen automatically by F1 score |
| 📊 **Rich EDA** | Genre distribution, word clouds, word frequencies, box plots |
| 🎯 **Confidence Scores** | Top-3 predictions with probability bars |
| 📜 **Prediction History** | Persisted across sessions with download |
| 📥 **Report Download** | Exportable plain-text prediction report |
| 🌐 **REST API** | FastAPI endpoints for integration |
| 🐳 **Docker** | Single-command deployment |
| ⚙️ **CI/CD** | GitHub Actions — lint, train, test, build |

---

## 🏗️ Architecture

```
Raw CSV
  │
  ▼
Data Loader & Validator
  │
  ▼
NLP Text Cleaner (NLTK)
  │  lowercase → urls → html → punctuation → numbers → stopwords → lemmatize
  ▼
TF-IDF Vectorizer (unigrams + bigrams, 30k features)
  │
  ├──▶ Logistic Regression
  ├──▶ Multinomial Naive Bayes
  └──▶ Linear SVM
            │
            ▼
      Auto-select best (F1 weighted)
            │
            ▼
  best_model.pkl + tfidf_vectorizer.pkl
            │
            ├──▶ Streamlit Web App
            └──▶ FastAPI REST API


## ⚙️ Installation

### Prerequisites
- Python 3.11+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/movie-genre-classification.git
cd movie-genre-classification
cd project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the model

```bash
# Auto-generates dataset, trains all models, saves artifacts
python train.py

# Skip EDA charts for faster training
python train.py --no-eda

# Use your own dataset
python train.py --data path/to/your/movies.csv
```

Your CSV must have columns: `movie_title`, `plot`, `genre`.

### Launch the Streamlit app

python -m streamlit run streamlit_app\main.py

Open http://localhost:8501

### CLI prediction
```bash
# Interactive mode
python predict.py

# Single prediction
python predict.py --plot "A soldier races to stop a nuclear attack in Tokyo." --top3

# Batch prediction (one plot per line)
python predict.py --file my_plots.txt
```

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1 (Weighted) | CV F1 Mean |
|-------|----------|-----------|--------|---------------|------------|
| Logistic Regression | ~0.87 | ~0.88 | ~0.87 | ~0.87 | ~0.86 |
| Multinomial Naive Bayes | ~0.81 | ~0.83 | ~0.81 | ~0.81 | ~0.80 |
| **Linear SVM** | **~0.89** | **~0.89** | **~0.89** | **~0.89** | **~0.88** |

> Actual results depend on dataset size and composition. Train on a larger corpus for higher scores.

---

## 🌐 REST API

Start the API server:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: http://localhost:8000/docs

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + model info |
| POST | `/predict` | Single genre prediction |
| POST | `/predict/top3` | Top-3 genres with confidence |
| POST | `/predict/all` | All genre probabilities |
| POST | `/predict/batch` | Batch prediction (up to 100) |

### Example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"plot": "A soldier races to stop a nuclear attack."}'

# Response:
# {"genre": "Action", "confidence": 0.8731}
```

---

## 🐳 Docker

```bash
# Build
docker build -t cinegenre-ai .

# Run
docker run -p 8501:8501 cinegenre-ai

# With environment variable
docker run -p 8501:8501 -p 8000:8000 cinegenre-ai
```

---

## ☁️ Streamlit Cloud Deployment

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set **Main file path** to: `streamlit_app/main.py`
5. Click **Deploy**

> **Note:** Streamlit Cloud runs `requirements.txt` automatically. NLTK data is downloaded at first run via the `_download_nltk()` call in `text_cleaner.py`.

---

## 🔬 Running Tests

```bash
# All tests (requires trained model)
pytest tests/ -v

# Text cleaning & data loader only (no model needed)
pytest tests/test_pipeline.py::TestTextCleaner tests/test_pipeline.py::TestDataLoader -v
```

---

## 🗺️ Future Enhancements

- [ ] **BERT / DistilBERT classifier** for higher accuracy
- [ ] **Multi-label genre prediction** (a film can be both Romance and Comedy)
- [ ] **Hugging Face model hub** integration
- [ ] **PostgreSQL** prediction history persistence
- [ ] **MLflow** experiment tracking
- [ ] **Active learning** loop for continuous improvement
- [ ] **Gradio** alternative frontend

---

## 📸 Screenshots

| Home Page | Genre Predictor | Model Performance |
|-----------|----------------|-------------------|
| Dark theme hero with feature cards | Large text area + confidence bars | Interactive model comparison dashboard |

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

Built as a production-quality portfolio ML project demonstrating:
- End-to-end NLP pipeline engineering
- Classical ML model comparison & selection
- Premium UI/UX with Streamlit
- REST API design with FastAPI
- DevOps: Docker + GitHub Actions CI/CD

**Connect:** [GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile)

---

<div align="center">
<strong>⭐ Star this repository if you found it useful!</strong>
</div>
