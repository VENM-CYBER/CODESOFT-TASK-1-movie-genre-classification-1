# ── Dockerfile ────────────────────────────────────────────────────────────
# Movie Genre Classification System
# Build : docker build -t cinegenre-ai .
# Run   : docker run -p 8501:8501 cinegenre-ai
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="CineGenre AI"
LABEL description="NLP Movie Genre Classification System"

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Download NLTK data ─────────────────────────────────────────────────────
RUN python -c "\
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('omw-1.4', quiet=True)"

# ── Copy project files ─────────────────────────────────────────────────────
COPY . .

# ── Train model if artifacts don't exist ──────────────────────────────────
RUN python train.py --no-eda || echo "Training failed; pre-built artifacts may be missing."

# ── Expose Streamlit port ──────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Launch ─────────────────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "streamlit_app/main.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
