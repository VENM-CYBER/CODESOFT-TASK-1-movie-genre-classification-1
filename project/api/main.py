"""
api/main.py
------------
Optional FastAPI REST API for the Movie Genre Classification System.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                  — Health check
    POST /predict           — Predict single genre
    POST /predict/top3      — Top-3 genres
    POST /predict/all       — All genre probabilities
    POST /predict/batch     — Batch prediction
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from src.models.predictor import GenrePredictor
from src.utils.helpers    import load_model_meta

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CineGenre AI — Movie Genre Classification API",
    description="NLP-powered REST API that predicts movie genres from plot descriptions.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global predictor (loaded once) ────────────────────────────────────────
_predictor: Optional[GenrePredictor] = None


def get_predictor() -> GenrePredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = GenrePredictor()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _predictor


# ── Request / response schemas ─────────────────────────────────────────────

class PlotRequest(BaseModel):
    plot: str = Field(..., min_length=10, example="A soldier races to stop a nuclear threat in Tokyo.")


class BatchRequest(BaseModel):
    plots: list[str] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    genre:      str
    confidence: float


class Top3Response(BaseModel):
    predictions: list[PredictionResponse]


class AllProbsResponse(BaseModel):
    probabilities: dict[str, float]


class BatchResponse(BaseModel):
    results: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    meta = load_model_meta()
    return {
        "status":     "ok",
        "service":    "CineGenre AI",
        "best_model": meta.get("best_model", "unknown"),
        "best_f1":    meta.get("best_f1", None),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(req: PlotRequest):
    """Predict the primary genre for a movie plot."""
    p    = get_predictor()
    top3 = p.predict_top3_genres(req.plot)
    return {"genre": top3[0]["genre"], "confidence": top3[0]["confidence"]}


@app.post("/predict/top3", response_model=Top3Response, tags=["Prediction"])
def predict_top3(req: PlotRequest):
    """Return the top-3 most likely genres with confidence scores."""
    p    = get_predictor()
    top3 = p.predict_top3_genres(req.plot)
    return {"predictions": [{"genre": t["genre"], "confidence": t["confidence"]} for t in top3]}


@app.post("/predict/all", response_model=AllProbsResponse, tags=["Prediction"])
def predict_all_probs(req: PlotRequest):
    """Return probabilities for all genre classes."""
    p     = get_predictor()
    probs = p.predict_probability(req.plot)
    return {"probabilities": probs}


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(req: BatchRequest):
    """Predict genres for a list of plot descriptions."""
    p = get_predictor()
    results = p.batch_predict(req.plots)
    return {"results": results}
